//! Transport observables and statistical estimators for the single-vacancy Fe–Cr model.
//!
//! Tracer coefficients are `D*_a = slope(Sum_a |dr|^2 / N_a) / 6`, vacancy diffusivity
//! is `D_v = slope(|dR_v|^2) / 6`, and collective Onsager-related coefficients per vacancy
//! (`Lambda = L k_B T Omega / c_v`, `c_v = 1/N`) are `Lambda_ab = slope(dR_a . dR_b) / 6`.
//! The correlation factor `f = N_atoms * mean(D*_all) / mean(D_v)` is estimated as a ratio
//! of ensemble means with delete-1 jackknife standard error.

use std::collections::BTreeMap;
use std::error::Error;
use std::fmt;

use okrates_core::{
    AbsorptionPolicy, Checkpointable, GridError, SampleGrid, SampledTrajectory, SamplingError,
    Simulation, StreamRole, run_ensemble, sample_trajectory,
};
use okrates_schema::{Conditions, EstimatorSpec, RngPolicy, RunManifest};

use crate::fecr::{Energetics, FeCrError, FeCrModel, FeCrState, Species};
use crate::fecr_anneal::KawasakiAnnealer;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum FitMethod {
    #[default]
    WindowedOls,
    OriginSlope,
}

impl FitMethod {
    pub fn name(self) -> &'static str {
        match self {
            Self::WindowedOls => "windowed_ols",
            Self::OriginSlope => "origin_slope",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TransportConfig {
    pub t_equil: f64,
    pub t_end: f64,
    pub sample_count: usize,
    pub lattice_parameter: f64,
    pub fit_method: FitMethod,
    pub max_steps: Option<u64>,
}

impl TransportConfig {
    pub fn windowed(t_equil: f64, t_end: f64, sample_count: usize, lattice_parameter: f64) -> Self {
        Self {
            t_equil,
            t_end,
            sample_count,
            lattice_parameter,
            fit_method: FitMethod::WindowedOls,
            max_steps: None,
        }
    }

    pub fn from_origin(t_end: f64, sample_count: usize, lattice_parameter: f64) -> Self {
        Self {
            t_equil: 0.0,
            t_end,
            sample_count,
            lattice_parameter,
            fit_method: FitMethod::OriginSlope,
            max_steps: None,
        }
    }
}

#[derive(Debug)]
pub enum TransportError {
    Model(FeCrError),
    InvalidLatticeParameter(f64),
    InsufficientSamples {
        fit_method: FitMethod,
        sample_count: usize,
    },
    InsufficientReplicas(u64),
    Sampling(SamplingError<FeCrError>),
}

impl fmt::Display for TransportError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Model(err) => write!(f, "{err}"),
            Self::InvalidLatticeParameter(a) => {
                write!(f, "lattice_parameter must be finite and positive, got {a}")
            }
            Self::InsufficientSamples {
                fit_method,
                sample_count,
            } => write!(
                f,
                "fit method {} requires more samples than {sample_count}",
                fit_method.name()
            ),
            Self::InsufficientReplicas(runs) => {
                write!(
                    f,
                    "ensemble summary requires at least 2 replicas, got {runs}"
                )
            }
            Self::Sampling(err) => write!(f, "{err}"),
        }
    }
}

impl Error for TransportError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Model(err) => Some(err),
            Self::Sampling(err) => Some(err),
            Self::InvalidLatticeParameter(_)
            | Self::InsufficientSamples { .. }
            | Self::InsufficientReplicas(_) => None,
        }
    }
}

impl From<FeCrError> for TransportError {
    fn from(err: FeCrError) -> Self {
        Self::Model(err)
    }
}

impl From<SamplingError<FeCrError>> for TransportError {
    fn from(err: SamplingError<FeCrError>) -> Self {
        Self::Sampling(err)
    }
}

impl From<GridError> for TransportError {
    fn from(err: GridError) -> Self {
        Self::Sampling(SamplingError::Grid(err))
    }
}

/// Squared and collective dot-product displacements in `lattice_parameter^2` units.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Observables {
    pub tracer: [f64; 2],
    pub tracer_all: f64,
    /// `[dR_Fe.dR_Fe, dR_Fe.dR_Cr, dR_Cr.dR_Cr]`.
    pub collective: [f64; 3],
    pub s: f64,
    pub vacancy: f64,
}

fn dot(a: [i128; 3], b: [i128; 3]) -> f64 {
    (0..3).map(|axis| (a[axis] as f64) * (b[axis] as f64)).sum()
}

pub fn observables(state: &FeCrState, lattice_parameter: f64) -> Observables {
    let scale = lattice_parameter * lattice_parameter / 4.0;
    let [n_fe, n_cr, _] = state.counts();
    let sum_fe = state.tracer_squared_sum(Species::Fe);
    let sum_cr = state.tracer_squared_sum(Species::Cr);
    let per_atom = |sum: f64, count: usize| {
        if count == 0 {
            0.0
        } else {
            sum / count as f64 * scale
        }
    };
    let n_atoms = n_fe + n_cr;
    let r_fe = state.collective_displacement(Species::Fe);
    let r_cr = state.collective_displacement(Species::Cr);
    let vac = state.vacancy_displacement().map(i128::from);
    Observables {
        tracer: [per_atom(sum_fe, n_fe), per_atom(sum_cr, n_cr)],
        tracer_all: per_atom(sum_fe + sum_cr, n_atoms),
        collective: [
            dot(r_fe, r_fe) * scale,
            dot(r_fe, r_cr) * scale,
            dot(r_cr, r_cr) * scale,
        ],
        s: (sum_fe + sum_cr) * scale,
        vacancy: dot(vac, vac) * scale,
    }
}

/// Lazily samples [`observables`] on `grid` without evaluating displacements on intermediate jumps
/// (mid-trajectory errors such as `MaxStepsExceeded` leave `sim` at the last completed jump).
pub fn sample_fecr_run<E: Energetics>(
    sim: &mut Simulation<FeCrModel<E>>,
    grid: &SampleGrid,
    lattice_parameter: f64,
    max_steps: Option<u64>,
) -> Result<SampledTrajectory<Observables>, TransportError> {
    if !lattice_parameter.is_finite() || lattice_parameter <= 0.0 {
        return Err(TransportError::InvalidLatticeParameter(lattice_parameter));
    }
    Ok(sample_trajectory(
        sim,
        grid,
        max_steps,
        AbsorptionPolicy::Error,
        |_model, state| observables(state, lattice_parameter),
    )?)
}

/// Samples [`observables`] at `t_k = k * t_end / k_count` (`k = 1..=k_count`), returning `(times, samples, fired_steps)`.
pub fn sample_run<E: Energetics>(
    model: FeCrModel<E>,
    state: FeCrState,
    seed: u64,
    t_end: f64,
    k_count: usize,
    lattice_parameter: f64,
) -> Result<(Vec<f64>, Vec<Observables>, u64), TransportError> {
    let grid = SampleGrid::from_origin(t_end, k_count)?;
    let mut sim = Simulation::new(model, state, seed);
    let traj = sample_fecr_run(&mut sim, &grid, lattice_parameter, None)?;
    Ok((traj.times, traj.values, traj.fired_steps))
}

/// Through-origin slope `Sum(t_i v_i) / Sum(t_i^2)` (`NaN` on empty or mismatched slices).
pub fn origin_slope(times: &[f64], values: &[f64]) -> f64 {
    if times.is_empty() || times.len() != values.len() {
        return f64::NAN;
    }
    let num: f64 = times.iter().zip(values).map(|(t, v)| t * v).sum();
    let den: f64 = times.iter().map(|t| t * t).sum();
    num / den
}

/// Ordinary least-squares slope with free intercept (`NaN` if `< 2` points or mismatched slices).
pub fn ols_slope(times: &[f64], values: &[f64]) -> f64 {
    if times.len() < 2 || times.len() != values.len() {
        return f64::NAN;
    }
    let n = times.len() as f64;
    let mean_t = times.iter().sum::<f64>() / n;
    let mean_v = values.iter().sum::<f64>() / n;
    let num: f64 = times
        .iter()
        .zip(values)
        .map(|(t, v)| (t - mean_t) * (v - mean_v))
        .sum();
    let den: f64 = times.iter().map(|t| (t - mean_t).powi(2)).sum();
    num / den
}

/// Sample mean and standard error `s / sqrt(n)` with Bessel correction `n - 1`.
pub fn mean_se(values: &[f64]) -> (f64, f64) {
    if values.is_empty() {
        return (f64::NAN, f64::NAN);
    }
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0);
    (mean, (variance / n).sqrt())
}

/// Ratio-of-means `scale * mean(u) / mean(v)` and delete-1 jackknife standard error.
pub fn jackknife_ratio_mean_se(numerators: &[f64], denominators: &[f64], scale: f64) -> (f64, f64) {
    let r = numerators.len();
    if r == 0 || r != denominators.len() {
        return (f64::NAN, f64::NAN);
    }
    let sum_u: f64 = numerators.iter().sum();
    let sum_v: f64 = denominators.iter().sum();
    if sum_v == 0.0 || !sum_u.is_finite() || !sum_v.is_finite() {
        return (f64::NAN, f64::NAN);
    }
    let ratio = scale * (sum_u / sum_v);
    if r < 2 {
        return (ratio, f64::NAN);
    }
    let mut loo = Vec::with_capacity(r);
    for (&u, &v) in numerators.iter().zip(denominators) {
        let den_minus = sum_v - v;
        if den_minus == 0.0 {
            return (ratio, f64::NAN);
        }
        loo.push(scale * ((sum_u - u) / den_minus));
    }
    let mean_loo = loo.iter().sum::<f64>() / r as f64;
    let factor = (r - 1) as f64 / r as f64;
    let var = factor * loo.iter().map(|&x| (x - mean_loo).powi(2)).sum::<f64>();
    (ratio, var.sqrt())
}

/// Per-replica diffusivity estimates (`slope / 6`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ReplicaEstimate {
    pub tracer: [f64; 2],
    pub tracer_all: f64,
    pub collective: [f64; 3],
    pub s: f64,
    pub vacancy: f64,
}

pub fn estimate(
    times: &[f64],
    samples: &[Observables],
    slope: fn(&[f64], &[f64]) -> f64,
) -> ReplicaEstimate {
    let fit = |f: &dyn Fn(&Observables) -> f64| {
        let values: Vec<f64> = samples.iter().map(f).collect();
        slope(times, &values) / 6.0
    };
    ReplicaEstimate {
        tracer: [fit(&|o| o.tracer[0]), fit(&|o| o.tracer[1])],
        tracer_all: fit(&|o| o.tracer_all),
        collective: [
            fit(&|o| o.collective[0]),
            fit(&|o| o.collective[1]),
            fit(&|o| o.collective[2]),
        ],
        s: fit(&|o| o.s),
        vacancy: fit(&|o| o.vacancy),
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ReplicaRun {
    pub estimate: ReplicaEstimate,
    pub early_tracer_all: Option<f64>,
    pub late_tracer_all: Option<f64>,
    pub fired: u64,
    pub anneal_energy: Vec<f64>,
    pub final_sro_alpha1: Option<f64>,
}

/// Runs `runs` parallel replicas using domain-separated [`StreamRole`] RNG substreams and `config`.
pub fn run_transport<E: Energetics + Clone + Sync>(
    model: &FeCrModel<E>,
    cr_count: usize,
    seed_base: u64,
    runs: u64,
    config: &TransportConfig,
    anneal_sweeps: u64,
) -> Result<Vec<ReplicaRun>, TransportError> {
    if !config.lattice_parameter.is_finite() || config.lattice_parameter <= 0.0 {
        return Err(TransportError::InvalidLatticeParameter(
            config.lattice_parameter,
        ));
    }
    let min_samples = match config.fit_method {
        FitMethod::WindowedOls => 2,
        FitMethod::OriginSlope => 1,
    };
    if config.sample_count < min_samples {
        return Err(TransportError::InsufficientSamples {
            fit_method: config.fit_method,
            sample_count: config.sample_count,
        });
    }
    if cr_count >= model.lattice().site_count() {
        return Err(TransportError::Model(FeCrError::TooManyAtoms));
    }
    let grid = SampleGrid::windowed(config.t_equil, config.t_end, config.sample_count)?;
    let slope_fn: fn(&[f64], &[f64]) -> f64 = match config.fit_method {
        FitMethod::WindowedOls => ols_slope,
        FitMethod::OriginSlope => origin_slope,
    };

    run_ensemble(
        seed_base,
        runs,
        |ctx| -> Result<ReplicaRun, TransportError> {
            let mut init_rng = ctx.rng(StreamRole::InitialState);
            let mut state = model.random_state_with_rng(cr_count, &mut init_rng)?;
            let mut anneal_energy = Vec::new();
            let mut final_sro_alpha1 = None;
            if anneal_sweeps > 0 {
                let mut annealer = KawasakiAnnealer::new(
                    model,
                    state.occupancy().to_vec(),
                    ctx.rng(StreamRole::Equilibration),
                )?;
                annealer.run_sweeps(anneal_sweeps)?;
                let a = annealer.finish();
                final_sro_alpha1 = Some(a.diagnostics.final_sro_alpha1);
                anneal_energy = a.energy;
                state = model.state(a.occupancy)?;
            }
            let mut sim = Simulation::with_rng(model.clone(), state, ctx.rng(StreamRole::Dynamics));
            let traj =
                sample_fecr_run(&mut sim, &grid, config.lattice_parameter, config.max_steps)?;
            let est = estimate(&traj.times, &traj.values, slope_fn);
            let (early_tracer_all, late_tracer_all) = if traj.times.len() >= 4 {
                let mid = traj.times.len() / 2;
                let early = estimate(&traj.times[..mid], &traj.values[..mid], ols_slope).tracer_all;
                let late = estimate(&traj.times[mid..], &traj.values[mid..], ols_slope).tracer_all;
                (Some(early), Some(late))
            } else {
                (None, None)
            };
            Ok(ReplicaRun {
                estimate: est,
                early_tracer_all,
                late_tracer_all,
                fired: traj.fired_steps,
                anneal_energy,
                final_sro_alpha1,
            })
        },
    )
    .into_iter()
    .collect()
}

/// Convenience wrapper running `runs` replicas on `[0, t_end]` with [`FitMethod::OriginSlope`].
#[allow(clippy::too_many_arguments)]
pub fn run_composition<E: Energetics + Clone + Sync>(
    model: &FeCrModel<E>,
    cr_count: usize,
    seed_base: u64,
    runs: u64,
    t_end: f64,
    k_count: usize,
    lattice_parameter: f64,
    anneal_sweeps: u64,
) -> Result<Vec<ReplicaRun>, TransportError> {
    let config = TransportConfig::from_origin(t_end, k_count, lattice_parameter);
    run_transport(model, cr_count, seed_base, runs, &config, anneal_sweeps)
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Summary {
    pub vacancy: (f64, f64),
    pub tracer: [(f64, f64); 2],
    pub tracer_all: (f64, f64),
    pub collective: [(f64, f64); 3],
    /// Consistent ratio-of-means `n_atoms * mean(D*_all) / mean(D_v)` and delete-1 jackknife SE.
    pub f: (f64, f64),
    pub mean_jumps: f64,
    /// Relative drift `(mean_late - mean_early) / |mean_late|` of `D*_all` (`NaN` if `< 4` samples).
    pub window_drift_ratio: f64,
}

pub fn summarize(runs: &[ReplicaRun], n_cr: usize, n_atoms: usize) -> Summary {
    let column = |f: &dyn Fn(&ReplicaEstimate) -> f64| {
        let values: Vec<f64> = runs.iter().map(|r| f(&r.estimate)).collect();
        mean_se(&values)
    };
    let numerators: Vec<f64> = runs.iter().map(|r| r.estimate.tracer_all).collect();
    let denominators: Vec<f64> = runs.iter().map(|r| r.estimate.vacancy).collect();
    let window_drift_ratio = if runs
        .iter()
        .all(|r| r.early_tracer_all.is_some() && r.late_tracer_all.is_some())
        && !runs.is_empty()
    {
        let mean_early =
            runs.iter().filter_map(|r| r.early_tracer_all).sum::<f64>() / runs.len() as f64;
        let mean_late =
            runs.iter().filter_map(|r| r.late_tracer_all).sum::<f64>() / runs.len() as f64;
        if mean_late.abs() > 0.0 {
            (mean_late - mean_early) / mean_late.abs()
        } else {
            f64::NAN
        }
    } else {
        f64::NAN
    };

    Summary {
        vacancy: column(&|e| e.vacancy),
        tracer: [
            column(&|e| e.tracer[0]),
            if n_cr == 0 {
                (f64::NAN, f64::NAN)
            } else {
                column(&|e| e.tracer[1])
            },
        ],
        tracer_all: column(&|e| e.tracer_all),
        collective: [
            column(&|e| e.collective[0]),
            column(&|e| e.collective[1]),
            column(&|e| e.collective[2]),
        ],
        f: jackknife_ratio_mean_se(&numerators, &denominators, n_atoms as f64),
        mean_jumps: runs.iter().map(|r| u128::from(r.fired)).sum::<u128>() as f64
            / runs.len() as f64,
        window_drift_ratio,
    }
}

/// Builds a [`RunManifest`] recording model parameter digest, optional barrier digest, RNG policy, and estimator config.
#[allow(clippy::too_many_arguments)]
pub fn build_fecr_manifest<M: Checkpointable>(
    model: &M,
    cells: [usize; 3],
    cr_count: usize,
    temperature_k: f64,
    seed_base: u64,
    runs: u64,
    config: &TransportConfig,
    anneal_sweeps: u64,
    barrier_set_sha256: Option<String>,
) -> RunManifest {
    let mut encoded_params = Vec::new();
    model.encode_params(&mut encoded_params);
    let n_sites = 2_usize
        .saturating_mul(cells[0])
        .saturating_mul(cells[1])
        .saturating_mul(cells[2]);
    let n_fe = n_sites.saturating_sub(cr_count).saturating_sub(1);
    let mut composition = BTreeMap::new();
    composition.insert("Fe".to_string(), n_fe);
    composition.insert("Cr".to_string(), cr_count);
    composition.insert("Vacancy".to_string(), 1);
    let mut manifest = RunManifest::new(
        M::MODEL_ID,
        &encoded_params,
        "direct",
        RngPolicy::chacha12_substreams(seed_base),
        Conditions {
            cells,
            lattice_parameter: config.lattice_parameter,
            temperature_k,
            composition,
        },
    );
    manifest.barrier_set_sha256 = barrier_set_sha256;
    manifest.estimator = Some(EstimatorSpec {
        fit_method: config.fit_method.name().to_string(),
        t_equil: config.t_equil,
        t_end: config.t_end,
        sample_count: config.sample_count,
        replicas: runs,
        anneal_sweeps,
    });
    manifest
}
