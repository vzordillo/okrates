//! Diffusion observables and estimators for the single-vacancy Fe–Cr model.
//!
//! Tracer coefficients are D*_α = slope(Σ_α |Δr|² / N_α)/6 and the vacancy
//! coefficient is D_v = slope(|ΔR_v|²)/6. The collective coefficient per vacancy
//! is Λ_αβ = slope(ΔR_α·ΔR_β)/6, where ΔR_α is the summed displacement of all
//! atoms of species α. It is related to the Onsager coefficient by
//! Λ = L·k_B·T·Ω/c_v with the vacancy concentration c_v = 1/N; details are in the
//! local theory notes.

use okrates_core::ensemble::run_replicas;
use okrates_core::{Simulation, Step};

use crate::fecr::{Energetics, FeCrError, FeCrModel, FeCrState, Species};
use crate::fecr_anneal::anneal;

/// Offset between a replica's dynamics seed and its `random_state` seed, so that
/// the initial labels are independent of the dynamics stream.
pub const STATE_SEED_OFFSET: u64 = 1_000_000_000;

/// Offset between a replica's dynamics seed and its pre-anneal seed.
pub const ANNEAL_SEED_OFFSET: u64 = 2_000_000_000;

/// Squared and dot-product observables of one Fe–Cr state, computed in (a/2)² integer
/// units, converted to `f64` and multiplied by a²/4.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Observables {
    /// `[Σ_Fe |Δr|² / N_Fe, Σ_Cr |Δr|² / N_Cr]`; 0.0 when that species count is 0.
    pub tracer: [f64; 2],
    /// Σ over all atoms of |Δr|² divided by the number of atoms.
    pub tracer_all: f64,
    /// `[ΔR_Fe·ΔR_Fe, ΔR_Fe·ΔR_Cr, ΔR_Cr·ΔR_Cr]`.
    pub collective: [f64; 3],
    /// Σ over all atoms of |Δr|².
    pub s: f64,
    /// |ΔR_v|², the squared vacancy displacement.
    pub vacancy: f64,
}

fn dot(a: [i64; 3], b: [i64; 3]) -> i64 {
    (0..3).map(|axis| a[axis] * b[axis]).sum()
}

/// Computes the [`Observables`] of `state` for lattice parameter `lattice_parameter`.
pub fn observables(state: &FeCrState, lattice_parameter: f64) -> Observables {
    let scale = lattice_parameter * lattice_parameter / 4.0;
    let [n_fe, n_cr, _] = state.counts();
    let sum_fe = state.tracer_squared_sum(Species::Fe);
    let sum_cr = state.tracer_squared_sum(Species::Cr);
    let per_atom = |sum: i64, count: usize| {
        if count == 0 {
            0.0
        } else {
            sum as f64 / count as f64 * scale
        }
    };
    let n_atoms = n_fe + n_cr;
    let r_fe = state.collective_displacement(Species::Fe);
    let r_cr = state.collective_displacement(Species::Cr);
    Observables {
        tracer: [per_atom(sum_fe, n_fe), per_atom(sum_cr, n_cr)],
        tracer_all: per_atom(sum_fe + sum_cr, n_atoms),
        collective: [
            dot(r_fe, r_fe) as f64 * scale,
            dot(r_fe, r_cr) as f64 * scale,
            dot(r_cr, r_cr) as f64 * scale,
        ],
        s: (sum_fe + sum_cr) as f64 * scale,
        vacancy: dot(state.vacancy_displacement(), state.vacancy_displacement()) as f64 * scale,
    }
}

/// Samples [`observables`] at t_k = k·`t_end`/`k_count`, k = 1..=`k_count`, using the value
/// held before the first jump whose time exceeds t_k.
///
/// Returns the sample times, the samples and the number of fired steps.
///
/// # Panics
///
/// Panics if the simulation fails or is absorbed.
pub fn sample_run<E: Energetics + Clone>(
    model: FeCrModel<E>,
    state: FeCrState,
    seed: u64,
    t_end: f64,
    k_count: usize,
    lattice_parameter: f64,
) -> (Vec<f64>, Vec<Observables>, u64) {
    let mut sim = Simulation::new(model, state, seed);
    let times: Vec<f64> = (1..=k_count)
        .map(|k| k as f64 * t_end / k_count as f64)
        .collect();
    let mut samples = Vec::with_capacity(k_count);
    let mut fired = 0_u64;
    while samples.len() < k_count {
        let held = observables(sim.state(), lattice_parameter);
        match sim.step().expect("Fe-Cr events have valid rates") {
            Step::Fired { .. } => fired += 1,
            Step::Absorbed => panic!("Fe-Cr model unexpectedly absorbed"),
        }
        let t_after = sim.time();
        while samples.len() < k_count && times[samples.len()] < t_after {
            samples.push(held);
        }
    }
    (times, samples, fired)
}

/// Returns the slope of a line through the origin, Σ t v / Σ t².
pub fn origin_slope(times: &[f64], values: &[f64]) -> f64 {
    let num: f64 = times.iter().zip(values).map(|(t, v)| t * v).sum();
    let den: f64 = times.iter().map(|t| t * t).sum();
    num / den
}

/// Returns the ordinary least-squares slope of a line with intercept.
pub fn ols_slope(times: &[f64], values: &[f64]) -> f64 {
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

/// Returns the sample mean and its standard error sd/√n, with sd using n − 1.
pub fn mean_se(values: &[f64]) -> (f64, f64) {
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0);
    (mean, (variance / n).sqrt())
}

/// Per-replica estimates; every entry is a slope of the named quantity divided by 6.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ReplicaEstimate {
    /// D*_Fe and D*_Cr.
    pub tracer: [f64; 2],
    /// D*_all.
    pub tracer_all: f64,
    /// Λ_FeFe, Λ_FeCr, Λ_CrCr.
    pub collective: [f64; 3],
    /// Slope of S divided by 6.
    pub s: f64,
    /// The vacancy coefficient D_v.
    pub vacancy: f64,
}

/// Estimates every coefficient of one replica with the given `slope` fit.
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

/// The outcome of one replica of [`run_composition`].
#[derive(Clone, Debug, PartialEq)]
pub struct ReplicaRun {
    /// The coefficients estimated from the replica.
    pub estimate: ReplicaEstimate,
    /// The number of fired steps.
    pub fired: u64,
    /// Total configuration energy in eV after each anneal sweep (entry 0 is the
    /// initial energy); empty when no anneal ran.
    pub anneal_energy: Vec<f64>,
}

/// Runs `runs` replicas of one composition with `cr_count` Cr atoms, seeds
/// `seed_base`, `seed_base + 1`, ..., and returns them in replica order.
///
/// Each replica draws a random state with seed `seed + STATE_SEED_OFFSET`. If
/// `anneal_sweeps > 0` it is annealed for that many sweeps with seed
/// `seed + ANNEAL_SEED_OFFSET`. The dynamics then run through [`sample_run`] and
/// the coefficients are fitted with [`origin_slope`].
///
/// # Errors
///
/// Returns the [`FeCrError`] of the first failing replica, such as too many atoms.
///
/// # Panics
///
/// Panics if the replica seeds overflow `u64`, or under the conditions of [`sample_run`].
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
) -> Result<Vec<ReplicaRun>, FeCrError> {
    run_replicas(seed_base, runs, |_, seed| {
        let mut state = model.random_state(cr_count, seed + STATE_SEED_OFFSET)?;
        let mut anneal_energy = Vec::new();
        if anneal_sweeps > 0 {
            let a = anneal(
                model,
                state.occupancy().to_vec(),
                anneal_sweeps,
                seed + ANNEAL_SEED_OFFSET,
            )?;
            state = model.state(a.occupancy)?;
            anneal_energy = a.energy;
        }
        let (times, samples, fired) = sample_run(
            model.clone(),
            state,
            seed,
            t_end,
            k_count,
            lattice_parameter,
        );
        Ok(ReplicaRun {
            estimate: estimate(&times, &samples, origin_slope),
            fired,
            anneal_energy,
        })
    })
    .expect("seeds do not overflow")
    .into_iter()
    .map(|replica| replica.value)
    .collect()
}

/// Means and standard errors, each `(mean, se)`, over the replicas of one composition.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Summary {
    /// D_v.
    pub vacancy: (f64, f64),
    /// D*_Fe and D*_Cr; Cr is `(NaN, NaN)` when there is no Cr.
    pub tracer: [(f64, f64); 2],
    /// D*_all.
    pub tracer_all: (f64, f64),
    /// Λ_FeFe, Λ_FeCr, Λ_CrCr.
    pub collective: [(f64, f64); 3],
    /// The correlation factor `n_atoms · D*_all / D_v`, computed per replica.
    pub f: (f64, f64),
    /// Mean number of fired steps per replica.
    pub mean_jumps: f64,
}

/// Summarizes `runs` for a composition with `n_cr` Cr atoms among `n_atoms` atoms.
pub fn summarize(runs: &[ReplicaRun], n_cr: usize, n_atoms: usize) -> Summary {
    let column = |f: &dyn Fn(&ReplicaEstimate) -> f64| {
        let values: Vec<f64> = runs.iter().map(|r| f(&r.estimate)).collect();
        mean_se(&values)
    };
    let f_values: Vec<f64> = runs
        .iter()
        .map(|r| n_atoms as f64 * r.estimate.tracer_all / r.estimate.vacancy)
        .collect();
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
        f: mean_se(&f_values),
        mean_jumps: runs.iter().map(|r| r.fired).sum::<u64>() as f64 / runs.len() as f64,
    }
}
