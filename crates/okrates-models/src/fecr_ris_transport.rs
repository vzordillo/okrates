//! Single-defect tracer and Onsager transport estimators for the Fe–Cr RIS model.
//!
//! Computes point-defect tracer diffusivity `D_d = slope(|dr_d|^2) / 6` and reduced
//! defect-normalized Onsager coefficients `Lambda_ab = slope(dR_a . dR_b) / 6 = L_ab k_B T Omega / c_d`
//! from Senninger, Soisson, Martínez, Nastar, Fu, and Bréchet, *Acta Mater.* **103**, 1–11 (2016),
//! doi:10.1016/j.actamat.2015.09.058 (Section 3.3, Equations 5–6; Section 4.1–4.2), using
//! ratio-of-means ensemble estimators with delete-1 jackknife standard errors (Quenouille,
//! *J. R. Stat. Soc. B* **11**, 68–84 (1949), doi:10.1111/j.2517-6161.1949.tb00023.x; Quenouille,
//! *Biometrika* **43**, 353–360 (1956), doi:10.1093/biomet/43.3-4.353; Tukey, *Ann. Math. Stat.*
//! **29**, 614 (1958), doi:10.1214/aoms/1177706647) and 1NN Warren–Cowley SRO monitoring
//! (Cowley, *Phys. Rev.* **77**, 669–675 (1950), doi:10.1103/PhysRev.77.669; Warren,
//! *X-Ray Diffraction* (Addison-Wesley, 1969)).
//!
//! Because [`FeCrRis`] models bulk jumps with `k0_dpa_per_s == 0.0` and no sink planes, exact
//! half-lattice (`a/2`) displacements obey `dR_Fe + dR_Cr + dr_V = 0` for a single vacancy and
//! `dR_Fe + dR_Cr = dr_I` for a single `<110>` dumbbell on every jump.

use std::error::Error;
use std::fmt;

use okrates_core::{
    CatalogBackend, Clock, DirectCatalog, Engine, GridError, KmcRng, SampleGrid, SamplingError,
    Step, StreamRole, run_ensemble,
};

use crate::bcc::{BccLattice, DISPLACEMENTS};
use crate::fecr::Species;
use crate::fecr_diffusion::{FitMethod, jackknife_ratio_mean_se, mean_se, ols_slope, origin_slope};
use crate::fecr_ris::{Cell, Defect, FeCrRis, FeCrRisError, FeCrRisEvent, FeCrRisState, Record};

/// Single-defect species for RIS bulk transport measurements.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DefectKind {
    Vacancy,
    Dumbbell,
}

impl DefectKind {
    pub fn name(self) -> &'static str {
        match self {
            Self::Vacancy => "vacancy",
            Self::Dumbbell => "dumbbell",
        }
    }
}

/// Errors returned while configuring, sampling, or summarizing single-defect [`FeCrRis`] transport.
#[derive(Debug)]
pub enum RisTransportError {
    Model(FeCrRisError),
    Sampling(SamplingError<FeCrRisError>),
    Grid(GridError),
    NonZeroCreationRate(f64),
    NonEmptySinkPlanes(usize),
    InvalidCapacity {
        capacity: usize,
    },
    ExpectedSingleDefect {
        vacancies: usize,
        dumbbells: usize,
    },
    UnexpectedEvent,
    NonNearestNeighborJump {
        from: usize,
        to: usize,
    },
    DisplacementOverflow,
    InvalidLatticeParameter(f64),
    InvalidCrCount {
        defect: DefectKind,
        cr_count: usize,
        max: usize,
    },
    InsufficientSamples {
        fit_method: FitMethod,
        sample_count: usize,
    },
    InsufficientReplicas(usize),
}

impl fmt::Display for RisTransportError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Model(err) => write!(f, "model error: {err}"),
            Self::Sampling(err) => write!(f, "sampling error: {err}"),
            Self::Grid(err) => write!(f, "grid error: {err}"),
            Self::NonZeroCreationRate(k0) => {
                write!(f, "bulk transport requires k0_dpa_per_s == 0, got {k0}")
            }
            Self::NonEmptySinkPlanes(count) => {
                write!(f, "bulk transport requires no sink planes, got {count}")
            }
            Self::InvalidCapacity { capacity } => write!(
                f,
                "single-defect transport requires capacity == 1, got {capacity}"
            ),
            Self::ExpectedSingleDefect {
                vacancies,
                dumbbells,
            } => write!(
                f,
                "expected exactly one defect, found {vacancies} vacancies and {dumbbells} dumbbells"
            ),
            Self::UnexpectedEvent => {
                write!(
                    f,
                    "encountered unexpected event during single-defect transport"
                )
            }
            Self::NonNearestNeighborJump { from, to } => {
                write!(
                    f,
                    "jump endpoints {from} -> {to} are not nearest neighbours"
                )
            }
            Self::DisplacementOverflow => write!(f, "unwrapped displacement overflowed i64"),
            Self::InvalidLatticeParameter(a) => {
                write!(f, "lattice parameter must be finite and > 0, got {a}")
            }
            Self::InvalidCrCount {
                defect,
                cr_count,
                max,
            } => write!(
                f,
                "cr_count {cr_count} exceeds max {max} for single {}",
                defect.name()
            ),
            Self::InsufficientSamples {
                fit_method,
                sample_count,
            } => write!(
                f,
                "{} slope requires at least {} samples, got {sample_count}",
                fit_method.name(),
                min_samples_for(*fit_method)
            ),
            Self::InsufficientReplicas(runs) => {
                write!(
                    f,
                    "transport summary requires at least 2 replicas, got {runs}"
                )
            }
        }
    }
}

impl Error for RisTransportError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Model(err) => Some(err),
            Self::Sampling(err) => Some(err),
            Self::Grid(err) => Some(err),
            _ => None,
        }
    }
}

impl From<FeCrRisError> for RisTransportError {
    fn from(err: FeCrRisError) -> Self {
        Self::Model(err)
    }
}

impl From<SamplingError<FeCrRisError>> for RisTransportError {
    fn from(err: SamplingError<FeCrRisError>) -> Self {
        Self::Sampling(err)
    }
}

impl From<GridError> for RisTransportError {
    fn from(err: GridError) -> Self {
        Self::Grid(err)
    }
}

const fn min_samples_for(fit_method: FitMethod) -> usize {
    match fit_method {
        FitMethod::WindowedOls => 2,
        FitMethod::OriginSlope => 1,
    }
}

fn validate_model(model: &FeCrRis) -> Result<(), RisTransportError> {
    let params = model.params();
    if params.k0_dpa_per_s != 0.0 {
        return Err(RisTransportError::NonZeroCreationRate(params.k0_dpa_per_s));
    }
    if !params.sink_planes.is_empty() {
        return Err(RisTransportError::NonEmptySinkPlanes(
            params.sink_planes.len(),
        ));
    }
    if params.capacity != 1 {
        return Err(RisTransportError::InvalidCapacity {
            capacity: params.capacity,
        });
    }
    Ok(())
}

fn validate_lattice_parameter(lattice_parameter: f64) -> Result<(), RisTransportError> {
    if !lattice_parameter.is_finite() || lattice_parameter <= 0.0 {
        return Err(RisTransportError::InvalidLatticeParameter(
            lattice_parameter,
        ));
    }
    Ok(())
}

fn max_cr_count(defect: DefectKind, site_count: usize) -> usize {
    match defect {
        DefectKind::Vacancy => site_count.saturating_sub(1),
        DefectKind::Dumbbell => site_count.saturating_add(1),
    }
}

/// Configuration for fixed-time [`FeCrRis`] single-defect transport ensembles.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RisTransportConfig {
    pub t_equil: f64,
    pub t_end: f64,
    pub sample_count: usize,
    pub lattice_parameter: f64,
    pub fit_method: FitMethod,
    pub max_steps: Option<u64>,
}

impl RisTransportConfig {
    /// OLS fit over a uniform `[t_equil, t_end]` grid of `sample_count` samples.
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

    /// Through-origin fit on `[0, t_end]` with `sample_count` uniform samples.
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

    pub fn with_max_steps(mut self, max_steps: Option<u64>) -> Self {
        self.max_steps = max_steps;
        self
    }

    pub fn with_fit_method(mut self, fit_method: FitMethod) -> Self {
        self.fit_method = fit_method;
        self
    }
}

/// Unwrapped integer displacements in half-lattice (`a/2`) units.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RisDisplacements {
    pub defect: [i64; 3],
    pub fe: [i64; 3],
    pub cr: [i64; 3],
}

/// Squared and collective dot-product observables in `lattice_parameter^2` units.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RisObservables {
    pub defect_msd: f64,
    pub fe_fe: f64,
    pub fe_cr: f64,
    pub cr_cr: f64,
    pub fe_defect: f64,
    pub cr_defect: f64,
}

/// Sampled single-defect transport trajectory with SRO and exact endpoint displacements.
#[derive(Clone, Debug, PartialEq)]
pub struct RisSampledTrajectory {
    pub times: Vec<f64>,
    pub values: Vec<RisObservables>,
    pub fired_steps: u64,
    pub final_displacements: RisDisplacements,
    pub initial_sro_alpha1: f64,
    pub final_sro_alpha1: f64,
    pub floored_jumps: u64,
}

/// Per-replica diffusivity and reduced Onsager coefficients (`slope / 6`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RisReplicaEstimate {
    pub defect: f64,
    pub fe_fe: f64,
    pub fe_cr: f64,
    pub cr_cr: f64,
    pub fe_defect: f64,
    pub cr_defect: f64,
}

/// Output of one simulated replica in [`run_ris_transport`].
#[derive(Clone, Debug, PartialEq)]
pub struct RisReplicaRun {
    pub estimate: RisReplicaEstimate,
    pub fired: u64,
    pub floored_jumps: u64,
    pub initial_sro_alpha1: f64,
    pub final_sro_alpha1: f64,
    pub final_displacements: RisDisplacements,
}

/// Ensemble-averaged transport coefficients and delete-1 jackknife ratios across replicas.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RisTransportSummary {
    pub defect: (f64, f64),
    pub fe_fe: (f64, f64),
    pub fe_cr: (f64, f64),
    pub cr_cr: (f64, f64),
    pub fe_defect: (f64, f64),
    pub cr_defect: (f64, f64),
    pub l_cr_d_over_l_fe_d: (f64, f64),
    pub l_fe_d_over_l_fe_fe: (f64, f64),
    pub l_cr_d_over_l_cr_cr: (f64, f64),
    pub d_cr_over_d_fe: (f64, f64),
    pub mean_jumps: f64,
    pub mean_floored_jumps: f64,
    pub mean_initial_sro_alpha1: f64,
    pub mean_final_sro_alpha1: f64,
    pub mean_delta_sro_alpha1: f64,
}

fn dot_f64(a: [i64; 3], b: [i64; 3]) -> f64 {
    let sum: i128 = (0..3)
        .map(|axis| i128::from(a[axis]) * i128::from(b[axis]))
        .sum();
    sum as f64
}

fn observables(d: &RisDisplacements, lattice_parameter: f64) -> RisObservables {
    let scale = lattice_parameter * lattice_parameter / 4.0;
    RisObservables {
        defect_msd: dot_f64(d.defect, d.defect) * scale,
        fe_fe: dot_f64(d.fe, d.fe) * scale,
        fe_cr: dot_f64(d.fe, d.cr) * scale,
        cr_cr: dot_f64(d.cr, d.cr) * scale,
        fe_defect: dot_f64(d.fe, d.defect) * scale,
        cr_defect: dot_f64(d.cr, d.defect) * scale,
    }
}

fn add_vec(acc: &mut [i64; 3], delta: [i64; 3]) -> Result<(), RisTransportError> {
    for (dst, &d) in acc.iter_mut().zip(&delta) {
        *dst = dst
            .checked_add(d)
            .ok_or(RisTransportError::DisplacementOverflow)?;
    }
    Ok(())
}

fn sub_vec(acc: &mut [i64; 3], delta: [i64; 3]) -> Result<(), RisTransportError> {
    for (dst, &d) in acc.iter_mut().zip(&delta) {
        *dst = dst
            .checked_sub(d)
            .ok_or(RisTransportError::DisplacementOverflow)?;
    }
    Ok(())
}

fn min_image_nn_step(
    lattice: &BccLattice,
    from: usize,
    to: usize,
) -> Result<[i64; 3], RisTransportError> {
    let a = lattice.position(from);
    let b = lattice.position(to);
    let cells = lattice.cells();
    let mut delta = [0_i64; 3];
    for axis in 0..3 {
        let period = 2 * cells[axis] as i64;
        let mut d = b[axis] - a[axis];
        if d > 1 {
            d -= period;
        } else if d < -1 {
            d += period;
        }
        if d != -1 && d != 1 {
            return Err(RisTransportError::NonNearestNeighborJump { from, to });
        }
        delta[axis] = d;
    }
    Ok(delta)
}

/// Computes the 1NN Warren–Cowley short-range order parameter `alpha_1` over atom–atom bonds.
///
/// Defect cells (`Vacancy` and `Dumbbell`) are excluded from atom and neighbor bond counts.
pub fn warren_cowley_alpha1_ris(
    lattice: &BccLattice,
    state: &FeCrRisState,
) -> Result<f64, FeCrRisError> {
    let expected = lattice.site_count();
    let cells = state.cells();
    if cells.len() != expected {
        return Err(FeCrRisError::SiteCount {
            expected,
            found: cells.len(),
        });
    }
    let mut n_fe = 0_usize;
    let mut n_cr = 0_usize;
    let mut fe_atom_bonds = 0_usize;
    let mut fe_cr_bonds = 0_usize;

    for (site, &cell) in cells.iter().enumerate() {
        match cell {
            Cell::Fe => {
                n_fe += 1;
                for &nb in lattice.neighbors(site) {
                    match cells[nb] {
                        Cell::Fe => fe_atom_bonds += 1,
                        Cell::Cr => {
                            fe_atom_bonds += 1;
                            fe_cr_bonds += 1;
                        }
                        Cell::Vacancy | Cell::Dumbbell => {}
                    }
                }
            }
            Cell::Cr => n_cr += 1,
            Cell::Vacancy | Cell::Dumbbell => {}
        }
    }

    if n_fe == 0 || n_cr == 0 || fe_atom_bonds == 0 {
        return Ok(0.0);
    }
    let x_cr = n_cr as f64 / (n_fe + n_cr) as f64;
    let p_cr_given_fe = fe_cr_bonds as f64 / fe_atom_bonds as f64;
    Ok(1.0 - p_cr_given_fe / x_cr)
}

fn shuffled_atoms(total_atoms: usize, cr_count: usize, rng: &mut KmcRng) -> Vec<Species> {
    let mut order: Vec<usize> = (0..total_atoms).collect();
    for i in 0..cr_count {
        let rem = total_atoms - i;
        let j = i + ((rng.open01() * rem as f64) as usize).min(rem - 1);
        order.swap(i, j);
    }
    let mut atoms = vec![Species::Fe; total_atoms];
    for &idx in &order[..cr_count] {
        atoms[idx] = Species::Cr;
    }
    atoms
}

/// Builds a homogeneous random-alloy single-defect state with exact Fe/Cr atom counts.
pub fn random_single_defect_state(
    model: &FeCrRis,
    defect: DefectKind,
    cr_count: usize,
    rng: &mut KmcRng,
    run_seed: u64,
) -> Result<FeCrRisState, RisTransportError> {
    validate_model(model)?;
    let n = model.lattice().site_count();
    let max = max_cr_count(defect, n);
    if cr_count > max {
        return Err(RisTransportError::InvalidCrCount {
            defect,
            cr_count,
            max,
        });
    }
    let mut occ = vec![Species::Fe; n];
    let defect_entry = match defect {
        DefectKind::Vacancy => {
            let atoms = shuffled_atoms(n - 1, cr_count, rng);
            occ[1..n].copy_from_slice(&atoms);
            Defect::Vacancy { site: 0 }
        }
        DefectKind::Dumbbell => {
            let atoms = shuffled_atoms(n + 1, cr_count, rng);
            occ[1..n].copy_from_slice(&atoms[..n - 1]);
            let axis = ((rng.open01() * 6.0) as usize).min(5);
            Defect::Dumbbell {
                site: 0,
                axis,
                plus: atoms[n - 1],
                minus: atoms[n],
            }
        }
    };
    Ok(model.state_with_defects(occ, &[defect_entry], run_seed)?)
}

fn validate_single_defect_state(
    model: &FeCrRis,
    state: &FeCrRisState,
) -> Result<(), RisTransportError> {
    validate_model(model)?;
    model.check_invariants(state)?;
    let (vacancies, dumbbells) = state.defect_counts();
    if (vacancies, dumbbells) != (1, 0) && (vacancies, dumbbells) != (0, 1) {
        return Err(RisTransportError::ExpectedSingleDefect {
            vacancies,
            dumbbells,
        });
    }
    if matches!(state.records().first(), None | Some(Record::Empty)) {
        return Err(RisTransportError::ExpectedSingleDefect {
            vacancies,
            dumbbells,
        });
    }
    Ok(())
}

/// Samples right-continuous single-defect transport observables on `grid` without modifying
/// [`FeCrRisState`] schema or R2 regression behavior.
pub fn sample_ris_transport<C: CatalogBackend<FeCrRis>>(
    engine: &mut Engine<FeCrRis, C>,
    grid: &SampleGrid,
    lattice_parameter: f64,
    max_steps: Option<u64>,
) -> Result<RisSampledTrajectory, RisTransportError> {
    validate_lattice_parameter(lattice_parameter)?;
    validate_single_defect_state(engine.model(), engine.state())?;

    let times = grid.times();
    if engine.clock().cmp_f64(times[0]).is_gt() {
        return Err(RisTransportError::Sampling(
            SamplingError::GridBeforeSimTime {
                grid_start: times[0],
                sim_time: engine.clock().time(),
            },
        ));
    }

    let initial_sro_alpha1 = warren_cowley_alpha1_ris(engine.model().lattice(), engine.state())?;
    let initial_floored =
        engine.state().floored_vacancy_jumps() + engine.state().floored_dumbbell_jumps();

    let mut disp = RisDisplacements::default();
    let mut values = Vec::with_capacity(times.len());
    let mut fired_steps = 0_u64;

    while values.len() < times.len() {
        let current_obs = observables(&disp, lattice_parameter);
        if engine.clock().cmp_f64(times[values.len()]).is_eq() {
            while values.len() < times.len() && engine.clock().cmp_f64(times[values.len()]).is_eq()
            {
                values.push(current_obs);
            }
            if values.len() >= times.len() {
                break;
            }
        }

        if let Some(limit) = max_steps
            && fired_steps >= limit
        {
            if engine.is_absorbed().map_err(|source| SamplingError::Sim {
                steps: fired_steps,
                time: engine.clock().time(),
                source,
            })? {
                return Err(RisTransportError::Sampling(SamplingError::Absorbed {
                    steps: fired_steps,
                    time: engine.clock().time(),
                }));
            }
            return Err(RisTransportError::Sampling(
                SamplingError::MaxStepsExceeded {
                    max_steps: limit,
                    time: engine.clock().time(),
                },
            ));
        }

        let next_target = times[values.len()];
        let mut before_rec: Option<Record> = None;
        let mut crossed: Option<(RisObservables, Clock)> = None;
        let step = engine
            .step_observing(|_model, state_before, next_clock| {
                before_rec = Some(state_before.records()[0]);
                if next_clock.cmp_f64(next_target).is_gt() {
                    crossed = Some((current_obs, next_clock));
                }
            })
            .map_err(|source| SamplingError::Sim {
                steps: fired_steps,
                time: engine.clock().time(),
                source,
            })?;

        match step {
            Step::Fired { event, .. } => {
                if let Some((held, clock_after)) = crossed {
                    while values.len() < times.len()
                        && clock_after.cmp_f64(times[values.len()]).is_gt()
                    {
                        values.push(held);
                    }
                }
                let prev = before_rec.ok_or(RisTransportError::UnexpectedEvent)?;
                match (event, prev) {
                    (FeCrRisEvent::VacancyJump { rec: 0, dir }, Record::Vacancy { site: q }) => {
                        let d = DISPLACEMENTS[dir];
                        add_vec(&mut disp.defect, d)?;
                        match engine.state().cells()[q] {
                            Cell::Fe => sub_vec(&mut disp.fe, d)?,
                            Cell::Cr => sub_vec(&mut disp.cr, d)?,
                            _ => return Err(RisTransportError::UnexpectedEvent),
                        }
                    }
                    (
                        FeCrRisEvent::DumbbellJump { rec: 0, .. },
                        Record::Dumbbell {
                            site: o,
                            plus,
                            minus,
                            ..
                        },
                    ) => {
                        let Record::Dumbbell { site: k, .. } = engine.state().records()[0] else {
                            return Err(RisTransportError::UnexpectedEvent);
                        };
                        let d = min_image_nn_step(engine.model().lattice(), o, k)?;
                        let jumper = if plus == minus {
                            plus
                        } else {
                            // `plus` and `minus` split into `{jumper -> k, rest -> o}`.
                            match engine.state().cells()[o] {
                                Cell::Fe => Species::Cr,
                                Cell::Cr => Species::Fe,
                                _ => return Err(RisTransportError::UnexpectedEvent),
                            }
                        };
                        add_vec(&mut disp.defect, d)?;
                        match jumper {
                            Species::Fe => add_vec(&mut disp.fe, d)?,
                            Species::Cr => add_vec(&mut disp.cr, d)?,
                            Species::Vacancy => return Err(RisTransportError::UnexpectedEvent),
                        }
                    }
                    _ => return Err(RisTransportError::UnexpectedEvent),
                }
                fired_steps += 1;
            }
            Step::Absorbed => {
                return Err(RisTransportError::Sampling(SamplingError::Absorbed {
                    steps: fired_steps,
                    time: engine.clock().time(),
                }));
            }
        }
    }

    let final_sro_alpha1 = warren_cowley_alpha1_ris(engine.model().lattice(), engine.state())?;
    let final_floored =
        engine.state().floored_vacancy_jumps() + engine.state().floored_dumbbell_jumps();

    Ok(RisSampledTrajectory {
        times: times.to_vec(),
        values,
        fired_steps,
        final_displacements: disp,
        initial_sro_alpha1,
        final_sro_alpha1,
        floored_jumps: final_floored - initial_floored,
    })
}

/// Estimates diffusivity and reduced Onsager coefficients from a sampled trajectory.
pub fn estimate_ris(
    times: &[f64],
    samples: &[RisObservables],
    fit_method: FitMethod,
) -> RisReplicaEstimate {
    let slope: fn(&[f64], &[f64]) -> f64 = match fit_method {
        FitMethod::WindowedOls => ols_slope,
        FitMethod::OriginSlope => origin_slope,
    };
    let fit = |extract: &dyn Fn(&RisObservables) -> f64| {
        let series: Vec<f64> = samples.iter().map(extract).collect();
        slope(times, &series) / 6.0
    };
    RisReplicaEstimate {
        defect: fit(&|o| o.defect_msd),
        fe_fe: fit(&|o| o.fe_fe),
        fe_cr: fit(&|o| o.fe_cr),
        cr_cr: fit(&|o| o.cr_cr),
        fe_defect: fit(&|o| o.fe_defect),
        cr_defect: fit(&|o| o.cr_defect),
    }
}

/// Runs `runs` parallel single-defect [`FeCrRis`] replicas with domain-separated RNG streams.
pub fn run_ris_transport(
    model: &FeCrRis,
    defect: DefectKind,
    cr_count: usize,
    seed_base: u64,
    runs: u64,
    config: &RisTransportConfig,
) -> Result<Vec<RisReplicaRun>, RisTransportError> {
    if runs < 2 {
        return Err(RisTransportError::InsufficientReplicas(runs as usize));
    }
    validate_model(model)?;
    validate_lattice_parameter(config.lattice_parameter)?;
    let max = max_cr_count(defect, model.lattice().site_count());
    if cr_count > max {
        return Err(RisTransportError::InvalidCrCount {
            defect,
            cr_count,
            max,
        });
    }
    let min_samples = min_samples_for(config.fit_method);
    if config.sample_count < min_samples {
        return Err(RisTransportError::InsufficientSamples {
            fit_method: config.fit_method,
            sample_count: config.sample_count,
        });
    }
    let grid = SampleGrid::windowed(config.t_equil, config.t_end, config.sample_count)?;

    run_ensemble(
        seed_base,
        runs,
        |ctx| -> Result<RisReplicaRun, RisTransportError> {
            let mut init_rng = ctx.rng(StreamRole::InitialState);
            let run_seed = ctx.base_seed.wrapping_add(ctx.id);
            let state =
                random_single_defect_state(model, defect, cr_count, &mut init_rng, run_seed)?;
            let mut engine = Engine::<FeCrRis, DirectCatalog<FeCrRisEvent>>::with_rng(
                model.clone(),
                state,
                ctx.rng(StreamRole::Dynamics),
            )
            .map_err(|source| SamplingError::Sim {
                steps: 0,
                time: 0.0,
                source,
            })?;
            let traj = sample_ris_transport(
                &mut engine,
                &grid,
                config.lattice_parameter,
                config.max_steps,
            )?;
            let estimate = estimate_ris(&traj.times, &traj.values, config.fit_method);
            Ok(RisReplicaRun {
                estimate,
                fired: traj.fired_steps,
                floored_jumps: traj.floored_jumps,
                initial_sro_alpha1: traj.initial_sro_alpha1,
                final_sro_alpha1: traj.final_sro_alpha1,
                final_displacements: traj.final_displacements,
            })
        },
    )
    .into_iter()
    .collect()
}

/// Aggregates ensemble means, standard errors, and delete-1 jackknife Onsager ratios.
pub fn summarize_ris_transport(
    runs: &[RisReplicaRun],
    defect: DefectKind,
    site_count: usize,
    cr_count: usize,
) -> Result<RisTransportSummary, RisTransportError> {
    if runs.len() < 2 {
        return Err(RisTransportError::InsufficientReplicas(runs.len()));
    }
    let Some(n_atoms) = (match defect {
        DefectKind::Vacancy => site_count.checked_sub(1),
        DefectKind::Dumbbell => site_count.checked_add(1),
    }) else {
        return Err(RisTransportError::InvalidCrCount {
            defect,
            cr_count,
            max: 0,
        });
    };
    let max = max_cr_count(defect, site_count);
    if cr_count > max {
        return Err(RisTransportError::InvalidCrCount {
            defect,
            cr_count,
            max,
        });
    }
    let n_fe = n_atoms - cr_count;
    let n_cr = cr_count;

    let col = |f: &dyn Fn(&RisReplicaEstimate) -> f64| -> Vec<f64> {
        runs.iter().map(|r| f(&r.estimate)).collect()
    };
    let fe_fe_v = col(&|e| e.fe_fe);
    let fe_cr_v = col(&|e| e.fe_cr);
    let cr_cr_v = col(&|e| e.cr_cr);
    let fe_d_v = col(&|e| e.fe_defect);
    let cr_d_v = col(&|e| e.cr_defect);

    let l_cr_d_over_l_fe_d = jackknife_ratio_mean_se(&cr_d_v, &fe_d_v, 1.0);
    let l_fe_d_over_l_fe_fe = jackknife_ratio_mean_se(&fe_d_v, &fe_fe_v, 1.0);
    let l_cr_d_over_l_cr_cr = if n_cr == 0 {
        (f64::NAN, f64::NAN)
    } else {
        jackknife_ratio_mean_se(&cr_d_v, &cr_cr_v, 1.0)
    };
    let d_cr_over_d_fe = if n_cr == 0 || n_fe == 0 {
        (f64::NAN, f64::NAN)
    } else {
        jackknife_ratio_mean_se(&cr_d_v, &fe_d_v, n_fe as f64 / n_cr as f64)
    };

    let n = runs.len() as f64;
    let mean_jumps = runs.iter().map(|r| r.fired as f64).sum::<f64>() / n;
    let mean_floored_jumps = runs.iter().map(|r| r.floored_jumps as f64).sum::<f64>() / n;
    let mean_initial_sro_alpha1 = runs.iter().map(|r| r.initial_sro_alpha1).sum::<f64>() / n;
    let mean_final_sro_alpha1 = runs.iter().map(|r| r.final_sro_alpha1).sum::<f64>() / n;

    Ok(RisTransportSummary {
        defect: mean_se(&col(&|e| e.defect)),
        fe_fe: mean_se(&fe_fe_v),
        fe_cr: mean_se(&fe_cr_v),
        cr_cr: mean_se(&cr_cr_v),
        fe_defect: mean_se(&fe_d_v),
        cr_defect: mean_se(&cr_d_v),
        l_cr_d_over_l_fe_d,
        l_fe_d_over_l_fe_fe,
        l_cr_d_over_l_cr_cr,
        d_cr_over_d_fe,
        mean_jumps,
        mean_floored_jumps,
        mean_initial_sro_alpha1,
        mean_final_sro_alpha1,
        mean_delta_sro_alpha1: mean_final_sro_alpha1 - mean_initial_sro_alpha1,
    })
}
