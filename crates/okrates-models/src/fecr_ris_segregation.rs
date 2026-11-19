//! Grain-boundary radiation-induced segregation (RIS) observables and ensemble driver for
//! [`FeCrRis`].
//!
//! Measures planar Cr enrichment/depletion (`C_Cr^GB`, `C_Cr^bulk`), Gibbsian Cr excess
//! `S_Cr^tot = sum_x (C_Cr(x)/C_Fe(x) - bulk_ratio) dx` in nm following Senninger, Soisson,
//! Martínez, Nastar, Fu, and Bréchet, *Acta Mater.* **103**, 1–11 (2016),
//! doi:10.1016/j.actamat.2015.09.058 (Section 4.3, Figs. 5–8) and Wagner, Rehn, Wiedersich,
//! and Naundorf, *Phys. Rev. B* **28**, 6780 (1983), continuous-time per-plane point-defect
//! occupancies using Neumaier compensated summation (Neumaier, *Z. Angew. Math. Mech.* **54**,
//! 39–51 (1974), doi:10.1002/zamm.19740540106), and single-exponential relaxation fits
//! `y(phi) = y_inf + (y0 - y_inf) * exp(-phi / phi_rel)`.

use std::error::Error;
use std::fmt;

use okrates_core::{
    Clock, IncrementalSimulation, KmcRng, ReplicaContext, SimError, Step, StreamRole, run_ensemble,
};

use crate::fecr::Species;
use crate::fecr_diffusion::mean_se;
use crate::fecr_ris::{FeCrRis, FeCrRisError, FeCrRisState, Record};

/// Default iron lattice parameter `a` in meters (`2.8665 Å`).
pub const DEFAULT_LATTICE_A_M: f64 = 2.8665e-10;

/// Errors returned by RIS grain-boundary segregation configuration, stepping, and analysis.
#[derive(Debug)]
pub enum RisSegregationError {
    Model(FeCrRisError),
    Sim {
        steps: u64,
        time: f64,
        source: SimError<FeCrRisError>,
    },
    InvalidSinkPlanes(usize),
    SinkPlaneOutOfRange {
        sink_plane: i64,
        planes: usize,
    },
    NonPositiveCreationRate(f64),
    InvalidLatticeParameter(f64),
    InvalidCrCount {
        cr_count: usize,
        max: usize,
    },
    InvalidDoseGrid(&'static str),
    InvalidBulkWindow {
        min_distance: usize,
        half_planes: usize,
    },
    InvalidSteadyStart {
        steady_start_dpa: f64,
        dpa_end: f64,
    },
    InsufficientReplicas(usize),
    MaxStepsExceeded {
        max_steps: u64,
        dose: f64,
    },
    Absorbed {
        steps: u64,
        dose: f64,
    },
    ZeroFePlane {
        plane: usize,
    },
    ZeroFeBulk,
    Allocation(usize),
    InvalidSummary(&'static str),
}

impl fmt::Display for RisSegregationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Model(err) => write!(f, "model error: {err}"),
            Self::Sim {
                steps,
                time,
                source,
            } => write!(f, "step error after {steps} steps at t={time:e}: {source}"),
            Self::InvalidSinkPlanes(count) => write!(
                f,
                "grain-boundary RIS requires exactly one sink plane, got {count}"
            ),
            Self::SinkPlaneOutOfRange { sink_plane, planes } => write!(
                f,
                "sink plane {sink_plane} is outside [0, {planes}) for {planes} (100) planes"
            ),
            Self::NonPositiveCreationRate(k0) => {
                write!(f, "RIS irradiation requires k0_dpa_per_s > 0, got {k0}")
            }
            Self::InvalidLatticeParameter(a) => {
                write!(f, "lattice parameter must be finite and > 0, got {a}")
            }
            Self::InvalidCrCount { cr_count, max } => {
                write!(f, "cr_count {cr_count} exceeds site count {max}")
            }
            Self::InvalidDoseGrid(msg) => write!(f, "invalid dose grid: {msg}"),
            Self::InvalidBulkWindow {
                min_distance,
                half_planes,
            } => write!(
                f,
                "bulk window min_distance={min_distance} must lie in 1..={half_planes}"
            ),
            Self::InvalidSteadyStart {
                steady_start_dpa,
                dpa_end,
            } => write!(
                f,
                "steady_start_dpa={steady_start_dpa} must satisfy 0 <= steady_start_dpa < {dpa_end}"
            ),
            Self::InsufficientReplicas(n) => {
                write!(f, "segregation run requires at least 1 replica, got {n}")
            }
            Self::MaxStepsExceeded { max_steps, dose } => write!(
                f,
                "exceeded max_steps={max_steps} before target dose (reached {dose:e} dpa)"
            ),
            Self::Absorbed { steps, dose } => {
                write!(f, "simulation absorbed after {steps} steps at {dose:e} dpa")
            }
            Self::ZeroFePlane { plane } => {
                write!(f, "plane {plane} has zero Fe atoms; Cr/Fe ratio undefined")
            }
            Self::ZeroFeBulk => write!(f, "bulk window has zero Fe atoms; Cr/Fe ratio undefined"),
            Self::Allocation(n) => write!(f, "failed to allocate buffer of {n} elements"),
            Self::InvalidSummary(msg) => write!(f, "invalid replica summary input: {msg}"),
        }
    }
}

impl Error for RisSegregationError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Model(err) => Some(err),
            Self::Sim { source, .. } => Some(source),
            _ => None,
        }
    }
}

impl From<FeCrRisError> for RisSegregationError {
    fn from(err: FeCrRisError) -> Self {
        Self::Model(err)
    }
}

fn try_vec_zero<T: Clone>(len: usize, val: T) -> Result<Vec<T>, RisSegregationError> {
    let mut v = Vec::new();
    v.try_reserve_exact(len)
        .map_err(|_| RisSegregationError::Allocation(len))?;
    v.resize(len, val);
    Ok(v)
}

#[inline]
fn neumaier_add(sum: &mut f64, comp: &mut f64, x: f64) {
    let t = *sum + x;
    if sum.abs() >= x.abs() {
        *comp += (*sum - t) + x;
    } else {
        *comp += (x - t) + *sum;
    }
    *sum = t;
}

#[inline]
fn site_plane_x(cells: [usize; 3], site: usize) -> usize {
    let [lx, ly, lz] = cells;
    let plane_stride = ly * lz;
    let half_sites = lx * plane_stride;
    let (sub, within) = if site < half_sites {
        (0, site)
    } else {
        (1, site - half_sites)
    };
    2 * (within / plane_stride) + sub
}

#[inline]
fn abs_periodic_distance(x: usize, sink_plane: usize, nx: usize) -> usize {
    let d = x.abs_diff(sink_plane);
    d.min(nx - d)
}

/// Default minimum periodic plane distance from the grain boundary used to define the bulk window.
///
/// Chooses `(3 * nx) / 8 = (3 * cells[0]) / 4` clamped to `1..=cells[0]`, selecting the
/// central quarter of the box farthest from the planar sink.
pub fn default_bulk_min_distance_planes(cells: [usize; 3]) -> usize {
    (cells[0] * 3 / 4).clamp(1, cells[0].max(1))
}

/// Instantaneous per-(100)-plane Fe, Cr, vacancy, and dumbbell counts extracted from a state.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlaneCounts {
    pub fe: Vec<u64>,
    pub cr: Vec<u64>,
    pub vacancies: Vec<u64>,
    pub dumbbells: Vec<u64>,
    pub sites_per_plane: usize,
}

impl PlaneCounts {
    /// Aggregates per-(100)-plane counts directly from `model` and `state`.
    pub fn from_state(model: &FeCrRis, state: &FeCrRisState) -> Result<Self, RisSegregationError> {
        if state.cells().len() != model.lattice().site_count() {
            return Err(RisSegregationError::InvalidSummary(
                "state cell count does not match model lattice",
            ));
        }
        let cells = model.lattice().cells();
        let nx = 2 * cells[0];
        let sites_per_plane = cells[1] * cells[2];
        let mut fe = try_vec_zero(nx, 0_u64)?;
        let mut cr = try_vec_zero(nx, 0_u64)?;
        let mut vacancies = try_vec_zero(nx, 0_u64)?;
        let mut dumbbells = try_vec_zero(nx, 0_u64)?;

        for (site, &cell) in state.cells().iter().enumerate() {
            let x = site_plane_x(cells, site);
            match cell {
                crate::fecr_ris::Cell::Fe => fe[x] += 1,
                crate::fecr_ris::Cell::Cr => cr[x] += 1,
                crate::fecr_ris::Cell::Vacancy => vacancies[x] += 1,
                crate::fecr_ris::Cell::Dumbbell => {
                    dumbbells[x] += 1;
                    if let Some(rec) = state.record_at(site)
                        && let Some(&Record::Dumbbell { plus, minus, .. }) =
                            state.records().get(rec)
                    {
                        for sp in [plus, minus] {
                            match sp {
                                Species::Fe => fe[x] += 1,
                                Species::Cr => cr[x] += 1,
                                Species::Vacancy => {}
                            }
                        }
                    }
                }
            }
        }
        Ok(Self {
            fe,
            cr,
            vacancies,
            dumbbells,
            sites_per_plane,
        })
    }

    /// Number of (100) atomic planes `2 * cells[0]`.
    pub fn plane_count(&self) -> usize {
        self.fe.len()
    }

    /// Cr atomic fraction `Cr / (Fe + Cr)` on plane `x`, `0.0` if devoid of atoms, or `NaN` out of range.
    pub fn c_cr(&self, x: usize) -> f64 {
        let (Some(&n_fe), Some(&n_cr)) = (self.fe.get(x), self.cr.get(x)) else {
            return f64::NAN;
        };
        let total = n_fe + n_cr;
        if total == 0 {
            0.0
        } else {
            n_cr as f64 / total as f64
        }
    }
}

/// Scalar grain-boundary Cr segregation observables at one dose point.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SegregationMetrics {
    pub c_cr_gb: f64,
    pub c_cr_bulk: f64,
    pub s_cr_tot_nm: f64,
    pub s_cr_linear_nm: f64,
}

/// Computes `C_Cr^GB`, `C_Cr^bulk`, and Gibbsian excess `S_Cr^tot` (nm) from planar counts.
///
/// The baseline ratio in `S_Cr^tot` averages `Cr(x) / Fe(x)` across bulk planes whose
/// periodic distance from `sink_plane` is at least `bulk_min_distance_planes`, matching
/// the exact planewise integral definition of Senninger et al. (2016) Section 4.3 (p. 7).
pub fn compute_segregation_metrics(
    counts: &PlaneCounts,
    sink_plane: usize,
    bulk_min_distance_planes: usize,
    lattice_a_m: f64,
) -> Result<SegregationMetrics, RisSegregationError> {
    let nx = counts.plane_count();
    if nx == 0
        || counts.cr.len() != nx
        || counts.vacancies.len() != nx
        || counts.dumbbells.len() != nx
    {
        return Err(RisSegregationError::InvalidSummary(
            "mismatched plane count vectors",
        ));
    }
    if sink_plane >= nx {
        return Err(RisSegregationError::SinkPlaneOutOfRange {
            sink_plane: sink_plane as i64,
            planes: nx,
        });
    }
    let half = nx / 2;
    if bulk_min_distance_planes == 0 || bulk_min_distance_planes > half {
        return Err(RisSegregationError::InvalidBulkWindow {
            min_distance: bulk_min_distance_planes,
            half_planes: half,
        });
    }
    let dx_nm = 0.5 * lattice_a_m * 1.0e9;
    if !lattice_a_m.is_finite() || lattice_a_m <= 0.0 || !dx_nm.is_finite() || dx_nm <= 0.0 {
        return Err(RisSegregationError::InvalidLatticeParameter(lattice_a_m));
    }

    let mut bulk_fe = 0_u64;
    let mut bulk_cr = 0_u64;
    let mut bulk_planes = 0_usize;
    let mut bulk_ratio_sum = 0.0_f64;

    for x in 0..nx {
        if counts.fe[x] == 0 {
            return Err(RisSegregationError::ZeroFePlane { plane: x });
        }
        if abs_periodic_distance(x, sink_plane, nx) >= bulk_min_distance_planes {
            bulk_fe += counts.fe[x];
            bulk_cr += counts.cr[x];
            bulk_planes += 1;
            bulk_ratio_sum += counts.cr[x] as f64 / counts.fe[x] as f64;
        }
    }
    if bulk_planes == 0 || bulk_fe == 0 {
        return Err(RisSegregationError::ZeroFeBulk);
    }

    let c_cr_gb = counts.c_cr(sink_plane);
    let c_cr_bulk = bulk_cr as f64 / (bulk_fe + bulk_cr) as f64;
    let bulk_ratio = bulk_ratio_sum / bulk_planes as f64;

    let mut excess_ratio = 0.0_f64;
    let mut excess_ratio_c = 0.0_f64;
    let mut excess_linear = 0.0_f64;
    let mut excess_linear_c = 0.0_f64;
    for x in 0..nx {
        neumaier_add(
            &mut excess_ratio,
            &mut excess_ratio_c,
            counts.cr[x] as f64 / counts.fe[x] as f64 - bulk_ratio,
        );
        neumaier_add(
            &mut excess_linear,
            &mut excess_linear_c,
            counts.c_cr(x) - c_cr_bulk,
        );
    }

    Ok(SegregationMetrics {
        c_cr_gb,
        c_cr_bulk,
        s_cr_tot_nm: dx_nm * (excess_ratio + excess_ratio_c),
        s_cr_linear_nm: dx_nm * (excess_linear + excess_linear_c),
    })
}

/// Signed-distance spatial profiles (Cr atomic fraction and steady-state V/I concentrations).
#[derive(Clone, Debug, PartialEq)]
pub struct PlaneDefectProfile {
    pub d_planes: Vec<isize>,
    pub distance_nm: Vec<f64>,
    pub c_cr: Vec<f64>,
    pub c_v_mean: Vec<f64>,
    pub c_i_mean: Vec<f64>,
    pub steady_elapsed_s: f64,
    pub v_mean_total: f64,
    pub i_mean_total: f64,
    pub v_bulk_mean: f64,
    pub i_bulk_mean: f64,
}

/// Computes the overlap of `[clock_before, clock_after]` (of width `dt`) with `[t_start, t_end]`.
#[inline]
fn interval_overlap(
    clock_before: Clock,
    clock_after: Clock,
    dt: f64,
    t_start: f64,
    t_end: f64,
) -> f64 {
    if !dt.is_finite()
        || dt <= 0.0
        || clock_after.cmp_f64(t_start).is_le()
        || clock_before.cmp_f64(t_end).is_ge()
    {
        return 0.0;
    }
    let before_ge = clock_before.cmp_f64(t_start).is_ge();
    let after_le = clock_after.cmp_f64(t_end).is_le();
    if before_ge && after_le {
        return dt;
    }
    if !before_ge && !after_le {
        return (t_end - t_start).max(0.0);
    }
    if !before_ge {
        ((clock_after.time() - t_start) + clock_after.compensation()).clamp(0.0, dt)
    } else {
        ((t_end - clock_before.time()) - clock_before.compensation()).clamp(0.0, dt)
    }
}

/// Continuous-time Neumaier-compensated accumulator for planar vacancy and dumbbell counts.
#[derive(Clone, Debug)]
pub struct PlaneDefectTally {
    cells: [usize; 3],
    v_sum: Vec<f64>,
    v_comp: Vec<f64>,
    i_sum: Vec<f64>,
    i_comp: Vec<f64>,
    elapsed: f64,
    elapsed_comp: f64,
    total_v_sum: f64,
    total_v_comp: f64,
    total_i_sum: f64,
    total_i_comp: f64,
}

impl PlaneDefectTally {
    /// Allocates zeroed planar defect tallies for `cells` (`2 * cells[0]` planes).
    pub fn new(cells: [usize; 3]) -> Result<Self, RisSegregationError> {
        let nx = 2 * cells[0];
        Ok(Self {
            cells,
            v_sum: try_vec_zero(nx, 0.0)?,
            v_comp: try_vec_zero(nx, 0.0)?,
            i_sum: try_vec_zero(nx, 0.0)?,
            i_comp: try_vec_zero(nx, 0.0)?,
            elapsed: 0.0,
            elapsed_comp: 0.0,
            total_v_sum: 0.0,
            total_v_comp: 0.0,
            total_i_sum: 0.0,
            total_i_comp: 0.0,
        })
    }

    /// Accumulates active defects in `active_defects` (`(plane_x, is_vacancy)`) over `dt_steady`.
    pub fn update_active(&mut self, active_defects: &[(usize, bool)], dt_steady: f64) {
        if dt_steady <= 0.0 || !dt_steady.is_finite() {
            return;
        }
        neumaier_add(&mut self.elapsed, &mut self.elapsed_comp, dt_steady);
        let mut n_v = 0_usize;
        let mut n_i = 0_usize;
        for &(plane, is_vacancy) in active_defects {
            if plane >= self.v_sum.len() {
                continue;
            }
            if is_vacancy {
                n_v += 1;
                neumaier_add(&mut self.v_sum[plane], &mut self.v_comp[plane], dt_steady);
            } else {
                n_i += 1;
                neumaier_add(&mut self.i_sum[plane], &mut self.i_comp[plane], dt_steady);
            }
        }
        if n_v > 0 {
            neumaier_add(
                &mut self.total_v_sum,
                &mut self.total_v_comp,
                n_v as f64 * dt_steady,
            );
        }
        if n_i > 0 {
            neumaier_add(
                &mut self.total_i_sum,
                &mut self.total_i_comp,
                n_i as f64 * dt_steady,
            );
        }
    }

    /// Collects active `(plane_x, is_vacancy)` pairs from `state` into `buf` without heap churn.
    pub fn collect_active(cells: [usize; 3], state: &FeCrRisState, buf: &mut Vec<(usize, bool)>) {
        buf.clear();
        let records = state.records();
        for &rec in state.active_records() {
            match records[rec] {
                Record::Vacancy { site } => buf.push((site_plane_x(cells, site), true)),
                Record::Dumbbell { site, .. } => buf.push((site_plane_x(cells, site), false)),
                Record::Empty => {}
            }
        }
    }

    /// Returns the total elapsed steady-state sampling time in seconds.
    pub fn steady_elapsed_s(&self) -> f64 {
        self.elapsed + self.elapsed_comp
    }

    /// Builds a signed-distance spatial profile centered at `sink_plane` (`d = -(nx/2)..(nx/2 - 1)`).
    pub fn into_profile(
        self,
        final_counts: &PlaneCounts,
        sink_plane: usize,
        bulk_min_distance_planes: usize,
        lattice_a_m: f64,
    ) -> Result<PlaneDefectProfile, RisSegregationError> {
        let nx = 2 * self.cells[0];
        if final_counts.plane_count() != nx || self.v_sum.len() != nx {
            return Err(RisSegregationError::InvalidSummary(
                "plane count mismatch building profile",
            ));
        }
        if sink_plane >= nx {
            return Err(RisSegregationError::SinkPlaneOutOfRange {
                sink_plane: sink_plane as i64,
                planes: nx,
            });
        }
        let half = nx / 2;
        if bulk_min_distance_planes == 0 || bulk_min_distance_planes > half {
            return Err(RisSegregationError::InvalidBulkWindow {
                min_distance: bulk_min_distance_planes,
                half_planes: half,
            });
        }
        let dx_nm = 0.5 * lattice_a_m * 1.0e9;
        if !lattice_a_m.is_finite() || lattice_a_m <= 0.0 || !dx_nm.is_finite() || dx_nm <= 0.0 {
            return Err(RisSegregationError::InvalidLatticeParameter(lattice_a_m));
        }

        let steady_elapsed_s = (self.elapsed + self.elapsed_comp).max(0.0);
        let plane_sites = final_counts.sites_per_plane as f64;

        let mut d_planes = try_vec_zero(nx, 0_isize)?;
        let mut distance_nm = try_vec_zero(nx, 0.0_f64)?;
        let mut c_cr = try_vec_zero(nx, 0.0_f64)?;
        let mut c_v_mean = try_vec_zero(nx, 0.0_f64)?;
        let mut c_i_mean = try_vec_zero(nx, 0.0_f64)?;

        let mut bulk_planes = 0_usize;
        let mut bulk_v_sum = 0.0_f64;
        let mut bulk_i_sum = 0.0_f64;

        for idx in 0..nx {
            let d = idx as isize - half as isize;
            let x = (sink_plane as isize + d).rem_euclid(nx as isize) as usize;
            d_planes[idx] = d;
            distance_nm[idx] = if d == 0 { 0.0 } else { d as f64 * dx_nm };
            c_cr[idx] = final_counts.c_cr(x);
            if steady_elapsed_s > 0.0 && plane_sites > 0.0 {
                let v_t = self.v_sum[x] + self.v_comp[x];
                let i_t = self.i_sum[x] + self.i_comp[x];
                c_v_mean[idx] = (v_t / steady_elapsed_s) / plane_sites;
                c_i_mean[idx] = (i_t / steady_elapsed_s) / plane_sites;
            }
            if d.unsigned_abs() >= bulk_min_distance_planes {
                bulk_planes += 1;
                bulk_v_sum += c_v_mean[idx];
                bulk_i_sum += c_i_mean[idx];
            }
        }
        if bulk_planes == 0 {
            return Err(RisSegregationError::ZeroFeBulk);
        }

        let (v_mean_total, i_mean_total) = if steady_elapsed_s > 0.0 {
            (
                (self.total_v_sum + self.total_v_comp) / steady_elapsed_s,
                (self.total_i_sum + self.total_i_comp) / steady_elapsed_s,
            )
        } else {
            (0.0, 0.0)
        };

        Ok(PlaneDefectProfile {
            d_planes,
            distance_nm,
            c_cr,
            c_v_mean,
            c_i_mean,
            steady_elapsed_s,
            v_mean_total,
            i_mean_total,
            v_bulk_mean: bulk_v_sum / bulk_planes as f64,
            i_bulk_mean: bulk_i_sum / bulk_planes as f64,
        })
    }
}

/// Configuration for an ensemble RIS segregation run.
#[derive(Clone, Debug, PartialEq)]
pub struct RisSegregationConfig {
    pub cr_count: usize,
    pub sample_doses: Vec<f64>,
    pub steady_start_dpa: f64,
    pub bulk_min_distance_planes: usize,
    pub runs: usize,
    pub seed: u64,
    pub lattice_a_m: f64,
    pub max_steps: Option<u64>,
}

impl RisSegregationConfig {
    /// Builds a uniform strictly increasing dose grid `k * dpa_end / sample_count` (`k = 1..=sample_count`).
    pub fn uniform(
        cr_count: usize,
        dpa_end: f64,
        sample_count: usize,
        steady_start_dpa: f64,
        bulk_min_distance_planes: usize,
        runs: usize,
        seed: u64,
    ) -> Result<Self, RisSegregationError> {
        if !dpa_end.is_finite() || dpa_end <= 0.0 {
            return Err(RisSegregationError::InvalidDoseGrid(
                "dpa_end must be finite and > 0",
            ));
        }
        if sample_count == 0 {
            return Err(RisSegregationError::InvalidDoseGrid(
                "sample_count must be >= 1",
            ));
        }
        let mut sample_doses = Vec::new();
        sample_doses
            .try_reserve_exact(sample_count)
            .map_err(|_| RisSegregationError::Allocation(sample_count))?;
        for k in 1..=sample_count {
            let d = if k == sample_count {
                dpa_end
            } else {
                let prod = k as f64 * dpa_end;
                let v = if prod.is_finite() {
                    prod / sample_count as f64
                } else {
                    (k as f64 / sample_count as f64) * dpa_end
                };
                v.clamp(0.0, dpa_end)
            };
            sample_doses.push(d);
        }
        Ok(Self {
            cr_count,
            sample_doses,
            steady_start_dpa,
            bulk_min_distance_planes,
            runs,
            seed,
            lattice_a_m: DEFAULT_LATTICE_A_M,
            max_steps: None,
        })
    }

    pub fn with_lattice_a_m(mut self, lattice_a_m: f64) -> Self {
        self.lattice_a_m = lattice_a_m;
        self
    }

    pub fn with_max_steps(mut self, max_steps: Option<u64>) -> Self {
        self.max_steps = max_steps;
        self
    }
}

/// Segregation metrics observed at one sample dose on a single replica.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SegregationSample {
    pub dose: f64,
    pub time_s: f64,
    pub metrics: SegregationMetrics,
}

/// Full output of one RIS grain-boundary segregation replica.
#[derive(Clone, Debug, PartialEq)]
pub struct RisSegregationReplica {
    pub replica_id: u64,
    pub steps: u64,
    pub final_dose: f64,
    pub final_time_s: f64,
    pub created: u64,
    pub recombined: u64,
    pub absorbed_v: u64,
    pub absorbed_i: u64,
    pub borrowed_fe: u64,
    pub creation_attempts: u64,
    pub floored_dumbbell_jumps: u64,
    pub floored_vacancy_jumps: u64,
    pub samples: Vec<SegregationSample>,
    pub profile: PlaneDefectProfile,
}

/// Places `cr_count` Cr atoms on `sites` by partial Fisher-Yates shuffle without overflow.
pub fn place_random_cr(sites: usize, cr_count: usize, rng: &mut KmcRng) -> Vec<Species> {
    let mut order: Vec<usize> = (0..sites).collect();
    for i in 0..cr_count {
        let rem = sites - i;
        let j = i + ((rng.open01() * rem as f64) as usize).min(rem - 1);
        order.swap(i, j);
    }
    let mut occ = vec![Species::Fe; sites];
    for &site in &order[..cr_count] {
        occ[site] = Species::Cr;
    }
    occ
}

fn validate_run_inputs(
    model: &FeCrRis,
    config: &RisSegregationConfig,
) -> Result<(usize, f64, Vec<f64>), RisSegregationError> {
    let params = model.params();
    if params.sink_planes.len() != 1 {
        return Err(RisSegregationError::InvalidSinkPlanes(
            params.sink_planes.len(),
        ));
    }
    let nx = 2 * model.lattice().cells()[0];
    let raw_sink = params.sink_planes[0];
    let sink_plane = usize::try_from(raw_sink).ok().filter(|&p| p < nx).ok_or(
        RisSegregationError::SinkPlaneOutOfRange {
            sink_plane: raw_sink,
            planes: nx,
        },
    )?;
    let k0 = params.k0_dpa_per_s;
    if !k0.is_finite() || k0 <= 0.0 {
        return Err(RisSegregationError::NonPositiveCreationRate(k0));
    }
    let site_count = model.lattice().site_count();
    if config.cr_count > site_count {
        return Err(RisSegregationError::InvalidCrCount {
            cr_count: config.cr_count,
            max: site_count,
        });
    }
    let dx_nm = 0.5 * config.lattice_a_m * 1.0e9;
    if !config.lattice_a_m.is_finite()
        || config.lattice_a_m <= 0.0
        || !dx_nm.is_finite()
        || dx_nm <= 0.0
    {
        return Err(RisSegregationError::InvalidLatticeParameter(
            config.lattice_a_m,
        ));
    }
    let half = nx / 2;
    if config.bulk_min_distance_planes == 0 || config.bulk_min_distance_planes > half {
        return Err(RisSegregationError::InvalidBulkWindow {
            min_distance: config.bulk_min_distance_planes,
            half_planes: half,
        });
    }
    if config.runs == 0 {
        return Err(RisSegregationError::InsufficientReplicas(0));
    }
    if config.sample_doses.is_empty() {
        return Err(RisSegregationError::InvalidDoseGrid(
            "sample_doses must contain at least one dose",
        ));
    }

    let mut t_samples = Vec::new();
    t_samples
        .try_reserve_exact(config.sample_doses.len())
        .map_err(|_| RisSegregationError::Allocation(config.sample_doses.len()))?;
    let mut prev_d = 0.0_f64;
    let mut prev_t = 0.0_f64;
    for (idx, &d) in config.sample_doses.iter().enumerate() {
        if !d.is_finite() || d <= 0.0 {
            return Err(RisSegregationError::InvalidDoseGrid(
                "all sample doses must be finite and > 0",
            ));
        }
        if idx > 0 && d <= prev_d {
            return Err(RisSegregationError::InvalidDoseGrid(
                "sample doses must be strictly increasing",
            ));
        }
        let t = d / k0;
        if !t.is_finite() || t <= 0.0 || (idx > 0 && t <= prev_t) {
            return Err(RisSegregationError::InvalidDoseGrid(
                "sample times d / k0 must be finite and strictly increasing",
            ));
        }
        prev_d = d;
        prev_t = t;
        t_samples.push(t);
    }

    let dpa_end = *config
        .sample_doses
        .last()
        .ok_or(RisSegregationError::InvalidDoseGrid("empty dose grid"))?;
    if !config.steady_start_dpa.is_finite()
        || config.steady_start_dpa < 0.0
        || config.steady_start_dpa >= dpa_end
    {
        return Err(RisSegregationError::InvalidSteadyStart {
            steady_start_dpa: config.steady_start_dpa,
            dpa_end,
        });
    }
    let t_steady_start = config.steady_start_dpa / k0;
    if !t_steady_start.is_finite() {
        return Err(RisSegregationError::InvalidSteadyStart {
            steady_start_dpa: config.steady_start_dpa,
            dpa_end,
        });
    }

    Ok((sink_plane, t_steady_start, t_samples))
}

fn run_single_replica(
    model: &FeCrRis,
    config: &RisSegregationConfig,
    sink_plane: usize,
    t_steady_start: f64,
    t_samples: &[f64],
    ctx: ReplicaContext,
) -> Result<RisSegregationReplica, RisSegregationError> {
    let k0 = model.params().k0_dpa_per_s;
    let cells = model.lattice().cells();
    let sites = model.lattice().site_count();
    let mut init_rng = ctx.rng(StreamRole::InitialState);
    let occ = place_random_cr(sites, config.cr_count, &mut init_rng);
    let state = model.state_with_defects(occ, &[], ctx.base_seed.wrapping_add(ctx.id))?;
    let mut sim =
        IncrementalSimulation::with_rng(model.clone(), state, ctx.rng(StreamRole::Dynamics))
            .map_err(|source| RisSegregationError::Sim {
                steps: 0,
                time: 0.0,
                source,
            })?;

    let Some(&t_end) = t_samples.last() else {
        return Err(RisSegregationError::InvalidDoseGrid("empty dose grid"));
    };
    let mut tally = PlaneDefectTally::new(cells)?;
    let mut samples = Vec::new();
    samples
        .try_reserve_exact(config.sample_doses.len())
        .map_err(|_| RisSegregationError::Allocation(config.sample_doses.len()))?;
    let mut active_buf: Vec<(usize, bool)> = Vec::new();
    let mut last_counts: Option<PlaneCounts> = None;

    while samples.len() < t_samples.len() {
        if sim.clock().cmp_f64(t_samples[samples.len()]).is_eq() {
            let counts = PlaneCounts::from_state(sim.model(), sim.state())?;
            let metrics = compute_segregation_metrics(
                &counts,
                sink_plane,
                config.bulk_min_distance_planes,
                config.lattice_a_m,
            )?;
            let exact_time = sim.clock().time();
            while samples.len() < t_samples.len()
                && sim.clock().cmp_f64(t_samples[samples.len()]).is_eq()
            {
                let dose = config.sample_doses[samples.len()];
                samples.push(SegregationSample {
                    dose,
                    time_s: exact_time,
                    metrics,
                });
            }
            last_counts = Some(counts);
            if samples.len() >= t_samples.len() {
                break;
            }
        }

        if let Some(limit) = config.max_steps
            && sim.steps() >= limit
        {
            return Err(RisSegregationError::MaxStepsExceeded {
                max_steps: limit,
                dose: k0 * sim.time(),
            });
        }

        let clock_before = sim.clock();
        let next_idx = samples.len();
        let next_target = t_samples[next_idx];
        let mut crossed_counts: Result<Option<PlaneCounts>, RisSegregationError> = Ok(None);
        active_buf.clear();

        let step = sim
            .step_observing(|m, state_before, next_clock| {
                if next_clock.cmp_f64(t_steady_start).is_gt() && clock_before.cmp_f64(t_end).is_lt()
                {
                    PlaneDefectTally::collect_active(cells, state_before, &mut active_buf);
                }
                if next_clock.cmp_f64(next_target).is_gt() {
                    crossed_counts = PlaneCounts::from_state(m, state_before).map(Some);
                }
            })
            .map_err(|source| RisSegregationError::Sim {
                steps: sim.steps(),
                time: sim.time(),
                source,
            })?;

        match step {
            Step::Fired { dt, .. } => {
                let dt_steady =
                    interval_overlap(clock_before, sim.clock(), dt, t_steady_start, t_end);
                tally.update_active(&active_buf, dt_steady);

                if let Some(counts) = crossed_counts? {
                    let metrics = compute_segregation_metrics(
                        &counts,
                        sink_plane,
                        config.bulk_min_distance_planes,
                        config.lattice_a_m,
                    )?;
                    while samples.len() < t_samples.len()
                        && sim.clock().cmp_f64(t_samples[samples.len()]).is_gt()
                    {
                        let idx = samples.len();
                        let t_grid = t_samples[idx];
                        samples.push(SegregationSample {
                            dose: config.sample_doses[idx],
                            time_s: t_grid,
                            metrics,
                        });
                    }
                    last_counts = Some(counts);
                }
            }
            Step::Absorbed => {
                return Err(RisSegregationError::Absorbed {
                    steps: sim.steps(),
                    dose: k0 * sim.time(),
                });
            }
        }
    }

    let final_counts = match last_counts {
        Some(counts) => counts,
        None => PlaneCounts::from_state(sim.model(), sim.state())?,
    };
    let profile = tally.into_profile(
        &final_counts,
        sink_plane,
        config.bulk_min_distance_planes,
        config.lattice_a_m,
    )?;
    let s = sim.state();
    let final_time_s = sim.time();
    Ok(RisSegregationReplica {
        replica_id: ctx.id,
        steps: sim.steps(),
        final_dose: k0 * final_time_s,
        final_time_s,
        created: s.created(),
        recombined: s.recombined(),
        absorbed_v: s.absorbed_v(),
        absorbed_i: s.absorbed_i(),
        borrowed_fe: s.borrowed_fe(),
        creation_attempts: s.creation_attempts(),
        floored_dumbbell_jumps: s.floored_dumbbell_jumps(),
        floored_vacancy_jumps: s.floored_vacancy_jumps(),
        samples,
        profile,
    })
}

/// Runs an ensemble of `config.runs` independent grain-boundary RIS replicas in parallel.
pub fn run_ris_segregation(
    model: &FeCrRis,
    config: &RisSegregationConfig,
) -> Result<Vec<RisSegregationReplica>, RisSegregationError> {
    let (sink_plane, t_steady_start, t_samples) = validate_run_inputs(model, config)?;
    let mut probe = Vec::<RisSegregationReplica>::new();
    probe
        .try_reserve_exact(config.runs)
        .map_err(|_| RisSegregationError::Allocation(config.runs))?;
    drop(probe);

    run_ensemble(
        config.seed,
        config.runs as u64,
        |ctx| -> Result<RisSegregationReplica, RisSegregationError> {
            run_single_replica(model, config, sink_plane, t_steady_start, &t_samples, ctx)
        },
    )
    .into_iter()
    .collect()
}

#[inline]
fn mean_and_se_or_zero(values: &[f64]) -> (f64, f64) {
    if values.len() <= 1 {
        (values.first().copied().unwrap_or(0.0), 0.0)
    } else {
        mean_se(values)
    }
}

/// Ensemble mean and standard error summary across RIS replicas.
#[derive(Clone, Debug, PartialEq)]
pub struct RisSegregationSummary {
    pub runs: usize,
    pub doses: Vec<f64>,
    pub c_cr_gb: Vec<(f64, f64)>,
    pub c_cr_bulk: Vec<(f64, f64)>,
    pub s_cr_tot_nm: Vec<(f64, f64)>,
    pub s_cr_linear_nm: Vec<(f64, f64)>,
    pub d_planes: Vec<isize>,
    pub distance_nm: Vec<f64>,
    pub profile_c_cr: Vec<(f64, f64)>,
    pub profile_c_v: Vec<(f64, f64)>,
    pub profile_c_i: Vec<(f64, f64)>,
    pub v_mean_total: (f64, f64),
    pub i_mean_total: (f64, f64),
    pub v_bulk_mean: (f64, f64),
    pub i_bulk_mean: (f64, f64),
    pub mean_steps: f64,
    pub mean_created: f64,
    pub mean_recombined: f64,
    pub mean_absorbed_v: f64,
    pub mean_absorbed_i: f64,
    pub mean_borrowed_fe: f64,
    pub mean_floored_dumbbell_jumps: f64,
    pub mean_floored_vacancy_jumps: f64,
}

/// Aggregates ensemble means and standard errors over replicas returned by [`run_ris_segregation`].
pub fn summarize_ris_segregation(
    replicas: &[RisSegregationReplica],
) -> Result<RisSegregationSummary, RisSegregationError> {
    if replicas.is_empty() {
        return Err(RisSegregationError::InsufficientReplicas(0));
    }
    let first = &replicas[0];
    let n_samples = first.samples.len();
    let nx = first.profile.d_planes.len();
    if n_samples == 0 || nx == 0 {
        return Err(RisSegregationError::InvalidSummary(
            "replicas must contain at least one sample and plane",
        ));
    }
    let doses: Vec<f64> = first.samples.iter().map(|s| s.dose).collect();
    for r in replicas {
        if r.samples.len() != n_samples || r.profile.d_planes.len() != nx {
            return Err(RisSegregationError::InvalidSummary(
                "replicas have inconsistent sample or plane counts",
            ));
        }
        if r.samples
            .iter()
            .zip(&doses)
            .any(|(s, &d)| s.dose.to_bits() != d.to_bits())
        {
            return Err(RisSegregationError::InvalidSummary(
                "replicas have mismatched sample dose grids",
            ));
        }
        if r.profile.d_planes != first.profile.d_planes {
            return Err(RisSegregationError::InvalidSummary(
                "replicas have mismatched signed plane grids",
            ));
        }
    }

    let mut c_cr_gb = Vec::with_capacity(n_samples);
    let mut c_cr_bulk = Vec::with_capacity(n_samples);
    let mut s_cr_tot_nm = Vec::with_capacity(n_samples);
    let mut s_cr_linear_nm = Vec::with_capacity(n_samples);
    let mut scratch = Vec::with_capacity(replicas.len());

    for k in 0..n_samples {
        scratch.clear();
        scratch.extend(replicas.iter().map(|r| r.samples[k].metrics.c_cr_gb));
        c_cr_gb.push(mean_and_se_or_zero(&scratch));

        scratch.clear();
        scratch.extend(replicas.iter().map(|r| r.samples[k].metrics.c_cr_bulk));
        c_cr_bulk.push(mean_and_se_or_zero(&scratch));

        scratch.clear();
        scratch.extend(replicas.iter().map(|r| r.samples[k].metrics.s_cr_tot_nm));
        s_cr_tot_nm.push(mean_and_se_or_zero(&scratch));

        scratch.clear();
        scratch.extend(replicas.iter().map(|r| r.samples[k].metrics.s_cr_linear_nm));
        s_cr_linear_nm.push(mean_and_se_or_zero(&scratch));
    }

    let mut profile_c_cr = Vec::with_capacity(nx);
    let mut profile_c_v = Vec::with_capacity(nx);
    let mut profile_c_i = Vec::with_capacity(nx);
    for p in 0..nx {
        scratch.clear();
        scratch.extend(replicas.iter().map(|r| r.profile.c_cr[p]));
        profile_c_cr.push(mean_and_se_or_zero(&scratch));

        scratch.clear();
        scratch.extend(replicas.iter().map(|r| r.profile.c_v_mean[p]));
        profile_c_v.push(mean_and_se_or_zero(&scratch));

        scratch.clear();
        scratch.extend(replicas.iter().map(|r| r.profile.c_i_mean[p]));
        profile_c_i.push(mean_and_se_or_zero(&scratch));
    }

    scratch.clear();
    scratch.extend(replicas.iter().map(|r| r.profile.v_mean_total));
    let v_mean_total = mean_and_se_or_zero(&scratch);

    scratch.clear();
    scratch.extend(replicas.iter().map(|r| r.profile.i_mean_total));
    let i_mean_total = mean_and_se_or_zero(&scratch);

    scratch.clear();
    scratch.extend(replicas.iter().map(|r| r.profile.v_bulk_mean));
    let v_bulk_mean = mean_and_se_or_zero(&scratch);

    scratch.clear();
    scratch.extend(replicas.iter().map(|r| r.profile.i_bulk_mean));
    let i_bulk_mean = mean_and_se_or_zero(&scratch);

    let inv_r = 1.0 / replicas.len() as f64;
    let mean_of = |f: fn(&RisSegregationReplica) -> f64| -> f64 {
        replicas.iter().map(f).sum::<f64>() * inv_r
    };

    Ok(RisSegregationSummary {
        runs: replicas.len(),
        doses,
        c_cr_gb,
        c_cr_bulk,
        s_cr_tot_nm,
        s_cr_linear_nm,
        d_planes: first.profile.d_planes.clone(),
        distance_nm: first.profile.distance_nm.clone(),
        profile_c_cr,
        profile_c_v,
        profile_c_i,
        v_mean_total,
        i_mean_total,
        v_bulk_mean,
        i_bulk_mean,
        mean_steps: mean_of(|r| r.steps as f64),
        mean_created: mean_of(|r| r.created as f64),
        mean_recombined: mean_of(|r| r.recombined as f64),
        mean_absorbed_v: mean_of(|r| r.absorbed_v as f64),
        mean_absorbed_i: mean_of(|r| r.absorbed_i as f64),
        mean_borrowed_fe: mean_of(|r| r.borrowed_fe as f64),
        mean_floored_dumbbell_jumps: mean_of(|r| r.floored_dumbbell_jumps as f64),
        mean_floored_vacancy_jumps: mean_of(|r| r.floored_vacancy_jumps as f64),
    })
}

/// Fitted exponential relaxation `y(phi) = y_inf + (y0 - y_inf) * exp(-phi / phi_rel)`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RelaxationFit {
    pub phi_rel: f64,
    pub y_inf: f64,
    pub y0: f64,
    pub rmse: f64,
    pub r2: f64,
}

fn eval_relaxation_sse(doses: &[f64], values: &[f64], y0: f64, phi_rel: f64) -> Option<(f64, f64)> {
    if !phi_rel.is_finite() || phi_rel <= 0.0 {
        return None;
    }
    let mut sum_w2 = 0.0_f64;
    let mut sum_wr = 0.0_f64;
    for (&phi, &y) in doses.iter().zip(values) {
        let w = -(-phi / phi_rel).exp_m1();
        let dy = y - y0;
        sum_w2 += w * w;
        sum_wr += w * dy;
    }
    if !sum_w2.is_finite() || sum_w2 <= 0.0 || !sum_wr.is_finite() {
        return None;
    }
    let delta_inf = sum_wr / sum_w2;
    let y_inf = y0 + delta_inf;
    if !y_inf.is_finite() || !delta_inf.is_finite() {
        return None;
    }
    let mut sse = 0.0_f64;
    for (&phi, &y) in doses.iter().zip(values) {
        let w = -(-phi / phi_rel).exp_m1();
        let pred = y0 + delta_inf * w;
        let diff = y - pred;
        sse += diff * diff;
    }
    sse.is_finite().then_some((sse, y_inf))
}

/// Fits `y(phi) = y_inf + (y0 - y_inf) * exp(-phi / phi_rel)` with fixed `y0` by profile least squares.
///
/// Uses a closed-form optimal `y_inf(phi_rel)` evaluated on a log-spaced coarse grid across
/// `[1e-4 * doses[0], 100 * doses.last()]` followed by golden-section refinement. Returns `None`
/// if `doses.len() < 3`, inputs are non-finite or non-increasing, or values have zero variance.
pub fn fit_exponential_relaxation(doses: &[f64], values: &[f64], y0: f64) -> Option<RelaxationFit> {
    let n = doses.len();
    if n < 3 || n != values.len() || !y0.is_finite() {
        return None;
    }
    for i in 0..n {
        if !doses[i].is_finite() || doses[i] <= 0.0 || !values[i].is_finite() {
            return None;
        }
        if i > 0 && doses[i] <= doses[i - 1] {
            return None;
        }
    }
    if values.windows(2).all(|w| w[0] == w[1]) {
        return None;
    }
    let y_mean = values.iter().sum::<f64>() / n as f64;
    let sst = values
        .iter()
        .map(|&y| (y - y_mean) * (y - y_mean))
        .sum::<f64>();
    if !sst.is_finite() || sst <= 1.0e-24 * y_mean * y_mean * n as f64 {
        return None;
    }

    let u_min = (1.0e-4 * doses[0]).ln();
    let u_max = (100.0 * doses[n - 1]).ln();
    if !u_min.is_finite() || !u_max.is_finite() || u_max <= u_min {
        return None;
    }

    const GRID: usize = 200;
    let grid_u = |idx: usize| u_min + (u_max - u_min) * (idx as f64 / GRID as f64);
    let mut best_idx = None;
    let mut best_sse = f64::INFINITY;
    for i in 0..=GRID {
        let u = grid_u(i);
        if let Some((sse, _)) = eval_relaxation_sse(doses, values, y0, u.exp())
            && sse < best_sse
        {
            best_sse = sse;
            best_idx = Some(i);
        }
    }
    let idx = best_idx?;
    let mut a = grid_u(idx.saturating_sub(1));
    let mut b = grid_u((idx + 1).min(GRID));

    let eval_u = |u: f64| -> f64 {
        eval_relaxation_sse(doses, values, y0, u.exp()).map_or(f64::INFINITY, |(sse, _)| sse)
    };
    let inv_phi = 0.5 * (5.0_f64.sqrt() - 1.0);
    let mut c = b - inv_phi * (b - a);
    let mut d = a + inv_phi * (b - a);
    let mut fc = eval_u(c);
    let mut fd = eval_u(d);
    for _ in 0..80 {
        if fc < fd {
            b = d;
            d = c;
            fd = fc;
            c = b - inv_phi * (b - a);
            fc = eval_u(c);
        } else {
            a = c;
            c = d;
            fc = fd;
            d = a + inv_phi * (b - a);
            fd = eval_u(d);
        }
    }

    let u_opt = 0.5 * (a + b);
    let phi_rel = u_opt.exp();
    let (sse, y_inf) = eval_relaxation_sse(doses, values, y0, phi_rel)?;
    let rmse = (sse / n as f64).sqrt();
    let r2 = 1.0 - sse / sst;
    if !(phi_rel > 0.0
        && phi_rel.is_finite()
        && y_inf.is_finite()
        && rmse.is_finite()
        && r2.is_finite())
    {
        return None;
    }

    Some(RelaxationFit {
        phi_rel,
        y_inf,
        y0,
        rmse,
        r2,
    })
}
