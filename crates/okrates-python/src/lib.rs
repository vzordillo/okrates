//! Thin Python bindings for okrates.
//!
//! Native Rust owns all simulation state; results are returned to Python as copies.

use okrates_core::{Simulation, Step};
use okrates_models::bcc::BccLattice;
use okrates_models::fecr::{Energetics, FeCrModel, PairEnergetics, Species};
use okrates_models::fecr_diffusion::{run_composition, summarize};
use okrates_models::fecr_martinez::{Martinez2012, params};
use okrates_models::vacancy::VacancyDiffusion;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

/// Computes an Arrhenius jump rate in s^-1.
///
/// Arguments are the attempt frequency in s^-1, the migration energy in eV and
/// the temperature in K. Raises `ValueError` if any argument is invalid.
#[pyfunction]
fn jump_rate(
    attempt_frequency: f64,
    migration_energy_ev: f64,
    temperature_k: f64,
) -> PyResult<f64> {
    okrates_models::vacancy::jump_rate(attempt_frequency, migration_energy_ev, temperature_k)
        .map_err(|err| PyValueError::new_err(err.to_string()))
}

/// Runs a BCC single-vacancy trajectory and returns `(times, squared_displacements)`.
///
/// `times` are in seconds and `squared_displacements` in m^2; each has
/// `steps + 1` entries, starting with the initial point (shorter if the
/// system is absorbed). `jump_rate` is in s^-1 and `lattice_parameter_m` in
/// metres. Raises `ValueError` for a non-finite or non-positive `jump_rate` or
/// `lattice_parameter_m`, or for invalid `cells`, and `RuntimeError` if the
/// simulation fails.
#[pyfunction]
#[pyo3(signature = (*, cells, jump_rate, lattice_parameter_m, seed, steps))]
fn vacancy_trajectory(
    py: Python<'_>,
    cells: [usize; 3],
    jump_rate: f64,
    lattice_parameter_m: f64,
    seed: u64,
    steps: u64,
) -> PyResult<(Vec<f64>, Vec<f64>)> {
    if !jump_rate.is_finite() || jump_rate <= 0.0 {
        return Err(PyValueError::new_err(format!(
            "jump_rate must be finite and positive, got {jump_rate}"
        )));
    }
    if !lattice_parameter_m.is_finite() || lattice_parameter_m <= 0.0 {
        return Err(PyValueError::new_err(format!(
            "lattice_parameter_m must be finite and positive, got {lattice_parameter_m}"
        )));
    }
    let lattice = BccLattice::new(cells).map_err(|err| PyValueError::new_err(err.to_string()))?;

    let result = py.detach(move || -> Result<(Vec<f64>, Vec<f64>), String> {
        let model = VacancyDiffusion::new(lattice, jump_rate);
        let state = model.initial_state(0).map_err(|err| err.to_string())?;
        let mut sim = Simulation::new(model, state, seed);
        let mut times = vec![sim.time()];
        let mut sq = vec![sim.state().squared_displacement(lattice_parameter_m)];
        for _ in 0..steps {
            match sim.step() {
                Ok(Step::Fired { .. }) => {
                    times.push(sim.time());
                    sq.push(sim.state().squared_displacement(lattice_parameter_m));
                }
                Ok(Step::Absorbed) => break,
                Err(err) => return Err(err.to_string()),
            }
        }
        Ok((times, sq))
    });
    result.map_err(PyRuntimeError::new_err)
}

/// Returns the ILLUSTRATIVE Fe-Cr energetics as `(pair_energies_ev, kra_fe, kra_cr)`.
///
/// The pair energies are a symmetric 3x3 nested list in eV indexed Fe, Cr,
/// vacancy. The values are round numbers with no source, not DFT or literature.
#[pyfunction]
fn fecr_illustrative_energetics() -> ([[f64; 3]; 3], f64, f64) {
    let e = PairEnergetics::illustrative();
    (*e.pair(), e.kra(Species::Fe), e.kra(Species::Cr))
}

fn value_error(message: impl Into<String>) -> PyErr {
    PyValueError::new_err(message.into())
}

/// Raw output of the Fe-Cr diffusion pipeline: ten scalars
/// `[D_v, se, D_Fe, se, D_Cr, se, D_all, se, f, se]`, the Lambda means
/// `[FeFe, FeCr, CrCr]`, their standard errors, the mean jump count, and the
/// barrier-floor activations during the call (0 for the illustrative model).
type FeCrRaw = (Vec<f64>, Vec<f64>, Vec<f64>, f64, u64);

/// Runs the pipeline for one model and flattens the summary into [`FeCrRaw`].
#[allow(clippy::too_many_arguments)]
fn run_fecr<E: Energetics + Clone + Sync + Send>(
    py: Python<'_>,
    model: FeCrModel<E>,
    n_sites: usize,
    cr_count: usize,
    t_end: f64,
    k_count: usize,
    lattice_parameter_m: f64,
    seed: u64,
    runs: u64,
    anneal_sweeps: u64,
    floor_activations: impl FnOnce() -> u64,
) -> PyResult<FeCrRaw> {
    if cr_count + 1 > n_sites {
        return Err(value_error(format!(
            "cr_count {cr_count} plus one vacancy exceeds the {n_sites} lattice sites"
        )));
    }
    let result = py.detach(move || {
        run_composition(
            &model,
            cr_count,
            seed,
            runs,
            t_end,
            k_count,
            lattice_parameter_m,
            anneal_sweeps,
        )
        .map(|replicas| summarize(&replicas, cr_count, n_sites - 1))
    });
    let s = result.map_err(|err| value_error(err.to_string()))?;
    let scalars = [s.vacancy, s.tracer[0], s.tracer[1], s.tracer_all, s.f]
        .iter()
        .flat_map(|&(mean, se)| [mean, se])
        .collect();
    let lambda = s.collective.iter().map(|&(mean, _)| mean).collect();
    let lambda_se = s.collective.iter().map(|&(_, se)| se).collect();
    Ok((
        scalars,
        lambda,
        lambda_se,
        s.mean_jumps,
        floor_activations(),
    ))
}

/// Runs the Fe-Cr single-vacancy diffusion pipeline for one composition.
///
/// Returns `(scalars, lambda, lambda_se, mean_jumps, barrier_floor_activations)`; see the Python wrapper
/// `okrates.fecr_diffusion`. Raises `ValueError` for invalid arguments.
#[pyfunction]
#[pyo3(signature = (
    *, cells, cr_count, temperature_k, attempt_frequency = None, t_end, k_count,
    lattice_parameter_m, seed, runs, anneal_sweeps = 0,
    pair_energies_ev = None, kra_ev = None, energetics = "illustrative"
))]
#[allow(clippy::too_many_arguments)]
fn fecr_diffusion(
    py: Python<'_>,
    cells: [usize; 3],
    cr_count: usize,
    temperature_k: f64,
    attempt_frequency: Option<f64>,
    t_end: f64,
    k_count: usize,
    lattice_parameter_m: f64,
    seed: u64,
    runs: u64,
    anneal_sweeps: u64,
    pair_energies_ev: Option<[[f64; 3]; 3]>,
    kra_ev: Option<(f64, f64)>,
    energetics: &str,
) -> PyResult<FeCrRaw> {
    if runs < 2 {
        return Err(value_error(format!("runs must be at least 2, got {runs}")));
    }
    if k_count < 1 {
        return Err(value_error("k_count must be at least 1"));
    }
    if !t_end.is_finite() || t_end <= 0.0 {
        return Err(value_error(format!(
            "t_end must be finite and positive, got {t_end}"
        )));
    }
    if !lattice_parameter_m.is_finite() || lattice_parameter_m <= 0.0 {
        return Err(value_error(format!(
            "lattice_parameter_m must be finite and positive, got {lattice_parameter_m}"
        )));
    }
    match energetics {
        "illustrative" => {
            let attempt_frequency = attempt_frequency.ok_or_else(|| {
                value_error("attempt_frequency is required for the illustrative energetics")
            })?;
            let e = match (pair_energies_ev, kra_ev) {
                (None, None) => PairEnergetics::illustrative(),
                (Some(pair), Some((kra_fe, kra_cr))) => {
                    PairEnergetics::new(pair, kra_fe, kra_cr)
                        .map_err(|err| value_error(err.to_string()))?
                }
                _ => {
                    return Err(value_error(
                        "pair_energies_ev and kra_ev must be given together or both omitted",
                    ));
                }
            };
            let lattice = BccLattice::new(cells).map_err(|err| value_error(err.to_string()))?;
            let n_sites = lattice.site_count();
            let model = FeCrModel::new(lattice, e, attempt_frequency, temperature_k)
                .map_err(|err| value_error(err.to_string()))?;
            run_fecr(
                py,
                model,
                n_sites,
                cr_count,
                t_end,
                k_count,
                lattice_parameter_m,
                seed,
                runs,
                anneal_sweeps,
                || 0,
            )
        }
        "martinez2012" => {
            if pair_energies_ev.is_some() || kra_ev.is_some() {
                return Err(value_error(
                    "pair_energies_ev and kra_ev apply only to the illustrative energetics",
                ));
            }
            if attempt_frequency.is_some() {
                return Err(value_error(
                    "attempt_frequency must be omitted for martinez2012: the attempt \
                     frequencies come from the Martinez 2012 table",
                ));
            }
            let e = Martinez2012::new(temperature_k).map_err(|err| value_error(err.to_string()))?;
            let lattice = BccLattice::new(cells).map_err(|err| value_error(err.to_string()))?;
            let n_sites = lattice.site_count();
            let probe = e.clone();
            let model = FeCrModel::new(lattice, e, params::NU_FE, temperature_k)
                .map_err(|err| value_error(err.to_string()))?;
            run_fecr(
                py,
                model,
                n_sites,
                cr_count,
                t_end,
                k_count,
                lattice_parameter_m,
                seed,
                runs,
                anneal_sweeps,
                move || probe.floor_activations(),
            )
        }
        other => Err(value_error(format!(
            "energetics must be \"illustrative\" or \"martinez2012\", got {other:?}"
        ))),
    }
}

#[pymodule]
fn _okrates(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(jump_rate, m)?)?;
    m.add_function(wrap_pyfunction!(vacancy_trajectory, m)?)?;
    m.add_function(wrap_pyfunction!(fecr_illustrative_energetics, m)?)?;
    m.add_function(wrap_pyfunction!(fecr_diffusion, m)?)?;
    Ok(())
}
