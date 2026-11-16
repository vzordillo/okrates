//! Thin Python bindings for okrates.
//!
//! Native Rust owns all simulation state; results are returned to Python as copies.

use okrates_core::{Simulation, Step};
use okrates_models::bcc::BccLattice;
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

#[pymodule]
fn _okrates(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(jump_rate, m)?)?;
    m.add_function(wrap_pyfunction!(vacancy_trajectory, m)?)?;
    Ok(())
}
