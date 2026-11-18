//! PyO3 extension module backing the `okrates` Python package.

use okrates_core::{
    Checkpointable, KmcRng, SampleGrid, SamplingError, Simulation, Step, StreamRole,
};
use okrates_models::bcc::BccLattice;
use okrates_models::fecr::{Energetics, FeCrModel, PairEnergetics, Species};
use okrates_models::fecr_anneal::KawasakiAnnealer;
use okrates_models::fecr_diffusion::{
    FitMethod, TransportConfig, TransportError, build_fecr_manifest, run_transport, summarize,
};
use okrates_models::fecr_martinez::{BarrierFloorPolicy, Martinez2012, params};
use okrates_models::tabulated::TabulatedEnergetics;
use okrates_models::vacancy::VacancyDiffusion;
use okrates_schema::{BarrierSet, import_aiida_qe_json, import_pymatgen_vasp_json};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

type FeCrTransportRaw = (
    Vec<f64>,
    Vec<f64>,
    Vec<f64>,
    f64,
    f64,
    u64,
    Option<f64>,
    String,
);

type AnnealRaw = (Vec<f64>, Vec<f64>, u64, u64, f64, f64, f64);

fn value_error(message: impl Into<String>) -> PyErr {
    PyValueError::new_err(message.into())
}

fn runtime_error(message: impl Into<String>) -> PyErr {
    PyRuntimeError::new_err(message.into())
}

fn transport_error_to_py(err: TransportError) -> PyErr {
    match err {
        TransportError::Model(_)
        | TransportError::InvalidLatticeParameter(_)
        | TransportError::InsufficientSamples { .. }
        | TransportError::InsufficientReplicas(_)
        | TransportError::Sampling(SamplingError::Grid(_)) => value_error(err.to_string()),
        TransportError::Sampling(
            SamplingError::Sim { .. }
            | SamplingError::Absorbed { .. }
            | SamplingError::MaxStepsExceeded { .. }
            | SamplingError::GridBeforeSimTime { .. },
        ) => runtime_error(err.to_string()),
    }
}

fn parse_fit_method(name: &str) -> PyResult<FitMethod> {
    match name {
        "windowed_ols" => Ok(FitMethod::WindowedOls),
        "origin_slope" => Ok(FitMethod::OriginSlope),
        other => Err(value_error(format!(
            "fit_method must be \"windowed_ols\" or \"origin_slope\", got {other:?}"
        ))),
    }
}

fn parse_barrier_policy(name: &str) -> PyResult<BarrierFloorPolicy> {
    match name {
        "symmetric_floor" => Ok(BarrierFloorPolicy::SymmetricFloor),
        "unfloored" => Ok(BarrierFloorPolicy::Unfloored),
        "strict" => Ok(BarrierFloorPolicy::Strict),
        other => Err(value_error(format!(
            "barrier_policy must be \"symmetric_floor\", \"unfloored\", or \"strict\", got {other:?}"
        ))),
    }
}

struct EnergeticsArgs<'a> {
    attempt_frequency: Option<f64>,
    pair_energies_ev: Option<[[f64; 3]; 3]>,
    kra_ev: Option<(f64, f64)>,
    barrier_policy: &'a str,
    attempt_frequencies_hz: Option<(f64, f64)>,
    saddle_shift_ev: f64,
    barrier_set_json: Option<&'a str>,
}

fn build_pair_model(
    cells: [usize; 3],
    temperature_k: f64,
    args: &EnergeticsArgs<'_>,
) -> PyResult<FeCrModel<PairEnergetics>> {
    if args.attempt_frequencies_hz.is_some()
        || args.barrier_set_json.is_some()
        || args.saddle_shift_ev != 0.0
        || args.barrier_policy != "symmetric_floor"
    {
        return Err(value_error(
            "martinez2012/tabulated options are not valid for pair energetics",
        ));
    }
    let nu = args.attempt_frequency.ok_or_else(|| {
        value_error("attempt_frequency is required for pair/illustrative energetics")
    })?;
    let energetics = match (args.pair_energies_ev, args.kra_ev) {
        (None, None) => PairEnergetics::illustrative(),
        (Some(pair), Some((kra_fe, kra_cr))) => {
            PairEnergetics::new(pair, kra_fe, kra_cr).map_err(|e| value_error(e.to_string()))?
        }
        _ => {
            return Err(value_error(
                "pair_energies_ev and kra_ev must be given together or both omitted",
            ));
        }
    };
    let lattice = BccLattice::new(cells).map_err(|e| value_error(e.to_string()))?;
    FeCrModel::new(lattice, energetics, nu, temperature_k).map_err(|e| value_error(e.to_string()))
}

fn build_martinez_model(
    cells: [usize; 3],
    temperature_k: f64,
    args: &EnergeticsArgs<'_>,
) -> PyResult<(FeCrModel<Martinez2012>, Martinez2012)> {
    if args.pair_energies_ev.is_some() || args.kra_ev.is_some() || args.barrier_set_json.is_some() {
        return Err(value_error(
            "pair_energies_ev, kra_ev, and barrier_set_json do not apply to martinez2012",
        ));
    }
    if args.attempt_frequency.is_some() {
        return Err(value_error(
            "attempt_frequency must be omitted for martinez2012; use attempt_frequencies_hz if overriding species rates",
        ));
    }
    if !args.saddle_shift_ev.is_finite() {
        return Err(value_error("saddle_shift_ev must be finite"));
    }
    let policy = parse_barrier_policy(args.barrier_policy)?;
    let (nu_fe, nu_cr) = args
        .attempt_frequencies_hz
        .unwrap_or((params::NU_FE, params::NU_CR));
    let mut energetics = Martinez2012::with_policy(temperature_k, nu_fe, nu_cr, policy)
        .map_err(|e| value_error(e.to_string()))?;
    if args.saddle_shift_ev != 0.0 {
        energetics = energetics.with_saddle_shift(args.saddle_shift_ev);
    }
    let probe = energetics.clone();
    let lattice = BccLattice::new(cells).map_err(|e| value_error(e.to_string()))?;
    let model = FeCrModel::new(lattice, energetics, nu_fe, temperature_k)
        .map_err(|e| value_error(e.to_string()))?;
    Ok((model, probe))
}

fn build_tabulated_model(
    cells: [usize; 3],
    temperature_k: f64,
    args: &EnergeticsArgs<'_>,
) -> PyResult<(FeCrModel<TabulatedEnergetics>, String)> {
    if args.attempt_frequency.is_some() {
        return Err(value_error(
            "attempt_frequency does not apply to tabulated energetics; each barrier record defines attempt_frequency_hz",
        ));
    }
    if args.pair_energies_ev.is_some()
        || args.kra_ev.is_some()
        || args.attempt_frequencies_hz.is_some()
        || args.saddle_shift_ev != 0.0
        || args.barrier_policy != "symmetric_floor"
    {
        return Err(value_error(
            "pair and martinez2012 parameters do not apply to tabulated energetics",
        ));
    }
    let raw = args
        .barrier_set_json
        .ok_or_else(|| value_error("barrier_set_json is required for tabulated energetics"))?;
    let set: BarrierSet = serde_json::from_str(raw).map_err(|e| value_error(e.to_string()))?;
    set.validate_at_temperature(temperature_k)
        .map_err(|e| value_error(e.to_string()))?;
    let energetics = TabulatedEnergetics::new(set).map_err(|e| value_error(e.to_string()))?;
    let sha = energetics.barrier_set_sha256().to_string();
    let lattice = BccLattice::new(cells).map_err(|e| value_error(e.to_string()))?;
    let model = FeCrModel::new(lattice, energetics, 1e13, temperature_k)
        .map_err(|e| value_error(e.to_string()))?;
    Ok((model, sha))
}

fn validate_transport_config(config: &TransportConfig, runs: u64) -> PyResult<()> {
    if runs < 2 {
        return Err(transport_error_to_py(TransportError::InsufficientReplicas(
            runs,
        )));
    }
    if !config.lattice_parameter.is_finite() || config.lattice_parameter <= 0.0 {
        return Err(transport_error_to_py(
            TransportError::InvalidLatticeParameter(config.lattice_parameter),
        ));
    }
    let min_samples = match config.fit_method {
        FitMethod::WindowedOls => 2,
        FitMethod::OriginSlope => 1,
    };
    if config.sample_count < min_samples {
        return Err(transport_error_to_py(TransportError::InsufficientSamples {
            fit_method: config.fit_method,
            sample_count: config.sample_count,
        }));
    }
    SampleGrid::windowed(config.t_equil, config.t_end, config.sample_count.min(2))
        .map_err(|e| transport_error_to_py(TransportError::Sampling(SamplingError::Grid(e))))?;
    let mut grid_buf = Vec::<f64>::new();
    if grid_buf.try_reserve_exact(config.sample_count).is_err() {
        return Err(value_error(format!(
            "cannot allocate sample grid for {} samples",
            config.sample_count
        )));
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn execute_transport<E>(
    py: Python<'_>,
    model: FeCrModel<E>,
    cells: [usize; 3],
    cr_count: usize,
    temperature_k: f64,
    config: TransportConfig,
    seed: u64,
    runs: u64,
    anneal_sweeps: u64,
    barrier_sha256: Option<String>,
    floor_activations: impl FnOnce() -> u64,
) -> PyResult<FeCrTransportRaw>
where
    E: Energetics + Clone + Sync + Send,
    FeCrModel<E>: Checkpointable,
{
    if runs < 2 {
        return Err(transport_error_to_py(TransportError::InsufficientReplicas(
            runs,
        )));
    }
    let n_sites = model.lattice().site_count();
    if cr_count >= n_sites {
        return Err(value_error(format!(
            "cr_count {cr_count} plus one vacancy exceeds the {n_sites} lattice sites"
        )));
    }
    let manifest = build_fecr_manifest(
        &model,
        cells,
        cr_count,
        temperature_k,
        seed,
        runs,
        &config,
        anneal_sweeps,
        barrier_sha256,
    );
    let manifest_json = manifest
        .to_json_pretty()
        .map_err(|e| runtime_error(e.to_string()))?;

    let replicas = py
        .detach(move || run_transport(&model, cr_count, seed, runs, &config, anneal_sweeps))
        .map_err(transport_error_to_py)?;

    let s = summarize(&replicas, cr_count, n_sites - 1);
    let scalars = [s.vacancy, s.tracer[0], s.tracer[1], s.tracer_all, s.f]
        .iter()
        .flat_map(|&(mean, se)| [mean, se])
        .collect();
    let lambda = s.collective.iter().map(|&(mean, _)| mean).collect();
    let lambda_se = s.collective.iter().map(|&(_, se)| se).collect();
    let mean_final_sro = if replicas.iter().all(|r| r.final_sro_alpha1.is_some()) {
        Some(
            replicas
                .iter()
                .filter_map(|r| r.final_sro_alpha1)
                .sum::<f64>()
                / replicas.len() as f64,
        )
    } else {
        None
    };

    Ok((
        scalars,
        lambda,
        lambda_se,
        s.mean_jumps,
        s.window_drift_ratio,
        floor_activations(),
        mean_final_sro,
        manifest_json,
    ))
}

fn execute_anneal<E: Energetics + Send + Sync>(
    py: Python<'_>,
    model: FeCrModel<E>,
    cr_count: usize,
    sweeps: u64,
    seed: u64,
) -> PyResult<AnnealRaw> {
    let n_sites = model.lattice().site_count();
    if cr_count >= n_sites {
        return Err(value_error(format!(
            "cr_count {cr_count} plus one vacancy exceeds the {n_sites} lattice sites"
        )));
    }
    let anneal = py
        .detach(move || {
            let mut init_rng = KmcRng::for_replica(seed, 0, StreamRole::InitialState);
            let state = model.random_state_with_rng(cr_count, &mut init_rng)?;
            let mut annealer = KawasakiAnnealer::new(
                &model,
                state.occupancy().to_vec(),
                KmcRng::for_replica(seed, 0, StreamRole::Equilibration),
            )?;
            annealer.run_sweeps(sweeps)?;
            Ok(annealer.finish())
        })
        .map_err(|e: okrates_models::fecr::FeCrError| value_error(e.to_string()))?;

    Ok((
        anneal.energy,
        anneal.sro_alpha1,
        anneal.attempts,
        anneal.accepted,
        anneal.diagnostics.acceptance_rate,
        anneal.diagnostics.final_sro_alpha1,
        anneal.diagnostics.tail_energy_drift_ev,
    ))
}

#[pyfunction]
fn jump_rate(
    attempt_frequency: f64,
    migration_energy_ev: f64,
    temperature_k: f64,
) -> PyResult<f64> {
    okrates_models::vacancy::jump_rate(attempt_frequency, migration_energy_ev, temperature_k)
        .map_err(|err| value_error(err.to_string()))
}

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
        return Err(value_error(format!(
            "jump_rate must be finite and positive, got {jump_rate}"
        )));
    }
    if !lattice_parameter_m.is_finite() || lattice_parameter_m <= 0.0 {
        return Err(value_error(format!(
            "lattice_parameter_m must be finite and positive, got {lattice_parameter_m}"
        )));
    }
    let cap = usize::try_from(steps)
        .ok()
        .and_then(|n| n.checked_add(1))
        .ok_or_else(|| {
            value_error(format!(
                "steps {steps} exceeds addressable trajectory length"
            ))
        })?;
    let mut times = Vec::new();
    let mut sq = Vec::new();
    if times.try_reserve_exact(cap).is_err() || sq.try_reserve_exact(cap).is_err() {
        return Err(value_error(format!(
            "cannot allocate trajectory buffers for {steps} steps"
        )));
    }
    let lattice = BccLattice::new(cells).map_err(|err| value_error(err.to_string()))?;

    let result = py.detach(move || -> Result<(Vec<f64>, Vec<f64>), String> {
        let model = VacancyDiffusion::new(lattice, jump_rate);
        let state = model.initial_state(0).map_err(|err| err.to_string())?;
        let mut sim = Simulation::new(model, state, seed);
        times.push(sim.time());
        sq.push(sim.state().squared_displacement(lattice_parameter_m));
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
    result.map_err(runtime_error)
}

#[pyfunction]
fn fecr_illustrative_energetics() -> ([[f64; 3]; 3], f64, f64) {
    let e = PairEnergetics::illustrative();
    (*e.pair(), e.kra(Species::Fe), e.kra(Species::Cr))
}

#[pyfunction]
#[pyo3(signature = (
    *,
    cells,
    cr_count,
    temperature_k,
    t_equil,
    t_end,
    sample_count,
    fit_method = "windowed_ols",
    max_steps = None,
    lattice_parameter_m,
    seed,
    runs,
    anneal_sweeps = 0,
    energetics = "illustrative",
    attempt_frequency = None,
    pair_energies_ev = None,
    kra_ev = None,
    barrier_policy = "symmetric_floor",
    attempt_frequencies_hz = None,
    saddle_shift_ev = 0.0,
    barrier_set_json = None
))]
#[allow(clippy::too_many_arguments)]
fn run_fecr_transport(
    py: Python<'_>,
    cells: [usize; 3],
    cr_count: usize,
    temperature_k: f64,
    t_equil: f64,
    t_end: f64,
    sample_count: usize,
    fit_method: &str,
    max_steps: Option<u64>,
    lattice_parameter_m: f64,
    seed: u64,
    runs: u64,
    anneal_sweeps: u64,
    energetics: &str,
    attempt_frequency: Option<f64>,
    pair_energies_ev: Option<[[f64; 3]; 3]>,
    kra_ev: Option<(f64, f64)>,
    barrier_policy: &str,
    attempt_frequencies_hz: Option<(f64, f64)>,
    saddle_shift_ev: f64,
    barrier_set_json: Option<&str>,
) -> PyResult<FeCrTransportRaw> {
    let config = TransportConfig {
        t_equil,
        t_end,
        sample_count,
        lattice_parameter: lattice_parameter_m,
        fit_method: parse_fit_method(fit_method)?,
        max_steps,
    };
    validate_transport_config(&config, runs)?;
    let args = EnergeticsArgs {
        attempt_frequency,
        pair_energies_ev,
        kra_ev,
        barrier_policy,
        attempt_frequencies_hz,
        saddle_shift_ev,
        barrier_set_json,
    };
    match energetics {
        "illustrative" | "pair" => {
            let model = build_pair_model(cells, temperature_k, &args)?;
            execute_transport(
                py,
                model,
                cells,
                cr_count,
                temperature_k,
                config,
                seed,
                runs,
                anneal_sweeps,
                None,
                || 0,
            )
        }
        "martinez2012" => {
            let (model, probe) = build_martinez_model(cells, temperature_k, &args)?;
            execute_transport(
                py,
                model,
                cells,
                cr_count,
                temperature_k,
                config,
                seed,
                runs,
                anneal_sweeps,
                None,
                move || probe.floor_activations(),
            )
        }
        "tabulated" => {
            let (model, sha) = build_tabulated_model(cells, temperature_k, &args)?;
            execute_transport(
                py,
                model,
                cells,
                cr_count,
                temperature_k,
                config,
                seed,
                runs,
                anneal_sweeps,
                Some(sha),
                || 0,
            )
        }
        other => Err(value_error(format!(
            "energetics must be \"illustrative\", \"pair\", \"martinez2012\", or \"tabulated\", got {other:?}"
        ))),
    }
}

#[pyfunction]
#[pyo3(signature = (
    *,
    cells,
    cr_count,
    temperature_k,
    sweeps,
    seed,
    energetics = "illustrative",
    attempt_frequency = None,
    pair_energies_ev = None,
    kra_ev = None,
    barrier_policy = "symmetric_floor",
    attempt_frequencies_hz = None,
    saddle_shift_ev = 0.0,
    barrier_set_json = None
))]
#[allow(clippy::too_many_arguments)]
fn run_kawasaki_anneal(
    py: Python<'_>,
    cells: [usize; 3],
    cr_count: usize,
    temperature_k: f64,
    sweeps: u64,
    seed: u64,
    energetics: &str,
    attempt_frequency: Option<f64>,
    pair_energies_ev: Option<[[f64; 3]; 3]>,
    kra_ev: Option<(f64, f64)>,
    barrier_policy: &str,
    attempt_frequencies_hz: Option<(f64, f64)>,
    saddle_shift_ev: f64,
    barrier_set_json: Option<&str>,
) -> PyResult<AnnealRaw> {
    let args = EnergeticsArgs {
        attempt_frequency,
        pair_energies_ev,
        kra_ev,
        barrier_policy,
        attempt_frequencies_hz,
        saddle_shift_ev,
        barrier_set_json,
    };
    match energetics {
        "illustrative" | "pair" => {
            let model = build_pair_model(cells, temperature_k, &args)?;
            execute_anneal(py, model, cr_count, sweeps, seed)
        }
        "martinez2012" => {
            let (model, _) = build_martinez_model(cells, temperature_k, &args)?;
            execute_anneal(py, model, cr_count, sweeps, seed)
        }
        "tabulated" => {
            let (model, _) = build_tabulated_model(cells, temperature_k, &args)?;
            execute_anneal(py, model, cr_count, sweeps, seed)
        }
        other => Err(value_error(format!(
            "energetics must be \"illustrative\", \"pair\", \"martinez2012\", or \"tabulated\", got {other:?}"
        ))),
    }
}

#[pyfunction]
#[pyo3(signature = (barrier_set_json, *, temperature_k = None))]
fn validate_barrier_set_json(
    barrier_set_json: &str,
    temperature_k: Option<f64>,
) -> PyResult<String> {
    let set: BarrierSet =
        serde_json::from_str(barrier_set_json).map_err(|e| value_error(e.to_string()))?;
    match temperature_k {
        Some(t) => set
            .validate_at_temperature(t)
            .map_err(|e| value_error(e.to_string()))?,
        None => set.validate().map_err(|e| value_error(e.to_string()))?,
    }
    set.content_sha256().map_err(|e| value_error(e.to_string()))
}

#[pyfunction]
fn import_pymatgen_vasp_barriers_json(payload_json: &str) -> PyResult<String> {
    let set = import_pymatgen_vasp_json(payload_json).map_err(|e| value_error(e.to_string()))?;
    serde_json::to_string_pretty(&set).map_err(|e| value_error(e.to_string()))
}

#[pyfunction]
fn import_aiida_qe_barriers_json(payload_json: &str) -> PyResult<String> {
    let set = import_aiida_qe_json(payload_json).map_err(|e| value_error(e.to_string()))?;
    serde_json::to_string_pretty(&set).map_err(|e| value_error(e.to_string()))
}

#[pymodule]
fn _okrates(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(jump_rate, m)?)?;
    m.add_function(wrap_pyfunction!(vacancy_trajectory, m)?)?;
    m.add_function(wrap_pyfunction!(fecr_illustrative_energetics, m)?)?;
    m.add_function(wrap_pyfunction!(run_fecr_transport, m)?)?;
    m.add_function(wrap_pyfunction!(run_kawasaki_anneal, m)?)?;
    m.add_function(wrap_pyfunction!(validate_barrier_set_json, m)?)?;
    m.add_function(wrap_pyfunction!(import_pymatgen_vasp_barriers_json, m)?)?;
    m.add_function(wrap_pyfunction!(import_aiida_qe_barriers_json, m)?)?;
    Ok(())
}
