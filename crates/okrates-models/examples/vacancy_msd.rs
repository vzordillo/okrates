use std::env;
use std::process;

use okrates_core::{Simulation, Step};
use okrates_models::bcc::BccLattice;
use okrates_models::vacancy::{VacancyDiffusion, jump_rate};

const DEFAULT_RUNS: usize = 2000;
const DEFAULT_SEED_BASE: u64 = 20261117;
const CELLS: [usize; 3] = [4, 4, 4];
const ATTEMPT_FREQUENCY: f64 = 1e13;
const MIGRATION_ENERGY_EV: f64 = 0.65;
const TEMPERATURE_K: f64 = 800.0;
const LATTICE_PARAMETER_M: f64 = 2.8665e-10;
const SAMPLE_COUNT: usize = 10;
const CI_95_Z: f64 = 1.959_963_984_540_054;

fn usage() -> ! {
    eprintln!("usage: vacancy_msd [RUNS] [SEED_BASE]");
    process::exit(2);
}

fn arguments() -> (usize, u64) {
    let mut args = env::args().skip(1);
    let runs = match args.next() {
        Some(value) => match value.parse::<usize>() {
            Ok(runs) if runs >= 2 => runs,
            _ => usage(),
        },
        None => DEFAULT_RUNS,
    };
    let seed_base = match args.next() {
        Some(value) => match value.parse::<u64>() {
            Ok(seed_base) => seed_base,
            Err(_) => usage(),
        },
        None => DEFAULT_SEED_BASE,
    };
    if args.next().is_some()
        || u64::try_from(runs - 1)
            .ok()
            .and_then(|last| seed_base.checked_add(last))
            .is_none()
    {
        usage();
    }
    (runs, seed_base)
}

fn estimate_one_run(
    lattice: &BccLattice,
    seed: u64,
    rate: f64,
    lattice_parameter: f64,
    t_end: f64,
) -> (f64, u64) {
    let model = VacancyDiffusion::new(lattice.clone(), rate);
    let state = model.initial_state(0).expect("site zero is in the lattice");
    let mut sim = Simulation::new(model, state, seed);
    let mut samples = [0.0; SAMPLE_COUNT];
    let mut k = 1;
    let mut fired_steps = 0;

    loop {
        let held = sim.state().squared_displacement(lattice_parameter);
        match sim.step().expect("vacancy events have valid rates") {
            Step::Fired { .. } => fired_steps += 1,
            Step::Absorbed => panic!("vacancy model unexpectedly absorbed"),
        }
        let t_after = sim.time();
        while k <= SAMPLE_COUNT && (k as f64 * t_end / SAMPLE_COUNT as f64) < t_after {
            samples[k - 1] = held;
            k += 1;
        }
        if k > SAMPLE_COUNT {
            break;
        }
    }

    let numerator = (1..=SAMPLE_COUNT)
        .map(|sample| {
            let time = sample as f64 * t_end / SAMPLE_COUNT as f64;
            time * samples[sample - 1]
        })
        .sum::<f64>();
    let denominator = (1..=SAMPLE_COUNT)
        .map(|sample| {
            let time = sample as f64 * t_end / SAMPLE_COUNT as f64;
            time * time
        })
        .sum::<f64>();
    ((numerator / denominator) / 6.0, fired_steps)
}

fn main() {
    let (runs, seed_base) = arguments();
    let rate = jump_rate(ATTEMPT_FREQUENCY, MIGRATION_ENERGY_EV, TEMPERATURE_K)
        .expect("the fixed Arrhenius parameters are valid");
    let t_end = 1000.0 / (8.0 * rate);
    let lattice = BccLattice::new(CELLS).expect("the fixed cell dimensions are valid");
    let mut estimates = Vec::with_capacity(runs);
    let mut total_fired_steps = 0_u64;

    for run in 0..runs {
        let seed = seed_base + run as u64;
        let (estimate, fired_steps) =
            estimate_one_run(&lattice, seed, rate, LATTICE_PARAMETER_M, t_end);
        estimates.push(estimate);
        total_fired_steps += fired_steps;
    }

    let d_hat = estimates.iter().sum::<f64>() / runs as f64;
    let sample_variance = estimates
        .iter()
        .map(|estimate| (estimate - d_hat).powi(2))
        .sum::<f64>()
        / (runs - 1) as f64;
    let se = sample_variance.sqrt() / (runs as f64).sqrt();
    let d_theory = rate * LATTICE_PARAMETER_M * LATTICE_PARAMETER_M;
    let ci_half_width = CI_95_Z * se;
    let z_score = (d_hat - d_theory) / se;
    let relative_error = (d_hat - d_theory).abs() / d_theory;

    println!("parameterization = illustrative parameters, not a validated Fe parameterization");
    println!("cells = [{}, {}, {}]", CELLS[0], CELLS[1], CELLS[2]);
    println!("nu = 1e13 s^-1");
    println!("Em = 0.65 eV");
    println!("T = 800 K");
    println!("a = 2.8665e-10 m");
    println!("r = {rate:.12e} s^-1");
    println!("T_end = {t_end:.12e} s");
    println!("D_theory = {d_theory:.12e} m^2/s");
    println!("d_hat = {d_hat:.12e} m^2/s");
    println!("se = {se:.12e} m^2/s");
    println!(
        "CI_95 = [{:.12e}, {:.12e}] m^2/s",
        d_hat - ci_half_width,
        d_hat + ci_half_width
    );
    println!("z_score_vs_theory = {z_score:.12}");
    println!("relative_error = {relative_error:.12e}");
    println!("runs = {runs}");
    println!("seed_base = {seed_base}");
    println!("total_fired_steps = {total_fired_steps}");
}
