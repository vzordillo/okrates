use std::env;
use std::process;

use okrates_core::{Simulation, Step};
use okrates_models::bcc::BccLattice;
use okrates_models::vacancy::{VacancyDiffusion, jump_rate};

const CELLS: [usize; 3] = [4, 4, 4];
const ATTEMPT_FREQUENCY: f64 = 1e13;
const MIGRATION_ENERGY_EV: f64 = 0.65;
const TEMPERATURE_K: f64 = 800.0;
const LATTICE_PARAMETER_M: f64 = 2.8665e-10;

fn usage() -> ! {
    eprintln!("usage: vacancy_trajectory SEED STEPS");
    process::exit(2);
}

fn arguments() -> (u64, u64) {
    let mut args = env::args().skip(1);
    let seed = match args.next().map(|value| value.parse::<u64>()) {
        Some(Ok(seed)) => seed,
        _ => usage(),
    };
    let steps = match args.next().map(|value| value.parse::<u64>()) {
        Some(Ok(steps)) => steps,
        _ => usage(),
    };
    if args.next().is_some() {
        usage();
    }
    (seed, steps)
}

fn main() {
    let (seed, steps) = arguments();
    let rate = jump_rate(ATTEMPT_FREQUENCY, MIGRATION_ENERGY_EV, TEMPERATURE_K)
        .expect("the fixed Arrhenius parameters are valid");
    let lattice = BccLattice::new(CELLS).expect("the fixed cell dimensions are valid");
    let model = VacancyDiffusion::new(lattice, rate);
    let state = model.initial_state(0).expect("site zero is in the lattice");
    let mut sim = Simulation::new(model, state, seed);

    println!("rate {rate:?}");
    println!(
        "{:?} {:?}",
        sim.time(),
        sim.state().squared_displacement(LATTICE_PARAMETER_M)
    );
    for _ in 0..steps {
        match sim.step().expect("vacancy events have valid rates") {
            Step::Fired { .. } => println!(
                "{:?} {:?}",
                sim.time(),
                sim.state().squared_displacement(LATTICE_PARAMETER_M)
            ),
            Step::Absorbed => break,
        }
    }
}
