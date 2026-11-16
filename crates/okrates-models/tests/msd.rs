//! Validates the vacancy MSD against D_v = r a^2.

use okrates_core::{Simulation, Step};
use okrates_models::bcc::{BccLattice, DISPLACEMENTS};
use okrates_models::vacancy::VacancyDiffusion;

const M: usize = 2000;
const SEED_BASE: u64 = 20261117;
const CELLS: [usize; 3] = [4, 4, 4];
const RATE: f64 = 1.0;
const A: f64 = 1.0;
const T_END: f64 = 125.0;
const K: usize = 10;
const Z_CRIT: f64 = 4.89163847569859;

fn estimate_one_run(lattice: &BccLattice, seed: u64, rate: f64, a: f64, t_end: f64) -> f64 {
    let model = VacancyDiffusion::new(lattice.clone(), rate);
    let state = model.initial_state(0).unwrap();
    let mut sim = Simulation::new(model, state, seed);
    let mut samples = [0.0; K];
    let mut k = 1;

    loop {
        let held = sim.state().squared_displacement(a);
        assert!(matches!(sim.step().unwrap(), Step::Fired { .. }));
        let t_after = sim.time();
        while k <= K && (k as f64 * t_end / K as f64) < t_after {
            samples[k - 1] = held;
            k += 1;
        }
        if k > K {
            break;
        }
    }

    let numerator = (1..=K)
        .map(|sample| {
            let time = sample as f64 * t_end / K as f64;
            time * samples[sample - 1]
        })
        .sum::<f64>();
    let denominator = (1..=K)
        .map(|sample| {
            let time = sample as f64 * t_end / K as f64;
            time * time
        })
        .sum::<f64>();
    (numerator / denominator) / 6.0
}

#[test]
fn invariants_hold_after_every_event() {
    let lattice = BccLattice::new([3, 3, 3]).unwrap();
    let model = VacancyDiffusion::new(lattice, 1.0);
    let state = model.initial_state(0).unwrap();
    let mut sim = Simulation::new(model, state, 5);

    for _ in 0..20_000 {
        let before = sim.state().displacement();
        let event = match sim.step().unwrap() {
            Step::Fired { event, .. } => event,
            Step::Absorbed => panic!("vacancy model unexpectedly absorbed"),
        };
        let after = sim.state().displacement();
        assert_eq!(
            std::array::from_fn::<_, 3, _>(|axis| after[axis] - before[axis]),
            DISPLACEMENTS[event.direction]
        );
        assert_eq!(sim.model().check_invariants(sim.state()), Ok(()));
        assert_eq!(
            sim.state()
                .occupancy()
                .iter()
                .filter(|&&species| species == okrates_models::vacancy::Species::Fe)
                .count(),
            sim.model().lattice().site_count() - 1
        );
    }
}

#[test]
fn vacancy_msd_matches_rate_times_a_squared() {
    let lattice = BccLattice::new(CELLS).unwrap();
    let estimates = (0..M)
        .map(|run| estimate_one_run(&lattice, SEED_BASE + run as u64, RATE, A, T_END))
        .collect::<Vec<_>>();
    let d_hat = estimates.iter().sum::<f64>() / M as f64;
    let variance = estimates
        .iter()
        .map(|estimate| (estimate - d_hat).powi(2))
        .sum::<f64>()
        / (M - 1) as f64;
    let se = variance.sqrt() / (M as f64).sqrt();
    let z = (d_hat - RATE * A * A) / se;
    let ci_half_width = 1.959_963_984_540_054 * se;

    println!(
        "d_hat={d_hat:.12} se={se:.12} z={z:.12} CI=[{:.12}, {:.12}]",
        d_hat - ci_half_width,
        d_hat + ci_half_width
    );
    assert!(z.abs() <= Z_CRIT, "z = {z}, threshold = {Z_CRIT}");
}
