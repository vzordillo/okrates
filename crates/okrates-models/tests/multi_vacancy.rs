//! Validates the many-vacancy model on the direct and incremental schedulers.

use okrates_core::ensemble::run_replicas;
use okrates_core::{IncrementalModel, IncrementalSimulation, Model, Simulation, Step, SumTree};
use okrates_models::bcc::{BccLattice, DISPLACEMENTS};
use okrates_models::multi_vacancy::MultiVacancyDiffusion;

const Z_CRIT: f64 = 4.89163847569859;
const K: usize = 10;

const CATALOG_CELLS: [usize; 3] = [4, 4, 4];
const CATALOG_VACANCIES: usize = 40;
const CATALOG_RUNS: u64 = 20;
const CATALOG_STEPS: usize = 500;
const CATALOG_SITE_SEED_BASE: u64 = 1000;
const CATALOG_SIM_SEED_BASE: u64 = 2000;

const ONE_CELLS: [usize; 3] = [4, 4, 4];
const ONE_RATE: f64 = 1.0;
const ONE_A: f64 = 1.0;
const ONE_T_END: f64 = 125.0;
const ONE_RUNS: u64 = 2000;
const ONE_SEED_BASE: u64 = 40_000_000;

const DENSE_CELLS: [usize; 3] = [5, 5, 5];
const DENSE_VACANCIES: usize = 32;
const DENSE_RATE: f64 = 1.0;
const DENSE_A: f64 = 1.0;
const DENSE_T_END: f64 = 12.5;
const DENSE_RUNS: u64 = 300;
const DENSE_DIRECT_SEED_BASE: u64 = 50_000_000;
const DENSE_INCREMENTAL_SEED_BASE: u64 = 60_000_000;

const RESTART_CELLS: [usize; 3] = [4, 4, 4];
const RESTART_VACANCIES: usize = 40;
const RESTART_SEED: u64 = 77;
const RESTART_STEPS: usize = 300;
const RESTART_SPLITS: [usize; 9] = [0, 1, 15, 16, 17, 31, 32, 33, 100];

const DIRECT_CELLS: [usize; 3] = [3, 3, 3];
const DIRECT_VACANCIES: usize = 20;
const DIRECT_SEED: u64 = 5;
const DIRECT_STEPS: usize = 5000;

fn model(cells: [usize; 3], rate: f64) -> MultiVacancyDiffusion {
    MultiVacancyDiffusion::new(BccLattice::new(cells).unwrap(), rate)
}

/// MSD-based diffusivity estimate for one run, averaged over vacancies.
///
/// `step` fires one event and returns the mean squared displacement over
/// vacancies held before the step and the time after it.
fn estimate_one_run(mut step: impl FnMut() -> (f64, f64), t_end: f64) -> f64 {
    let mut samples = [0.0; K];
    let mut k = 1;
    loop {
        let (held, t_after) = step();
        while k <= K && (k as f64 * t_end / K as f64) < t_after {
            samples[k - 1] = held;
            k += 1;
        }
        if k > K {
            break;
        }
    }
    let numerator = (1..=K)
        .map(|s| s as f64 * t_end / K as f64 * samples[s - 1])
        .sum::<f64>();
    let denominator = (1..=K)
        .map(|s| (s as f64 * t_end / K as f64).powi(2))
        .sum::<f64>();
    (numerator / denominator) / 6.0
}

fn mean_squared_displacement(
    state: &okrates_models::multi_vacancy::MultiVacancyState,
    a: f64,
) -> f64 {
    let n = state.vacancy_count();
    (0..n)
        .map(|v| state.squared_displacement(v, a))
        .sum::<f64>()
        / n as f64
}

fn mean_and_se(values: &[f64]) -> (f64, f64) {
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0);
    (mean, (variance / n).sqrt())
}

#[test]
fn catalog_matches_rebuild_after_random_sequences() {
    let model = model(CATALOG_CELLS, 1.0);
    let mut pair_changes = 0usize;
    for c in 0..CATALOG_RUNS {
        let sites = model
            .random_sites(CATALOG_VACANCIES, CATALOG_SITE_SEED_BASE + c)
            .unwrap();
        let state = model.initial_state(&sites).unwrap();
        let mut sim =
            IncrementalSimulation::new(model.clone(), state, CATALOG_SIM_SEED_BASE + c).unwrap();
        for _ in 0..CATALOG_STEPS {
            let pairs_before = sim.state().vv_neighbor_pairs(model.lattice());
            assert!(matches!(sim.step().unwrap(), Step::Fired { .. }));
            let state = sim.state();
            let count = model.slot_count(state);
            let rates: Vec<f64> = (0..count).map(|s| model.slot_rate(state, s)).collect();
            assert_eq!(*sim.tree(), SumTree::new(&rates));

            let from_slots: Vec<(usize, f64)> = rates
                .iter()
                .enumerate()
                .filter(|(_, r)| **r > 0.0)
                .map(|(s, r)| (s, *r))
                .collect();
            let mut from_enumerate = Vec::new();
            let mut k = 0;
            model.enumerate_events(state, &mut |event, rate| {
                assert_eq!(event, model.slot_event(state, k));
                if rate > 0.0 {
                    from_enumerate.push((k, rate));
                }
                k += 1;
            });
            assert_eq!(k, count);
            assert_eq!(from_slots, from_enumerate);

            assert_eq!(model.check_invariants(state), Ok(()));
            if state.vv_neighbor_pairs(model.lattice()) != pairs_before {
                pair_changes += 1;
            }
        }
    }
    println!("vv pair count changed in {pair_changes} steps");
    assert!(pair_changes > 0);
}

#[test]
fn one_vacancy_msd_on_incremental_scheduler() {
    let model = model(ONE_CELLS, ONE_RATE);
    let estimates = run_replicas(ONE_SEED_BASE, ONE_RUNS, |_, seed| {
        let state = model.initial_state(&[0]).unwrap();
        let mut sim = IncrementalSimulation::new(model.clone(), state, seed).unwrap();
        estimate_one_run(
            || {
                let held = mean_squared_displacement(sim.state(), ONE_A);
                assert!(matches!(sim.step().unwrap(), Step::Fired { .. }));
                (held, sim.time())
            },
            ONE_T_END,
        )
    })
    .unwrap()
    .into_iter()
    .map(|r| r.value)
    .collect::<Vec<_>>();
    let (d_hat, se) = mean_and_se(&estimates);
    let z = (d_hat - ONE_RATE * ONE_A * ONE_A) / se;
    println!("d_hat={d_hat:.12} se={se:.12} z={z:.12}");
    assert!(z.abs() <= Z_CRIT, "z = {z}, threshold = {Z_CRIT}");
}

#[test]
fn high_density_direct_and_incremental_agree() {
    let model = model(DENSE_CELLS, DENSE_RATE);
    let direct = run_replicas(DENSE_DIRECT_SEED_BASE, DENSE_RUNS, |_, seed| {
        let sites = model.random_sites(DENSE_VACANCIES, seed).unwrap();
        let state = model.initial_state(&sites).unwrap();
        let mut sim = Simulation::new(model.clone(), state, seed);
        estimate_one_run(
            || {
                let held = mean_squared_displacement(sim.state(), DENSE_A);
                assert!(matches!(sim.step().unwrap(), Step::Fired { .. }));
                (held, sim.time())
            },
            DENSE_T_END,
        )
    })
    .unwrap()
    .into_iter()
    .map(|r| r.value)
    .collect::<Vec<_>>();
    let incremental = run_replicas(DENSE_INCREMENTAL_SEED_BASE, DENSE_RUNS, |_, seed| {
        let sites = model.random_sites(DENSE_VACANCIES, seed).unwrap();
        let state = model.initial_state(&sites).unwrap();
        let mut sim = IncrementalSimulation::new(model.clone(), state, seed).unwrap();
        estimate_one_run(
            || {
                let held = mean_squared_displacement(sim.state(), DENSE_A);
                assert!(matches!(sim.step().unwrap(), Step::Fired { .. }));
                (held, sim.time())
            },
            DENSE_T_END,
        )
    })
    .unwrap()
    .into_iter()
    .map(|r| r.value)
    .collect::<Vec<_>>();
    let (m_d, se_d) = mean_and_se(&direct);
    let (m_i, se_i) = mean_and_se(&incremental);
    let z = (m_d - m_i) / (se_d * se_d + se_i * se_i).sqrt();
    println!("direct={m_d:.12} se_d={se_d:.12} incremental={m_i:.12} se_i={se_i:.12} z={z:.12}");
    assert!(z.abs() <= Z_CRIT, "z = {z}, threshold = {Z_CRIT}");
}

#[test]
fn incremental_restart_is_bitwise() {
    let model = model(RESTART_CELLS, 1.0);
    let sites = model.random_sites(RESTART_VACANCIES, RESTART_SEED).unwrap();
    let fresh = || {
        let state = model.initial_state(&sites).unwrap();
        IncrementalSimulation::new(model.clone(), state, RESTART_SEED).unwrap()
    };
    let fire = |sim: &mut IncrementalSimulation<MultiVacancyDiffusion>| match sim.step().unwrap() {
        Step::Fired { event, dt } => (event, dt.to_bits()),
        Step::Absorbed => panic!("unexpected absorption"),
    };

    let mut reference = fresh();
    let expected: Vec<_> = (0..RESTART_STEPS).map(|_| fire(&mut reference)).collect();
    let expected_final = reference.checkpoint();

    for split in RESTART_SPLITS {
        let mut first = fresh();
        for _ in 0..split {
            fire(&mut first);
        }
        let bytes = first.checkpoint();
        let mut resumed = IncrementalSimulation::resume(model.clone(), &bytes).unwrap();
        let rest: Vec<_> = (split..RESTART_STEPS).map(|_| fire(&mut resumed)).collect();
        assert_eq!(rest, expected[split..], "split {split}");
        assert_eq!(resumed.checkpoint(), expected_final, "split {split}");
    }
}

#[test]
fn invariants_hold_on_direct_scheduler() {
    let model = model(DIRECT_CELLS, 1.0);
    let sites = model.random_sites(DIRECT_VACANCIES, DIRECT_SEED).unwrap();
    let state = model.initial_state(&sites).unwrap();
    let mut sim = Simulation::new(model.clone(), state, DIRECT_SEED);
    for _ in 0..DIRECT_STEPS {
        let before = sim.state().displacements().to_vec();
        let event = match sim.step().unwrap() {
            Step::Fired { event, .. } => event,
            Step::Absorbed => panic!("unexpected absorption"),
        };
        assert_eq!(model.check_invariants(sim.state()), Ok(()));
        let after = sim.state().displacements();
        for v in 0..before.len() {
            let delta: [i64; 3] = std::array::from_fn(|axis| after[v][axis] - before[v][axis]);
            if v == event.vacancy {
                assert_eq!(delta, DISPLACEMENTS[event.direction]);
            } else {
                assert_eq!(delta, [0; 3]);
            }
        }
    }
}
