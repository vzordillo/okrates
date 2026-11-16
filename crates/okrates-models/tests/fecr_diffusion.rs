//! Validates displacement tracking and the diffusion estimators of the Fe–Cr model.

use okrates_core::ensemble::run_replicas;
use okrates_core::{Simulation, Step};
use okrates_models::bcc::{BccLattice, DISPLACEMENTS};
use okrates_models::fecr::{FeCrModel, PairEnergetics, Species};
use okrates_models::fecr_diffusion::{
    ReplicaEstimate, estimate, mean_se, origin_slope, sample_run,
};

/// Offset between a replica's dynamics seed and its `random_state` seed, so that
/// the initial labels are independent of the dynamics stream.
const STATE_SEED_OFFSET: u64 = 1_000_000_000;

const Z_CRIT: f64 = 4.89163847569859;
const M: u64 = 2000;

fn equal_energy_model() -> FeCrModel<PairEnergetics> {
    let energetics = PairEnergetics::new([[0.0; 3]; 3], 0.0, 0.0).unwrap();
    FeCrModel::new(BccLattice::new([4, 4, 4]).unwrap(), energetics, 1.0, 1000.0).unwrap()
}

fn add(a: [i64; 3], b: [i64; 3]) -> [i64; 3] {
    std::array::from_fn(|axis| a[axis] + b[axis])
}

fn neg(a: [i64; 3]) -> [i64; 3] {
    a.map(|c| -c)
}

fn sub(a: [i64; 3], b: [i64; 3]) -> [i64; 3] {
    add(a, neg(b))
}

#[test]
fn tracking_invariants_hold_after_every_event() {
    let model = FeCrModel::new(
        BccLattice::new([3, 3, 3]).unwrap(),
        PairEnergetics::illustrative(),
        1e13,
        800.0,
    )
    .unwrap();
    let state = model.random_state(11, 9).unwrap();
    let sites = model.lattice().site_count();
    let mut sim = Simulation::new(model, state, 9);
    for _ in 0..5000 {
        let before: Vec<[i64; 3]> = (0..sites)
            .map(|s| sim.state().displacement_of_site(s))
            .collect();
        let old_vacancy = sim.state().vacancy();
        let event = match sim.step().unwrap() {
            Step::Fired { event, .. } => event,
            Step::Absorbed => panic!("unexpectedly absorbed"),
        };
        let state = sim.state();
        assert_eq!(sim.model().check_invariants(state), Ok(()));
        let shift = DISPLACEMENTS[event.direction];
        let new_vacancy = state.vacancy();
        // Entity bookkeeping: the vacancy entity was on `old_vacancy`, the atom on `new_vacancy`.
        assert_eq!(
            sub(state.vacancy_displacement(), before[old_vacancy]),
            shift
        );
        assert_eq!(
            sub(state.displacement_of_site(old_vacancy), before[new_vacancy]),
            neg(shift)
        );
        for (site, &d) in before.iter().enumerate() {
            if site != old_vacancy && site != new_vacancy {
                assert_eq!(state.displacement_of_site(site), d);
            }
        }
        let vac = state.vacancy_displacement();
        let total = state
            .atom_displacements()
            .fold([0; 3], |acc, (_, d)| add(acc, d));
        assert_eq!(total, neg(vac));
        assert_eq!(
            add(
                state.collective_displacement(Species::Fe),
                state.collective_displacement(Species::Cr)
            ),
            neg(vac)
        );
    }
}

fn gate(name: &str, values: &[f64]) {
    let (mean, se) = mean_se(values);
    let z = mean / se;
    println!("{name}: mean={mean:.9e} se={se:.9e} z={z:.6}");
    assert!(z.abs() <= Z_CRIT, "{name}: z = {z}, threshold = {Z_CRIT}");
}

#[test]
fn equal_energy_limit_gates() {
    let model = equal_energy_model();
    let estimates: Vec<ReplicaEstimate> = run_replicas(20261118, M, |_, seed| {
        let state = model.random_state(63, seed + STATE_SEED_OFFSET).unwrap();
        let (times, samples, _) = sample_run(model.clone(), state, seed, 125.0, 10, 1.0);
        estimate(&times, &samples, origin_slope)
    })
    .unwrap()
    .into_iter()
    .map(|r| r.value)
    .collect();

    let (n_a, n_fe, n_cr) = (127.0, 64.0, 63.0);
    let p = |e: &ReplicaEstimate| e.vacancy - e.s;

    let g1: Vec<f64> = estimates.iter().map(|e| e.vacancy - 1.0).collect();
    gate("G1 D_v - 1", &g1);
    let g2: Vec<f64> = estimates
        .iter()
        .map(|e| e.tracer[0] - e.tracer[1])
        .collect();
    gate("G2 D*_Fe - D*_Cr", &g2);
    let g3: Vec<f64> = estimates
        .iter()
        .map(|e| e.collective[1] - (n_fe * n_cr) / (n_a * (n_a - 1.0)) * p(e))
        .collect();
    gate("G3 L_FeCr", &g3);
    let g4: Vec<f64> = estimates
        .iter()
        .map(|e| {
            e.collective[2]
                - (n_cr / n_a) * e.s
                - (n_cr * (n_cr - 1.0)) / (n_a * (n_a - 1.0)) * p(e)
        })
        .collect();
    gate("G4 L_CrCr", &g4);

    for e in &estimates {
        let sum = e.collective[0] + 2.0 * e.collective[1] + e.collective[2];
        assert!(
            (sum - e.vacancy).abs() <= 1e-9 * e.vacancy.max(1.0),
            "sum rule violated: {sum} vs {}",
            e.vacancy
        );
    }

    let mean = |f: &dyn Fn(&ReplicaEstimate) -> f64| {
        estimates.iter().map(f).sum::<f64>() / estimates.len() as f64
    };
    let (ff, fc, cc) = (
        mean(&|e| e.collective[0]),
        mean(&|e| e.collective[1]),
        mean(&|e| e.collective[2]),
    );
    println!("Lambda mean: FeFe={ff:.9e} FeCr={fc:.9e} CrCr={cc:.9e}");
    let trace = ff + cc;
    let det = ff * cc - fc * fc;
    assert!(ff >= 0.0 && cc >= 0.0, "negative diagonal");
    assert!(det >= -1e-9 * trace * trace, "det = {det}, trace = {trace}");
}

#[test]
fn pure_fe_vacancy_gate() {
    let model = FeCrModel::new(
        BccLattice::new([4, 4, 4]).unwrap(),
        PairEnergetics::illustrative(),
        1.0,
        1000.0,
    )
    .unwrap();
    let probe = model.random_state(0, 0).unwrap();
    let r = model.rate(&probe, 0);
    for d in 0..8 {
        let rd = model.rate(&probe, d);
        assert!(
            ((rd - r) / r).abs() <= 1e-12,
            "rate {d} = {rd}, rate 0 = {r}"
        );
    }
    let t_end = 1000.0 / (8.0 * r);
    let estimates: Vec<ReplicaEstimate> = run_replicas(20261119, M, |_, seed| {
        let state = model.random_state(0, seed + STATE_SEED_OFFSET).unwrap();
        let (times, samples, _) = sample_run(model.clone(), state, seed, t_end, 10, 1.0);
        estimate(&times, &samples, origin_slope)
    })
    .unwrap()
    .into_iter()
    .map(|r| r.value)
    .collect();
    for e in &estimates {
        assert!(
            (e.collective[0] - e.vacancy).abs() <= 1e-9 * e.vacancy.abs(),
            "pure Fe: {} vs {}",
            e.collective[0],
            e.vacancy
        );
    }
    let values: Vec<f64> = estimates.iter().map(|e| e.vacancy - r).collect();
    gate("G5 D_v - r", &values);
}
