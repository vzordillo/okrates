//! Validates displacement tracking and the diffusion estimators of the Fe–Cr model.

use okrates_core::ensemble::run_replicas;
use okrates_core::{KmcRng, Simulation, Step};
use okrates_models::bcc::{BccLattice, DISPLACEMENTS};
use okrates_models::fecr::{FeCrModel, PairEnergetics, Species};
use okrates_models::fecr_diffusion::{
    ReplicaEstimate, TransportConfig, build_fecr_manifest, estimate, jackknife_ratio_mean_se,
    mean_se, origin_slope, run_transport, sample_run, summarize,
};

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
    assert!(z.abs() <= Z_CRIT, "{name}: z = {z}, threshold = {Z_CRIT}");
}

fn equal_energy_estimates(base_seed: u64, m: u64, t_end: f64) -> Vec<ReplicaEstimate> {
    let model = equal_energy_model();
    let estimates: Vec<ReplicaEstimate> = run_replicas(base_seed, m, |_, seed| {
        let state = model.random_state(63, seed + STATE_SEED_OFFSET).unwrap();
        let (times, samples, _) = sample_run(model.clone(), state, seed, t_end, 10, 1.0).unwrap();
        estimate(&times, &samples, origin_slope)
    })
    .unwrap()
    .into_iter()
    .map(|r| r.value)
    .collect();

    let g1: Vec<f64> = estimates.iter().map(|e| e.vacancy - 1.0).collect();
    gate("G1 D_v - 1", &g1);
    let g2: Vec<f64> = estimates
        .iter()
        .map(|e| e.tracer[0] - e.tracer[1])
        .collect();
    gate("G2 D*_Fe - D*_Cr", &g2);
    for e in &estimates {
        let sum = e.collective[0] + 2.0 * e.collective[1] + e.collective[2];
        assert!(
            (sum - e.vacancy).abs() <= 1e-9 * e.vacancy.max(1.0),
            "sum rule violated: {sum} vs {}",
            e.vacancy
        );
    }
    estimates
}

#[test]
fn equal_energy_limit_smoke() {
    equal_energy_estimates(30_000_000, 100, 25.0);
}

#[test]
#[ignore = "full tier: statistical"]
fn equal_energy_limit_gates() {
    let estimates = equal_energy_estimates(20261118, M, 125.0);

    let (n_a, n_fe, n_cr) = (127.0, 64.0, 63.0);
    let p = |e: &ReplicaEstimate| e.vacancy - e.s;

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

    let mean = |f: &dyn Fn(&ReplicaEstimate) -> f64| {
        estimates.iter().map(f).sum::<f64>() / estimates.len() as f64
    };
    let (ff, fc, cc) = (
        mean(&|e| e.collective[0]),
        mean(&|e| e.collective[1]),
        mean(&|e| e.collective[2]),
    );
    let trace = ff + cc;
    let det = ff * cc - fc * fc;
    assert!(ff >= 0.0 && cc >= 0.0, "negative diagonal");
    assert!(det >= -1e-9 * trace * trace, "det = {det}, trace = {trace}");
}

fn pure_fe_vacancy(base_seed: u64, m: u64, hops: f64) {
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
    let t_end = hops / (8.0 * r);
    let estimates: Vec<ReplicaEstimate> = run_replicas(base_seed, m, |_, seed| {
        let state = model.random_state(0, seed + STATE_SEED_OFFSET).unwrap();
        let (times, samples, _) = sample_run(model.clone(), state, seed, t_end, 10, 1.0).unwrap();
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

#[test]
fn pure_fe_vacancy_smoke() {
    pure_fe_vacancy(31_000_000, 100, 200.0);
}

#[test]
#[ignore = "full tier: statistical"]
fn pure_fe_vacancy_gate() {
    pure_fe_vacancy(20261119, M, 1000.0);
}

#[test]
fn jackknife_ratio_eliminates_per_replica_jensen_bias() {
    // True E[u] = 0.727, E[v] = 1.0 with independent exponential draws:
    // per-replica u_i / v_i has divergent/huge positive Jensen bias from 1/v_i,
    // whereas ratio-of-means + jackknife SE concentrates cleanly around 0.727.
    let mut rng = KmcRng::from_seed(12345);
    let n = 2000;
    let mut u = Vec::with_capacity(n);
    let mut v = Vec::with_capacity(n);
    for _ in 0..n {
        u.push(-0.727 * rng.open01().ln());
        v.push(-rng.open01().ln());
    }
    let (ratio, se) = jackknife_ratio_mean_se(&u, &v, 1.0);
    assert!(((ratio - 0.727) / se).abs() <= 4.0, "ratio={ratio} se={se}");

    let naive_per_replica: Vec<f64> = u.iter().zip(&v).map(|(a, b)| a / b).collect();
    let (naive_mean, _) = mean_se(&naive_per_replica);
    assert!(
        naive_mean > 1.5,
        "naive mean {naive_mean} should exhibit Jensen bias"
    );

    // Single zero-displacement replica does not poison the ensemble ratio.
    let (r_zero, se_zero) = jackknife_ratio_mean_se(&[0.0, 0.7, 0.8], &[0.0, 1.0, 1.0], 1.0);
    assert!((r_zero - 0.75).abs() < 1e-12);
    assert!(se_zero.is_finite() && se_zero > 0.0);
}

#[test]
fn windowed_ols_transport_handles_u64_max_seed_and_builds_manifest() {
    let model = equal_energy_model();
    let cfg = TransportConfig::windowed(5.0, 25.0, 8, 1.0);
    let runs = run_transport(&model, 32, u64::MAX, 4, &cfg, 2).unwrap();
    assert_eq!(runs.len(), 4);
    assert!(runs.iter().all(|r| r.final_sro_alpha1.is_some()));
    let summary = summarize(&runs, 32, 127);
    assert!(summary.f.0.is_finite() && summary.f.1.is_finite());
    assert!(summary.window_drift_ratio.is_finite());

    let manifest = build_fecr_manifest(&model, [4, 4, 4], 32, 1000.0, u64::MAX, 4, &cfg, 2, None);
    let json = manifest.to_json_pretty().unwrap();
    let decoded = okrates_schema::RunManifest::from_json(&json).unwrap();
    assert_eq!(decoded, manifest);
}
