//! Validates the Fe–Cr single-vacancy model: detailed balance, barrier bounds,
//! conservation and equilibrium occupation.

use std::collections::HashMap;

use okrates_core::ensemble::run_replicas;
use okrates_core::{KmcRng, Model, Simulation, Step};
use okrates_models::bcc::BccLattice;
use okrates_models::fecr::{Energetics, FeCrError, FeCrJump, FeCrModel, PairEnergetics, Species};

const ATTEMPT_FREQUENCY: f64 = 1e13;
const TEMPERATURE_K: f64 = 800.0;
const BASE: u64 = 80_000_000;
const INIT_OFFSET: u64 = 10_000_000;
const R: u64 = 10_000;
const ALPHA: f64 = 1e-6;

fn model(cells: [usize; 3]) -> FeCrModel<PairEnergetics> {
    FeCrModel::new(
        BccLattice::new(cells).unwrap(),
        PairEnergetics::illustrative(),
        ATTEMPT_FREQUENCY,
        TEMPERATURE_K,
    )
    .unwrap()
}

#[test]
fn detailed_balance_is_exact() {
    let model = model([3, 3, 3]);
    let kt = model.kt();
    for c in 0..20 {
        let state = model.random_state(11, 3000 + c).unwrap();
        for d in 0..8 {
            let e0 = model
                .energetics()
                .total_energy(model.lattice(), state.occupancy());
            let kf = model.rate(&state, d);
            let bf = model.barrier(&state, d);
            let target = model.lattice().neighbor(state.vacancy(), d);
            let local = model.energetics().exchange_delta(
                model.lattice(),
                state.occupancy(),
                state.vacancy(),
                target,
            );
            let mut after = state.clone();
            model.apply(&mut after, &FeCrJump { direction: d }).unwrap();
            let e1 = model
                .energetics()
                .total_energy(model.lattice(), after.occupancy());
            let kr = model.rate(&after, 7 - d);
            assert!(
                (local - (e1 - e0)).abs() <= 1e-9,
                "local {local} vs {}",
                e1 - e0
            );
            assert!(
                ((kf / kr).ln() + (e1 - e0) / kt).abs() <= 1e-9,
                "detailed balance violated at c={c} d={d}"
            );
            assert!(bf >= 0.0);
        }
    }
}

fn with_kra(atom: Species, kra: f64) -> PairEnergetics {
    let base = PairEnergetics::illustrative();
    let (fe, cr) = (base.kra(Species::Fe), base.kra(Species::Cr));
    match atom {
        Species::Fe => PairEnergetics::new(*base.pair(), kra, cr).unwrap(),
        _ => PairEnergetics::new(*base.pair(), fe, kra).unwrap(),
    }
}

fn build(energetics: PairEnergetics) -> Result<FeCrModel<PairEnergetics>, FeCrError> {
    FeCrModel::new(
        BccLattice::new([3, 3, 3]).unwrap(),
        energetics,
        ATTEMPT_FREQUENCY,
        TEMPERATURE_K,
    )
}

#[test]
fn barrier_bound_boundary() {
    for atom in [Species::Fe, Species::Cr] {
        let bound = PairEnergetics::illustrative().delta_bound(atom);
        assert!(build(with_kra(atom, bound / 2.0 + 1e-9)).is_ok());
        assert!(matches!(
            build(with_kra(atom, bound / 2.0 - 1e-9)),
            Err(FeCrError::NegativeBarrier { atom: a, .. }) if a == atom
        ));
    }
}

#[test]
fn barrier_bound_is_attained() {
    for atom in [Species::Fe, Species::Cr] {
        let base = PairEnergetics::illustrative();
        let bound = base.delta_bound(atom);
        let kra = bound / 2.0 + 1e-6;
        let model = build(with_kra(atom, kra)).unwrap();
        let pair = base.pair();
        let d =
            |s: Species| pair[atom.index()][s.index()] - pair[Species::Vacancy.index()][s.index()];
        let (s_max, s_min) = if d(Species::Fe) >= d(Species::Cr) {
            (Species::Fe, Species::Cr)
        } else {
            (Species::Cr, Species::Fe)
        };
        let lattice = model.lattice();
        let j = lattice.neighbor(0, 0);
        let mut occupancy = vec![Species::Fe; lattice.site_count()];
        // The minimum ΔE = 7·d(s_min) − 7·d(s_max) = −bound: weakest X-bonds
        // gained around the vacancy site, strongest lost around the atom site.
        for &k in lattice.neighbors(0) {
            occupancy[k] = s_min;
        }
        for &k in lattice.neighbors(j) {
            occupancy[k] = s_max;
        }
        occupancy[0] = Species::Vacancy;
        occupancy[j] = atom;
        let state = model.state(occupancy).unwrap();
        let barrier = model.barrier(&state, 0);
        assert!(
            (barrier - (kra - bound / 2.0)).abs() <= 1e-12,
            "{atom:?}: barrier {barrier}, expected {}",
            kra - bound / 2.0
        );
        assert!(barrier >= 0.0);
    }
}

#[test]
fn composition_is_conserved() {
    let model = model([3, 3, 3]);
    let state = model.random_state(11, 9).unwrap();
    let initial = state.counts();
    let mut sim = Simulation::new(model, state, 9);
    for _ in 0..5000 {
        assert!(matches!(sim.step().unwrap(), Step::Fired { .. }));
        assert_eq!(sim.state().counts(), initial);
        assert_eq!(sim.model().check_invariants(sim.state()), Ok(()));
    }
}

#[test]
fn chi2_quantile_matches_textbook() {
    for (df, alpha, expected) in [
        (1, 0.05, 3.841458820694124),
        (3, 0.05, 7.814727903251178),
        (3, 1e-6, 30.6648497062136),
    ] {
        let got = okrates_core::testing::chi2_upper_quantile(df, alpha);
        assert!(
            (got - expected).abs() <= 1e-6,
            "df {df}: {got} vs {expected}"
        );
    }
}

#[test]
fn tabulated_energetics_detailed_balance_and_missing_env_error() {
    use okrates_core::SimError;
    use okrates_models::tabulated::TabulatedEnergetics;
    use okrates_schema::{BarrierRecord, BarrierSet};

    let records = vec![
        BarrierRecord {
            id: "fe_asym".to_string(),
            migrating_species: "Fe".to_string(),
            env_key: "cr_atom=0,cr_vac=1".to_string(),
            forward_barrier_ev: 0.62,
            reverse_barrier_ev: 0.68,
            delta_energy_ev: -0.06,
            attempt_frequency_hz: 1.0e13,
            reverse_attempt_frequency_hz: None,
            barrier_std_ev: Some(0.01),
            temp_range_k: Some([300.0, 1200.0]),
            provenance: None,
        },
        BarrierRecord {
            id: "fe_default".to_string(),
            migrating_species: "Fe".to_string(),
            env_key: "*".to_string(),
            forward_barrier_ev: 0.65,
            reverse_barrier_ev: 0.65,
            delta_energy_ev: 0.0,
            attempt_frequency_hz: 1.0e13,
            reverse_attempt_frequency_hz: None,
            barrier_std_ev: None,
            temp_range_k: Some([300.0, 1200.0]),
            provenance: None,
        },
        BarrierRecord {
            id: "cr_default".to_string(),
            migrating_species: "Cr".to_string(),
            env_key: "*".to_string(),
            forward_barrier_ev: 0.58,
            reverse_barrier_ev: 0.58,
            delta_energy_ev: 0.0,
            attempt_frequency_hz: 1.2e13,
            reverse_attempt_frequency_hz: None,
            barrier_std_ev: None,
            temp_range_k: Some([300.0, 1200.0]),
            provenance: None,
        },
    ];
    let set = BarrierSet::new("test_fecr", 1e-9, records).unwrap();
    let energetics = TabulatedEnergetics::new(set).unwrap();
    let m = FeCrModel::new(
        BccLattice::new([3, 3, 3]).unwrap(),
        energetics,
        ATTEMPT_FREQUENCY,
        TEMPERATURE_K,
    )
    .unwrap();
    let kt = m.kt();
    let state = m.random_state(5, 77).unwrap();
    let nb0 = m.lattice().neighbor(state.vacancy(), 0);
    assert!(
        m.energetics()
            .total_energy(m.lattice(), state.occupancy())
            .is_nan()
    );
    assert!(
        m.energetics()
            .exchange_delta(m.lattice(), state.occupancy(), state.vacancy(), nb0)
            .is_nan()
    );
    assert!(
        m.energetics()
            .swap_delta(m.lattice(), state.occupancy(), 0, 1)
            .is_nan()
    );
    assert!(
        m.energetics()
            .swap_delta(m.lattice(), &[Species::Vacancy], 0, 1)
            .is_nan()
    );
    for d in 0..8 {
        let kf = m.rate(&state, d);
        let target = m.lattice().neighbor(state.vacancy(), d);
        let de = m.energetics().transition_delta_ev(
            m.lattice(),
            state.occupancy(),
            state.vacancy(),
            target,
        );
        let mut after = state.clone();
        m.apply(&mut after, &FeCrJump { direction: d }).unwrap();
        let kr = m.rate(&after, 7 - d);
        assert!(((kf / kr).ln() + de / kt).abs() <= 1e-12);
    }

    // Missing Cr record surfaces SimError::InvalidRate without panicking.
    let fe_only = BarrierSet::new(
        "fe_only",
        1e-9,
        vec![BarrierRecord {
            id: "fe_default".to_string(),
            migrating_species: "Fe".to_string(),
            env_key: "*".to_string(),
            forward_barrier_ev: 0.65,
            reverse_barrier_ev: 0.65,
            delta_energy_ev: 0.0,
            attempt_frequency_hz: 1.0e13,
            reverse_attempt_frequency_hz: None,
            barrier_std_ev: None,
            temp_range_k: None,
            provenance: None,
        }],
    )
    .unwrap();
    let incomplete = FeCrModel::new(
        BccLattice::new([2, 2, 2]).unwrap(),
        TabulatedEnergetics::new(fe_only).unwrap(),
        ATTEMPT_FREQUENCY,
        TEMPERATURE_K,
    )
    .unwrap();
    let mut occ = vec![Species::Cr; 16];
    occ[0] = Species::Vacancy;
    let init = incomplete.state(occ).unwrap();
    let mut sim = Simulation::new(incomplete, init.clone(), 1);
    assert!(matches!(sim.step(), Err(SimError::InvalidRate { .. })));
    assert_eq!(sim.state(), &init);
}

/// Runs `replicas` replicas started from the Boltzmann distribution and applies
/// a chi-square test to the occupation of energy groups at time `t_s`.
fn equilibrium_chi2(base_seed: u64, replicas: u64) {
    let model = model([2, 2, 2]);
    let kt = model.kt();
    let n = model.lattice().site_count();
    assert_eq!(n, 16);

    let mut states: Vec<Vec<Species>> = Vec::new();
    let mut index: HashMap<Vec<Species>, usize> = HashMap::new();
    for v in 0..n {
        let others: Vec<usize> = (0..n).filter(|&s| s != v).collect();
        for x in 0..others.len() {
            for y in x + 1..others.len() {
                let mut occupancy = vec![Species::Fe; n];
                occupancy[v] = Species::Vacancy;
                occupancy[others[x]] = Species::Cr;
                occupancy[others[y]] = Species::Cr;
                index.insert(occupancy.clone(), states.len());
                states.push(occupancy);
            }
        }
    }
    assert_eq!(states.len(), 1680);

    let energies: Vec<f64> = states
        .iter()
        .map(|o| model.energetics().total_energy(model.lattice(), o))
        .collect();
    let e_min = energies.iter().cloned().fold(f64::INFINITY, f64::min);
    let weights: Vec<f64> = energies.iter().map(|e| (-(e - e_min) / kt).exp()).collect();
    let z: f64 = weights.iter().sum();
    let pi: Vec<f64> = weights.iter().map(|w| w / z).collect();
    let mut cumulative = Vec::with_capacity(pi.len());
    let mut acc = 0.0;
    for p in &pi {
        acc += p;
        cumulative.push(acc);
    }

    // Classes of equal energy.
    let mut sorted = energies.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mut class_energies: Vec<f64> = Vec::new();
    for e in sorted {
        match class_energies.last() {
            Some(&first) if (e - first).abs() <= 1e-9 => {}
            _ => class_energies.push(e),
        }
    }
    let class_of = |e: f64| -> usize {
        class_energies
            .iter()
            .rposition(|&first| e >= first - 1e-9)
            .unwrap()
    };
    let mut class_prob = vec![0.0; class_energies.len()];
    let state_class: Vec<usize> = energies.iter().map(|&e| class_of(e)).collect();
    for (i, &c) in state_class.iter().enumerate() {
        class_prob[c] += pi[i];
    }

    // Groups with expected count at least 5.
    let r = replicas as f64;
    let mut class_group = vec![0_usize; class_prob.len()];
    let mut group_prob = vec![0.0];
    for (c, &p) in class_prob.iter().enumerate() {
        let g = group_prob.len() - 1;
        class_group[c] = g;
        group_prob[g] += p;
        if r * group_prob[g] >= 5.0 {
            group_prob.push(0.0);
        }
    }
    if r * group_prob.last().unwrap() < 5.0 {
        let last = group_prob.pop().unwrap();
        let g = group_prob.len() - 1;
        group_prob[g] += last;
        for group in class_group.iter_mut() {
            if *group > g {
                *group = g;
            }
        }
    }
    let groups = group_prob.len();
    let state_group: Vec<usize> = state_class.iter().map(|&c| class_group[c]).collect();

    let t_s = 1000.0 / (8.0 * ATTEMPT_FREQUENCY * (-0.60 / kt).exp());
    let results = run_replicas(base_seed, replicas, |_, seed| {
        let u = KmcRng::from_seed(seed + INIT_OFFSET).open01();
        let start = cumulative
            .iter()
            .position(|&c| u < c)
            .unwrap_or(cumulative.len() - 1);
        let state = model.state(states[start].clone()).unwrap();
        let mut sim = Simulation::new(model.clone(), state, seed);
        loop {
            let held = state_group[index[sim.state().occupancy()]];
            assert!(matches!(sim.step().unwrap(), Step::Fired { .. }));
            if sim.time() > t_s {
                return held;
            }
        }
    })
    .unwrap();

    let mut observed = vec![0_u64; groups];
    for replica in &results {
        observed[replica.value] += 1;
    }
    let mut chi2 = 0.0;
    for g in 0..groups {
        let expected = r * group_prob[g];
        chi2 += (observed[g] as f64 - expected).powi(2) / expected;
        println!("group {g}: observed {} expected {expected:.3}", observed[g]);
    }
    let df = groups - 1;
    let critical = okrates_core::testing::chi2_upper_quantile(df, ALPHA);
    println!("chi2={chi2:.6} df={df} critical={critical:.6} groups={groups} t_s={t_s:.6e}");
    assert!(chi2 <= critical, "chi2 = {chi2}, critical = {critical}");
}

#[test]
fn equilibrium_boltzmann_smoke() {
    equilibrium_chi2(81_000_000, 400);
}

#[test]
#[ignore = "full tier: statistical"]
fn equilibrium_matches_boltzmann() {
    equilibrium_chi2(BASE, R);
}

#[test]
fn imported_dft_temp_range_is_enforced_by_fecr_model() {
    use okrates_models::tabulated::TabulatedEnergetics;
    use okrates_schema::{import_aiida_qe_json, import_pymatgen_vasp_json};

    let vasp_json = r#"{
        "name": "vasp-fecr-window",
        "tolerance_ev": 1e-9,
        "tasks": [
            {
                "task_id": "fe-wildcard",
                "migrating_specie": "Fe",
                "env_key": "*",
                "image_energies_ev": [-10.0, -9.35, -10.0],
                "v0_hz": 1.0e13,
                "functional": "PBE",
                "temp_range_k": [500.0, 1100.0]
            },
            {
                "task_id": "cr-wildcard",
                "migrating_specie": "Cr",
                "env_key": "*",
                "image_energies_ev": [-10.0, -9.42, -10.0],
                "v0_hz": 1.05e13,
                "functional": "PBE"
            }
        ]
    }"#;
    let vasp_set = import_pymatgen_vasp_json(vasp_json).unwrap();
    let vasp_tab = TabulatedEnergetics::new(vasp_set).unwrap();
    let lattice = BccLattice::new([3, 3, 3]).unwrap();
    assert!(FeCrModel::new(lattice.clone(), vasp_tab.clone(), ATTEMPT_FREQUENCY, 500.0).is_ok());
    assert!(FeCrModel::new(lattice.clone(), vasp_tab.clone(), ATTEMPT_FREQUENCY, 773.0).is_ok());
    assert!(matches!(
        FeCrModel::new(lattice.clone(), vasp_tab, ATTEMPT_FREQUENCY, 300.0),
        Err(FeCrError::BarrierTable(msg)) if msg.contains("outside [500, 1100] K")
    ));

    let aiida_json = r#"{
        "workchain_uuid": "uuid-fecr",
        "name": "qe-fecr-window",
        "dft_functional": "PBEsol",
        "tolerance_ev": 1e-9,
        "paths": [
            {
                "label": "fe-wildcard",
                "species": "Fe",
                "env_key": "*",
                "forward_barrier_ev": 0.65,
                "reverse_barrier_ev": 0.65,
                "reaction_energy_ev": 0.0,
                "attempt_frequency_thz": 10.0,
                "barrier_uncertainty_ev": 0.01,
                "temp_range_k": [500.0, 1100.0]
            },
            {
                "label": "cr-wildcard",
                "species": "Cr",
                "env_key": "*",
                "forward_barrier_ev": 0.58,
                "reverse_barrier_ev": 0.58,
                "reaction_energy_ev": 0.0,
                "attempt_frequency_thz": 10.5,
                "barrier_uncertainty_ev": null
            }
        ]
    }"#;
    let aiida_set = import_aiida_qe_json(aiida_json).unwrap();
    let aiida_tab = TabulatedEnergetics::new(aiida_set).unwrap();
    assert!(FeCrModel::new(lattice.clone(), aiida_tab.clone(), ATTEMPT_FREQUENCY, 500.0).is_ok());
    assert!(FeCrModel::new(lattice.clone(), aiida_tab.clone(), ATTEMPT_FREQUENCY, 773.0).is_ok());
    assert!(matches!(
        FeCrModel::new(lattice, aiida_tab, ATTEMPT_FREQUENCY, 300.0),
        Err(FeCrError::BarrierTable(msg)) if msg.contains("outside [500, 1100] K")
    ));
}
