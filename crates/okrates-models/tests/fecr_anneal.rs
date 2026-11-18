//! Tests of the Kawasaki anneal: local swap energies against full recomputation,
//! conservation laws, detailed balance against the exact Boltzmann distribution
//! on a 16-site lattice, and preservation of the random alloy at zero interaction.

use std::collections::HashMap;

use okrates_core::KmcRng;
use okrates_core::ensemble::run_replicas;
use okrates_core::testing::chi2_upper_quantile;
use okrates_models::bcc::BccLattice;
use okrates_models::fecr::{Energetics, FeCrError, FeCrModel, PairEnergetics, Species};
use okrates_models::fecr_anneal::{KawasakiAnnealer, anneal, warren_cowley_alpha1};
use okrates_models::fecr_martinez::Martinez2012;
use okrates_models::tabulated::TabulatedEnergetics;
use okrates_schema::{BarrierRecord, BarrierSet};

const ATTEMPT_FREQUENCY: f64 = 1e13;
const TEMPERATURE_K: f64 = 800.0;
const ALPHA: f64 = 1e-6;
const Z_CRIT: f64 = 4.89163847569859;

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
fn chunked_annealer_matches_single_call_bitwise_and_rejects_mismatched_models() {
    let m = model([4, 4, 4]);
    let init = m.random_state(32, 42).unwrap().occupancy().to_vec();
    let one_shot = anneal(&m, init.clone(), 24, 99).unwrap();

    let mut chunked = KawasakiAnnealer::new(&m, init.clone(), KmcRng::from_seed(99)).unwrap();
    for step in [5, 7, 12] {
        chunked.run_sweeps(&m, step).unwrap();
    }
    assert_eq!(chunked.sweeps_completed(), 24);
    assert_eq!(chunked.finish(), one_shot);

    let m_hot = FeCrModel::new(
        BccLattice::new([4, 4, 4]).unwrap(),
        PairEnergetics::illustrative(),
        ATTEMPT_FREQUENCY,
        950.0,
    )
    .unwrap();
    let mut guard_check = KawasakiAnnealer::new(&m, init, KmcRng::from_seed(5)).unwrap();
    assert_eq!(
        guard_check.run_sweeps(&m_hot, 1),
        Err(FeCrError::AnnealerModelMismatch)
    );
}

#[test]
fn tabulated_energetics_rejects_annealing_malformed_keys_and_inconsistent_prefactors() {
    let rec = |id: &str, sp: &str, key: &str, ef: f64, er: f64, de: f64| BarrierRecord {
        id: id.to_string(),
        migrating_species: sp.to_string(),
        env_key: key.to_string(),
        forward_barrier_ev: ef,
        reverse_barrier_ev: er,
        delta_energy_ev: de,
        attempt_frequency_hz: 1e13,
        reverse_attempt_frequency_hz: None,
        barrier_std_ev: None,
        temp_range_k: None,
        provenance: None,
    };

    // Valid non-canonical key order (`cr_vac=2,cr_atom=1`) is canonicalized and accepted,
    // while Kawasaki equilibration is explicitly rejected.
    let set = BarrierSet::new(
        "tab",
        1e-9,
        vec![
            rec("fe_star", "Fe", "*", 0.65, 0.65, 0.0),
            rec("cr_star", "Cr", "*", 0.58, 0.58, 0.0),
            rec("fe_asym", "Fe", "cr_vac=2,cr_atom=1", 0.67, 0.64, 0.03),
        ],
    )
    .unwrap();
    let tab = FeCrModel::new(
        BccLattice::new([3, 3, 3]).unwrap(),
        TabulatedEnergetics::new(set).unwrap(),
        ATTEMPT_FREQUENCY,
        TEMPERATURE_K,
    )
    .unwrap();
    let occ = tab.random_state(8, 12).unwrap().occupancy().to_vec();
    assert!(matches!(
        KawasakiAnnealer::new(&tab, occ, KmcRng::from_seed(1)),
        Err(FeCrError::UnsupportedEquilibration(_))
    ));

    // Malformed / out-of-range env_key strings are rejected instead of silently shadowed by "*".
    for bad_key in ["cr=0", "cr_atom=1", "cr_atom=8,cr_vac=0", "cr_saddle=7"] {
        let bad_set =
            BarrierSet::new("bad", 1e-6, vec![rec("r", "Fe", bad_key, 0.6, 0.6, 0.0)]).unwrap();
        assert!(matches!(
            TabulatedEnergetics::new(bad_set),
            Err(FeCrError::BarrierTable(_))
        ));
    }

    // Duplicate after canonicalization (`cr_atom=1,cr_vac=2` vs `cr_vac=2,cr_atom=1`) is rejected.
    let dup_set = BarrierSet::new(
        "dup",
        1e-6,
        vec![
            rec("r1", "Fe", "cr_atom=1,cr_vac=2", 0.67, 0.64, 0.03),
            rec("r2", "Fe", "cr_vac=2,cr_atom=1", 0.67, 0.64, 0.03),
        ],
    )
    .unwrap();
    assert!(matches!(
        TabulatedEnergetics::new(dup_set),
        Err(FeCrError::BarrierTable(_))
    ));

    // Symmetric key with distinct reverse_attempt_frequency_hz violates symmetry.
    let mut bad_sym = rec("sym", "Fe", "*", 0.65, 0.65, 0.0);
    bad_sym.reverse_attempt_frequency_hz = Some(2e13);
    let sym_set = BarrierSet::new("sym", 1e-6, vec![bad_sym]).unwrap();
    assert!(matches!(
        TabulatedEnergetics::new(sym_set),
        Err(FeCrError::BarrierTable(_))
    ));
}

#[test]
fn warren_cowley_alpha1_distinguishes_ordering_and_clustering() {
    let lattice = BccLattice::new([4, 4, 4]).unwrap();
    // B2 ordered alloy (corner sites Fe, body-centre sites Cr, one vacancy at 0):
    // every 1NN of Fe is on the Cr sublattice -> P(Cr | Fe) = 1 > x_Cr -> alpha_1 < 0.
    let mut b2 = vec![Species::Fe; lattice.site_count()];
    for (s, sp) in b2.iter_mut().enumerate() {
        if lattice.position(s)[0] % 2 != 0 {
            *sp = Species::Cr;
        }
    }
    b2[0] = Species::Vacancy;
    assert!(warren_cowley_alpha1(&lattice, &b2) < -0.8);

    // Phase-separated slab (x < 4 Fe, x >= 4 Cr): few Fe-Cr bonds -> alpha_1 > 0.
    let mut slab = vec![Species::Fe; lattice.site_count()];
    for (s, sp) in slab.iter_mut().enumerate() {
        if lattice.position(s)[0] >= 4 {
            *sp = Species::Cr;
        }
    }
    slab[0] = Species::Vacancy;
    assert!(warren_cowley_alpha1(&lattice, &slab) > 0.5);
}
#[test]
fn swap_delta_matches_total_energy() {
    let model = model([3, 3, 3]);
    let lattice = model.lattice();
    let energetics = model.energetics();
    let cr_count = (0.2_f64 * 53.0).round() as usize;
    assert_eq!(cr_count, 11);
    let (mut neighbour_pairs, mut vacancy_pairs) = (0, 0);
    for c in 0..20 {
        let state = model.random_state(cr_count, 5000 + c).unwrap();
        let occ = state.occupancy();
        let vacancy = state.vacancy();
        let before = energetics.total_energy(lattice, occ);
        let sites: Vec<usize> = (0..occ.len()).filter(|&s| s != vacancy).collect();
        for (x, &i) in sites.iter().enumerate() {
            for &j in &sites[x + 1..] {
                let mut after = occ.to_vec();
                after.swap(i, j);
                let reference = energetics.total_energy(lattice, &after) - before;
                let local = energetics.swap_delta(lattice, occ, i, j);
                assert!(
                    (local - reference).abs() <= 1e-9,
                    "state {c} sites {i},{j}: local {local} vs reference {reference}"
                );
                if lattice.neighbors(i).contains(&j) {
                    neighbour_pairs += 1;
                }
                if lattice.neighbors(vacancy).contains(&i)
                    || lattice.neighbors(vacancy).contains(&j)
                {
                    vacancy_pairs += 1;
                }
            }
        }
    }
    assert!(neighbour_pairs > 0);
    assert!(vacancy_pairs > 0);
}

fn counts(occupancy: &[Species]) -> [usize; 3] {
    let mut counts = [0; 3];
    for species in occupancy {
        counts[species.index()] += 1;
    }
    counts
}

#[test]
fn composition_and_energy_are_consistent() {
    let model = model([4, 4, 4]);
    let state = model.random_state(32, 7).unwrap();
    let start = state.occupancy().to_vec();
    let result = anneal(&model, start.clone(), 50, 7).unwrap();
    assert_eq!(counts(&result.occupancy), counts(&start));
    assert_eq!(state.vacancy(), {
        result
            .occupancy
            .iter()
            .position(|&s| s == Species::Vacancy)
            .unwrap()
    });
    let direct = model
        .energetics()
        .total_energy(model.lattice(), &result.occupancy);
    let tracked = *result.energy.last().unwrap();
    assert!(
        (tracked - direct).abs() <= 1e-9,
        "tracked {tracked} vs direct {direct}"
    );
    assert_eq!(result.energy.len(), 51);
    assert_eq!(result.attempts, 50 * 127);
}

#[test]
fn no_op_without_both_species() {
    let model = model([3, 3, 3]);
    let state = model.random_state(0, 3).unwrap();
    let start = state.occupancy().to_vec();
    let result = anneal(&model, start.clone(), 5, 3).unwrap();
    assert_eq!(result.attempts, 0);
    assert_eq!(result.accepted, 0);
    assert_eq!(result.occupancy, start);
}

/// Anneals from the highest-energy state and applies a chi-square test of the
/// final state's energy group against the exact Boltzmann distribution over all
/// states with the vacancy on site 0 and two Cr atoms.
fn anneal_chi2<E: Energetics + Sync>(
    model: &FeCrModel<E>,
    expected_states: usize,
    base_seed: u64,
    replicas: u64,
) {
    let kt = model.kt();
    let n = model.lattice().site_count();

    let mut states: Vec<Vec<Species>> = Vec::new();
    let mut index: HashMap<Vec<Species>, usize> = HashMap::new();
    for x in 1..n {
        for y in x + 1..n {
            let mut occupancy = vec![Species::Fe; n];
            occupancy[0] = Species::Vacancy;
            occupancy[x] = Species::Cr;
            occupancy[y] = Species::Cr;
            index.insert(occupancy.clone(), states.len());
            states.push(occupancy);
        }
    }
    assert_eq!(states.len(), expected_states);

    let energies: Vec<f64> = states
        .iter()
        .map(|o| model.energetics().total_energy(model.lattice(), o))
        .collect();
    let e_min = energies.iter().cloned().fold(f64::INFINITY, f64::min);
    let weights: Vec<f64> = energies.iter().map(|e| (-(e - e_min) / kt).exp()).collect();
    let z: f64 = weights.iter().sum();
    let pi: Vec<f64> = weights.iter().map(|w| w / z).collect();

    let mut start = 0;
    for (i, &e) in energies.iter().enumerate() {
        if e > energies[start] {
            start = i;
        }
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

    let results = run_replicas(base_seed, replicas, |_, seed| {
        let result = anneal(model, states[start].clone(), 10, seed).unwrap();
        state_group[index[&result.occupancy]]
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
    let critical = chi2_upper_quantile(df, ALPHA);
    println!("chi2={chi2:.6} df={df} critical={critical:.6} groups={groups}");
    assert!(chi2 <= critical, "chi2 = {chi2}, critical = {critical}");
}

#[test]
fn anneal_boltzmann_smoke() {
    let model = model([2, 2, 2]);
    assert_eq!(model.lattice().site_count(), 16);
    anneal_chi2(&model, 105, 83_000_000, 2000);
}

#[test]
#[ignore = "full tier: statistical"]
fn anneal_matches_boltzmann() {
    let model = model([2, 2, 2]);
    assert_eq!(model.lattice().site_count(), 16);
    anneal_chi2(&model, 105, 82_000_000, 20000);
}

fn martinez_model() -> FeCrModel<Martinez2012> {
    FeCrModel::new(
        BccLattice::new([3, 3, 3]).unwrap(),
        Martinez2012::new(TEMPERATURE_K).unwrap(),
        ATTEMPT_FREQUENCY,
        TEMPERATURE_K,
    )
    .unwrap()
}

#[test]
fn martinez_anneal_boltzmann_smoke() {
    let model = martinez_model();
    assert_eq!(model.lattice().site_count(), 54);
    anneal_chi2(&model, 1378, 87_000_000, 300);
}

#[test]
#[ignore = "full tier: statistical"]
fn martinez_anneal_matches_boltzmann() {
    let model = martinez_model();
    assert_eq!(model.lattice().site_count(), 54);
    anneal_chi2(&model, 1378, 86_000_000, 20000);
}

/// With all pair energies zero every swap is accepted and the chain must keep the
/// uniform random alloy; checks two exact means of that distribution.
fn equal_energy(base_seed: u64, replicas: u64) {
    let energetics = PairEnergetics::new([[0.0; 3]; 3], 0.0, 0.0).unwrap();
    let model = FeCrModel::new(
        BccLattice::new([4, 4, 4]).unwrap(),
        energetics,
        ATTEMPT_FREQUENCY,
        TEMPERATURE_K,
    )
    .unwrap();
    let lattice = model.lattice();
    assert_eq!(lattice.site_count(), 128);
    let results = run_replicas(base_seed, replicas, |_, seed| {
        let start = model
            .random_state(32, seed + 1_000_000_000)
            .unwrap()
            .occupancy()
            .to_vec();
        let result = anneal(&model, start, 10, seed).unwrap();
        assert_eq!(result.accepted, result.attempts);
        assert_eq!(result.attempts, 10 * 127);
        let occ = &result.occupancy;
        let vacancy = occ.iter().position(|&s| s == Species::Vacancy).unwrap();
        let cr_v = lattice
            .neighbors(vacancy)
            .iter()
            .filter(|&&k| occ[k] == Species::Cr)
            .count();
        let mut cr_cr = 0;
        for (site, &species) in occ.iter().enumerate() {
            if species == Species::Cr {
                cr_cr += lattice
                    .neighbors(site)
                    .iter()
                    .filter(|&&k| k > site && occ[k] == Species::Cr)
                    .count();
            }
        }
        (cr_v as f64, cr_cr as f64)
    })
    .unwrap();

    let exact_v = 8.0 * 32.0 / 127.0;
    let exact_cc = (4.0 * 128.0 - 8.0) * 32.0 * 31.0 / (127.0 * 126.0);
    for (name, exact, values) in [
        (
            "cr_v",
            exact_v,
            results.iter().map(|r| r.value.0).collect::<Vec<_>>(),
        ),
        (
            "cr_cr",
            exact_cc,
            results.iter().map(|r| r.value.1).collect::<Vec<_>>(),
        ),
    ] {
        let r = values.len() as f64;
        let mean = values.iter().sum::<f64>() / r;
        let var = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (r - 1.0);
        let se = (var / r).sqrt();
        let z = (mean - exact) / se;
        println!("{name}: mean={mean:.6} exact={exact:.6} se={se:.6} z={z:.4}");
        assert!(z.abs() <= Z_CRIT, "{name}: z = {z}");
    }
}

#[test]
fn equal_energy_smoke() {
    equal_energy(85_000_000, 400);
}

#[test]
#[ignore = "full tier: statistical"]
fn equal_energy_preserves_random_alloy() {
    equal_energy(84_000_000, 4000);
}
