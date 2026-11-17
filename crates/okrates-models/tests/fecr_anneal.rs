//! Tests of the Kawasaki anneal: local swap energies against full recomputation,
//! conservation laws, detailed balance against the exact Boltzmann distribution
//! on a 16-site lattice, and preservation of the random alloy at zero interaction.

use std::collections::HashMap;

use okrates_core::ensemble::run_replicas;
use okrates_models::bcc::BccLattice;
use okrates_models::fecr::{Energetics, FeCrModel, PairEnergetics, Species};
use okrates_models::fecr_anneal::anneal;
use okrates_models::fecr_martinez::Martinez2012;

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

fn ln_gamma(z: f64) -> f64 {
    const G: f64 = 7.0;
    const C: [f64; 9] = [
        0.999_999_999_999_809_9,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];
    if z < 0.5 {
        return (std::f64::consts::PI / (std::f64::consts::PI * z).sin()).ln() - ln_gamma(1.0 - z);
    }
    let z = z - 1.0;
    let mut sum = C[0];
    for (i, c) in C.iter().enumerate().skip(1) {
        sum += c / (z + i as f64);
    }
    let t = z + G + 0.5;
    0.5 * (2.0 * std::f64::consts::PI).ln() + (z + 0.5) * t.ln() - t + sum.ln()
}

/// Regularized upper incomplete gamma function Q(a, x).
fn gamma_q(a: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 1.0;
    }
    if x < a + 1.0 {
        let mut term = 1.0;
        let mut sum = 1.0;
        let mut n = 1.0;
        while term > sum * 1e-17 {
            term *= x / (a + n);
            sum += term;
            n += 1.0;
        }
        let p = (a * x.ln() - x - ln_gamma(a + 1.0) + sum.ln()).exp();
        1.0 - p
    } else {
        const TINY: f64 = 1e-300;
        let mut b = x + 1.0 - a;
        let mut c = 1.0 / TINY;
        let mut d = 1.0 / b;
        let mut h = d;
        for i in 1..1000 {
            let an = -(i as f64) * (i as f64 - a);
            b += 2.0;
            d = an * d + b;
            if d.abs() < TINY {
                d = TINY;
            }
            c = b + an / c;
            if c.abs() < TINY {
                c = TINY;
            }
            d = 1.0 / d;
            let delta = d * c;
            h *= delta;
            if (delta - 1.0).abs() < 1e-16 {
                break;
            }
        }
        (a * x.ln() - x - ln_gamma(a)).exp() * h
    }
}

/// Upper `alpha` quantile of the chi-square distribution with `df` degrees of freedom.
fn chi2_upper_quantile(df: usize, alpha: f64) -> f64 {
    let a = df as f64 / 2.0;
    let (mut lo, mut hi) = (0.0, 1000.0);
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if gamma_q(a, mid / 2.0) > alpha {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
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
