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
fn chi2_quantile_matches_textbook() {
    for (df, alpha, expected) in [
        (1, 0.05, 3.841458820694124),
        (3, 0.05, 7.814727903251178),
        (3, 1e-6, 30.6648497062136),
    ] {
        let got = chi2_upper_quantile(df, alpha);
        assert!(
            (got - expected).abs() <= 1e-6,
            "df {df}: {got} vs {expected}"
        );
    }
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
    let critical = chi2_upper_quantile(df, ALPHA);
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
