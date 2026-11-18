//! Tests of the Martínez et al. (2012) Fe–Cr energetics: parameter transcription,
//! oracle agreement at 0 K and 800 K, local-delta consistency, detailed balance,
//! absolute rates, barrier floor policies, and validation.

use std::collections::HashSet;
use std::fs;

use okrates_core::{Model, SimError, Simulation};
use okrates_models::bcc::BccLattice;
use okrates_models::fecr::{Energetics, FeCrError, FeCrModel, Species};
use okrates_models::fecr_martinez::{BarrierFloorPolicy, Martinez2012, params};
use okrates_models::vacancy::BOLTZMANN_EV_PER_K;

const NU: f64 = 1e13;

fn data(name: &str) -> String {
    let path = format!("{}/tests/data/{name}", env!("CARGO_MANIFEST_DIR"));
    fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path}: {e}"))
}

fn model(cells: [usize; 3], temperature_k: f64) -> FeCrModel<Martinez2012> {
    FeCrModel::new(
        BccLattice::new(cells).unwrap(),
        Martinez2012::new(temperature_k).unwrap(),
        NU,
        temperature_k,
    )
    .unwrap()
}

struct Xorshift(u64);

impl Xorshift {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

fn triple(tokens: &[&str]) -> [i64; 3] {
    [
        tokens[0].parse().unwrap(),
        tokens[1].parse().unwrap(),
        tokens[2].parse().unwrap(),
    ]
}

#[derive(Default)]
struct Block {
    header: String,
    vacancy: Option<[i64; 3]>,
    cr: Vec<[i64; 3]>,
    energy: Option<f64>,
    hmix: Option<f64>,
    barriers: Vec<([i64; 3], f64)>,
}

fn parse_blocks(text: &str, saddle_tag: &str) -> (Vec<Block>, [[f64; 2]; 2]) {
    let mut blocks: Vec<Block> = Vec::new();
    let mut saddle = [[f64::NAN; 2]; 2];
    for line in text.lines() {
        let line = line.trim();
        if let Some(rest) = line.strip_prefix('#') {
            if let Some(values) = rest.trim().strip_prefix(saddle_tag) {
                for item in values.split_whitespace() {
                    let (key, value) = item.split_once('=').unwrap();
                    let key = key.as_bytes();
                    saddle[(key[0] - b'0') as usize][(key[1] - b'0') as usize] =
                        value.parse().unwrap();
                }
            }
            continue;
        }
        if line.is_empty() {
            continue;
        }
        let tokens: Vec<&str> = line.split_whitespace().collect();
        match tokens[0] {
            "config" | "mix" => blocks.push(Block {
                header: line.to_string(),
                ..Block::default()
            }),
            "vacancy" => blocks.last_mut().unwrap().vacancy = Some(triple(&tokens[1..])),
            "cr" => {
                let body = line.strip_prefix("cr").unwrap();
                blocks.last_mut().unwrap().cr = body
                    .split(';')
                    .map(|p| triple(&p.split_whitespace().collect::<Vec<_>>()))
                    .collect();
            }
            "energy" => blocks.last_mut().unwrap().energy = Some(tokens[1].parse().unwrap()),
            "hmix" => blocks.last_mut().unwrap().hmix = Some(tokens[1].parse().unwrap()),
            "barrier" => blocks
                .last_mut()
                .unwrap()
                .barriers
                .push((triple(&tokens[1..4]), tokens[4].parse().unwrap())),
            other => panic!("unknown fixture line {other}"),
        }
    }
    (blocks, saddle)
}

fn occupancy_of(lattice: &BccLattice, block: &Block) -> Vec<Species> {
    let site = |p: [i64; 3]| lattice.site_at(p).unwrap();
    let mut occ = vec![Species::Fe; lattice.site_count()];
    for &p in &block.cr {
        occ[site(p)] = Species::Cr;
    }
    if let Some(v) = block.vacancy {
        occ[site(v)] = Species::Vacancy;
    }
    occ
}

fn json_number(text: &str, key: &str) -> f64 {
    let pattern = format!("\"{key}\":");
    let start = text
        .find(&pattern)
        .unwrap_or_else(|| panic!("key {key} not found"))
        + pattern.len();
    let rest = text[start..].trim_start();
    let end = rest
        .find(|c: char| !(c.is_ascii_digit() || "+-.eE".contains(c)))
        .unwrap();
    rest[..end].parse().unwrap()
}

#[test]
fn constants_match_json() {
    let json = data("fecr-martinez2012.json");
    let omega = &json[json.find("\"PWSCF-PAW\"").unwrap()..];
    let table = [
        ("H0_Fe_eV", params::H0_FE),
        ("H0_Cr_eV", params::H0_CR),
        ("cP_Fe_theta_K", params::CP_THETA_FE),
        ("cP_Fe_linear_kB_per_K", params::CP_LINEAR_FE),
        ("cP_Cr_theta_K", params::CP_THETA_CR),
        ("cP_Cr_linear_kB_per_K", params::CP_LINEAR_CR),
        ("dHf_V_Fe_eV", params::DHF_V_FE),
        ("dHf_V_Cr_eV", params::DHF_V_CR),
        ("dSf_V_Fe_kB", params::DSF_V_FE),
        ("dSf_V_Cr_kB", params::DSF_V_CR),
        ("Theta_K", params::THETA_K),
        ("dH0_Fe_eV", params::DH0_FE),
        ("dH2_Fe_eV", params::DH2_FE),
        ("dH0_Cr_eV", params::DH0_CR),
        ("dH2_Cr_eV", params::DH2_CR),
        ("dSmig_kB", params::DS_MIG),
        ("nu_Fe_s-1", params::NU_FE),
        ("nu_Cr_s-1", params::NU_CR),
    ];
    for (key, value) in table {
        assert_eq!(json_number(&json, key), value, "{key}");
    }
    for (key, value) in [
        ("alpha", params::OMEGA_ALPHA),
        ("beta", params::OMEGA_BETA),
        ("gamma", params::OMEGA_GAMMA),
        ("delta", params::OMEGA_DELTA),
    ] {
        assert_eq!(json_number(omega, key), value, "{key}");
    }
}

#[test]
fn vacancy_formation() {
    let lattice = BccLattice::new([6, 6, 6]).unwrap();
    let n = lattice.site_count() as f64;
    for temperature in [0.0, 800.0] {
        let energetics = Martinez2012::new(temperature).unwrap();
        for (species, dhf, dsf) in [
            (Species::Fe, params::DHF_V_FE, params::DSF_V_FE),
            (Species::Cr, params::DHF_V_CR, params::DSF_V_CR),
        ] {
            let pure = vec![species; lattice.site_count()];
            let mut with_vacancy = pure.clone();
            with_vacancy[0] = Species::Vacancy;
            let e_pure = energetics.total_energy(&lattice, &pure);
            let e_vac = energetics.total_energy(&lattice, &with_vacancy);
            let formation = e_vac - e_pure * (n - 1.0) / n;
            let expected = dhf - temperature * BOLTZMANN_EV_PER_K * dsf;
            assert!((formation - expected).abs() < 1e-9);
        }
    }
}

#[test]
fn table2_barriers() {
    let lattice = BccLattice::new([6, 6, 6]).unwrap();
    let energetics = Martinez2012::new(0.0).unwrap();
    let site = |p: [i64; 3]| lattice.site_at(p).unwrap();
    let (mut h0, mut h2) = ([f64::NAN; 2], [f64::NAN; 2]);
    let mut lines = 0;
    for line in data("martinez2012_table2.txt").lines() {
        if !line.starts_with("table2") {
            continue;
        }
        let t: Vec<&str> = line.split_whitespace().collect();
        let (matrix, solute, other) = match t[1] {
            "Fe" => (0, Species::Cr, Species::Fe),
            _ => (1, Species::Fe, Species::Cr),
        };
        let name = t[2];
        let jumper = triple(&t[4..7]);
        let vacancy = triple(&t[8..11]);
        let value: f64 = *t.last().map(|v| v.parse().unwrap()).as_ref().unwrap();
        let mut occ = vec![other; lattice.site_count()];
        if t[12] == "4" {
            occ[site([4, 4, 4])] = solute;
        }
        occ[site(vacancy)] = Species::Vacancy;
        let rust = energetics.barrier(&lattice, &occ, site(vacancy), site(jumper));
        assert!((rust - value).abs() < 1e-6, "{} {name}", t[1]);
        if name == "H0" {
            h0[matrix] = rust;
        }
        if name == "H2" {
            h2[matrix] = rust;
        }
        lines += 1;
    }
    assert_eq!(lines, 20);
    for (m, (dh0, dh2)) in [
        (params::DH0_FE, params::DH2_FE),
        (params::DH0_CR, params::DH2_CR),
    ]
    .into_iter()
    .enumerate()
    {
        assert!((h0[m] - dh0).abs() < 1e-9, "H0 matrix {m}");
        assert!((h2[m] - dh2).abs() < 1e-9, "H2 matrix {m}");
    }
}

#[test]
fn saddle_fit() {
    for (file, tag, temperature, tolerance) in [
        ("martinez2012_0k.txt", "ht", 0.0, 1e-9),
        ("martinez2012_t800.txt", "gt", 800.0, 1e-8),
    ] {
        let (_, expected) = parse_blocks(&data(file), tag);
        let fitted = Martinez2012::new(temperature)
            .unwrap()
            .saddle_interactions();
        for a in 0..2 {
            for b in 0..2 {
                assert!((fitted[a][b] - expected[a][b]).abs() < tolerance, "{a}{b}");
            }
        }
    }
}

fn check_configs(file: &str, temperature: f64, energy_tol: f64, barrier_tol: f64) {
    let lattice = BccLattice::new([6, 6, 6]).unwrap();
    let energetics = Martinez2012::new(temperature).unwrap();
    let (blocks, _) = parse_blocks(&data(file), "ht");
    let configs: Vec<&Block> = blocks
        .iter()
        .filter(|b| b.header.starts_with("config"))
        .collect();
    assert!(!configs.is_empty());
    for block in configs {
        let occ = occupancy_of(&lattice, block);
        let vacancy = lattice.site_at(block.vacancy.unwrap()).unwrap();
        let energy = energetics.total_energy(&lattice, &occ);
        let expected = block.energy.unwrap();
        assert!(
            (energy - expected).abs() < energy_tol,
            "{}: energy {energy} vs {expected}",
            block.header
        );
        assert_eq!(block.barriers.len(), 8);
        for &(jumper, value) in &block.barriers {
            let b = energetics.barrier(&lattice, &occ, vacancy, lattice.site_at(jumper).unwrap());
            assert!(
                (b - value).abs() < barrier_tol,
                "{} jumper {jumper:?}: {b} vs {value}",
                block.header
            );
        }
    }
}

#[test]
fn oracle_0k() {
    check_configs("martinez2012_0k.txt", 0.0, 1e-8, 1e-9);

    let lattice = BccLattice::new([6, 6, 6]).unwrap();
    let energetics = Martinez2012::new(0.0).unwrap();
    let n = lattice.site_count() as f64;
    let e_fe = energetics.total_energy(&lattice, &vec![Species::Fe; lattice.site_count()]);
    let e_cr = energetics.total_energy(&lattice, &vec![Species::Cr; lattice.site_count()]);
    let (blocks, _) = parse_blocks(&data("martinez2012_0k.txt"), "ht");
    let mixes: Vec<&Block> = blocks
        .iter()
        .filter(|b| b.header.starts_with("mix"))
        .collect();
    assert_eq!(mixes.len(), 4);
    for block in mixes {
        let occ = occupancy_of(&lattice, block);
        let x = block.cr.len() as f64 / n;
        let hmix =
            energetics.total_energy(&lattice, &occ) / n - (1.0 - x) * e_fe / n - x * e_cr / n;
        let expected = block.hmix.unwrap();
        assert!((hmix - expected).abs() < 1e-9, "{}", block.header);
    }
}

#[test]
fn oracle_t800() {
    check_configs("martinez2012_t800.txt", 800.0, 1e-8, 1e-8);
}

fn energy_with(
    energetics: &Martinez2012,
    lattice: &BccLattice,
    occ: &[Species],
    swaps: &[(usize, usize)],
) -> f64 {
    let mut after = occ.to_vec();
    for &(i, j) in swaps {
        after.swap(i, j);
    }
    energetics.total_energy(lattice, &after)
}

fn local_deltas(cells: [usize; 3], fractions: &[f64], seeds: u64, random_pairs: usize) {
    let model = model(cells, 800.0);
    let (lattice, energetics) = (model.lattice(), model.energetics());
    let n = lattice.site_count();
    for &x in fractions {
        let cr_count = (x * (n - 1) as f64).round() as usize;
        for seed in 0..seeds {
            let state = model.random_state(cr_count, 7000 + seed).unwrap();
            let occ = state.occupancy();
            let vacancy = state.vacancy();
            let before = energetics.total_energy(lattice, occ);
            for &a in lattice.neighbors(vacancy) {
                let reference = energy_with(energetics, lattice, occ, &[(vacancy, a)]) - before;
                let local = energetics.exchange_delta(lattice, occ, vacancy, a);
                assert!(
                    (local - reference).abs() <= 1e-9,
                    "exchange {cells:?} x={x} seed={seed}: {local} vs {reference}"
                );
            }

            let mut pairs: Vec<(usize, usize)> = Vec::new();
            let fe: Vec<usize> = (0..n).filter(|&s| occ[s] == Species::Fe).collect();
            let cr: Vec<usize> = (0..n).filter(|&s| occ[s] == Species::Cr).collect();
            let mut rng = Xorshift(0x9E37_79B9 + seed * 7919 + n as u64);
            for _ in 0..random_pairs {
                pairs.push((fe[rng.below(fe.len())], cr[rng.below(cr.len())]));
            }
            let mut seen = HashSet::new();
            for &a in lattice.neighbors(vacancy) {
                for &b in lattice.neighbors(a) {
                    if b != vacancy
                        && occ[a] != Species::Vacancy
                        && occ[b] != occ[a]
                        && seen.insert((a.min(b), a.max(b)))
                    {
                        pairs.push((a, b));
                    }
                }
            }
            for (i, j) in pairs {
                let reference = energy_with(energetics, lattice, occ, &[(i, j)]) - before;
                let local = energetics.swap_delta(lattice, occ, i, j);
                assert!(
                    (local - reference).abs() <= 1e-9,
                    "swap {cells:?} x={x} seed={seed} {i},{j}: {local} vs {reference}"
                );
            }
        }
    }
}

#[test]
#[ignore = "full tier: equivalence"]
fn local_deltas_match_total() {
    for cells in [[3, 3, 3], [4, 4, 4]] {
        local_deltas(cells, &[0.1, 0.5, 0.9], 3, 200);
    }
}

#[test]
fn local_deltas_match_total_smoke() {
    local_deltas([3, 3, 3], &[0.5], 1, 20);
}

fn check_detailed_balance_with(
    model: &FeCrModel<Martinez2012>,
    fractions: &[f64],
    seeds: u64,
) -> u64 {
    let mut floored = 0;
    let (lattice, energetics) = (model.lattice(), model.energetics());
    let n = lattice.site_count();
    let kt = model.kt();
    for &x in fractions {
        let cr_count = (x * (n - 1) as f64).round() as usize;
        for seed in 0..seeds {
            let state = model.random_state(cr_count, 9000 + seed).unwrap();
            let e_before = energetics.total_energy(lattice, state.occupancy());
            for direction in 0..8 {
                let kf = model.rate(&state, direction);
                let atom = lattice.neighbor(state.vacancy(), direction);
                let raw = energetics.raw_barrier(lattice, state.occupancy(), state.vacancy(), atom);
                let de =
                    energetics.exchange_delta(lattice, state.occupancy(), state.vacancy(), atom);
                if raw < de.max(0.0) {
                    floored += 1;
                }
                let mut next = state.clone();
                model
                    .apply(&mut next, &okrates_models::fecr::FeCrJump { direction })
                    .unwrap();
                let e_after = energetics.total_energy(lattice, next.occupancy());
                let back = (0..8)
                    .find(|&d| lattice.neighbor(next.vacancy(), d) == state.vacancy())
                    .unwrap();
                let kr = model.rate(&next, back);
                let check = kf / kr * ((e_after - e_before) / kt).exp();
                assert!(
                    (check - 1.0).abs() <= 1e-9,
                    "x={x} seed={seed} dir={direction}: {check}"
                );
            }
        }
    }
    floored
}

#[test]
fn detailed_balance() {
    check_detailed_balance_with(&model([4, 4, 4], 800.0), &[0.1, 0.5, 0.9], 5);
}

#[test]
fn floor_preserves_detailed_balance() {
    let shift = -0.12;
    let energetics = Martinez2012::new(800.0).unwrap().with_saddle_shift(shift);
    let model = FeCrModel::new(BccLattice::new([4, 4, 4]).unwrap(), energetics, NU, 800.0).unwrap();
    let floored = check_detailed_balance_with(&model, &[0.1, 0.5, 0.9], 5);
    let counted = model.energetics().floor_activations();
    assert!(floored >= 1, "floor never active at shift {shift}");
    assert!(counted > 0);
}

#[test]
fn floor_policies_behave_without_panicking() {
    let shift = -0.15;
    let lattice = BccLattice::new([4, 4, 4]).unwrap();

    // Unfloored returns raw barrier directly even below max(0, dE).
    let unfloored = Martinez2012::with_policy(800.0, NU, NU, BarrierFloorPolicy::Unfloored)
        .unwrap()
        .with_saddle_shift(shift);
    let m_unfloored = FeCrModel::new(lattice.clone(), unfloored, NU, 800.0).unwrap();
    let state = m_unfloored.random_state(63, 9000).unwrap();
    let mut saw_below_floor = false;
    for d in 0..8 {
        let atom = lattice.neighbor(state.vacancy(), d);
        let raw = m_unfloored.energetics().raw_barrier(
            &lattice,
            state.occupancy(),
            state.vacancy(),
            atom,
        );
        let b = m_unfloored.barrier(&state, d);
        let de = m_unfloored.energetics().exchange_delta(
            &lattice,
            state.occupancy(),
            state.vacancy(),
            atom,
        );
        assert_eq!(b, raw);
        if raw < de.max(0.0) {
            saw_below_floor = true;
        }
    }
    assert!(saw_below_floor);
    assert!(m_unfloored.energetics().floor_activations() >= 1);

    // Strict rejects floored jumps via SimError::InvalidRate without panicking or mutating state.
    let strict = Martinez2012::with_policy(800.0, NU, NU, BarrierFloorPolicy::Strict)
        .unwrap()
        .with_saddle_shift(shift);
    let probe = strict.clone();
    let m_strict = FeCrModel::new(lattice.clone(), strict, NU, 800.0).unwrap();
    let initial = state.clone();
    let mut sim = Simulation::new(m_strict, state, 1);
    assert!(matches!(sim.step(), Err(SimError::InvalidRate { .. })));
    assert_eq!(sim.state(), &initial);
    assert!(probe.floor_activations() >= 1);

    // Invalid multi-vacancy input to raw_barrier returns NaN rather than panicking.
    let mut two_vac = initial.occupancy().to_vec();
    let nb = lattice.neighbors(initial.vacancy())[0];
    two_vac[nb] = Species::Vacancy;
    assert!(
        probe
            .raw_barrier(&lattice, &two_vac, initial.vacancy(), nb)
            .is_nan()
    );
}

#[test]
fn floor_inactive_on_oracles() {
    let lattice = BccLattice::new([6, 6, 6]).unwrap();
    let site = |p: [i64; 3]| lattice.site_at(p).unwrap();
    let mut checked = 0;
    for temperature in [0.0, 800.0] {
        let energetics = Martinez2012::new(temperature).unwrap();
        energetics.reset_floor_activations();
        let mut cases: Vec<(Vec<Species>, usize, usize)> = Vec::new();
        if temperature == 0.0 {
            for line in data("martinez2012_table2.txt").lines() {
                if !line.starts_with("table2") {
                    continue;
                }
                let t: Vec<&str> = line.split_whitespace().collect();
                let (solute, other) = match t[1] {
                    "Fe" => (Species::Cr, Species::Fe),
                    _ => (Species::Fe, Species::Cr),
                };
                let vacancy = site(triple(&t[8..11]));
                let mut occ = vec![other; lattice.site_count()];
                if t[12] == "4" {
                    occ[site([4, 4, 4])] = solute;
                }
                occ[vacancy] = Species::Vacancy;
                cases.push((occ, vacancy, site(triple(&t[4..7]))));
            }
            cases.extend(config_cases(&lattice, "martinez2012_0k.txt"));
        } else {
            cases.extend(config_cases(&lattice, "martinez2012_t800.txt"));
        }
        for (occ, vacancy, jumper) in &cases {
            let b = energetics.barrier(&lattice, occ, *vacancy, *jumper);
            let raw = energetics.raw_barrier(&lattice, occ, *vacancy, *jumper);
            assert_eq!(b, raw, "T={temperature} vacancy {vacancy} jumper {jumper}");
            checked += 1;
        }
        assert_eq!(energetics.floor_activations(), 0, "T={temperature}");
    }
    assert!(checked > 20);
}

fn config_cases(lattice: &BccLattice, file: &str) -> Vec<(Vec<Species>, usize, usize)> {
    let (blocks, _) = parse_blocks(&data(file), "ht");
    let mut cases = Vec::new();
    for block in blocks.iter().filter(|b| b.header.starts_with("config")) {
        let occ = occupancy_of(lattice, block);
        let vacancy = lattice.site_at(block.vacancy.unwrap()).unwrap();
        for &(jumper, _) in &block.barriers {
            cases.push((occ.clone(), vacancy, lattice.site_at(jumper).unwrap()));
        }
    }
    cases
}

#[test]
fn absolute_rates() {
    let temperature = 800.0;
    let kt = BOLTZMANN_EV_PER_K * temperature;
    let ts = temperature * BOLTZMANN_EV_PER_K * params::DS_MIG;
    let rel = |a: f64, b: f64| ((a - b) / b).abs();
    let lattice = BccLattice::new([6, 6, 6]).unwrap();
    let vacancy = lattice.site_at([5, 5, 5]).unwrap();
    let solute = lattice.site_at([4, 4, 4]).unwrap();

    let build = |energetics: Martinez2012| {
        FeCrModel::new(lattice.clone(), energetics, NU, temperature).unwrap()
    };
    let pure = |species: Species| {
        let mut occ = vec![species; lattice.site_count()];
        occ[vacancy] = Species::Vacancy;
        occ
    };

    let m = build(Martinez2012::new(temperature).unwrap());
    for (species, dh0) in [(Species::Fe, params::DH0_FE), (Species::Cr, params::DH0_CR)] {
        let state = m.state(pure(species)).unwrap();
        let expected = 1e13 * (-(dh0 - ts) / kt).exp();
        for d in 0..8 {
            let rate = m.rate(&state, d);
            assert!(rel(rate, expected) <= 1e-12, "{species:?} dir {d}");
        }
    }

    let m = build(Martinez2012::with_attempt_frequencies(temperature, 2e13, 3e13).unwrap());
    let state = m.state(pure(Species::Fe)).unwrap();
    let expected = 2e13 * (-(params::DH0_FE - ts) / kt).exp();
    for d in 0..8 {
        assert!(rel(m.rate(&state, d), expected) <= 1e-12, "Fe dir {d}");
    }
    let mut occ = pure(Species::Fe);
    occ[solute] = Species::Cr;
    let state = m.state(occ).unwrap();
    let d = (0..8)
        .find(|&d| lattice.neighbor(vacancy, d) == solute)
        .unwrap();
    let expected = 3e13 * (-(params::DH2_FE - ts) / kt).exp();
    let rate = m.rate(&state, d);
    assert!(rel(rate, expected) <= 1e-12);
}

fn barriers_positive(fractions: &[f64], seeds: u64) {
    let n = 2 * 6 * 6 * 6;
    for temperature in [0.0, 800.0] {
        let energetics = Martinez2012::new(temperature).unwrap();
        let lattice = BccLattice::new([6, 6, 6]).unwrap();
        let helper = model([6, 6, 6], 800.0);
        for &x in fractions {
            let cr_count = (x * (n - 1) as f64).round() as usize;
            for seed in 0..seeds {
                let state = helper.random_state(cr_count, 11_000 + seed).unwrap();
                let occ = state.occupancy();
                let v = state.vacancy();
                for &a in lattice.neighbors(v) {
                    let b = energetics.barrier(&lattice, occ, v, a);
                    let delta = energetics.exchange_delta(&lattice, occ, v, a);
                    assert!(b > 0.0, "T={temperature} x={x} seed={seed}: barrier {b}");
                    assert!(
                        b > delta,
                        "T={temperature} x={x} seed={seed}: barrier {b} <= delta {delta}"
                    );
                }
            }
        }
    }
}

const T10_FRACTIONS: [f64; 7] = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95];

#[test]
#[ignore = "full tier: statistical"]
fn barriers_positive_full() {
    barriers_positive(&T10_FRACTIONS, 20);
}

#[test]
fn barriers_positive_smoke() {
    barriers_positive(&[0.1, 0.5, 0.9], 1);
}

#[test]
fn validate_rejects() {
    let small = FeCrModel::new(
        BccLattice::new([2, 2, 2]).unwrap(),
        Martinez2012::new(800.0).unwrap(),
        NU,
        800.0,
    );
    assert_eq!(
        small.unwrap_err(),
        FeCrError::LatticeTooSmall { min_cells: 3 }
    );

    let mismatch = FeCrModel::new(
        BccLattice::new([3, 3, 3]).unwrap(),
        Martinez2012::new(800.0).unwrap(),
        NU,
        900.0,
    );
    assert_eq!(
        mismatch.unwrap_err(),
        FeCrError::TemperatureMismatch {
            model: 900.0,
            energetics: 800.0
        }
    );

    assert_eq!(
        Martinez2012::new(-1.0).err(),
        Some(FeCrError::Temperature(-1.0))
    );
    assert!(matches!(
        Martinez2012::new(f64::NAN),
        Err(FeCrError::Temperature(t)) if t.is_nan()
    ));
    assert_eq!(
        Martinez2012::with_attempt_frequencies(800.0, 0.0, 1e13).err(),
        Some(FeCrError::AttemptFrequency(0.0))
    );
    assert!(Martinez2012::new(0.0).is_ok());
}
