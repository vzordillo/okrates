//! Integration tests for the Fe–Cr RIS model: energetics parity, detailed balance,
//! jump and creation geometry, recombination, sink absorption, catalog equivalence,
//! checkpoints, and the dumbbell saddle refit at T (T16).

use std::collections::HashMap;
use std::fs;

use okrates_core::{Checkpointable, IncrementalModel, IncrementalSimulation, Model, Step};
use okrates_models::bcc::{BccLattice, DISPLACEMENTS};
use okrates_models::fecr::{Energetics, Species};
use okrates_models::fecr_martinez::Martinez2012;
use okrates_models::fecr_ris::{
    AXES, Cell, Defect, DumbbellParams, FeCrRis, FeCrRisError, FeCrRisEvent, FeCrRisParams,
    FeCrRisState, RC_OFFSETS, Record,
};
use okrates_models::vacancy::BOLTZMANN_EV_PER_K;

fn model(cells: [usize; 3], temperature_k: f64, capacity: usize) -> FeCrRis {
    FeCrRis::new(FeCrRisParams {
        cells,
        sink_planes: vec![],
        temperature_k,
        k0_dpa_per_s: 0.0,
        capacity,
        dumbbell: DumbbellParams::senninger2016_r1(),
    })
    .unwrap()
}

/// Deterministic xorshift generator for test draws.
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

fn species(name: &str) -> Species {
    match name {
        "Fe" => Species::Fe,
        "Cr" => Species::Cr,
        other => panic!("unknown species {other}"),
    }
}

fn dot(a: [i64; 3], b: [i64; 3]) -> i64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// The Fe/Cr occupancy of a state; vacancy and dumbbell sites hold an Fe placeholder.
fn occupancy(s: &FeCrRisState) -> Vec<Species> {
    s.cells()
        .iter()
        .map(|c| match c {
            Cell::Cr => Species::Cr,
            _ => Species::Fe,
        })
        .collect()
}

fn defect_of(r: Record) -> Option<Defect> {
    match r {
        Record::Empty => None,
        Record::Vacancy { site } => Some(Defect::Vacancy { site }),
        Record::Dumbbell {
            site,
            axis,
            plus,
            minus,
        } => Some(Defect::Dumbbell {
            site,
            axis,
            plus,
            minus,
        }),
    }
}

fn defects_of(s: &FeCrRisState) -> Vec<Defect> {
    s.records().iter().filter_map(|&r| defect_of(r)).collect()
}

fn site_of(d: Defect) -> usize {
    match d {
        Defect::Vacancy { site } | Defect::Dumbbell { site, .. } => site,
    }
}

fn data(name: &str) -> String {
    let path = format!("{}/tests/data/{name}", env!("CARGO_MANIFEST_DIR"));
    fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path}: {e}"))
}

struct R1 {
    model: FeCrRis,
}

impl R1 {
    fn new() -> Self {
        Self {
            model: model([6, 6, 6], 0.0, 1),
        }
    }

    /// Pure `matrix` with solutes at `solutes` and an optional dumbbell at [4,4,4], axis 0.
    fn state(
        &self,
        matrix: Species,
        ends: Option<(Species, Species)>,
        solutes: &[[i64; 3]],
    ) -> FeCrRisState {
        let lattice = self.model.lattice();
        let solute = if matrix == Species::Fe {
            Species::Cr
        } else {
            Species::Fe
        };
        let mut occ = vec![matrix; lattice.site_count()];
        for p in solutes {
            occ[lattice.site_at(*p).unwrap()] = solute;
        }
        let defects: Vec<Defect> = ends
            .map(|(minus, plus)| Defect::Dumbbell {
                site: lattice.site_at([4, 4, 4]).unwrap(),
                axis: 0,
                plus,
                minus,
            })
            .into_iter()
            .collect();
        self.model.state_with_defects(occ, &defects, 1).unwrap()
    }

    fn energy(
        &self,
        matrix: Species,
        ends: Option<(Species, Species)>,
        solutes: &[[i64; 3]],
    ) -> f64 {
        self.model.total_energy(&self.state(matrix, ends, solutes))
    }
}

/// Parses `<name> <matrix> <minus> <plus> <n> <offsets...> <value>` after the first token.
fn config(tokens: &[&str]) -> (Species, Species, Species, Vec<[i64; 3]>, f64) {
    let n: usize = tokens[3].parse().unwrap();
    let offsets = (0..n)
        .map(|i| {
            let t = &tokens[4 + 3 * i..7 + 3 * i];
            [
                t[0].parse().unwrap(),
                t[1].parse().unwrap(),
                t[2].parse().unwrap(),
            ]
        })
        .collect();
    let value = tokens[4 + 3 * n].parse().unwrap();
    assert_eq!(tokens.len(), 5 + 3 * n);
    (
        species(tokens[0]),
        species(tokens[1]),
        species(tokens[2]),
        offsets,
        value,
    )
}

#[test]
fn t1_r1_parity() {
    let r1 = R1::new();
    let text = data("fecr_ris_r1.txt");
    let p = DumbbellParams::senninger2016_r1();

    let expected: HashMap<&str, f64> = HashMap::from([
        ("g_FF", p.g[0]),
        ("g_FC", p.g[1]),
        ("g_CC", p.g[2]),
        ("e_FF_C_Fe", p.e[0][0][0]),
        ("e_FF_C_Cr", p.e[0][0][1]),
        ("e_FF_T_Fe", p.e[0][1][0]),
        ("e_FF_T_Cr", p.e[0][1][1]),
        ("e_CC_C_Fe", p.e[2][0][0]),
        ("e_CC_C_Cr", p.e[2][0][1]),
        ("e_CC_T_Fe", p.e[2][1][0]),
        ("e_CC_T_Cr", p.e[2][1][1]),
        ("e_FC_CFe_Fe", p.e[1][0][0]),
        ("e_FC_CFe_Cr", p.e[1][0][1]),
        ("e_FC_CCr_Fe", p.e[1][2][0]),
        ("e_FC_CCr_Cr", p.e[1][2][1]),
        ("e_FC_T_Fe", p.e[1][1][0]),
        ("e_FC_T_Cr", p.e[1][1][1]),
        ("FeFe", p.saddle[0][0]),
        ("FeCr", p.saddle[0][1]),
        ("CrFe", p.saddle[1][0]),
        ("CrCr", p.saddle[1][1]),
    ]);
    let mut seen = 0;
    let (mut bindings, mut hfs, mut barriers) = (0, 0, 0);
    let mut max_dev: f64 = 0.0;

    let energies = |matrix: Species| {
        (
            r1.energy(matrix, Some((matrix, matrix)), &[]),
            r1.energy(matrix, None, &[[1, 1, 1]]),
            r1.energy(matrix, None, &[]),
        )
    };
    let fe = energies(Species::Fe);
    let cr = energies(Species::Cr);

    for line in text.lines().filter(|l| !l.starts_with('#')) {
        let tokens: Vec<&str> = line.split_whitespace().collect();
        match tokens[0] {
            "param" | "saddle" => {
                let value: f64 = tokens[2].parse().unwrap();
                assert_eq!(value, expected[tokens[1]], "{line}");
                seen += 1;
            }
            "binding" => {
                let (matrix, minus, plus, offsets, value) = config(&tokens[2..]);
                let (e_d, e_s, e_r) = if matrix == Species::Fe { fe } else { cr };
                let solutes: Vec<[i64; 3]> = offsets
                    .iter()
                    .map(|o| [4 + o[0], 4 + o[1], 4 + o[2]])
                    .collect();
                let e_cfg = r1.energy(matrix, Some((minus, plus)), &solutes);
                let n = [minus, plus].iter().filter(|&&s| s != matrix).count() + offsets.len();
                let eb = e_d + n as f64 * (e_s - e_r) - e_cfg;
                max_dev = max_dev.max((eb - value).abs());
                assert!((eb - value).abs() <= 1e-9, "{line}: got {eb}");
                bindings += 1;
            }
            "hf" => {
                let matrix = species(tokens[1]);
                let value: f64 = tokens[2].parse().unwrap();
                let (e_d, _, e_r) = if matrix == Species::Fe { fe } else { cr };
                let hf = e_d - (433.0 / 432.0) * e_r;
                assert!((hf - value).abs() <= 1e-9, "{line}: got {hf}");
                hfs += 1;
            }
            "barrier" => {
                let (matrix, minus, plus, offsets, value) = config(&tokens[2..]);
                let solutes: Vec<[i64; 3]> = offsets
                    .iter()
                    .map(|o| [4 + o[0], 4 + o[1], 4 + o[2]])
                    .collect();
                let s = r1.state(matrix, Some((minus, plus)), &solutes);
                for jump in [6, 7] {
                    for b in [
                        r1.model.raw_barrier(&s, 0, jump),
                        r1.model.barrier(&s, 0, jump),
                    ] {
                        max_dev = max_dev.max((b - value).abs());
                        assert!((b - value).abs() <= 1e-9, "{line} jump {jump}: got {b}");
                    }
                }
                barriers += 1;
            }
            other => panic!("unknown line kind {other}"),
        }
    }
    assert_eq!((seen, bindings, hfs, barriers), (21, 14, 2, 25));
    println!("T1 max |deviation| = {max_dev:.3e} eV");
}

/// Runs the forward/reverse comparison for one jump; returns (k_f / k_r, exp(-ΔE/kT), ΔE).
fn ratio_check(m: &FeCrRis, s: &FeCrRisState, rec: usize, jump: usize) -> (f64, f64, f64) {
    let (new, left) = m.jump_result(s, rec, jump).expect("jump must be possible");
    let old = defect_of(s.records()[rec]).unwrap();
    let (old_site, new_site) = (site_of(old), site_of(new));
    let mut occ = occupancy(s);
    occ[old_site] = left;
    let species_on_new = match s.cells()[new_site] {
        Cell::Cr => Species::Cr,
        _ => Species::Fe,
    };
    let mut defects = defects_of(s);
    let position = defects.iter().position(|&d| d == old).unwrap();
    defects[position] = new;
    let fin = m.state_with_defects(occ, &defects, 7).unwrap();
    assert_eq!(fin.records()[rec], Record::from(new));

    let reverse: Vec<usize> = (0..8)
        .filter(|&j| m.jump_result(&fin, rec, j) == Some((old, species_on_new)))
        .collect();
    assert_eq!(reverse.len(), 1, "reverse of jump {jump}: {reverse:?}");
    let (kf, kr) = (m.rate(s, rec, jump), m.rate(&fin, rec, reverse[0]));
    assert!(kf > 0.0 && kr > 0.0, "rates {kf} {kr}");
    let de = m.energy_change(s, rec, jump);
    let de_rev = m.energy_change(&fin, rec, reverse[0]);
    assert!((de + de_rev).abs() < 1e-12, "dE {de} reverse {de_rev}");
    assert!(m.barrier(s, rec, jump).is_finite());
    let kt = BOLTZMANN_EV_PER_K * m.params().temperature_k;
    (kf / kr, (-de / kt).exp(), de)
}

fn assert_ratio(m: &FeCrRis, s: &FeCrRisState, rec: usize, jump: usize) -> f64 {
    let (ratio, expected, de) = ratio_check(m, s, rec, jump);
    let err = (ratio / expected - 1.0).abs();
    assert!(
        err < 1e-9,
        "jump {jump}: ratio {ratio} vs {expected}, dE {de}"
    );
    err
}

fn random_dumbbell(rng: &mut Xorshift, n: usize) -> Defect {
    let sp = |r: usize| if r == 0 { Species::Fe } else { Species::Cr };
    Defect::Dumbbell {
        site: rng.below(n),
        axis: rng.below(6),
        plus: sp(rng.below(2)),
        minus: sp(rng.below(2)),
    }
}

/// Random single-dumbbell trials; returns (max relative error, floored count, floored with ΔE > 0).
fn dumbbell_trials(m: &FeCrRis, trials: usize, seed: u64) -> (f64, usize, usize) {
    let base = m.random_state(102, 11, 3).unwrap();
    let occ = occupancy(&base);
    let mut rng = Xorshift(seed);
    let (mut max_err, mut floored, mut floored_up) = (0.0_f64, 0, 0);
    let shifted = m.clone().with_dumbbell_saddle_shift(0.1);
    for _ in 0..trials {
        let d = random_dumbbell(&mut rng, occ.len());
        let s = m.state_with_defects(occ.clone(), &[d], 5).unwrap();
        let jump = (0..40)
            .map(|_| rng.below(8))
            .find(|&j| m.rate(&s, 0, j) > 0.0)
            .expect("a jump with non-zero rate");
        let (raw, de, barrier) = (
            m.raw_barrier(&s, 0, jump),
            m.energy_change(&s, 0, jump),
            m.barrier(&s, 0, jump),
        );
        assert_eq!(barrier, raw.max(de.max(0.0)));
        // A lone dumbbell has 8 saddle terms: the 6 saddle sites plus R on O and A on k.
        let raw_shifted = shifted.raw_barrier(&s, 0, jump);
        assert!(
            (raw_shifted - raw - 0.8).abs() < 1e-12,
            "saddle shift moved the raw barrier by {}",
            raw_shifted - raw
        );
        if raw < de.max(0.0) {
            floored += 1;
            if de > 0.0 {
                floored_up += 1;
            }
        }
        max_err = max_err.max(assert_ratio(m, &s, 0, jump));
    }
    (max_err, floored, floored_up)
}

#[test]
fn t2_detailed_balance() {
    let m = model([8, 8, 8], 650.0, 1);
    let (max_err, floored, _) = dumbbell_trials(&m, 200, 0x1234_5678_9abc_def1);
    println!("T2 max relative error {max_err:.3e}, floored trials {floored}");
}

#[test]
fn t15a_floor_geometry() {
    let m = model([8, 8, 8], 650.0, 1).with_dumbbell_saddle_shift(-1.0);
    let (max_err, floored, floored_up) = dumbbell_trials(&m, 100, 0x0fed_cba9_8765_4321);
    println!("T15a max relative error {max_err:.3e}, floored {floored}, with dE > 0: {floored_up}");
    assert!(floored > 0, "no floored jump");
    assert!(floored_up > 0, "no floored jump with positive dE");
}

/// Raw barriers of the four defining pure-Fe jumps of r2-spec §2.5 (FeFe / Fe, mixed / Cr,
/// mixed / Fe, CrCr / Cr), from one dumbbell at site `site` with axis 3.
fn defining_barriers(m: &FeCrRis, site: usize) -> [f64; 4] {
    use Species::{Cr, Fe};
    let mut out = [f64::NAN; 4];
    for (i, (plus, minus, rest)) in [(Fe, Fe, Fe), (Cr, Fe, Fe), (Cr, Fe, Cr), (Cr, Cr, Cr)]
        .into_iter()
        .enumerate()
    {
        let d = Defect::Dumbbell {
            site,
            axis: 3,
            plus,
            minus,
        };
        let occ = vec![Fe; m.lattice().site_count()];
        let s = m.state_with_defects(occ, &[d], 0).unwrap();
        let jump = (0..8)
            .find(|&j| m.jump_result(&s, 0, j).is_some_and(|(_, r)| r == rest))
            .unwrap();
        out[i] = m.raw_barrier(&s, 0, jump);
    }
    out
}

#[test]
fn t16_defining_barriers_hold_at_t() {
    let site = 137;
    let cold = model([8, 8, 8], 0.0, 1);
    assert_eq!(cold.dumbbell_saddle(), cold.params().dumbbell.saddle);
    let target = defining_barriers(&cold, site);
    for (got, want) in target.iter().zip([0.340, 0.230, 0.3030, 0.050]) {
        assert!((got - want).abs() < 1e-4, "0 K barrier {got} vs {want}");
    }
    for t in [0.0, 300.0, 573.0, 773.0, 950.0] {
        let m = model([8, 8, 8], t, 1);
        let got = defining_barriers(&m, site);
        println!(
            "T16 {t} K: barriers {got:?}, saddle {:?}",
            m.dumbbell_saddle()
        );
        for (g, w) in got.iter().zip(target) {
            assert!((g - w).abs() <= 1e-12, "T = {t} K: barrier {g} vs 0 K {w}");
        }
    }
}

#[test]
fn t3_geometry() {
    let m = model([8, 8, 8], 650.0, 1);
    let lattice = m.lattice();
    let base = m.random_state(0, 1, 1).unwrap();
    let occ = occupancy(&base);
    let o = lattice.site_at([8, 8, 8]).unwrap();
    let period = 16;
    let wrap = |x: i64| {
        let r = x.rem_euclid(period);
        if r > period / 2 { r - period } else { r }
    };
    for (axis, &u) in AXES.iter().enumerate() {
        let original = Defect::Dumbbell {
            site: o,
            axis,
            plus: Species::Fe,
            minus: Species::Cr,
        };
        let s = m.state_with_defects(occ.clone(), &[original], 1).unwrap();
        let mut per_target: HashMap<usize, usize> = HashMap::new();
        for jump in 0..8 {
            let (new, rest) = m.jump_result(&s, 0, jump).unwrap();
            let Defect::Dumbbell {
                site: k,
                axis: b,
                plus,
                minus,
            } = new
            else {
                panic!("expected a dumbbell");
            };
            *per_target.entry(k).or_default() += 1;
            let (po, pk) = (lattice.position(o), lattice.position(k));
            let d: [i64; 3] = std::array::from_fn(|i| wrap(pk[i] - po[i]));
            let v = AXES[b];
            assert_eq!(dot(u, v).abs(), 1, "axis {axis} jump {jump}");
            assert_eq!(dot(d, v).abs(), 2, "axis {axis} jump {jump}");
            assert_eq!(dot(d, u).abs(), 2);
            // The jumper is the end facing k; the other original end stays on O.
            let (jumper, remaining) = if dot(d, u) > 0 {
                (Species::Fe, Species::Cr)
            } else {
                (Species::Cr, Species::Fe)
            };
            assert_eq!(rest, remaining);
            let (facing_o, facing_away) = if dot(d, v) < 0 {
                (plus, minus)
            } else {
                (minus, plus)
            };
            assert_eq!(facing_o, jumper, "axis {axis} jump {jump}");
            assert_eq!(facing_away, Species::Fe);

            let mut occ_final = occ.clone();
            occ_final[o] = rest;
            let fin = m.state_with_defects(occ_final, &[new], 1).unwrap();
            let back = (0..8)
                .filter(|&j| m.jump_result(&fin, 0, j) == Some((original, Species::Fe)))
                .count();
            assert_eq!(back, 1, "axis {axis} jump {jump}");
        }
        assert_eq!(per_target.len(), 4);
        assert!(per_target.values().all(|&c| c == 2));
    }
}

#[test]
fn t14b_vacancy_parity_with_m12() {
    let m = model([8, 8, 8], 650.0, 3);
    let m12 = Martinez2012::new(650.0).unwrap();
    let lattice: &BccLattice = m.lattice();
    let mut rng = Xorshift(0xdead_beef_cafe_f00d);
    let mut compared = 0;
    for seed in 0..5_u64 {
        let base = m.random_state(102, 100 + seed, 1).unwrap();
        let mut sites: Vec<usize> = Vec::new();
        while sites.len() < 3 {
            let c = rng.below(lattice.site_count());
            let far = sites.iter().all(|&o| {
                let (a, b) = (lattice.position(c), lattice.position(o));
                (0..3).any(|i| {
                    let d = (a[i] - b[i]).rem_euclid(16);
                    d.min(16 - d) >= 6
                })
            });
            if far {
                sites.push(c);
            }
        }
        let defects: Vec<Defect> = sites.iter().map(|&site| Defect::Vacancy { site }).collect();
        let s = m.state_with_defects(occupancy(&base), &defects, 1).unwrap();
        let occ: Vec<Species> = s
            .cells()
            .iter()
            .map(|c| match c {
                Cell::Fe => Species::Fe,
                Cell::Cr => Species::Cr,
                _ => Species::Vacancy,
            })
            .collect();
        for (rec, &q) in sites.iter().enumerate() {
            for dir in 0..8 {
                let p = lattice.neighbor(q, dir);
                if occ[p] == Species::Vacancy {
                    continue;
                }
                let checks = [
                    (
                        m.raw_barrier(&s, rec, dir),
                        m12.raw_barrier(lattice, &occ, q, p),
                    ),
                    (m.barrier(&s, rec, dir), m12.barrier(lattice, &occ, q, p)),
                    (
                        m.energy_change(&s, rec, dir),
                        m12.exchange_delta(lattice, &occ, q, p),
                    ),
                ];
                for (mine, theirs) in checks {
                    assert!((mine - theirs).abs() < 1e-12, "rec {rec} dir {dir}");
                }
                let nu = m12.attempt_frequency(occ[p]).unwrap();
                let kt = BOLTZMANN_EV_PER_K * 650.0;
                let rate = nu * (-m12.barrier(lattice, &occ, q, p) / kt).exp();
                assert!((m.rate(&s, rec, dir) / rate - 1.0).abs() < 1e-12);
                compared += 1;
            }
        }
    }
    assert_eq!(compared, 5 * 3 * 8);
}

fn saddle_sites(lattice: &BccLattice, a: usize, b: usize) -> Vec<usize> {
    let mut out: Vec<usize> = lattice
        .neighbors(a)
        .iter()
        .filter(|k| lattice.second_neighbors(b).contains(k))
        .chain(
            lattice
                .second_neighbors(a)
                .iter()
                .filter(|k| lattice.neighbors(b).contains(k)),
        )
        .copied()
        .collect();
    out.retain(|&k| k != a && k != b);
    out
}

#[test]
fn i3_two_vacancies_at_first_neighbour_distance() {
    let m = model([8, 8, 8], 650.0, 2);
    let lattice = m.lattice();
    let base = m.random_state(102, 21, 1).unwrap();
    let (q1, q2) = (
        lattice.site_at([8, 8, 8]).unwrap(),
        lattice.site_at([9, 9, 9]).unwrap(),
    );
    let s = m
        .state_with_defects(
            occupancy(&base),
            &[Defect::Vacancy { site: q1 }, Defect::Vacancy { site: q2 }],
            1,
        )
        .unwrap();
    let mut with_other_in_saddle = 0;
    for (rec, q) in [(0, q1), (1, q2)] {
        let other = if rec == 0 { q2 } else { q1 };
        for dir in 0..8 {
            let p = lattice.neighbor(q, dir);
            if p == other {
                assert_eq!(m.rate(&s, rec, dir), 0.0);
                continue;
            }
            assert!(m.barrier(&s, rec, dir).is_finite());
            assert!(m.rate(&s, rec, dir) > 0.0);
            if saddle_sites(lattice, q, p).contains(&other) {
                with_other_in_saddle += 1;
                assert_ratio(&m, &s, rec, dir);
            }
        }
    }
    assert!(with_other_in_saddle > 0);
}

#[test]
fn i3_dumbbell_inside_the_saddle_set_of_another() {
    let m = model([8, 8, 8], 650.0, 2);
    let lattice = m.lattice();
    let base = m.random_state(102, 22, 1).unwrap();
    let occ = occupancy(&base);
    let o = lattice.site_at([8, 8, 8]).unwrap();
    let first = Defect::Dumbbell {
        site: o,
        axis: 0,
        plus: Species::Fe,
        minus: Species::Cr,
    };
    let mut checked = 0;
    for jump in 0..8 {
        let probe = m.state_with_defects(occ.clone(), &[first], 1).unwrap();
        let (new, _) = m.jump_result(&probe, 0, jump).unwrap();
        let k = site_of(new);
        for (axis, plus, minus) in [(3, Species::Cr, Species::Fe), (4, Species::Fe, Species::Fe)] {
            for site in saddle_sites(lattice, o, k) {
                let second = Defect::Dumbbell {
                    site,
                    axis,
                    plus,
                    minus,
                };
                let s = m
                    .state_with_defects(occ.clone(), &[first, second], 1)
                    .unwrap();
                assert!(m.barrier(&s, 0, jump).is_finite());
                assert_ratio(&m, &s, 0, jump);
                checked += 1;
            }
        }
        // The second dumbbell on the target itself blocks both jumps to that target.
        let second = Defect::Dumbbell {
            site: k,
            axis: 5,
            plus: Species::Cr,
            minus: Species::Cr,
        };
        let s = m
            .state_with_defects(occ.clone(), &[first, second], 1)
            .unwrap();
        assert_eq!(m.rate(&s, 0, jump), 0.0);
        assert_eq!(m.jump_result(&s, 0, jump), None);
    }
    assert_eq!(checked, 8 * 2 * 6);
}

#[test]
fn state_validation() {
    let m = model([8, 8, 8], 650.0, 2);
    let lattice = m.lattice();
    let occ = occupancy(&m.random_state(10, 1, 1).unwrap());
    let at = |p: [i64; 3]| lattice.site_at(p).unwrap();
    let dumbbell = |site| Defect::Dumbbell {
        site,
        axis: 0,
        plus: Species::Fe,
        minus: Species::Fe,
    };
    let vacancy = |site| Defect::Vacancy { site };
    let origin = at([8, 8, 8]);

    // Every I–V pair at an Rc offset is rejected; the 5th shell (|o|² = 12) is accepted.
    assert_eq!(RC_OFFSETS.len(), 50);
    for o in RC_OFFSETS.iter() {
        let v = at([8 + o[0], 8 + o[1], 8 + o[2]]);
        let err = m
            .state_with_defects(occ.clone(), &[dumbbell(origin), vacancy(v)], 1)
            .unwrap_err();
        assert!(matches!(
            err,
            FeCrRisError::WithinRecombinationRadius { .. }
        ));
    }
    let far = at([10, 10, 10]);
    m.state_with_defects(occ.clone(), &[dumbbell(origin), vacancy(far)], 1)
        .unwrap();

    assert!(matches!(
        m.state_with_defects(occ.clone(), &[vacancy(origin), vacancy(origin)], 1),
        Err(FeCrRisError::DuplicateDefect { .. })
    ));
    assert!(matches!(
        m.state_with_defects(
            occ.clone(),
            &[vacancy(origin), vacancy(far), vacancy(at([0, 0, 0]))],
            1
        ),
        Err(FeCrRisError::TooManyDefects { .. })
    ));
    let mut with_vacancy = occ.clone();
    with_vacancy[0] = Species::Vacancy;
    assert!(matches!(
        m.state_with_defects(with_vacancy, &[], 1),
        Err(FeCrRisError::VacancyInOccupancy { site: 0 })
    ));
    let bad_axis = Defect::Dumbbell {
        site: origin,
        axis: 6,
        plus: Species::Fe,
        minus: Species::Fe,
    };
    assert!(matches!(
        m.state_with_defects(occ.clone(), &[bad_axis], 1),
        Err(FeCrRisError::InvalidDefect { .. })
    ));

    let sink = FeCrRis::new(FeCrRisParams {
        sink_planes: vec![0],
        ..m.params().clone()
    })
    .unwrap();
    assert!(matches!(
        sink.state_with_defects(occ.clone(), &[vacancy(at([0, 0, 0]))], 1),
        Err(FeCrRisError::DefectOnSink { .. })
    ));

    let s = m
        .state_with_defects(occ, &[dumbbell(origin), vacancy(far)], 1)
        .unwrap();
    m.check_invariants(&s).unwrap();
    assert_eq!(s.defect_counts(), (1, 1));
    assert_eq!(s.peak_defects(), 2);
}

#[test]
fn constructor_validation() {
    let ok = FeCrRisParams {
        cells: [6, 6, 6],
        sink_planes: vec![],
        temperature_k: 300.0,
        k0_dpa_per_s: 0.0,
        capacity: 1,
        dumbbell: DumbbellParams::senninger2016_r1(),
    };
    assert!(FeCrRis::new(ok.clone()).is_ok());
    let bad = |f: &dyn Fn(&mut FeCrRisParams)| {
        let mut p = ok.clone();
        f(&mut p);
        FeCrRis::new(p).is_err()
    };
    assert!(bad(&|p| p.cells = [5, 6, 6]));
    assert!(bad(&|p| p.temperature_k = -1.0));
    assert!(bad(&|p| p.temperature_k = f64::NAN));
    assert!(bad(&|p| p.k0_dpa_per_s = -1.0));
    assert!(bad(&|p| p.capacity = 0));
    assert!(bad(&|p| p.sink_planes = vec![12]));
    assert!(bad(&|p| p.dumbbell.e[0][2][0] = 0.1));
    assert!(bad(&|p| p.dumbbell.e[2][2][1] = 0.1));
}

fn ris_model(
    cells: [usize; 3],
    temperature_k: f64,
    k0: f64,
    capacity: usize,
    sinks: &[i64],
) -> FeCrRis {
    FeCrRis::new(FeCrRisParams {
        cells,
        sink_planes: sinks.to_vec(),
        temperature_k,
        k0_dpa_per_s: k0,
        capacity,
        dumbbell: DumbbellParams::senninger2016_r1(),
    })
    .unwrap()
}

/// Creation rate per site (s⁻¹) for the P = 64 runs and for the P = 16 smokes (T4, T5b, T7, T12).
const FULL_K0: f64 = 1.0e8;
const SMOKE_K0: f64 = 1.0e7;
/// The T4 full run: creation is a large share of the total rate.
const T4_K0: f64 = 1.0e9;

fn free_records(s: &FeCrRisState) -> usize {
    s.records()
        .iter()
        .filter(|r| matches!(r, Record::Empty))
        .count()
}

/// Picks an event with probability proportional to its rate; creation is skipped when the pool is short.
fn choose(m: &FeCrRis, s: &FeCrRisState, rng: &mut Xorshift) -> Option<FeCrRisEvent> {
    let mut events = Vec::new();
    m.enumerate_events(s, &mut |e, r| {
        assert!(r > 0.0 && r.is_finite());
        if !(e == FeCrRisEvent::Create && free_records(s) < 2) {
            events.push((e, r));
        }
    });
    let total: f64 = events.iter().map(|&(_, r)| r).sum();
    let mut target = (rng.next() >> 11) as f64 / (1_u64 << 53) as f64 * total;
    for &(e, r) in &events {
        if target < r {
            return Some(e);
        }
        target -= r;
    }
    events.last().map(|&(e, _)| e)
}

fn pos_site(m: &FeCrRis, p: [i64; 3]) -> usize {
    m.lattice().site_at(p).unwrap()
}

fn species_totals(s: &FeCrRisState) -> [i64; 2] {
    let c = s.atom_counts();
    [
        (c[0] + c[2] + c[4]) as i64 - s.borrowed_fe() as i64,
        (c[1] + c[3] + c[5]) as i64,
    ]
}

/// T5a: no creation; six defects, atom totals conserved and invariants held after every step.
fn t5a_run(steps: usize) {
    let m = ris_model([8, 8, 8], 650.0, 0.0, 6, &[0]);
    let base = m.random_state(102, 3, 1).unwrap();
    let p = |x, y, z| pos_site(&m, [x, y, z]);
    let mixed = |site, axis| Defect::Dumbbell {
        site,
        axis,
        plus: Species::Cr,
        minus: Species::Fe,
    };
    // Two vacancy-dumbbell pairs just outside the recombination radius (r² = 12) and one far pair.
    let defects = [
        Defect::Vacancy { site: p(4, 4, 4) },
        mixed(p(6, 6, 6), 0),
        Defect::Vacancy { site: p(12, 12, 4) },
        mixed(p(10, 10, 6), 1),
        Defect::Vacancy { site: p(4, 12, 12) },
        mixed(p(11, 5, 11), 2),
    ];
    let mut s = m.state_with_defects(occupancy(&base), &defects, 1).unwrap();
    let before = species_totals(&s);
    let mut rng = Xorshift(0x5eed_0001);
    for step in 0..steps {
        let Some(e) = choose(&m, &s, &mut rng) else {
            break;
        };
        m.apply(&mut s, &e).unwrap();
        assert_eq!(species_totals(&s), before, "step {step}, event {e:?}");
        m.check_invariants(&s).unwrap();
    }
    println!(
        "T5a recombined {} absorbed I {} V {} borrowed {}",
        s.recombined(),
        s.absorbed_i(),
        s.absorbed_v(),
        s.borrowed_fe()
    );
    assert!(s.recombined() > 0, "no recombination happened");
}

#[test]
fn t5a_atom_conservation_smoke() {
    t5a_run(250);
}

#[test]
#[ignore = "full tier: conservation"]
fn t5a_atom_conservation_without_creation() {
    t5a_run(2000);
}

/// T5b: creation on; created = recombined + absorbed + present, per kind, after every step.
fn t5b_run(capacity: usize, k0: f64, steps: usize, full: bool) {
    let m = ris_model([8, 8, 8], 650.0, k0, capacity, &[0]);
    let s = m.random_state(102, 3, 1).unwrap();
    let mut sim = IncrementalSimulation::new(m, s, 0x5eed_0002).unwrap();
    for step in 0..steps {
        let event = match sim.step().unwrap() {
            Step::Fired { event, .. } => event,
            Step::Absorbed => panic!("creation keeps the model live"),
        };
        let s = sim.state();
        let (v, i) = s.defect_counts();
        assert_eq!(
            s.created(),
            s.recombined() + s.absorbed_i() + i as u64,
            "interstitials, step {step}, event {event:?}"
        );
        assert_eq!(
            s.created(),
            s.recombined() + s.absorbed_v() + v as u64,
            "vacancies, step {step}, event {event:?}"
        );
        sim.model().check_invariants(s).unwrap();
    }
    let s = sim.state();
    println!(
        "T5b P={capacity} created {} recombined {} absorbed I {} V {} peak {} attempts {}",
        s.created(),
        s.recombined(),
        s.absorbed_i(),
        s.absorbed_v(),
        s.peak_defects(),
        s.creation_attempts()
    );
    assert!(s.created() > 0 && s.peak_defects() >= 2);
    if full {
        assert!(s.recombined() > 0, "no recombination");
        assert!(s.absorbed_i() + s.absorbed_v() > 0, "no absorption");
    }
}

#[test]
fn t5b_defect_bookkeeping_smoke() {
    t5b_run(16, SMOKE_K0, 100, false);
}

#[test]
#[ignore = "full tier: bookkeeping"]
fn t5b_defect_bookkeeping_with_creation() {
    t5b_run(64, FULL_K0, 2000, true);
}

#[test]
fn t6_pool_exhaustion() {
    let m = ris_model([8, 8, 8], 650.0, 1.0e5, 3, &[]);
    let mut s = m.random_state(0, 1, 1).unwrap();
    m.apply(&mut s, &FeCrRisEvent::Create).unwrap();
    assert_eq!(s.created(), 1);
    assert_eq!(free_records(&s), 1);
    assert_eq!(s.defect_counts(), (1, 1));
    let snapshot = (
        s.cells().to_vec(),
        s.records().to_vec(),
        s.created(),
        s.recombined(),
        s.creation_attempts(),
        s.creation_word_pos(),
        s.peak_defects(),
    );
    let err = m.apply(&mut s, &FeCrRisEvent::Create).unwrap_err();
    assert_eq!(
        err,
        FeCrRisError::PoolExhausted {
            capacity: 3,
            defects: 2
        }
    );
    let after = (
        s.cells().to_vec(),
        s.records().to_vec(),
        s.created(),
        s.recombined(),
        s.creation_attempts(),
        s.creation_word_pos(),
        s.peak_defects(),
    );
    assert_eq!(snapshot, after);
}

#[test]
fn t8_run_seed_changes_creation_site() {
    let m = ris_model([8, 8, 8], 650.0, 1.0e5, 4, &[]);
    let vacancy_site = |run_seed| {
        let mut s = m.random_state(0, 1, run_seed).unwrap();
        m.apply(&mut s, &FeCrRisEvent::Create).unwrap();
        match s.records()[0] {
            Record::Vacancy { site } => site,
            r => panic!("record 0 is {r:?}"),
        }
    };
    assert_ne!(vacancy_site(1), vacancy_site(2));
    assert_eq!(vacancy_site(1), vacancy_site(1));
}

/// The chain of a creation on `s` (vacancy record 0 or the first new vacancy `v_rec`, dumbbell `d_rec`),
/// re-derived from the result: (s_0..s_10, t).
fn derived_chain(m: &FeCrRis, v_site: usize, d_site: usize) -> (Vec<usize>, [i64; 3]) {
    let lattice = m.lattice();
    let p0 = lattice.position(v_site);
    let t = DISPLACEMENTS
        .iter()
        .copied()
        .find(|t| {
            lattice.site_at([p0[0] + 10 * t[0], p0[1] + 10 * t[1], p0[2] + 10 * t[2]])
                == Some(d_site)
        })
        .expect("the dumbbell is not 10 steps from the vacancy");
    let chain = (0..=10)
        .map(|i| pos_site(m, [p0[0] + i * t[0], p0[1] + i * t[1], p0[2] + i * t[2]]))
        .collect();
    (chain, t)
}

fn new_sites(s: &FeCrRisState) -> (usize, usize) {
    let mut v = None;
    let mut d = None;
    for r in s.records() {
        match *r {
            Record::Vacancy { site } => v = Some(site),
            Record::Dumbbell { site, .. } => d = Some(site),
            Record::Empty => {}
        }
    }
    (v.unwrap(), d.unwrap())
}

fn dumbbell_ends(s: &FeCrRisState) -> (Species, Species) {
    s.records()
        .iter()
        .find_map(|r| match *r {
            Record::Dumbbell { plus, minus, .. } => Some((plus, minus)),
            _ => None,
        })
        .unwrap()
}

#[test]
fn t9_creation_chain_geometry() {
    let m = ris_model([8, 8, 8], 650.0, 1.0e5, 4, &[]);
    let n = m.lattice().site_count();
    let run = |markers: &[usize]| {
        let mut occ = vec![Species::Fe; n];
        for &c in markers {
            occ[c] = Species::Cr;
        }
        let mut s = m.state_with_defects(occ, &[], 9).unwrap();
        m.apply(&mut s, &FeCrRisEvent::Create).unwrap();
        s
    };
    let probe = run(&[]);
    assert_eq!(probe.creation_attempts(), 1);
    let (v, d) = new_sites(&probe);
    let (chain, _) = derived_chain(&m, v, d);
    assert_eq!(chain.len(), 11);

    // Cr markers on s_5 and s_9: the chain shifts by one site along t.
    let s = run(&[chain[5], chain[9]]);
    assert_eq!(s.creation_attempts(), 1);
    assert_eq!((v, d), new_sites(&s));
    assert_eq!(s.cells()[chain[0]], Cell::Vacancy);
    assert_eq!(s.cells()[chain[10]], Cell::Dumbbell);
    for (i, &c) in chain.iter().enumerate().take(10).skip(1) {
        let expected = if i == 6 { Cell::Cr } else { Cell::Fe };
        assert_eq!(s.cells()[c], expected, "chain site {i}");
    }
    let (plus, minus) = dumbbell_ends(&s);
    assert_eq!(
        {
            let mut ends = [plus, minus];
            ends.sort_by_key(|sp| sp.index());
            ends
        },
        [Species::Fe, Species::Cr]
    );
    assert_eq!(species_totals(&s), [1022, 2]);
    m.check_invariants(&s).unwrap();

    // The end draw is deterministic: moving the marker from a_9 to a_10 swaps the ends.
    let a = dumbbell_ends(&run(&[chain[9]]));
    let b = dumbbell_ends(&run(&[chain[10]]));
    assert_eq!(a, (b.1, b.0));
    assert_ne!(a.0, a.1);

    // A vacancy on the chain rejects the first draw; the accepted chain avoids it.
    let blocker = Defect::Vacancy { site: chain[5] };
    let mut s = m
        .state_with_defects(vec![Species::Fe; n], &[blocker], 9)
        .unwrap();
    m.apply(&mut s, &FeCrRisEvent::Create).unwrap();
    assert!(
        s.creation_attempts() > 1,
        "attempts {}",
        s.creation_attempts()
    );
    assert_eq!(s.records()[0], Record::from(blocker));
    let (Record::Vacancy { site: v2 }, Record::Dumbbell { site: d2, .. }) =
        (s.records()[1], s.records()[2])
    else {
        panic!("new records are {:?}", &s.records()[1..3]);
    };
    let (chain2, _) = derived_chain(&m, v2, d2);
    assert!(!chain2.contains(&chain[5]));
    assert!(
        chain2
            .iter()
            .all(|&c| c == v2 || c == d2 || s.cells()[c] != Cell::Vacancy)
    );
    m.check_invariants(&s).unwrap();

    // A sink plane through s_3 rejects the first draw; the accepted chain stays off the plane.
    let plane = m.lattice().position(chain[3])[0];
    let ms = ris_model([8, 8, 8], 650.0, 1.0e5, 4, &[plane]);
    let mut s = ms.state_with_defects(vec![Species::Fe; n], &[], 9).unwrap();
    ms.apply(&mut s, &FeCrRisEvent::Create).unwrap();
    assert!(
        s.creation_attempts() > 1,
        "attempts {}",
        s.creation_attempts()
    );
    let (v3, d3) = new_sites(&s);
    let (chain3, _) = derived_chain(&ms, v3, d3);
    assert!(chain3.iter().all(|&c| ms.lattice().position(c)[0] != plane));
    ms.check_invariants(&s).unwrap();
}

/// Fe matrix with a mixed dumbbell (plus Cr, minus Fe, axis 0) at [8,8,8] and a vacancy at
/// `[8,8,8] + offset - d`; the vacancy then jumps by `d` to `[8,8,8] + offset`.
fn recombination_fixture(offset: [i64; 3], d: [i64; 3]) -> (FeCrRis, FeCrRisState, usize, usize) {
    let m = ris_model([8, 8, 8], 650.0, 0.0, 2, &[]);
    let i_site = pos_site(&m, [8, 8, 8]);
    let v_site = pos_site(
        &m,
        [
            8 + offset[0] - d[0],
            8 + offset[1] - d[1],
            8 + offset[2] - d[2],
        ],
    );
    let n = m.lattice().site_count();
    let s = m
        .state_with_defects(
            vec![Species::Fe; n],
            &[
                Defect::Dumbbell {
                    site: i_site,
                    axis: 0,
                    plus: Species::Cr,
                    minus: Species::Fe,
                },
                Defect::Vacancy { site: v_site },
            ],
            1,
        )
        .unwrap();
    (m, s, i_site, v_site)
}

fn fire_vacancy_jump(m: &FeCrRis, s: &mut FeCrRisState, d: [i64; 3]) {
    let dir = DISPLACEMENTS.iter().position(|&x| x == d).unwrap();
    m.apply(s, &FeCrRisEvent::VacancyJump { rec: 1, dir })
        .unwrap();
}

#[test]
fn t10_recombination_shells_and_fill_rule() {
    // r² = 11 (4th shell): recombines. u = (1,1,0), offset (3,1,1): u·δ > 0, so the plus atom (Cr)
    // fills the vacancy and the minus atom (Fe) stays.
    let d = [-1, -1, -1];
    let (m, mut s, i_site, _) = recombination_fixture([3, 1, 1], d);
    fire_vacancy_jump(&m, &mut s, d);
    assert_eq!(s.recombined(), 1);
    assert_eq!(free_records(&s), 2);
    let v_new = pos_site(&m, [11, 9, 9]);
    assert_eq!(s.cells()[v_new], Cell::Cr);
    assert_eq!(s.cells()[i_site], Cell::Fe);
    assert_eq!(species_totals(&s), [1023, 1]);
    m.check_invariants(&s).unwrap();

    // Offset (-3,-1,1): u·δ < 0, so the minus atom (Fe) fills the vacancy and the plus atom stays.
    let d = [1, 1, -1];
    let (m, mut s, i_site, _) = recombination_fixture([-3, -1, 1], d);
    fire_vacancy_jump(&m, &mut s, d);
    assert_eq!(s.recombined(), 1);
    assert_eq!(s.cells()[pos_site(&m, [5, 7, 9])], Cell::Fe);
    assert_eq!(s.cells()[i_site], Cell::Cr);
    assert_eq!(species_totals(&s), [1023, 1]);
    m.check_invariants(&s).unwrap();

    // r² = 12 (5th shell): no recombination.
    let d = [-1, -1, -1];
    let (m, mut s, _, _) = recombination_fixture([2, 2, 2], d);
    fire_vacancy_jump(&m, &mut s, d);
    assert_eq!(s.recombined(), 0);
    assert_eq!(free_records(&s), 0);
    m.check_invariants(&s).unwrap();
}

#[test]
fn t11_sink_absorption() {
    let m = ris_model([8, 8, 8], 650.0, 0.0, 6, &[0]);
    let n = m.lattice().site_count();
    let p = |x, y, z| pos_site(&m, [x, y, z]);
    let mut occ = vec![Species::Fe; n];
    occ[p(0, 0, 0)] = Species::Cr;
    let dumbbell = |site| Defect::Dumbbell {
        site,
        axis: 0,
        plus: Species::Cr,
        minus: Species::Fe,
    };
    let defects = [
        dumbbell(p(1, 1, 1)),
        dumbbell(p(1, 5, 5)),
        Defect::Vacancy { site: p(1, 9, 9) },
        Defect::Vacancy { site: p(1, 13, 13) },
        Defect::Vacancy { site: p(1, 3, 11) },
    ];
    let mut s = m.state_with_defects(occ, &defects, 1).unwrap();
    let totals = species_totals(&s);

    // Dumbbell 0 jumps from [1,1,1] to the sink site [0,0,0] (Cr). The jumper (Fe) stays there;
    // the sink atom goes to the reservoir; Cr remains on [1,1,1].
    let jump_to = |s: &FeCrRisState, rec: usize, k: usize| {
        (0..8)
            .find(|&j| matches!(m.jump_result(s, rec, j), Some((Defect::Dumbbell { site, .. }, _)) if site == k))
            .unwrap()
    };
    let j = jump_to(&s, 0, p(0, 0, 0));
    m.apply(&mut s, &FeCrRisEvent::DumbbellJump { rec: 0, jump: j })
        .unwrap();
    assert_eq!(s.absorbed_i(), 1);
    assert_eq!(s.records()[0], Record::Empty);
    assert_eq!(s.cells()[p(0, 0, 0)], Cell::Fe);
    assert_eq!(s.cells()[p(1, 1, 1)], Cell::Cr);
    assert_eq!(
        s.reservoir().iter().copied().collect::<Vec<_>>(),
        [Species::Cr]
    );
    assert_eq!(s.last_records(), [0]);
    assert!(s.last_changed().contains(&p(0, 0, 0)));

    // Dumbbell 1 queues an Fe behind the Cr.
    let j = jump_to(&s, 1, p(0, 4, 4));
    m.apply(&mut s, &FeCrRisEvent::DumbbellJump { rec: 1, jump: j })
        .unwrap();
    assert_eq!(s.absorbed_i(), 2);
    assert_eq!(
        s.reservoir().iter().copied().collect::<Vec<_>>(),
        [Species::Cr, Species::Fe]
    );
    assert_eq!(species_totals(&s), totals);

    // Vacancies reach the sink: filled first-in first-out, then with borrowed Fe.
    let expected = [
        (2, p(0, 8, 8), Cell::Cr, 0),
        (3, p(0, 12, 12), Cell::Fe, 0),
        (4, p(0, 2, 10), Cell::Fe, 1),
    ];
    for (rec, sink_site, fill, borrowed) in expected {
        let dir = (0..8)
            .find(|&dir| {
                m.lattice().neighbor(
                    match s.records()[rec] {
                        Record::Vacancy { site } => site,
                        r => panic!("{r:?}"),
                    },
                    dir,
                ) == sink_site
            })
            .unwrap();
        m.apply(&mut s, &FeCrRisEvent::VacancyJump { rec, dir })
            .unwrap();
        assert_eq!(s.records()[rec], Record::Empty);
        assert_eq!(s.cells()[sink_site], fill, "record {rec}");
        assert_eq!(s.borrowed_fe(), borrowed, "record {rec}");
        m.check_invariants(&s).unwrap();
    }
    assert_eq!(s.absorbed_v(), 3);
    assert!(s.reservoir().is_empty());
    assert_eq!(species_totals(&s), totals);
}

#[test]
fn t15b_floored_dumbbell_jumps_are_counted() {
    let m = ris_model([8, 8, 8], 650.0, 0.0, 1, &[]).with_dumbbell_saddle_shift(-1.0);
    let base = m.random_state(102, 5, 1).unwrap();
    let d = Defect::Dumbbell {
        site: 100,
        axis: 3,
        plus: Species::Fe,
        minus: Species::Cr,
    };
    let mut s = m.state_with_defects(occupancy(&base), &[d], 1).unwrap();
    let mut rng = Xorshift(0x5eed_0015);
    for _ in 0..20 {
        let e = choose(&m, &s, &mut rng).unwrap();
        assert!(matches!(e, FeCrRisEvent::DumbbellJump { .. }));
        m.apply(&mut s, &e).unwrap();
    }
    println!("T15b floored {}", s.floored_dumbbell_jumps());
    assert!(s.floored_dumbbell_jumps() > 0);
    assert_eq!(s.floored_vacancy_jumps(), 0);
}

/// Compares the tree's slot rates bitwise with a rebuild by `slot_rate`, and, if `enumerate`,
/// the nonzero slots with `enumerate_events`.
fn assert_catalog_matches_rebuild(
    sim: &IncrementalSimulation<FeCrRis>,
    enumerate: bool,
    context: &str,
) {
    let (m, s) = (sim.model(), sim.state());
    let count = m.slot_count(s);
    assert_eq!(sim.tree().len(), count);
    let mut nonzero = Vec::new();
    for slot in 0..count {
        let rebuilt = m.slot_rate(s, slot);
        assert_eq!(
            sim.tree().rate(slot).to_bits(),
            rebuilt.to_bits(),
            "{context}: slot {slot} of {count}"
        );
        if rebuilt > 0.0 {
            nonzero.push((m.slot_event(s, slot), rebuilt));
        }
    }
    if enumerate {
        let mut emitted = Vec::new();
        m.enumerate_events(s, &mut |e, r| emitted.push((e, r)));
        assert_eq!(emitted, nonzero, "{context}: enumerate_events");
    }
}

/// Runs `steps` steps from a random 10 % Cr state and checks the catalog after each step.
fn t4_run(capacity: usize, k0: f64, steps: usize, full: bool) {
    let m = ris_model([8, 8, 8], 650.0, k0, capacity, &[0]);
    let s = m.random_state(102, 3, 1).unwrap();
    let mut sim = IncrementalSimulation::new(m, s, 0x5eed_0004).unwrap();
    assert_catalog_matches_rebuild(&sim, true, "initial");
    let mut creations = 0_u64;
    let mut share_sum = 0.0;
    for step in 0..steps {
        let s = sim.state();
        let slot = sim.model().slot_count(s) - 1;
        share_sum += sim.tree().rate(slot) / sim.tree().total();
        let event = match sim.step().unwrap() {
            Step::Fired { event, .. } => event,
            Step::Absorbed => panic!("creation keeps the model live"),
        };
        creations += u64::from(event == FeCrRisEvent::Create);
        assert_catalog_matches_rebuild(
            &sim,
            step % 20 == 0,
            &format!("step {step}, event {event:?}"),
        );
    }
    let s = sim.state();
    println!(
        "T4 P={capacity} steps {steps}: Create events {creations}, created {} recombined {} absorbed I {} V {}, \
         mean creation share of the total rate {:.3}",
        s.created(),
        s.recombined(),
        s.absorbed_i(),
        s.absorbed_v(),
        share_sum / steps as f64
    );
    assert!(creations > 0 && creations == s.created(), "no creation");
    if full {
        assert!(s.recombined() > 0, "no recombination");
        assert!(s.absorbed_i() > 0 && s.absorbed_v() > 0, "no absorption");
    }
}

#[test]
fn t4_catalog_matches_rebuild_smoke() {
    t4_run(16, SMOKE_K0, 40, false);
}

#[test]
#[ignore = "full tier: equivalence"]
fn t4_catalog_matches_rebuild() {
    t4_run(64, T4_K0, 2000, true);
}

const T7_SEED: u64 = 0x5eed_0007;

fn t7_model(k0: f64) -> FeCrRis {
    ris_model([8, 8, 8], 650.0, k0, 16, &[0])
}

fn t7_fresh(k0: f64) -> IncrementalSimulation<FeCrRis> {
    let m = t7_model(k0);
    let s = m.random_state(102, 3, 1).unwrap();
    IncrementalSimulation::new(m, s, T7_SEED).unwrap()
}

fn advance(sim: &mut IncrementalSimulation<FeCrRis>, steps: usize) -> Vec<(FeCrRisEvent, u64)> {
    (0..steps)
        .map(|_| match sim.step().unwrap() {
            Step::Fired { event, dt } => (event, dt.to_bits()),
            Step::Absorbed => panic!("creation keeps the model live"),
        })
        .collect()
}

fn state_bytes(sim: &IncrementalSimulation<FeCrRis>) -> Vec<u8> {
    let mut out = Vec::new();
    sim.model().encode_state(sim.state(), &mut out);
    out
}

/// A checkpoint at each split point, taken just before and just after Create events too,
/// resumes to the bytes, time and creation-stream position of an uninterrupted run.
fn t7_run(k0: f64, total: usize) {
    let mut reference = t7_fresh(k0);
    let events = advance(&mut reference, total);
    let creates: Vec<usize> = (0..total)
        .filter(|&i| events[i].0 == FeCrRisEvent::Create)
        .collect();
    assert!(
        creates.len() >= 2,
        "need two Create events, got {creates:?}"
    );
    // After `k` steps the next event is index `k`: split k = c is just before Create c, k = c + 1 just after.
    let mut splits = vec![0, 1, total / 2, total];
    for &c in &creates[..2] {
        splits.extend([c, c + 1]);
    }
    splits.sort_unstable();
    splits.dedup();
    let reference_bytes = state_bytes(&reference);
    assert!(reference.state().creation_word_pos() > 0);
    for k in splits {
        let mut sim = t7_fresh(k0);
        let mut recorded = advance(&mut sim, k);
        let bytes = sim.checkpoint();
        drop(sim);
        let mut resumed = IncrementalSimulation::resume(t7_model(k0), &bytes).unwrap();
        recorded.extend(advance(&mut resumed, total - k));
        assert_eq!(recorded, events, "split {k}");
        assert_eq!(state_bytes(&resumed), reference_bytes, "split {k}: state");
        assert_eq!(resumed.time().to_bits(), reference.time().to_bits());
        assert_eq!(resumed.steps(), reference.steps());
        assert_eq!(
            resumed.state().creation_word_pos(),
            reference.state().creation_word_pos(),
            "split {k}: word_pos"
        );
        assert_eq!(resumed.checkpoint(), reference.checkpoint(), "split {k}");
    }
    println!(
        "T7 {total} steps, Creates at {creates:?}, final word_pos {}",
        reference.state().creation_word_pos()
    );
}

#[test]
fn t7_bitwise_restart_smoke() {
    t7_run(SMOKE_K0, 12);
}

#[test]
#[ignore = "full tier: restart"]
fn t7_bitwise_restart() {
    t7_run(SMOKE_K0, 150);
}

/// A model and a state with two vacancies and two dumbbells far apart: records 0, 2 vacancies, 1, 3 dumbbells.
fn t12_fixture() -> (FeCrRis, FeCrRisState, Vec<u8>) {
    let m = ris_model([8, 8, 8], 650.0, 1.0e7, 6, &[0]);
    let base = m.random_state(102, 3, 1).unwrap();
    let p = |x, y, z| pos_site(&m, [x, y, z]);
    let mixed = |site, axis| Defect::Dumbbell {
        site,
        axis,
        plus: Species::Cr,
        minus: Species::Fe,
    };
    let defects = [
        Defect::Vacancy { site: p(4, 4, 4) },
        mixed(p(6, 6, 6), 0),
        Defect::Vacancy { site: p(12, 12, 4) },
        mixed(p(10, 10, 6), 1),
    ];
    let s = m.state_with_defects(occupancy(&base), &defects, 1).unwrap();
    let mut bytes = Vec::new();
    m.encode_state(&s, &mut bytes);
    (m, s, bytes)
}

/// Byte layout of the encoded fixture state (N = 1024 cells, P = 6 records of 8 bytes).
struct Layout {
    n: usize,
}

impl Layout {
    fn record(&self, rec: usize) -> usize {
        self.n + 8 * rec
    }
    fn reservoir_len(&self, capacity: usize) -> usize {
        self.n + 8 * capacity
    }
}

fn rejection(m: &FeCrRis, bytes: &[u8]) -> String {
    match m.decode_state(bytes) {
        Ok(_) => panic!("corrupted state was accepted"),
        Err(e) => e.to_string(),
    }
}

#[test]
fn t12_round_trip_and_corruption() {
    // Round trip of an evolved state: re-encoded bytes are equal and the catalog is rebuilt identically.
    let mut sim = t7_fresh(SMOKE_K0);
    advance(&mut sim, 30);
    let bytes = state_bytes(&sim);
    let decoded = sim.model().decode_state(&bytes).unwrap();
    let mut again = Vec::new();
    sim.model().encode_state(&decoded, &mut again);
    assert_eq!(again, bytes);
    assert!(decoded.last_changed().is_empty() && decoded.last_records().is_empty());
    let m = sim.model();
    for site in 0..m.lattice().site_count() {
        assert_eq!(decoded.record_at(site), sim.state().record_at(site));
    }
    for slot in 0..m.slot_count(&decoded) {
        assert_eq!(
            m.slot_rate(&decoded, slot).to_bits(),
            m.slot_rate(sim.state(), slot).to_bits()
        );
    }

    let (m, s, bytes) = t12_fixture();
    let (n, cap) = (m.lattice().site_count(), 6);
    let lay = Layout { n };
    let mut again = Vec::new();
    m.encode_state(&m.decode_state(&bytes).unwrap(), &mut again);
    assert_eq!(again, bytes);
    let rec_site = |rec: usize| site_of(defect_of(s.records()[rec]).unwrap());
    let corrupt = |f: &dyn Fn(&mut Vec<u8>)| {
        let mut b = bytes.clone();
        f(&mut b);
        rejection(&m, &b)
    };
    let set_site = |b: &mut Vec<u8>, rec: usize, site: usize| {
        let at = lay.record(rec) + 1;
        b[at..at + 4].copy_from_slice(&(site as u32).to_le_bytes());
    };

    // Lengths.
    assert!(corrupt(&|b| b.truncate(bytes.len() - 1)).contains("ends early"));
    assert!(corrupt(&|b| b.truncate(n - 1)).contains("ends early"));
    assert!(corrupt(&|b| b.push(0)).contains("trailing"));
    let len_at = lay.reservoir_len(cap);
    assert!(
        corrupt(&|b| b[len_at..len_at + 8].copy_from_slice(&u64::MAX.to_le_bytes()))
            .contains("reservoir")
    );
    // Tags.
    assert!(corrupt(&|b| b[lay.record(0)] = 3).contains("kind tag"));
    assert!(corrupt(&|b| b[0] = 4).contains("cell byte"));
    assert!(corrupt(&|b| b[lay.record(1) + 6] = 2).contains("species byte"));
    assert!(corrupt(&|b| b[lay.record(1) + 5] = 6).contains("invalid defect"));
    // Cell/record mismatch: a record whose cell has another content, a swap that keeps the counts,
    // a stray vacancy cell, a record on a site that holds a plain atom.
    assert!(corrupt(&|b| b[rec_site(0)] = 0).contains("does not match"));
    assert!(corrupt(&|b| b.swap(rec_site(0), rec_site(1))).contains("does not match"));
    let plain = (0..n)
        .find(|&x| s.record_at(x).is_none() && x != 0)
        .unwrap();
    assert!(corrupt(&|b| b[plain] = 2).contains("differ in number"));
    assert!(corrupt(&|b| set_site(b, 0, plain)).contains("does not match"));
    // Duplicate site (two vacancies).
    assert!(corrupt(&|b| set_site(b, 2, rec_site(0))).contains("two records"));
    // Defect on the sink plane (x = 0): move vacancy 0 there, keeping cells and record consistent.
    let sink_site = pos_site(&m, [0, 6, 6]);
    assert!(
        corrupt(&|b| {
            b[rec_site(0)] = 0;
            b[sink_site] = 2;
            set_site(b, 0, sink_site);
        })
        .contains("sink")
    );
    // I-V pair within Rc: move vacancy 0 next to dumbbell 1.
    let near = pos_site(&m, [7, 7, 7]);
    assert!(
        corrupt(&|b| {
            b[rec_site(0)] = 0;
            b[near] = 2;
            set_site(b, 0, near);
        })
        .contains("recombination radius")
    );
    // Creation stream other than 1.
    let stream_at = bytes.len() - 32;
    assert!(corrupt(&|b| b[stream_at] = 0).contains("stream"));
}

/// The pilot example's placement code, compiled into this test crate.
#[allow(dead_code)]
#[path = "../examples/fecr_ris_pilot.rs"]
mod pilot;

/// The pilot places Cr on stream 2, not on the engine's stream 0, so its first draw is not
/// the engine's first uniform, and the same seed still gives the same placement.
#[test]
fn pilot_placement_uses_its_own_stream() {
    use okrates_core::KmcRng;
    use rand_chacha::rand_core::Rng;

    let seed = 7;
    let engine_u = KmcRng::from_seed(seed).open01();
    let placement_bits = pilot::placement_rng(seed).next_u64();
    let placement_u = ((placement_bits >> 12) as f64 + 0.5) * f64::EPSILON;
    assert_ne!(placement_u, engine_u);

    let (sites, cr) = (432, 43);
    let first = pilot::place_cr(sites, cr, &mut pilot::placement_rng(seed));
    let again = pilot::place_cr(sites, cr, &mut pilot::placement_rng(seed));
    assert_eq!(first, again);
    assert_eq!(first.iter().filter(|&&s| s == Species::Cr).count(), cr);

    // Placement on the engine's stream (what `random_state` does) gives another layout.
    let m = model([6, 6, 6], 773.0, 64);
    let on_engine_stream = occupancy(&m.random_state(cr, seed, seed).unwrap());
    assert_ne!(first, on_engine_stream);
}

/// A saddle table so large that the refit overflows is rejected instead of building a
/// model with non-finite saddle interactions (PR #9).
#[test]
fn non_finite_saddle_refit_is_rejected() {
    let mut dumbbell = DumbbellParams::senninger2016_r1();
    dumbbell.saddle = [[f64::MAX; 2]; 2];
    let result = FeCrRis::new(FeCrRisParams {
        cells: [6, 6, 6],
        sink_planes: vec![],
        temperature_k: 773.0,
        k0_dpa_per_s: 0.0,
        capacity: 4,
        dumbbell,
    });
    assert_eq!(
        result.err(),
        Some(FeCrRisError::Parameter(
            "the refitted dumbbell saddle interactions are not finite"
        ))
    );
}
