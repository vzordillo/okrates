//! Tests of the Fe–Cr RIS model, part 1: R1 parity (T1), detailed balance (T2),
//! jump geometry (T3), vacancy parity with M12 (T14b), the barrier floor (T15a),
//! defects inside each other's saddle sets (I3) and state validation.

use std::collections::HashMap;
use std::fs;

use okrates_models::bcc::BccLattice;
use okrates_models::fecr::{Energetics, Species};
use okrates_models::fecr_martinez::Martinez2012;
use okrates_models::fecr_ris::{
    AXES, Cell, Defect, DumbbellParams, FeCrRis, FeCrRisError, FeCrRisParams, FeCrRisState,
    RC_OFFSETS, Record,
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

// ------------------------------------------------------------------ T1

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

// ------------------------------------------------------------------ detailed balance

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

// ------------------------------------------------------------------ T3

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

// ------------------------------------------------------------------ T14b

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

// ------------------------------------------------------------------ I3

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

// ------------------------------------------------------------------ validation

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
