//! Deterministic regression pins: short seeded trajectories as `(event, time.to_bits())`
//! and reduced-size vacancy/Fe–Cr diffusivity estimators as `f64::to_bits()`.

use okrates_core::ensemble::run_replicas;
use okrates_core::{IncrementalSimulation, Simulation, Step};
use okrates_models::bcc::BccLattice;
use okrates_models::fecr::{FeCrModel, PairEnergetics};
use okrates_models::fecr_diffusion::{estimate, mean_se, ols_slope, origin_slope, sample_run};
use okrates_models::fecr_ris::{DumbbellParams, FeCrRis, FeCrRisEvent, FeCrRisParams};
use okrates_models::multi_vacancy::MultiVacancyDiffusion;
use okrates_models::vacancy::{VacancyDiffusion, jump_rate};

const SEED: u64 = 20261117;
const STEPS: usize = 30;
const STATE_SEED_OFFSET: u64 = 1_000_000_000;

fn record<E>(
    mut step: impl FnMut() -> (Step<E>, f64),
    code: impl Fn(E) -> usize,
) -> Vec<(usize, u64)> {
    (0..STEPS)
        .map(|_| match step() {
            (Step::Fired { event, .. }, time) => (code(event), time.to_bits()),
            (Step::Absorbed, _) => panic!("unexpected absorption"),
        })
        .collect()
}

fn vacancy_rate() -> f64 {
    jump_rate(1e13, 0.65, 800.0).unwrap()
}

#[test]
fn single_vacancy() {
    let model = VacancyDiffusion::new(BccLattice::new([4, 4, 4]).unwrap(), vacancy_rate());
    let state = model.initial_state(0).unwrap();
    let mut sim = Simulation::new(model, state, SEED);
    let got = record(|| (sim.step().unwrap(), sim.time()), |e| e.direction);
    println!("single_vacancy {got:?}");
    assert_eq!(got, SINGLE_VACANCY);
}

fn multi_vacancy() -> (
    MultiVacancyDiffusion,
    okrates_models::multi_vacancy::MultiVacancyState,
) {
    let model = MultiVacancyDiffusion::new(BccLattice::new([4, 4, 4]).unwrap(), 1.3);
    let sites = model.random_sites(6, SEED).unwrap();
    let state = model.initial_state(&sites).unwrap();
    (model, state)
}

#[test]
fn multi_vacancy_direct() {
    let (model, state) = multi_vacancy();
    let mut sim = Simulation::new(model, state, SEED);
    let got = record(
        || (sim.step().unwrap(), sim.time()),
        |e| e.vacancy * 8 + e.direction,
    );
    println!("multi_vacancy_direct {got:?}");
    assert_eq!(got, MULTI_VACANCY_DIRECT);
}

#[test]
fn multi_vacancy_incremental() {
    let (model, state) = multi_vacancy();
    let mut sim = IncrementalSimulation::new(model, state, SEED).unwrap();
    let got = record(
        || (sim.step().unwrap(), sim.time()),
        |e| e.vacancy * 8 + e.direction,
    );
    println!("multi_vacancy_incremental {got:?}");
    assert_eq!(got, MULTI_VACANCY_INCREMENTAL);
}

fn illustrative_fecr() -> FeCrModel<PairEnergetics> {
    FeCrModel::new(
        BccLattice::new([4, 4, 4]).unwrap(),
        PairEnergetics::illustrative(),
        1e13,
        800.0,
    )
    .unwrap()
}

#[test]
fn fecr_trajectory() {
    let model = illustrative_fecr();
    let state = model.random_state(32, SEED + STATE_SEED_OFFSET).unwrap();
    let mut sim = Simulation::new(model, state, SEED);
    let got = record(|| (sim.step().unwrap(), sim.time()), |e| e.direction);
    println!("fecr_trajectory {got:?}");
    assert_eq!(got, FECR_TRAJECTORY);
}

fn fnv1a(hash: u64, word: u64) -> u64 {
    (hash ^ word).wrapping_mul(0x0000_0100_0000_01b3)
}

/// Fe-Cr RIS model, 6x6x6 cells with a sink plane at x = 0, T = 773 K, K0 = 1e-3 dpa/s,
/// 2000 incremental steps. Pins two digests, one over the events (kind, record, index) and
/// one over the bits of every dt, plus the final counters and time bits. Everything but
/// the dt digest is shared by all targets; see `FECR_RIS_DTS`.
#[test]
fn fecr_ris_trajectory() {
    let model = FeCrRis::new(FeCrRisParams {
        cells: [6, 6, 6],
        sink_planes: vec![0],
        temperature_k: 773.0,
        k0_dpa_per_s: 1e-3,
        capacity: 64,
        dumbbell: DumbbellParams::senninger2016_r1(),
    })
    .unwrap();
    let state = model.random_state(43, SEED, SEED).unwrap();
    let mut sim = IncrementalSimulation::new(model, state, SEED).unwrap();
    let mut events = 0xcbf2_9ce4_8422_2325_u64;
    let mut dts = 0xcbf2_9ce4_8422_2325_u64;
    for _ in 0..2000 {
        let Step::Fired { event, dt } = sim.step().unwrap() else {
            panic!("unexpected absorption");
        };
        let (kind, rec, index) = match event {
            FeCrRisEvent::VacancyJump { rec, dir } => (1, rec, dir),
            FeCrRisEvent::DumbbellJump { rec, jump } => (2, rec, jump),
            FeCrRisEvent::Create => (0, 0, 0),
        };
        for word in [kind, rec as u64, index as u64] {
            events = fnv1a(events, word);
        }
        dts = fnv1a(dts, dt.to_bits());
    }
    let s = sim.state();
    let counters = [s.created(), s.recombined(), s.absorbed_v(), s.absorbed_i()];
    let time_bits = sim.time().to_bits();
    println!("fecr_ris_trajectory events {events}");
    println!("fecr_ris_trajectory counters {counters:?}");
    println!("fecr_ris_trajectory time_bits {time_bits}");
    println!("fecr_ris_trajectory dts {dts}");
    assert_eq!(events, FECR_RIS_EVENTS, "event digest differs");
    assert_eq!(
        counters.as_slice(),
        FECR_RIS_COUNTERS,
        "counters (created, recombined, absorbed_v, absorbed_i) differ"
    );
    assert_eq!(time_bits, FECR_RIS_TIME_BITS, "final time bits differ");
    #[cfg(any(
        all(target_arch = "x86_64", target_os = "linux"),
        all(target_arch = "aarch64", target_os = "macos")
    ))]
    assert_eq!(dts, FECR_RIS_DTS, "dt digest differs");
    #[cfg(not(any(
        all(target_arch = "x86_64", target_os = "linux"),
        all(target_arch = "aarch64", target_os = "macos")
    )))]
    println!(
        "fecr_ris_trajectory: no dt digest pinned for this target (arch + OS); dt check skipped"
    );
}

#[test]
fn vacancy_msd_reduced() {
    let (runs, a, k_count) = (8_u64, 2.8665e-10, 10);
    let rate = vacancy_rate();
    let t_end = 200.0 / (8.0 * rate);
    let lattice = BccLattice::new([4, 4, 4]).unwrap();
    let replicas = run_replicas(20261117, runs, |_, seed| {
        let model = VacancyDiffusion::new(lattice.clone(), rate);
        let state = model.initial_state(0).unwrap();
        let mut sim = Simulation::new(model, state, seed);
        let mut samples = vec![0.0; k_count];
        let mut k = 1;
        while k <= k_count {
            let held = sim.state().squared_displacement(a);
            sim.step().unwrap();
            while k <= k_count && (k as f64 * t_end / k_count as f64) < sim.time() {
                samples[k - 1] = held;
                k += 1;
            }
        }
        let times: Vec<f64> = (1..=k_count)
            .map(|k| k as f64 * t_end / k_count as f64)
            .collect();
        origin_slope(&times, &samples) / 6.0
    })
    .unwrap();
    let estimates: Vec<f64> = replicas.into_iter().map(|r| r.value).collect();
    let (d_hat, se) = mean_se(&estimates);
    let got = [d_hat.to_bits(), se.to_bits()];
    println!("vacancy_msd_reduced {got:?}");
    assert_eq!(got.as_slice(), VACANCY_MSD_REDUCED);
}

#[test]
fn fecr_estimators_reduced() {
    let runs = 4_u64;
    let equal = FeCrModel::new(
        BccLattice::new([4, 4, 4]).unwrap(),
        PairEnergetics::new([[0.0; 3]; 3], 0.0, 0.0).unwrap(),
        1.0,
        1000.0,
    )
    .unwrap();
    let f_n: Vec<f64> = run_replicas(20261120, runs, |_, seed| {
        let state = equal.random_state(64, seed + STATE_SEED_OFFSET).unwrap();
        let (times, samples, _) = sample_run(equal.clone(), state, seed, 50.0, 20, 1.0).unwrap();
        127.0 * estimate(&times[10..], &samples[10..], ols_slope).tracer_all
    })
    .unwrap()
    .into_iter()
    .map(|r| r.value)
    .collect();

    let model = illustrative_fecr();
    let r0 = 1e13 * (-0.60 / (okrates_models::vacancy::BOLTZMANN_EV_PER_K * 800.0)).exp();
    let d_cr: Vec<f64> = run_replicas(20261121, runs, |_, seed| {
        let state = model.random_state(13, seed + STATE_SEED_OFFSET).unwrap();
        let (times, samples, _) =
            sample_run(model.clone(), state, seed, 100.0 / (8.0 * r0), 10, 1.0).unwrap();
        estimate(&times, &samples, origin_slope).tracer[1]
    })
    .unwrap()
    .into_iter()
    .map(|r| r.value)
    .collect();

    let (f_mean, f_se) = mean_se(&f_n);
    let (cr_mean, cr_se) = mean_se(&d_cr);
    let got = [
        f_mean.to_bits(),
        f_se.to_bits(),
        cr_mean.to_bits(),
        cr_se.to_bits(),
    ];
    println!("fecr_estimators_reduced {got:?}");
    assert_eq!(got.as_slice(), FECR_ESTIMATORS_REDUCED);
}

const SINGLE_VACANCY: &[(usize, u64)] = &[
    (0, 4450578207487319775),
    (6, 4458115016692571594),
    (3, 4468371911039400504),
    (4, 4469351429524061411),
    (4, 4472907350433383041),
    (1, 4473460339841781673),
    (5, 4475211887373916338),
    (5, 4476874792125394247),
    (2, 4477133233745006790),
    (3, 4477689699374703942),
    (5, 4477724506871948194),
    (3, 4477910705233252470),
    (7, 4478121651700116602),
    (1, 4478193592112324482),
    (4, 4478201187659502600),
    (6, 4480156933701002443),
    (3, 4480299538791138563),
    (7, 4480427744920509370),
    (3, 4480535063530653432),
    (7, 4480869619928073985),
    (0, 4481103580522072028),
    (7, 4481355060031103586),
    (5, 4481812948427580128),
    (4, 4482172857132367006),
    (0, 4482215876555509693),
    (3, 4482335097893526319),
    (1, 4482631723904438679),
    (1, 4483373082110236368),
    (3, 4483481337055796103),
    (4, 4484369050335726857),
];
const MULTI_VACANCY_DIRECT: &[(usize, u64)] = &[
    (3, 4571000654989052202),
    (38, 4578222132279019826),
    (20, 4588714168296803716),
    (27, 4589800662457201499),
    (29, 4593652053084528309),
    (8, 4594147107680550063),
    (34, 4595684163406514317),
    (35, 4597325685147197067),
    (15, 4597758654094177079),
    (21, 4598433062086395778),
    (34, 4598460951022154843),
    (23, 4598610139402474709),
    (43, 4598779156796103809),
    (8, 4598834396158966197),
    (29, 4598840481966452412),
    (37, 4600342199317486094),
    (22, 4600451698472109924),
    (42, 4600550141401252891),
    (20, 4600632545872755012),
    (47, 4600889434618146756),
    (1, 4601085936263054047),
    (43, 4601472132767230536),
    (32, 4602175310925575149),
    (26, 4602703420534587061),
    (2, 4602739455896007758),
    (22, 4602839322042288675),
    (8, 4603099623689933442),
    (11, 4603693624782021781),
    (21, 4603788622994107838),
    (28, 4604532218945503001),
];
const MULTI_VACANCY_INCREMENTAL: &[(usize, u64)] = &[
    (3, 4571000654989052194),
    (38, 4578222132279019820),
    (20, 4588714168296803709),
    (27, 4589800662457201495),
    (29, 4593652053084528301),
    (8, 4594147107680550058),
    (34, 4595684163406514311),
    (35, 4597325685147197060),
    (15, 4597758654094177072),
    (21, 4598433062086395774),
    (34, 4598460951022154839),
    (23, 4598610139402474705),
    (43, 4598779156796103804),
    (8, 4598834396158966192),
    (29, 4598840481966452407),
    (37, 4600342199317486088),
    (22, 4600451698472109917),
    (42, 4600550141401252885),
    (20, 4600632545872755005),
    (47, 4600889434618146749),
    (1, 4601085936263054040),
    (43, 4601472132767230528),
    (32, 4602175310925575141),
    (26, 4602703420534587056),
    (2, 4602739455896007753),
    (22, 4602839322042288671),
    (8, 4603099623689933437),
    (11, 4603693624782021776),
    (21, 4603788622994107833),
    (28, 4604532218945502996),
];
const FECR_TRAJECTORY: &[(usize, u64)] = &[
    (0, 4439421333564381059),
    (4, 4452542391721444105),
    (4, 4456383279435944337),
    (5, 4459737960478015951),
    (6, 4463456957036608877),
    (0, 4465351652919606784),
    (7, 4467094450759664457),
    (4, 4470689823761332770),
    (2, 4470962311818486457),
    (3, 4471874634118659339),
    (5, 4471916164684854383),
    (2, 4472475863174522282),
    (5, 4472601708565591845),
    (1, 4472787382083718722),
    (6, 4472790186914270194),
    (4, 4477207934833843928),
    (3, 4477227870964767824),
    (4, 4477393316799418599),
    (3, 4477408319895453673),
    (7, 4477840054030792165),
    (0, 4477858703473198850),
    (4, 4478507755706091040),
    (3, 4478635780912091407),
    (2, 4479564681836778454),
    (1, 4479590346149670053),
    (5, 4479787848835319329),
    (2, 4479897384814526331),
    (4, 4481103575944586190),
    (4, 4481116624308049350),
    (4, 4481557275662921210),
];
const VACANCY_MSD_REDUCED: &[u64] = &[4455383623338886256, 4445696229053450704];
const FECR_ESTIMATORS_REDUCED: &[u64] = &[
    4604275095040570454,
    4587890298691939716,
    4714297785282910612,
    4698817579328050235,
];
const FECR_RIS_EVENTS: u64 = 3311104404065515709;
const FECR_RIS_COUNTERS: &[u64] = &[301, 278, 22, 23];
const FECR_RIS_TIME_BITS: u64 = 4649199847678246177;
// The dt digest is per target (ADR-0006 rule 4, PR #9): the event sequence, counters and
// final time bits agree on x86_64 Linux and aarch64 macOS, but the last bits of single
// dt values differ with the platform ln/exp, so the digest over their bits does too.
// A target is architecture plus OS (the maths library comes with the OS); other targets skip only the dt check.
#[cfg(all(target_arch = "x86_64", target_os = "linux"))]
const FECR_RIS_DTS: u64 = 1061706550594519766;
// Pending macOS CI update after D11 (`z_eff = 4`, 0.5 orientation rate split).
#[cfg(all(target_arch = "aarch64", target_os = "macos"))]
const FECR_RIS_DTS: u64 = 9886202463999487029;
