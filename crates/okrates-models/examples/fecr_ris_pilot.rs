//! Smoke driver for the Fe–Cr RIS model: a box with a sink plane at x = 0 and 10 % Cr,
//! irradiated at a fixed damage rate on the incremental scheduler until the target dose.
//!
//! Usage: `fecr_ris_pilot --temperature K --k0 dpa_per_s --dpa-end x --capacity P --seed s
//! [--cells nx ny nz] [--report-every-steps n]`.
//!
//! The seed drives three independent ChaCha12 streams of one key: stream 0 engine (the
//! scheduler), 1 creation (Frenkel-pair creation inside the model), 2 placement (the Cr
//! shuffle), so placement draws never repeat event draws. Output lines are
//! `key=value` pairs: steps, time (s), dose (dpa), current and peak vacancies V and
//! interstitials I, their time-weighted means, the event counters, `borrowed_fe`,
//! creation attempts, the floored-jump counters and the wall time per step. Exit status 1
//! on an error, including `PoolExhausted` (raise `--capacity`); 2 on a usage error.

use std::env;
use std::process::exit;
use std::time::Instant;

use okrates_core::{IncrementalSimulation, SimError, Step};
use okrates_models::fecr::Species;
use okrates_models::fecr_ris::{
    DumbbellParams, FeCrRis, FeCrRisError, FeCrRisParams, FeCrRisState,
};

#[path = "support/tally.rs"]
mod tally;
use tally::Tally;

use rand_chacha::ChaCha12Rng;
use rand_chacha::rand_core::{Rng, SeedableRng};

const USAGE: &str = "usage: fecr_ris_pilot --temperature K --k0 dpa_per_s --dpa-end x --capacity P --seed s [--cells nx ny nz] [--report-every-steps n]";

fn usage() -> ! {
    eprintln!("{USAGE}");
    exit(2);
}

fn fail(message: impl std::fmt::Display) -> ! {
    eprintln!("error: {message}");
    exit(1);
}

struct Args {
    temperature: f64,
    k0: f64,
    dpa_end: f64,
    capacity: usize,
    seed: u64,
    cells: [usize; 3],
    report_every: u64,
}

fn parse() -> Args {
    let argv: Vec<String> = env::args().skip(1).collect();
    let (mut temperature, mut k0, mut dpa_end, mut capacity, mut seed) =
        (None, None, None, None, None);
    let mut cells = [128, 32, 32];
    let mut report_every = 100_000_u64;
    let mut i = 0;
    let next = |i: &mut usize| -> &str {
        *i += 1;
        argv.get(*i).map(String::as_str).unwrap_or_else(|| usage())
    };
    while i < argv.len() {
        match argv[i].as_str() {
            "--temperature" => temperature = next(&mut i).parse::<f64>().ok(),
            "--k0" => k0 = next(&mut i).parse::<f64>().ok(),
            "--dpa-end" => dpa_end = next(&mut i).parse::<f64>().ok(),
            "--capacity" => capacity = next(&mut i).parse::<usize>().ok(),
            "--seed" => seed = next(&mut i).parse::<u64>().ok(),
            "--report-every-steps" => {
                report_every = next(&mut i).parse().unwrap_or_else(|_| usage());
            }
            "--cells" => {
                for c in &mut cells {
                    *c = next(&mut i).parse().unwrap_or_else(|_| usage());
                }
            }
            _ => usage(),
        }
        i += 1;
    }
    match (temperature, k0, dpa_end, capacity, seed) {
        (Some(temperature), Some(k0), Some(dpa_end), Some(capacity), Some(seed))
            if k0 > 0.0 && dpa_end > 0.0 && report_every > 0 =>
        {
            Args {
                temperature,
                k0,
                dpa_end,
                capacity,
                seed,
                cells,
                report_every,
            }
        }
        _ => usage(),
    }
}

/// The placement stream: ChaCha12 keyed like the engine (`seed` LE, zero padded) but on
/// stream 2. Stream 0 is the engine and stream 1 the model's creation stream.
pub fn placement_rng(seed: u64) -> ChaCha12Rng {
    let mut bytes = [0_u8; 32];
    bytes[..8].copy_from_slice(&seed.to_le_bytes());
    let mut rng = ChaCha12Rng::from_seed(bytes);
    rng.set_stream(2);
    rng
}

/// Fe with `cr_count` Cr atoms on sites drawn by a partial Fisher-Yates shuffle.
pub fn place_cr(sites: usize, cr_count: usize, rng: &mut ChaCha12Rng) -> Vec<Species> {
    let mut order: Vec<usize> = (0..sites).collect();
    for i in 0..cr_count {
        let u = ((rng.next_u64() >> 12) as f64 + 0.5) * f64::EPSILON;
        let j = i + ((u * (sites - i) as f64) as usize).min(sites - i - 1);
        order.swap(i, j);
    }
    let mut occ = vec![Species::Fe; sites];
    for &site in &order[..cr_count] {
        occ[site] = Species::Cr;
    }
    occ
}

/// Vacancies and dumbbells in use, from the counters (`created − recombined − absorbed`).
fn counts(s: &FeCrRisState) -> (u64, u64) {
    (
        s.created() - s.recombined() - s.absorbed_v(),
        s.created() - s.recombined() - s.absorbed_i(),
    )
}

fn main() {
    let a = parse();
    let model = FeCrRis::new(FeCrRisParams {
        cells: a.cells,
        sink_planes: vec![0],
        temperature_k: a.temperature,
        k0_dpa_per_s: a.k0,
        capacity: a.capacity,
        dumbbell: DumbbellParams::senninger2016_r1(),
    })
    .unwrap_or_else(|e| fail(e));
    let sites = model.lattice().site_count();
    let cr_count = (0.1 * sites as f64).round() as usize;
    let occ = place_cr(sites, cr_count, &mut placement_rng(a.seed));
    let state = model
        .state_with_defects(occ, &[], a.seed)
        .unwrap_or_else(|e| fail(e));
    let mut sim = IncrementalSimulation::new(model, state, a.seed).unwrap_or_else(|e| fail(e));
    println!(
        "box={:?} sites={sites} cr={cr_count} T={} K0={:e} dpa_end={:e} P={} seed={}",
        a.cells, a.temperature, a.k0, a.dpa_end, a.capacity, a.seed
    );

    let (mut vac, mut int) = (Tally::default(), Tally::default());
    let clock = Instant::now();
    let report = |sim: &IncrementalSimulation<FeCrRis>, vac: &Tally, int: &Tally, label: &str| {
        let (s, t) = (sim.state(), sim.time());
        let (v_now, i_now) = counts(s);
        let us = clock.elapsed().as_secs_f64() * 1e6 / sim.steps().max(1) as f64;
        println!(
            "{label} steps={} t={:.6e} dose={:.6e} V={} I={} V_peak={} I_peak={} V_mean={:.4} I_mean={:.4} \
             created={} recombined={} absorbed_V={} absorbed_I={} borrowed_fe={} creation_attempts={} \
             floored_dumbbell={} floored_vacancy={} us_per_step={us:.2}",
            sim.steps(),
            t,
            a.k0 * t,
            v_now,
            i_now,
            vac.peak,
            int.peak,
            vac.mean(),
            int.mean(),
            s.created(),
            s.recombined(),
            s.absorbed_v(),
            s.absorbed_i(),
            s.borrowed_fe(),
            s.creation_attempts(),
            s.floored_dumbbell_jumps(),
            s.floored_vacancy_jumps(),
        );
    };
    while a.k0 * sim.time() < a.dpa_end {
        let (v_before, i_before) = counts(sim.state());
        let dt = match sim.step() {
            // The counts before the step held for the sampled waiting time.
            Ok(Step::Fired { dt, .. }) => dt,
            Ok(Step::Absorbed) => fail("the model has no event left"),
            Err(SimError::Model(FeCrRisError::PoolExhausted { capacity, defects })) => {
                report(&sim, &vac, &int, "failed");
                fail(format!(
                    "defect pool exhausted at dose {:.6e} dpa: {defects} of {capacity} records in use; raise --capacity",
                    a.k0 * sim.time()
                ))
            }
            Err(e) => fail(e),
        };
        let (v_after, i_after) = counts(sim.state());
        vac.update(v_before, v_after, dt);
        int.update(i_before, i_after, dt);
        if sim.steps().is_multiple_of(a.report_every) {
            report(&sim, &vac, &int, "progress");
        }
    }
    report(&sim, &vac, &int, "final");
}
