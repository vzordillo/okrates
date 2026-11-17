use std::error::Error;
use std::time::Instant;

use okrates_core::{IncrementalSimulation, Simulation};
use okrates_models::bcc::BccLattice;
use okrates_models::multi_vacancy::MultiVacancyDiffusion;

const CELLS: [usize; 3] = [32, 32, 32];
const COUNTS: [usize; 7] = [1, 4, 16, 64, 256, 1024, 4096];
const REPS: u64 = 5;
const WARMUP: u64 = 100;

/// Reads an optional positive integer from the environment; exits 2 on a bad value.
fn env_u64(name: &str, default: u64) -> u64 {
    match std::env::var(name) {
        Ok(value) => match value.parse::<u64>() {
            Ok(v) if v >= 1 => v,
            _ => {
                eprintln!("{name} must be an integer >= 1");
                std::process::exit(2);
            }
        },
        Err(_) => default,
    }
}

fn stats(mut samples: Vec<f64>) -> (f64, f64, f64) {
    samples.sort_by(f64::total_cmp);
    (
        samples[samples.len() / 2],
        samples[0],
        samples[samples.len() - 1],
    )
}

fn main() -> Result<(), Box<dyn Error>> {
    // Optional limits for quick runs; the defaults give the full benchmark.
    let max_n = env_u64("MAX_N", u64::MAX);
    let budget = env_u64("STEP_BUDGET", 20_000_000);
    println!(
        "N,M,steps,direct_median_ns,direct_min_ns,direct_max_ns,incremental_median_ns,incremental_min_ns,incremental_max_ns,speedup"
    );
    let model = MultiVacancyDiffusion::new(BccLattice::new(CELLS)?, 1.0);
    for n in COUNTS.into_iter().filter(|&n| n as u64 <= max_n) {
        let m = 8 * n;
        let steps = (budget / m as u64).max(2000);
        let sites = model.random_sites(n, 1)?;
        let mut direct = Vec::new();
        let mut incremental = Vec::new();
        for rep in 0..REPS {
            let mut sim = Simulation::new(model.clone(), model.initial_state(&sites)?, 1 + rep);
            assert_eq!(sim.run(WARMUP)?.steps, WARMUP);
            let start = Instant::now();
            let summary = sim.run(steps)?;
            let elapsed = start.elapsed();
            assert!(!summary.absorbed && summary.steps == steps);
            direct.push(elapsed.as_nanos() as f64 / steps as f64);
        }
        for rep in 0..REPS {
            let mut sim =
                IncrementalSimulation::new(model.clone(), model.initial_state(&sites)?, 1 + rep)?;
            assert_eq!(sim.run(WARMUP)?.steps, WARMUP);
            let start = Instant::now();
            let summary = sim.run(steps)?;
            let elapsed = start.elapsed();
            assert!(!summary.absorbed && summary.steps == steps);
            incremental.push(elapsed.as_nanos() as f64 / steps as f64);
        }
        let (d_med, d_min, d_max) = stats(direct);
        let (i_med, i_min, i_max) = stats(incremental);
        println!(
            "{n},{m},{steps},{d_med:.2},{d_min:.2},{d_max:.2},{i_med:.2},{i_min:.2},{i_max:.2},{:.2}",
            d_med / i_med
        );
    }
    Ok(())
}
