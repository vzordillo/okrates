use std::error::Error;
use std::time::Instant;

use okrates_core::Simulation;
use okrates_core::reference::GraphRandomWalk;

const SIZES: [usize; 9] = [2, 8, 32, 128, 512, 2048, 8192, 32768, 131072];
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

fn build_model(m: usize) -> Result<GraphRandomWalk, Box<dyn Error>> {
    let edges = |target: usize| -> Vec<(usize, f64)> {
        (0..m).map(|j| (target, 1.0 + (j % 7) as f64)).collect()
    };
    Ok(GraphRandomWalk::new(vec![edges(1), edges(0)])?)
}

fn main() -> Result<(), Box<dyn Error>> {
    // Optional limits for quick runs; the defaults give the full benchmark.
    let max_m = env_u64("MAX_M", u64::MAX);
    let budget = env_u64("STEP_BUDGET", 20_000_000);
    println!("M,steps,reps,median_ns_per_step,min_ns_per_step,max_ns_per_step");
    for m in SIZES.into_iter().filter(|&m| m as u64 <= max_m) {
        let steps = (budget / m as u64).max(1000);
        let mut samples = Vec::new();
        for rep in 0..REPS {
            let mut simulation = Simulation::new(build_model(m)?, 0, 1 + rep);
            let warmup = simulation.run(WARMUP)?;
            assert_eq!(warmup.steps, WARMUP);
            let start = Instant::now();
            let summary = simulation.run(steps)?;
            let elapsed = start.elapsed();
            assert!(!summary.absorbed && summary.steps == steps);
            samples.push(elapsed.as_nanos() as f64 / steps as f64);
        }
        samples.sort_by(f64::total_cmp);
        println!(
            "{m},{steps},{REPS},{:.2},{:.2},{:.2}",
            samples[samples.len() / 2],
            samples[0],
            samples[samples.len() - 1]
        );
    }
    Ok(())
}
