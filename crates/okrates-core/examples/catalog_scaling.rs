use std::error::Error;
use std::time::Instant;

use okrates_core::Simulation;
use okrates_core::reference::GraphRandomWalk;

const SIZES: [usize; 9] = [2, 8, 32, 128, 512, 2048, 8192, 32768, 131072];
const REPS: u64 = 5;
const WARMUP: u64 = 100;

fn build_model(m: usize) -> Result<GraphRandomWalk, Box<dyn Error>> {
    let edges = |target: usize| -> Vec<(usize, f64)> {
        (0..m).map(|j| (target, 1.0 + (j % 7) as f64)).collect()
    };
    Ok(GraphRandomWalk::new(vec![edges(1), edges(0)])?)
}

fn main() -> Result<(), Box<dyn Error>> {
    println!("M,steps,reps,median_ns_per_step,min_ns_per_step,max_ns_per_step");
    for m in SIZES {
        let steps = (20_000_000 / m as u64).max(1000);
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
