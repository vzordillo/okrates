use std::error::Error;
use std::time::Instant;

use okrates_core::reference::GraphRandomWalk;
use okrates_core::{IncrementalSimulation, Simulation};

const SIZES: [usize; 8] = [8, 32, 128, 512, 2048, 8192, 32768, 131072];
const REPS: u64 = 5;
const WARMUP: u64 = 100;

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

/// Degree-4 directed ring with `m` total slots (`nodes = m / 4`), so each hop updates
/// `8 / m` of the catalog (`rho_update = 8.0 / m`) with rate skew `max / min = 7.0`.
fn build_ring_model(m: usize) -> Result<GraphRandomWalk, Box<dyn Error>> {
    let nodes = (m / 4).max(2);
    let adjacency = (0..nodes)
        .map(|i| {
            (0..4)
                .map(|d| {
                    let target = (i + 1 + d) % nodes;
                    let rate = 1.0 + ((i * 4 + d) % 7) as f64;
                    (target, rate)
                })
                .collect()
        })
        .collect();
    Ok(GraphRandomWalk::new(adjacency)?)
}

fn measure<F: FnMut(u64) -> Result<(), Box<dyn Error>>>(
    steps: u64,
    mut run_rep: F,
) -> Result<[f64; 3], Box<dyn Error>> {
    let mut samples = Vec::with_capacity(REPS as usize);
    for rep in 0..REPS {
        let start = Instant::now();
        run_rep(rep)?;
        let elapsed = start.elapsed();
        samples.push(elapsed.as_nanos() as f64 / steps as f64);
    }
    samples.sort_by(f64::total_cmp);
    Ok([
        samples[samples.len() / 2],
        samples[0],
        samples[samples.len() - 1],
    ])
}

fn main() -> Result<(), Box<dyn Error>> {
    let max_m = env_u64("MAX_M", u64::MAX);
    let budget = env_u64("STEP_BUDGET", 20_000_000);
    println!(
        "backend,M,rho_update,rate_skew,bytes_per_slot,steps,reps,median_ns_per_step,min_ns_per_step,max_ns_per_step"
    );
    for m in SIZES.into_iter().filter(|&m| m as u64 <= max_m) {
        let steps = (budget / m as u64).max(1000);
        let model = build_ring_model(m)?;
        let rho = (8.0 / m as f64).min(1.0);
        let rate_skew = 7.0;

        let direct_bytes = std::mem::size_of::<(usize, f64)>();
        let direct_ns = measure(steps, |rep| {
            let mut sim = Simulation::new(model.clone(), 0, 1 + rep);
            assert_eq!(sim.run(WARMUP)?.steps, WARMUP);
            let summary = sim.run(steps)?;
            assert!(!summary.absorbed && summary.steps == steps);
            Ok(())
        })?;
        println!(
            "direct,{m},{rho:.6},{rate_skew:.1},{direct_bytes},{steps},{REPS},{:.2},{:.2},{:.2}",
            direct_ns[0], direct_ns[1], direct_ns[2]
        );

        let tree_sim = IncrementalSimulation::new(model.clone(), 0, 1)?;
        let tree_bytes = std::mem::size_of_val(tree_sim.tree().nodes()) / m;
        let tree_ns = measure(steps, |rep| {
            let mut sim = IncrementalSimulation::new(model.clone(), 0, 1 + rep)?;
            assert_eq!(sim.run(WARMUP)?.steps, WARMUP);
            let summary = sim.run(steps)?;
            assert!(!summary.absorbed && summary.steps == steps);
            Ok(())
        })?;
        println!(
            "sum_tree,{m},{rho:.6},{rate_skew:.1},{tree_bytes},{steps},{REPS},{:.2},{:.2},{:.2}",
            tree_ns[0], tree_ns[1], tree_ns[2]
        );
    }
    Ok(())
}
