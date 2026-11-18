use std::convert::Infallible;
use std::error::Error;
use std::time::Instant;

use okrates_core::{
    CoupledModel, EnumeratedTopology, IncrementalSimulation, RateModel, RunSummary, Simulation,
    SlottedTopology, TransitionTopology,
};

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

/// Coupled ring with `nodes = (m / 4).max(2)` sites and 4 channels per site (`M = 4 * nodes`
/// simultaneously active events). Advancing `phase[node]` perturbs the 4 channels on `node` and
/// the 4 channels on `(node + nodes - 1) % nodes` (`k = 8`, `rho_update = min(1, 8 / M)`).
#[derive(Clone, Debug)]
struct RingTopology {
    nodes: usize,
}

impl TransitionTopology for RingTopology {
    type State = Vec<u8>;
    type Event = (usize, u8);
    type Undo = (usize, u8);
    type Error = Infallible;

    fn apply(
        &self,
        state: &mut Self::State,
        &(node, ch): &Self::Event,
    ) -> Result<Self::Undo, Self::Error> {
        let prev = state[node];
        state[node] = (prev + 1 + ch) % 7;
        Ok((node, prev))
    }

    fn undo(&self, state: &mut Self::State, (node, prev): Self::Undo) {
        state[node] = prev;
    }
}

impl EnumeratedTopology for RingTopology {
    fn enumerate_candidates(&self, _state: &Self::State, emit: &mut dyn FnMut(Self::Event)) {
        for node in 0..self.nodes {
            for ch in 0..4_u8 {
                emit((node, ch));
            }
        }
    }
}

impl SlottedTopology for RingTopology {
    fn slot_count(&self, _state: &Self::State) -> usize {
        4 * self.nodes
    }

    fn slot_event(&self, _state: &Self::State, slot: usize) -> Self::Event {
        (slot / 4, (slot % 4) as u8)
    }

    fn affected_slots(&self, _state: &Self::State, &(node, _): &Self::Event, out: &mut Vec<usize>) {
        let prev = (node + self.nodes - 1) % self.nodes;
        out.extend(4 * node..4 * node + 4);
        if prev != node {
            out.extend(4 * prev..4 * prev + 4);
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct SkewedRingRates;

impl RateModel<RingTopology> for SkewedRingRates {
    fn rate(&self, topology: &RingTopology, state: &Vec<u8>, &(node, ch): &(usize, u8)) -> f64 {
        let next = (node + 1) % topology.nodes;
        let bucket = (state[node] as usize + state[next] as usize + node + ch as usize) % 7;
        1.0 + bucket as f64
    }
}

fn verify_run(summary: RunSummary, expected_steps: u64) -> Result<(), Box<dyn Error>> {
    if summary.absorbed || summary.steps != expected_steps {
        return Err(format!(
            "expected {expected_steps} non-absorbing steps, got steps={} absorbed={}",
            summary.steps, summary.absorbed
        )
        .into());
    }
    Ok(())
}

fn measure<S, Setup, Run>(
    steps: u64,
    mut setup: Setup,
    mut run_timed: Run,
) -> Result<[f64; 3], Box<dyn Error>>
where
    Setup: FnMut(u64) -> Result<S, Box<dyn Error>>,
    Run: FnMut(&mut S) -> Result<RunSummary, Box<dyn Error>>,
{
    let mut samples = Vec::with_capacity(REPS as usize);
    for rep in 0..REPS {
        let mut sim = setup(rep)?;
        let start = Instant::now();
        let summary = run_timed(&mut sim)?;
        let elapsed = start.elapsed();
        verify_run(summary, steps)?;
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
    let budget = env_u64("STEP_BUDGET", 2_000_000);
    println!(
        "backend,M,rho_update,rate_skew,bytes_per_slot,steps,reps,median_ns_per_step,min_ns_per_step,max_ns_per_step"
    );
    for m in SIZES.into_iter().filter(|&m| m as u64 <= max_m) {
        let nodes = (m / 4).max(2);
        let actual_m = 4 * nodes;
        let model = CoupledModel::new(RingTopology { nodes }, SkewedRingRates);
        let init_state: Vec<u8> = (0..nodes).map(|i| (i % 7) as u8).collect();
        let steps = (budget / actual_m as u64).max(1000);
        let rho = (8.0 / actual_m as f64).min(1.0);
        let rate_skew = 7.0;

        let mut direct_probe = Simulation::new(model.clone(), init_state.clone(), 1);
        verify_run(direct_probe.run(WARMUP)?, WARMUP)?;
        let direct_bytes =
            direct_probe.engine().catalog().allocated_bytes() as f64 / actual_m as f64;
        let direct_ns = measure(
            steps,
            |rep| {
                let mut sim = Simulation::new(model.clone(), init_state.clone(), 1 + rep);
                verify_run(sim.run(WARMUP)?, WARMUP)?;
                Ok(sim)
            },
            |sim| Ok(sim.run(steps)?),
        )?;
        println!(
            "direct,{actual_m},{rho:.6},{rate_skew:.1},{direct_bytes:.2},{steps},{REPS},{:.2},{:.2},{:.2}",
            direct_ns[0], direct_ns[1], direct_ns[2]
        );

        let mut tree_probe = IncrementalSimulation::new(model.clone(), init_state.clone(), 1)?;
        verify_run(tree_probe.run(WARMUP)?, WARMUP)?;
        let tree_bytes = tree_probe.engine().catalog().allocated_bytes() as f64 / actual_m as f64;
        let tree_ns = measure(
            steps,
            |rep| {
                let mut sim =
                    IncrementalSimulation::new(model.clone(), init_state.clone(), 1 + rep)?;
                verify_run(sim.run(WARMUP)?, WARMUP)?;
                Ok(sim)
            },
            |sim| Ok(sim.run(steps)?),
        )?;
        println!(
            "sum_tree,{actual_m},{rho:.6},{rate_skew:.1},{tree_bytes:.2},{steps},{REPS},{:.2},{:.2},{:.2}",
            tree_ns[0], tree_ns[1], tree_ns[2]
        );
    }
    Ok(())
}
