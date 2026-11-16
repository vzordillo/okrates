use std::env;
use std::error::Error;
use std::process;

use okrates_core::reference::GraphRandomWalk;
use okrates_core::{Simulation, Step};

fn usage() -> ! {
    eprintln!("usage: random_walk [SEED] [STEPS]");
    process::exit(2);
}

fn main() -> Result<(), Box<dyn Error>> {
    let arguments: Vec<String> = env::args().skip(1).collect();
    if arguments.len() > 2 {
        usage();
    }
    let seed = arguments
        .first()
        .map_or(Ok(42), |value| value.parse::<u64>())
        .unwrap_or_else(|_| usage());
    let steps = arguments
        .get(1)
        .map_or(Ok(10), |value| value.parse::<u64>())
        .unwrap_or_else(|_| usage());

    let mut adjacency = vec![Vec::new(); 8];
    for (node, edges) in adjacency.iter_mut().enumerate() {
        edges.push(((node + 1) % 8, 1.0));
        edges.push(((node + 7) % 8, 1.0));
    }

    let model = GraphRandomWalk::new(adjacency)?;
    let mut simulation = Simulation::new(model, 0, seed);
    for k in 1..=steps {
        match simulation.step()? {
            Step::Fired { event, .. } => println!(
                "step {k} t={:.6} {} -> {}",
                simulation.time(),
                event.from,
                event.to
            ),
            Step::Absorbed => break,
        }
    }
    println!("steps={} time={:.6}", simulation.steps(), simulation.time());
    Ok(())
}
