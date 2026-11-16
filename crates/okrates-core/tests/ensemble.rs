//! The ensemble runner returns identical results for any thread count.

use okrates_core::ensemble::thread_count;
use okrates_core::reference::GraphRandomWalk;
use okrates_core::{Replica, Simulation, run_replicas};

fn star() -> GraphRandomWalk {
    GraphRandomWalk::new(vec![
        vec![(1, 1.0), (2, 2.0), (3, 3.0), (4, 4.0), (5, 0.0)],
        vec![(0, 0.5)],
        vec![(0, 0.5)],
        vec![(0, 0.5)],
        vec![(0, 0.5)],
        vec![(0, 0.5)],
    ])
    .unwrap()
}

type Outcome = (usize, u64, u64);

fn run_in_pool(threads: usize) -> (Vec<Replica<Outcome>>, f64) {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .unwrap();
    pool.install(|| {
        assert_eq!(thread_count(), threads);
        let replicas = run_replicas(7, 64, |_, seed| {
            let mut simulation = Simulation::new(star(), 0, seed);
            simulation.run(200).unwrap();
            (
                *simulation.state(),
                simulation.time().to_bits(),
                simulation.steps(),
            )
        })
        .unwrap();
        let aggregate = replicas
            .iter()
            .map(|replica| f64::from_bits(replica.value.1))
            .sum::<f64>();
        (replicas, aggregate)
    })
}

#[test]
fn results_are_identical_for_one_and_four_threads() {
    let (one, sum_one) = run_in_pool(1);
    let (four, sum_four) = run_in_pool(4);

    assert_eq!(one.len(), 64);
    assert_eq!(four.len(), 64);
    for (index, (a, b)) in one.iter().zip(&four).enumerate() {
        assert_eq!(a, b);
        assert_eq!(a.id, index as u64);
        assert_eq!(a.seed, 7 + a.id);
    }
    assert_eq!(sum_one.to_bits(), sum_four.to_bits());
}
