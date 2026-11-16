//! Tests for the incremental sum-tree scheduler.

use okrates_core::reference::{GraphRandomWalk, Hop};
use okrates_core::{IncrementalModel, IncrementalSimulation, Simulation, Step, SumTree};

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

#[test]
fn incremental_catalog_matches_rebuild_on_star_graph() {
    let mut sim = IncrementalSimulation::new(star(), 0, 3).unwrap();
    for _ in 0..5000 {
        sim.step().unwrap();
        let model = sim.model();
        let state = sim.state();
        let rates: Vec<f64> = (0..model.slot_count(state))
            .map(|s| model.slot_rate(state, s))
            .collect();
        assert_eq!(*sim.tree(), SumTree::new(&rates));
    }
}

#[test]
fn absorbing_state_draws_nothing() {
    let model = || GraphRandomWalk::new(vec![vec![]]).unwrap();
    let mut sim = IncrementalSimulation::new(model(), 0, 99).unwrap();
    for _ in 0..3 {
        assert_eq!(sim.step().unwrap(), Step::Absorbed);
    }
    let fresh = IncrementalSimulation::new(model(), 0, 99).unwrap();
    assert_eq!(sim.checkpoint(), fresh.checkpoint());
}

fn advance(sim: &mut IncrementalSimulation<GraphRandomWalk>, steps: u64) -> Vec<(Hop, u64)> {
    (0..steps)
        .map(|_| match sim.step().unwrap() {
            Step::Fired { event, dt } => (event, dt.to_bits()),
            Step::Absorbed => panic!("star must not absorb"),
        })
        .collect()
}

#[test]
fn restart_is_bitwise() {
    const N: u64 = 300;
    let mut reference = IncrementalSimulation::new(star(), 0, 11).unwrap();
    let expected = advance(&mut reference, N);

    for split in [0_u64, 1, 15, 16, 17, 31, 32, 33, 100] {
        let mut sim = IncrementalSimulation::new(star(), 0, 11).unwrap();
        let mut recorded = advance(&mut sim, split);
        let bytes = sim.checkpoint();
        drop(sim);
        let mut resumed = IncrementalSimulation::resume(star(), &bytes).unwrap();
        recorded.extend(advance(&mut resumed, N - split));
        assert_eq!(recorded, expected, "split {split}");
        assert_eq!(
            resumed.checkpoint(),
            reference.checkpoint(),
            "split {split}"
        );
    }
}

#[test]
fn incremental_matches_direct_checkpoint_layout() {
    let direct = Simulation::new(star(), 0, 5);
    let incremental = IncrementalSimulation::new(star(), 0, 5).unwrap();
    assert_eq!(direct.checkpoint(), incremental.checkpoint());
}
