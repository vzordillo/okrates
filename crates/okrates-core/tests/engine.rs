use okrates_core::reference::{GraphError, GraphRandomWalk, Hop};
use okrates_core::{Rate, SimError, Simulation, Step};

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

fn trajectory(seed: u64) -> Vec<(Hop, u64)> {
    let mut simulation = Simulation::new(star(), 0, seed);
    let mut trajectory = Vec::with_capacity(1000);
    for _ in 0..1000 {
        match simulation.step().unwrap() {
            Step::Fired { event, dt } => trajectory.push((event, dt.to_bits())),
            Step::Absorbed => panic!("star graph unexpectedly absorbed"),
        }
    }
    trajectory
}

#[test]
fn same_seed_same_trajectory() {
    assert_eq!(trajectory(7), trajectory(7));
    assert_ne!(trajectory(7), trajectory(8));
}

#[test]
fn absorbing_state() {
    let model = GraphRandomWalk::new(vec![vec![(1, 2.0)], vec![]]).unwrap();
    let mut simulation = Simulation::new(model, 0, 1);
    match simulation.step().unwrap() {
        Step::Fired { event, dt } => {
            assert_eq!(event, Hop { from: 0, to: 1 });
            assert!(dt > 0.0);
        }
        Step::Absorbed => panic!("expected the only event to fire"),
    }
    let t1 = simulation.time();
    assert!(t1 > 0.0);
    assert_eq!(simulation.step().unwrap(), Step::Absorbed);
    assert_eq!(simulation.step().unwrap(), Step::Absorbed);
    assert_eq!(simulation.time(), t1);
    assert_eq!(simulation.steps(), 1);
    assert_eq!(
        simulation.run(5).unwrap(),
        okrates_core::RunSummary {
            steps: 0,
            time: t1,
            absorbed: true,
        }
    );
}

#[test]
fn zero_rate_edges_are_absorbing() {
    let model = GraphRandomWalk::new(vec![vec![(0, 0.0), (0, -0.0)]]).unwrap();
    let mut simulation = Simulation::new(model, 0, 1);
    assert_eq!(simulation.step().unwrap(), Step::Absorbed);
}

#[test]
fn invalid_rate_rejected() {
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -1.0] {
        let model = GraphRandomWalk::new(vec![vec![(1, 1.0), (2, bad)], vec![], vec![]]).unwrap();
        let mut simulation = Simulation::new(model, 0, 3);
        match simulation.step().unwrap_err() {
            SimError::InvalidRate { index, source } => {
                assert_eq!(index, 1);
                assert_eq!(source, Rate::new(bad).unwrap_err());
            }
            error => panic!("expected InvalidRate, got {error:?}"),
        }
        assert_eq!(simulation.time(), 0.0);
        assert_eq!(*simulation.state(), 0);
        assert_eq!(simulation.steps(), 0);
    }
}

#[test]
fn total_rate_overflow_rejected() {
    let model =
        GraphRandomWalk::new(vec![vec![(1, f64::MAX), (2, f64::MAX)], vec![], vec![]]).unwrap();
    let mut simulation = Simulation::new(model, 0, 1);
    assert!(matches!(
        simulation.step(),
        Err(SimError::TotalRateNotFinite)
    ));
}

#[test]
fn graph_rejects_out_of_range_target() {
    assert_eq!(
        GraphRandomWalk::new(vec![vec![(2, 1.0)], vec![]]).unwrap_err(),
        GraphError::TargetOutOfRange {
            from: 0,
            to: 2,
            nodes: 2,
        }
    );
}
