//! Checkpoint and restart tests for schema version 1.

use okrates_core::reference::{GraphRandomWalk, Hop};
use okrates_core::{CheckpointError, Simulation, Step};

const N: u64 = 200;
const SEED: u64 = 7;
const SPLITS: [u64; 9] = [0, 1, 15, 16, 17, 31, 32, 33, 100];

fn ring_with(first_rate_scale: f64) -> GraphRandomWalk {
    let adjacency = (0..8)
        .map(|i| {
            let forward = 0.5 * (i + 1) as f64 * if i == 0 { first_rate_scale } else { 1.0 };
            vec![((i + 1) % 8, forward), ((i + 7) % 8, 1.0)]
        })
        .collect();
    GraphRandomWalk::new(adjacency).unwrap()
}

fn ring() -> GraphRandomWalk {
    ring_with(1.0)
}

fn advance(sim: &mut Simulation<GraphRandomWalk>, steps: u64) -> Vec<(Hop, u64)> {
    (0..steps)
        .map(|_| match sim.step().unwrap() {
            Step::Fired { event, dt } => (event, dt.to_bits()),
            Step::Absorbed => panic!("ring must not absorb"),
        })
        .collect()
}

#[test]
fn resumed_trajectory_matches_uninterrupted() {
    let mut reference = Simulation::new(ring(), 0, SEED);
    let expected = advance(&mut reference, N);

    for k in SPLITS {
        let mut sim = Simulation::new(ring(), 0, SEED);
        let mut recorded = advance(&mut sim, k);
        let bytes = sim.checkpoint();
        drop(sim);
        let mut resumed = Simulation::resume(ring(), &bytes).unwrap();
        recorded.extend(advance(&mut resumed, N - k));

        assert_eq!(recorded, expected, "split {k}");
        assert_eq!(resumed.state(), reference.state(), "split {k}");
        assert_eq!(resumed.time().to_bits(), reference.time().to_bits());
        assert_eq!(resumed.steps(), reference.steps());
        assert_eq!(resumed.checkpoint(), reference.checkpoint(), "split {k}");
    }
}

#[test]
fn resume_in_absorbed_state() {
    let model = || GraphRandomWalk::new(vec![vec![(1, 2.0)], vec![]]).unwrap();
    let mut sim = Simulation::new(model(), 0, 1);
    assert!(matches!(sim.step().unwrap(), Step::Fired { .. }));
    let bytes = sim.checkpoint();
    let mut resumed = Simulation::resume(model(), &bytes).unwrap();
    assert_eq!(resumed.step().unwrap(), Step::Absorbed);
    assert_eq!(resumed.checkpoint(), bytes);
}

#[test]
fn golden_checkpoint_bytes() {
    let model = GraphRandomWalk::new(vec![vec![(1, 1.0)], vec![(0, 3.0)]]).unwrap();
    let mut sim = Simulation::new(model, 0, 1);
    advance(&mut sim, 3);
    // Regression snapshot of schema 1, not a correctness oracle.
    assert_eq!(sim.checkpoint(), GOLDEN);
}

const GOLDEN: [u8; 193] = [
    79, 75, 82, 67, 75, 80, 84, 0, 1, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 71, 44, 54, 205, 29, 195,
    16, 64, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 12, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 25, 0, 0, 0,
    111, 107, 114, 97, 116, 101, 115, 46, 103, 114, 97, 112, 104, 95, 114, 97, 110, 100, 111, 109,
    95, 119, 97, 108, 107, 56, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
    1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 240, 63, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 8, 64, 8, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
];

fn good_bytes() -> Vec<u8> {
    let mut sim = Simulation::new(ring(), 0, SEED);
    advance(&mut sim, 5);
    sim.checkpoint()
}

fn resume_err(bytes: &[u8]) -> CheckpointError {
    match Simulation::resume(ring(), bytes) {
        Ok(_) => panic!("resume unexpectedly succeeded"),
        Err(error) => error,
    }
}

const VERSION_AT: usize = 8;
const TIME_AT: usize = 8 + 4 + 8;

#[test]
fn bad_magic_is_rejected() {
    let mut bytes = good_bytes();
    bytes[0] ^= 0xff;
    assert!(matches!(resume_err(&bytes), CheckpointError::BadMagic));
}

#[test]
fn other_version_is_rejected() {
    let mut bytes = good_bytes();
    bytes[VERSION_AT..VERSION_AT + 4].copy_from_slice(&2_u32.to_le_bytes());
    assert!(matches!(
        resume_err(&bytes),
        CheckpointError::UnsupportedVersion(2)
    ));
}

#[test]
fn truncated_input_is_rejected() {
    let mut bytes = good_bytes();
    bytes.pop();
    assert!(matches!(resume_err(&bytes), CheckpointError::Truncated));
}

#[test]
fn trailing_bytes_are_rejected() {
    let mut bytes = good_bytes();
    bytes.push(0);
    assert!(matches!(resume_err(&bytes), CheckpointError::TrailingBytes));
}

#[test]
fn invalid_time_is_rejected() {
    for time in [-1.0_f64, f64::NAN] {
        let mut bytes = good_bytes();
        bytes[TIME_AT..TIME_AT + 8].copy_from_slice(&time.to_bits().to_le_bytes());
        match resume_err(&bytes) {
            CheckpointError::InvalidTime(found) => {
                assert_eq!(found.to_bits(), time.to_bits());
            }
            other => panic!("expected InvalidTime, got {other:?}"),
        }
    }
}

#[test]
fn different_parameters_are_rejected() {
    let bytes = good_bytes();
    assert!(matches!(
        Simulation::resume(ring_with(2.0), &bytes),
        Err(CheckpointError::ParamsMismatch)
    ));
}

#[test]
fn invalid_state_is_rejected() {
    let mut bytes = good_bytes();
    let at = bytes.len() - 8;
    bytes[at..].copy_from_slice(&99_u64.to_le_bytes());
    assert!(matches!(
        resume_err(&bytes),
        CheckpointError::InvalidState(_)
    ));
}
