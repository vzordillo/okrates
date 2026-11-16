//! Checkpoint and restart tests for the vacancy model.

use okrates_core::reference::GraphRandomWalk;
use okrates_core::{CheckpointError, Simulation, Step};
use okrates_models::bcc::BccLattice;
use okrates_models::vacancy::{VacancyDiffusion, VacancyJump, VacancyState};

const N: u64 = 200;
const SEED: u64 = 20261117;
const SPLITS: [u64; 9] = [0, 1, 15, 16, 17, 31, 32, 33, 100];

fn model(rate: f64) -> VacancyDiffusion {
    VacancyDiffusion::new(BccLattice::new([2, 2, 2]).unwrap(), rate)
}

fn fresh() -> Simulation<VacancyDiffusion> {
    let model = model(1.0);
    let state = model.initial_state(0).unwrap();
    Simulation::new(model, state, SEED)
}

fn advance(sim: &mut Simulation<VacancyDiffusion>, steps: u64) -> Vec<(VacancyJump, u64)> {
    (0..steps)
        .map(|_| match sim.step().unwrap() {
            Step::Fired { event, dt } => (event, dt.to_bits()),
            Step::Absorbed => panic!("vacancy model must not absorb"),
        })
        .collect()
}

#[test]
fn resumed_trajectory_matches_uninterrupted() {
    let mut reference = fresh();
    let expected = advance(&mut reference, N);

    for k in SPLITS {
        let mut sim = fresh();
        let mut recorded = advance(&mut sim, k);
        let bytes = sim.checkpoint();
        drop(sim);
        let mut resumed = Simulation::resume(model(1.0), &bytes).unwrap();
        recorded.extend(advance(&mut resumed, N - k));

        assert_eq!(recorded, expected, "split {k}");
        let state: &VacancyState = resumed.state();
        assert_eq!(state, reference.state(), "split {k}");
        assert_eq!(resumed.time().to_bits(), reference.time().to_bits());
        assert_eq!(resumed.steps(), reference.steps());
        assert_eq!(resumed.checkpoint(), reference.checkpoint(), "split {k}");
    }
}

fn bytes_after(steps: u64) -> Vec<u8> {
    let mut sim = fresh();
    advance(&mut sim, steps);
    sim.checkpoint()
}

#[test]
fn different_rate_is_rejected() {
    assert!(matches!(
        Simulation::resume(model(2.0), &bytes_after(5)),
        Err(CheckpointError::ParamsMismatch)
    ));
}

#[test]
fn inconsistent_displacement_is_rejected() {
    let mut bytes = bytes_after(5);
    // The state section ends with vacancy, origin and three i64 displacements.
    let at = bytes.len() - 24;
    bytes[at..at + 8].copy_from_slice(&1000_i64.to_le_bytes());
    assert!(matches!(
        Simulation::resume(model(1.0), &bytes),
        Err(CheckpointError::InvalidState(_))
    ));
}

#[test]
fn other_model_is_rejected() {
    let walk = GraphRandomWalk::new(vec![vec![(0, 1.0)]]).unwrap();
    assert!(matches!(
        Simulation::resume(walk, &bytes_after(5)),
        Err(CheckpointError::ModelMismatch)
    ));
}
