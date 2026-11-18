//! Tests for the incremental sum-tree scheduler and transactional rollback guarantees.

use std::error::Error;
use std::fmt;

use okrates_core::reference::{GraphRandomWalk, Hop};
use okrates_core::{
    Checkpointable, IncrementalModel, IncrementalSimulation, Model, ReversibleModel, SimError,
    Simulation, Step, SumTree,
};

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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FaultMode {
    ApplyError,
    SlotOutOfRange,
    MidUpdateNaNRate,
    SlotCountMutated,
    ClockOverflow,
}

#[derive(Debug, PartialEq, Eq)]
struct FaultError;

impl fmt::Display for FaultError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "injected model fault")
    }
}

impl Error for FaultError {}

#[derive(Clone)]
struct FaultyIncremental {
    mode: FaultMode,
}

impl Model for FaultyIncremental {
    type State = [u64; 2];
    type Event = usize;
    type Error = FaultError;

    fn enumerate_events(&self, state: &Self::State, emit: &mut dyn FnMut(Self::Event, f64)) {
        for slot in 0..3 {
            emit(slot, self.slot_rate(state, slot));
        }
    }

    fn apply(&self, state: &mut Self::State, event: &Self::Event) -> Result<(), Self::Error> {
        self.apply_reversible(state, event).map(|_| ())
    }
}

impl ReversibleModel for FaultyIncremental {
    type Undo = [u64; 2];

    fn apply_reversible(
        &self,
        state: &mut Self::State,
        event: &Self::Event,
    ) -> Result<Self::Undo, Self::Error> {
        if self.mode == FaultMode::ApplyError {
            return Err(FaultError);
        }
        let prev = *state;
        state[0] += 1;
        state[1] = *event as u64 + 10;
        Ok(prev)
    }

    fn undo(&self, state: &mut Self::State, undo: Self::Undo) {
        *state = undo;
    }
}

impl IncrementalModel for FaultyIncremental {
    fn slot_count(&self, state: &Self::State) -> usize {
        if self.mode == FaultMode::SlotCountMutated && state[0] > 0 {
            4
        } else {
            3
        }
    }

    fn slot_rate(&self, state: &Self::State, slot: usize) -> f64 {
        if self.mode == FaultMode::ClockOverflow {
            return if slot == 0 { 1e-308 } else { 0.0 };
        }
        if self.mode == FaultMode::MidUpdateNaNRate && state[0] > 0 && slot == 2 {
            return f64::NAN;
        }
        (slot + 1) as f64 + state[0] as f64
    }

    fn slot_event(&self, _state: &Self::State, slot: usize) -> Self::Event {
        slot
    }

    fn affected_slots(&self, _state: &Self::State, _event: &Self::Event, out: &mut Vec<usize>) {
        // Slot 0 and 1 are valid first, then slot 2 or out-of-range slot 99 triggers rollback.
        out.push(0);
        out.push(1);
        if self.mode == FaultMode::SlotOutOfRange {
            out.push(99);
        } else {
            out.push(2);
        }
    }
}

impl Checkpointable for FaultyIncremental {
    const MODEL_ID: &'static str = "test.faulty_incremental";

    fn encode_params(&self, out: &mut Vec<u8>) {
        out.push(self.mode as u8);
    }

    fn encode_state(&self, state: &Self::State, out: &mut Vec<u8>) {
        out.extend_from_slice(&state[0].to_le_bytes());
        out.extend_from_slice(&state[1].to_le_bytes());
    }

    fn decode_state(&self, bytes: &[u8]) -> Result<Self::State, Box<dyn Error + Send + Sync>> {
        let a = u64::from_le_bytes(bytes[0..8].try_into()?);
        let b = u64::from_le_bytes(bytes[8..16].try_into()?);
        Ok([a, b])
    }
}

#[test]
fn failed_incremental_steps_roll_back_state_tree_clock_and_rng_atomically() {
    for mode in [
        FaultMode::ApplyError,
        FaultMode::SlotOutOfRange,
        FaultMode::MidUpdateNaNRate,
        FaultMode::SlotCountMutated,
        FaultMode::ClockOverflow,
    ] {
        let mut sim = IncrementalSimulation::new(FaultyIncremental { mode }, [0, 7], 42).unwrap();
        let state_before = *sim.state();
        let tree_before = sim.tree().clone();
        let ckpt_before = sim.checkpoint();

        let err = sim.step().unwrap_err();
        match mode {
            FaultMode::ApplyError => assert!(matches!(err, SimError::Model(FaultError))),
            FaultMode::SlotOutOfRange => assert!(matches!(
                err,
                SimError::InvalidSlot {
                    slot: 99,
                    slot_count: 3
                }
            )),
            FaultMode::MidUpdateNaNRate => {
                assert!(matches!(err, SimError::InvalidRate { index: 2, .. }));
            }
            FaultMode::SlotCountMutated => assert!(matches!(
                err,
                SimError::SlotCountChanged {
                    expected: 3,
                    found: 4
                }
            )),
            FaultMode::ClockOverflow => assert!(matches!(err, SimError::Clock(_))),
        }

        assert_eq!(*sim.state(), state_before, "mode={mode:?}");
        assert_eq!(*sim.tree(), tree_before, "mode={mode:?}");
        assert_eq!(sim.steps(), 0, "mode={mode:?}");
        assert_eq!(sim.time(), 0.0, "mode={mode:?}");
        assert_eq!(sim.checkpoint(), ckpt_before, "mode={mode:?}");
    }
}
