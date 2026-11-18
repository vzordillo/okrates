//! Deterministic property and hostile-input fuzz invariants for SumTree, CloneRollback, and checkpoints.

use std::error::Error;
use std::fmt;

use okrates_core::reference::{GraphRandomWalk, Hop};
use okrates_core::{
    CheckpointError, Checkpointable, CloneRollback, IncrementalSimulation, KmcRng, Model,
    ReversibleModel, Simulation, Step, SumTree,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PartialMutError;

impl fmt::Display for PartialMutError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "simulated failure after partial mutation")
    }
}

impl Error for PartialMutError {}

struct PartialMutatingModel;

impl Model for PartialMutatingModel {
    type State = Vec<u64>;
    type Event = bool;
    type Error = PartialMutError;

    fn enumerate_events(&self, _state: &Self::State, emit: &mut dyn FnMut(Self::Event, f64)) {
        emit(true, 1.0);
    }

    fn apply(&self, state: &mut Self::State, &succeed: &Self::Event) -> Result<(), Self::Error> {
        state.push(99);
        if succeed {
            Ok(())
        } else {
            Err(PartialMutError)
        }
    }
}

#[test]
fn sum_tree_property_invariants_across_sizes_and_random_mutations() {
    let mut rng = KmcRng::from_seed(0x5EED_CAFE);
    for len in [1_usize, 2, 3, 4, 7, 8, 9, 15, 16, 17, 31, 32, 63, 65, 129] {
        let mut reference = vec![0.0; len];
        for r in &mut reference {
            *r = if rng.index(4) == 0 {
                0.0
            } else {
                rng.open01() * 10.0
            };
        }
        let mut tree = SumTree::new(&reference);

        for _ in 0..80 {
            let slot = rng.index(len);
            let new_rate = if rng.index(3) == 0 {
                0.0
            } else {
                rng.open01() * 25.0
            };
            reference[slot] = new_rate;
            tree.try_set(slot, new_rate).unwrap();

            assert_eq!(tree, SumTree::new(&reference));
            if tree.total() > 0.0 {
                for _ in 0..8 {
                    let u = rng.open01();
                    let chosen = tree.select(u);
                    assert!(chosen < len);
                    assert!(
                        reference[chosen] > 0.0,
                        "selected zero-rate slot {chosen} for u={u}"
                    );
                }
            }
        }
    }
}

#[test]
fn clone_rollback_adapter_preserves_trajectories_and_restores_on_undo_or_err() {
    let walk = GraphRandomWalk::new(vec![
        vec![(1, 1.5), (2, 0.5)],
        vec![(2, 2.0), (0, 0.7)],
        vec![(0, 1.2)],
    ])
    .unwrap();
    let wrapped = CloneRollback(walk.clone());
    let mut state = 0_usize;
    let undo = wrapped
        .apply_reversible(&mut state, &Hop { from: 0, to: 2 })
        .unwrap();
    assert_eq!(state, 2);
    wrapped.undo(&mut state, undo);
    assert_eq!(state, 0);

    let fallible = CloneRollback(PartialMutatingModel);
    let mut buf = vec![1_u64, 2, 3];
    assert_eq!(
        fallible.apply_reversible(&mut buf, &false),
        Err(PartialMutError)
    );
    assert_eq!(buf, vec![1, 2, 3]);

    let mut a = Simulation::new(walk, 0, 55);
    let mut b = Simulation::new(wrapped, 0, 55);
    for _ in 0..100 {
        assert_eq!(a.step().unwrap(), b.step().unwrap());
        assert_eq!(a.state(), b.state());
        assert_eq!(a.time().to_bits(), b.time().to_bits());
    }
}

#[test]
fn checkpoint_decoder_never_panics_or_ooms_on_mutated_and_hostile_bytes() {
    let model = GraphRandomWalk::new(vec![
        vec![(1, 1.0), (2, 2.0)],
        vec![(0, 1.5), (2, 0.5)],
        vec![(0, 0.8), (1, 1.2)],
    ])
    .unwrap();
    let mut sim = Simulation::new(model.clone(), 0, 17);
    for _ in 0..15 {
        assert!(matches!(sim.step().unwrap(), Step::Fired { .. }));
    }
    let valid = sim.checkpoint();

    // Every truncation prefix must fail cleanly with BadMagic, Truncated, or Params/State error.
    for cut in 0..valid.len() {
        assert!(Simulation::resume(model.clone(), &valid[..cut]).is_err());
        assert!(IncrementalSimulation::resume(model.clone(), &valid[..cut]).is_err());
    }

    // Hostile length prefixes (`u32::MAX` / `u64::MAX`) inside section headers must return Truncated without allocating.
    let mut bad_id_len = valid.clone();
    bad_id_len[92..96].copy_from_slice(&u32::MAX.to_le_bytes());
    assert!(matches!(
        Simulation::resume(model.clone(), &bad_id_len),
        Err(CheckpointError::Truncated)
    ));
    assert!(matches!(
        IncrementalSimulation::resume(model.clone(), &bad_id_len),
        Err(CheckpointError::Truncated)
    ));

    let params_len_offset = 96 + GraphRandomWalk::MODEL_ID.len();
    let params_len = usize::try_from(u64::from_le_bytes(
        valid[params_len_offset..params_len_offset + 8]
            .try_into()
            .unwrap(),
    ))
    .unwrap();
    let state_len_offset = params_len_offset + 8 + params_len;
    for section_offset in [params_len_offset, state_len_offset] {
        let mut bad_section_len = valid.clone();
        bad_section_len[section_offset..section_offset + 8]
            .copy_from_slice(&u64::MAX.to_le_bytes());
        assert!(matches!(
            Simulation::resume(model.clone(), &bad_section_len),
            Err(CheckpointError::Truncated)
        ));
        assert!(matches!(
            IncrementalSimulation::resume(model.clone(), &bad_section_len),
            Err(CheckpointError::Truncated)
        ));
    }
    for offset in 0..valid.len().saturating_sub(8) {
        let mut hostile = valid.clone();
        hostile[offset..offset + 8].copy_from_slice(&u64::MAX.to_le_bytes());
        let _ = Simulation::resume(model.clone(), &hostile);
        let _ = IncrementalSimulation::resume(model.clone(), &hostile);
    }

    // Deterministic fuzzing with random byte mutations and random buffers.
    let mut rng = KmcRng::from_seed(2026);
    for _ in 0..500 {
        let mut mutated = valid.clone();
        let flips = 1 + rng.index(4);
        for _ in 0..flips {
            let pos = rng.index(mutated.len());
            mutated[pos] ^= (1 + rng.index(255)) as u8;
        }
        if let Ok(mut resumed) = Simulation::resume(model.clone(), &mutated) {
            // Any accepted checkpoint must still satisfy valid node & clock invariants and step cleanly.
            assert!(*resumed.state() < model.node_count());
            assert!(resumed.step().is_ok());
        }
        if let Ok(mut resumed) = IncrementalSimulation::resume(model.clone(), &mutated) {
            assert!(*resumed.state() < model.node_count());
            assert!(resumed.step().is_ok());
        }
    }

    // Directfailed step on Simulation also leaves checkpoint untouched.
    let bad = GraphRandomWalk::new(vec![vec![(1, 1.0), (1, -2.0)], vec![(0, 1.0)]]).unwrap();
    let mut bad_sim = Simulation::new(bad, 0, 3);
    let before = bad_sim.checkpoint();
    assert!(bad_sim.step().is_err());
    assert_eq!(bad_sim.checkpoint(), before);
    assert!(matches!(
        Simulation::resume(model, b"short"),
        Err(CheckpointError::Truncated)
    ));
}
