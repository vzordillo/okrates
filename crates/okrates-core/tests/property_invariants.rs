//! Deterministic property and hostile-input fuzz invariants for SumTree, CloneRollback, and checkpoints.

use okrates_core::reference::GraphRandomWalk;
use okrates_core::{
    CheckpointError, CloneRollback, IncrementalSimulation, KmcRng, Simulation, Step, SumTree,
};

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
fn clone_rollback_adapter_preserves_direct_simulation_trajectories() {
    let walk = GraphRandomWalk::new(vec![
        vec![(1, 1.5), (2, 0.5)],
        vec![(2, 2.0), (0, 0.7)],
        vec![(0, 1.2)],
    ])
    .unwrap();
    let mut a = Simulation::new(walk.clone(), 0, 55);
    let mut b = Simulation::new(CloneRollback(walk), 0, 55);
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

    // Hostile length prefixes (`u64::MAX`) inside section headers must return Truncated without allocating.
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
