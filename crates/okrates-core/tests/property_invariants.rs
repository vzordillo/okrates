//! Deterministic property and hostile-input fuzz invariants for SumTree, CloneRollback, and checkpoints.

use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;

use okrates_core::reference::{
    CoupledGraphWalk, EdgeRateTable, GraphRandomWalk, GraphTopology, Hop,
};
use okrates_core::{
    AbsorptionPolicy, CatalogBackend, CheckpointError, Checkpointable, CloneRollback,
    DirectCatalog, IncrementalModel, IncrementalSimulation, IndependentSnapshot, KmcRng, Model,
    ReversibleModel, SampleGrid, SamplingError, SimError, Simulation, Step, SumTree,
    sample_trajectory,
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
        emit(false, 1.0);
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

#[derive(Clone, Copy, Debug)]
struct StepOneFailModel;

impl Model for StepOneFailModel {
    type State = u64;
    type Event = u64;
    type Error = PartialMutError;

    fn enumerate_events(&self, state: &u64, emit: &mut dyn FnMut(u64, f64)) {
        emit(*state + 1, 2.0);
        emit(*state + 2, 3.0);
    }

    fn apply(&self, state: &mut u64, &next: &u64) -> Result<(), PartialMutError> {
        if *state >= 1 {
            return Err(PartialMutError);
        }
        *state = next;
        Ok(())
    }
}

impl Checkpointable for StepOneFailModel {
    const MODEL_ID: &'static str = "test.step_one_fail";

    fn encode_params(&self, _out: &mut Vec<u8>) {}

    fn encode_state(&self, state: &u64, out: &mut Vec<u8>) {
        out.extend_from_slice(&state.to_le_bytes());
    }

    fn decode_state(&self, bytes: &[u8]) -> Result<u64, Box<dyn Error + Send + Sync>> {
        let arr: [u8; 8] = bytes.try_into()?;
        Ok(u64::from_le_bytes(arr))
    }
}

#[test]
fn independent_snapshot_and_clone_rollback_preserve_trajectories_and_error_atomicity() {
    let mut map = BTreeMap::from([(String::from("a"), vec![1_u64, 2])]);
    let mut set = BTreeSet::from([10_i64, 20]);
    let mut boxed: Box<[u64]> = vec![7, 8].into_boxed_slice();
    let composite = (Some([true, false]), map.clone(), set.clone(), boxed.clone());
    let snap = composite.snapshot();
    map.get_mut("a").unwrap().push(3);
    set.insert(30);
    boxed[0] = 99;
    assert_eq!(snap.1["a"], vec![1, 2]);
    assert_eq!(snap.2.len(), 2);
    assert_eq!(&*snap.3, &[7, 8]);

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

    let mut direct: DirectCatalog<bool> = DirectCatalog::build(&fallible, &buf).unwrap();
    assert_eq!(direct.prepare_total(&fallible, &buf).unwrap(), Some(1.0));
    assert!(matches!(
        direct.apply_and_update(&fallible, &mut buf, 99),
        Err(SimError::InvalidSlot {
            slot: 99,
            slot_count: 1
        })
    ));
    assert_eq!(buf, vec![1, 2, 3]);
    assert!(matches!(
        direct.apply_and_update(&fallible, &mut buf, 0),
        Err(SimError::Model(PartialMutError))
    ));
    assert_eq!(buf, vec![1, 2, 3]);
    assert_eq!(
        <DirectCatalog<bool> as CatalogBackend<CloneRollback<PartialMutatingModel>>>::select(
            &direct, 0.5, 1.0
        ),
        0
    );

    let mut a = Simulation::new(walk.clone(), 0, 55);
    let mut b = Simulation::new(wrapped.clone(), 0, 55);
    for _ in 0..100 {
        assert_eq!(a.step().unwrap(), b.step().unwrap());
        assert_eq!(a.state(), b.state());
        assert_eq!(a.time().to_bits(), b.time().to_bits());
    }

    let mut inc_ref = IncrementalSimulation::new(walk.clone(), 0, 77).unwrap();
    let mut inc_clone = IncrementalSimulation::new(wrapped.clone(), 0, 77).unwrap();
    for _ in 0..25 {
        assert_eq!(inc_ref.step().unwrap(), inc_clone.step().unwrap());
    }
    let ckpt = inc_clone.checkpoint();
    let mut resumed = IncrementalSimulation::resume(wrapped, &ckpt).unwrap();
    for _ in 0..25 {
        assert_eq!(inc_ref.step().unwrap(), resumed.step().unwrap());
    }
    assert_eq!(inc_ref.time().to_bits(), resumed.time().to_bits());

    assert!(walk.slot_rate(&0, 99).is_nan());
    let topo = GraphTopology::new(vec![vec![1], vec![0]]).unwrap();
    let table = EdgeRateTable::new(&topo, vec![1.0, 2.0]).unwrap();
    let coupled = CoupledGraphWalk::new(topo, table);
    assert!(coupled.slot_rate(&0, 2).is_nan());
}

#[test]
fn direct_step_apply_error_and_step_counter_overflow_roll_back_atomically() {
    let mut sim = Simulation::new(StepOneFailModel, 0, 19);
    assert!(matches!(sim.step().unwrap(), Step::Fired { .. }));
    let state_after_first = *sim.state();
    let clock_after_first = sim.clock();
    let ckpt_after_first = sim.checkpoint();
    assert!(matches!(sim.step(), Err(SimError::Model(PartialMutError))));
    assert_eq!(*sim.state(), state_after_first);
    assert_eq!(sim.steps(), 1);
    assert_eq!(sim.clock(), clock_after_first);
    assert_eq!(sim.checkpoint(), ckpt_after_first);
    assert!(sim.engine().catalog().allocated_bytes() > 0);

    let walk = GraphRandomWalk::new(vec![vec![(1, 2.0)], vec![(0, 2.0)]]).unwrap();
    let mut max_ckpt = Simulation::new(walk.clone(), 0, 7).checkpoint();
    max_ckpt[12..20].copy_from_slice(&u64::MAX.to_le_bytes());

    let mut direct_max = Simulation::resume(walk.clone(), &max_ckpt).unwrap();
    assert_eq!(direct_max.steps(), u64::MAX);
    let before_direct = direct_max.checkpoint();
    assert!(matches!(
        direct_max.step(),
        Err(SimError::StepCounterOverflow)
    ));
    assert_eq!(direct_max.steps(), u64::MAX);
    assert_eq!(*direct_max.state(), 0);
    assert_eq!(direct_max.checkpoint(), before_direct);

    let mut observe_calls = 0_usize;
    let grid = SampleGrid::from_origin(10.0, 1).unwrap();
    assert!(matches!(
        sample_trajectory(
            &mut direct_max,
            &grid,
            None,
            AbsorptionPolicy::Error,
            |_, &s| {
                observe_calls += 1;
                s
            }
        ),
        Err(SamplingError::Sim {
            steps: 0,
            source: SimError::StepCounterOverflow,
            ..
        })
    ));
    assert_eq!(observe_calls, 0);
    assert_eq!(direct_max.checkpoint(), before_direct);

    let mut inc_max = IncrementalSimulation::resume(walk, &max_ckpt).unwrap();
    let before_inc = inc_max.checkpoint();
    let tree_before = inc_max.tree().clone();
    assert!(matches!(inc_max.step(), Err(SimError::StepCounterOverflow)));
    assert_eq!(inc_max.steps(), u64::MAX);
    assert_eq!(*inc_max.tree(), tree_before);
    assert_eq!(inc_max.checkpoint(), before_inc);

    let absorbed_walk = GraphRandomWalk::new(vec![vec![]]).unwrap();
    let mut abs_ckpt = Simulation::new(absorbed_walk.clone(), 0, 7).checkpoint();
    abs_ckpt[12..20].copy_from_slice(&u64::MAX.to_le_bytes());
    let mut abs_max = Simulation::resume(absorbed_walk, &abs_ckpt).unwrap();
    assert_eq!(abs_max.step().unwrap(), Step::Absorbed);
    assert_eq!(abs_max.steps(), u64::MAX);
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
