//! Checkpoint and restart tests across VacancyDiffusion and FeCrModel energetics backends.

use okrates_core::reference::GraphRandomWalk;
use okrates_core::{CheckpointError, Checkpointable, Simulation, Step};
use okrates_models::bcc::BccLattice;
use okrates_models::fecr::{FeCrModel, PairEnergetics};
use okrates_models::fecr_martinez::Martinez2012;
use okrates_models::tabulated::TabulatedEnergetics;
use okrates_models::vacancy::VacancyDiffusion;
use okrates_schema::{BarrierRecord, BarrierSet};

const N: u64 = 120;
const SEED: u64 = 20261117;
const SPLITS: [u64; 6] = [0, 1, 15, 16, 33, 60];

fn advance<M: Checkpointable>(sim: &mut Simulation<M>, steps: u64) -> Vec<(M::Event, u64)>
where
    M::Event: Clone,
{
    (0..steps)
        .map(|_| match sim.step().unwrap() {
            Step::Fired { event, dt } => (event, dt.to_bits()),
            Step::Absorbed => panic!("model must not absorb"),
        })
        .collect()
}

fn verify_roundtrip<M>(build_model: impl Fn() -> M, initial_state: impl Fn(&M) -> M::State)
where
    M: Checkpointable,
    M::Event: Clone + PartialEq + std::fmt::Debug,
    M::State: PartialEq + std::fmt::Debug,
{
    let m0 = build_model();
    let s0 = initial_state(&m0);
    let mut reference = Simulation::new(m0, s0, SEED);
    let expected = advance(&mut reference, N);

    for k in SPLITS {
        let m = build_model();
        let s = initial_state(&m);
        let mut sim = Simulation::new(m, s, SEED);
        let mut recorded = advance(&mut sim, k);
        let bytes = sim.checkpoint();
        drop(sim);

        let mut resumed = Simulation::resume(build_model(), &bytes).unwrap();
        recorded.extend(advance(&mut resumed, N - k));

        assert_eq!(recorded, expected, "split {k}");
        assert_eq!(resumed.state(), reference.state(), "split {k}");
        assert_eq!(resumed.time().to_bits(), reference.time().to_bits());
        assert_eq!(
            resumed.clock().compensation().to_bits(),
            reference.clock().compensation().to_bits()
        );
        assert_eq!(resumed.steps(), reference.steps());
        assert_eq!(resumed.checkpoint(), reference.checkpoint(), "split {k}");
    }
}

fn vacancy_model(rate: f64) -> VacancyDiffusion {
    VacancyDiffusion::new(BccLattice::new([2, 2, 2]).unwrap(), rate)
}

#[test]
fn vacancy_resumed_trajectory_matches_uninterrupted() {
    verify_roundtrip(|| vacancy_model(1.0), |m| m.initial_state(0).unwrap());
}

#[test]
fn fecr_pair_resumed_trajectory_matches_uninterrupted() {
    verify_roundtrip(
        || {
            FeCrModel::new(
                BccLattice::new([3, 3, 3]).unwrap(),
                PairEnergetics::illustrative(),
                1e13,
                800.0,
            )
            .unwrap()
        },
        |m| m.random_state(11, 77).unwrap(),
    );
}

#[test]
fn fecr_martinez_resumed_trajectory_matches_uninterrupted() {
    verify_roundtrip(
        || {
            FeCrModel::new(
                BccLattice::new([3, 3, 3]).unwrap(),
                Martinez2012::new(800.0).unwrap(),
                1e13,
                800.0,
            )
            .unwrap()
        },
        |m| m.random_state(11, 77).unwrap(),
    );
}

#[test]
fn fecr_tabulated_resumed_trajectory_matches_uninterrupted() {
    let make_tabulated = || {
        let set = BarrierSet::new(
            "tab",
            1e-9,
            vec![
                BarrierRecord {
                    id: "fe".to_string(),
                    migrating_species: "Fe".to_string(),
                    env_key: "*".to_string(),
                    forward_barrier_ev: 0.65,
                    reverse_barrier_ev: 0.65,
                    delta_energy_ev: 0.0,
                    attempt_frequency_hz: 1e13,
                    reverse_attempt_frequency_hz: None,
                    barrier_std_ev: None,
                    temp_range_k: None,
                    provenance: None,
                },
                BarrierRecord {
                    id: "cr".to_string(),
                    migrating_species: "Cr".to_string(),
                    env_key: "*".to_string(),
                    forward_barrier_ev: 0.58,
                    reverse_barrier_ev: 0.58,
                    delta_energy_ev: 0.0,
                    attempt_frequency_hz: 1e13,
                    reverse_attempt_frequency_hz: None,
                    barrier_std_ev: None,
                    temp_range_k: None,
                    provenance: None,
                },
            ],
        )
        .unwrap();
        FeCrModel::new(
            BccLattice::new([3, 3, 3]).unwrap(),
            TabulatedEnergetics::new(set).unwrap(),
            1e13,
            800.0,
        )
        .unwrap()
    };
    verify_roundtrip(make_tabulated, |m| m.random_state(11, 77).unwrap());
}

#[test]
fn checkpoint_rejects_mismatched_params_and_corrupted_state() {
    let m = vacancy_model(1.0);
    let mut sim = Simulation::new(m, vacancy_model(1.0).initial_state(0).unwrap(), SEED);
    advance(&mut sim, 5);
    let bytes = sim.checkpoint();

    assert!(matches!(
        Simulation::resume(vacancy_model(2.0), &bytes),
        Err(CheckpointError::ParamsMismatch)
    ));

    let walk = GraphRandomWalk::new(vec![vec![(0, 1.0)]]).unwrap();
    assert!(matches!(
        Simulation::resume(walk, &bytes),
        Err(CheckpointError::ModelMismatch)
    ));

    let mut bad_disp = bytes.clone();
    let at = bad_disp.len() - 24;
    bad_disp[at..at + 8].copy_from_slice(&1000_i64.to_le_bytes());
    assert!(matches!(
        Simulation::resume(vacancy_model(1.0), &bad_disp),
        Err(CheckpointError::InvalidState(_))
    ));

    let mut max_disp = bytes.clone();
    max_disp[at..at + 8].copy_from_slice(&i64::MAX.to_le_bytes());
    assert!(matches!(
        Simulation::resume(vacancy_model(1.0), &max_disp),
        Err(CheckpointError::InvalidState(_))
    ));

    // Shifting one entity's z-displacement by one full period (+6) preserves wrapped position
    // while breaking zero total displacement (`FeCrError::DisplacementSum`).
    let fecr = FeCrModel::new(
        BccLattice::new([3, 3, 3]).unwrap(),
        PairEnergetics::illustrative(),
        1e13,
        800.0,
    )
    .unwrap();
    let mut fecr_sim = Simulation::new(fecr.clone(), fecr.random_state(11, 7).unwrap(), SEED);
    advance(&mut fecr_sim, 5);
    let fecr_valid = fecr_sim.checkpoint();

    let mut fecr_bytes = fecr_valid.clone();
    let end = fecr_bytes.len() - 8;
    let orig_dz = i64::from_le_bytes(fecr_bytes[end..].try_into().unwrap());
    fecr_bytes[end..].copy_from_slice(&(orig_dz + 6).to_le_bytes());
    match Simulation::resume(fecr.clone(), &fecr_bytes) {
        Err(CheckpointError::InvalidState(err)) => {
            assert!(
                err.to_string().contains("displacements do not sum to zero"),
                "got: {err}"
            );
        }
        other => panic!(
            "expected InvalidState(DisplacementSum), got {}",
            other.is_ok()
        ),
    }

    let mut fecr_extreme = fecr_valid;
    fecr_extreme[end - 24..end - 16].copy_from_slice(&i64::MAX.to_le_bytes());
    fecr_extreme[end..].copy_from_slice(&i64::MAX.to_le_bytes());
    assert!(matches!(
        Simulation::resume(fecr, &fecr_extreme),
        Err(CheckpointError::InvalidState(_))
    ));
}
