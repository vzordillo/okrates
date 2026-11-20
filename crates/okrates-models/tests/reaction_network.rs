use okrates_core::testing::{chi2_statistic, chi2_upper_quantile, mat_exp, solve_stationary_ctmc};
use okrates_core::{
    AbsorptionPolicy, CheckpointError, Checkpointable, IncrementalModel, IncrementalSimulation,
    Model, ReversibleModel, SampleGrid, Simulation, Step, StreamRole, run_ensemble,
    sample_trajectory, verify_incremental_step_parity,
};
use okrates_models::reaction_network::{
    ReactionChannel, ReactionError, ReactionEvent, ReactionNetwork, ReactionState,
    StoichiometryRole,
};

const T1_KF: f64 = 0.1;
const T1_KR: f64 = 0.4;
const T1_NA_INIT: u64 = 8;

fn dimerization_model() -> (ReactionNetwork, ReactionState) {
    let channels = vec![
        ReactionChannel::new(vec![(0, 2)], vec![(1, 1)], T1_KF).with_reverse(1),
        ReactionChannel::new(vec![(1, 1)], vec![(0, 2)], T1_KR).with_reverse(0),
    ];
    let model = ReactionNetwork::new(2, channels)
        .unwrap()
        .with_conservation_laws(vec![vec![1, 2]])
        .unwrap();
    let init = model.initial_state(vec![T1_NA_INIT, 0]).unwrap();
    (model, init)
}

fn michaelis_menten_model() -> (ReactionNetwork, ReactionState) {
    // Species: 0 = E, 1 = S, 2 = ES, 3 = P
    let channels = vec![
        ReactionChannel::new(vec![(0, 1), (1, 1)], vec![(2, 1)], 0.02).with_reverse(1),
        ReactionChannel::new(vec![(2, 1)], vec![(0, 1), (1, 1)], 0.35).with_reverse(0),
        ReactionChannel::new(vec![(2, 1)], vec![(0, 1), (3, 1)], 0.25),
    ];
    let model = ReactionNetwork::new(4, channels)
        .unwrap()
        .with_conservation_laws(vec![vec![1, 0, 1, 0], vec![0, 1, 1, 1]])
        .unwrap();
    let init = model.initial_state(vec![10, 40, 2, 0]).unwrap();
    (model, init)
}

fn build_dimerization_rates(model: &ReactionNetwork) -> Vec<Vec<f64>> {
    // Macrostate m = N_B in 0..=4; N_A = 8 - 2*m.
    let n_states = (T1_NA_INIT / 2 + 1) as usize;
    let mut rates = vec![vec![0.0; n_states]; n_states];
    for m in 0..n_states {
        let n_b = m as u64;
        let n_a = T1_NA_INIT - 2 * n_b;
        let state = model.initial_state(vec![n_a, n_b]).unwrap();
        let fwd_analytical = if n_a >= 2 {
            T1_KF * ((n_a * (n_a - 1)) / 2) as f64
        } else {
            0.0
        };
        let rev_analytical = T1_KR * n_b as f64;
        assert_eq!(
            model.propensity(&state, 0).to_bits(),
            fwd_analytical.to_bits()
        );
        assert_eq!(
            model.propensity(&state, 1).to_bits(),
            rev_analytical.to_bits()
        );
        if m + 1 < n_states {
            rates[m][m + 1] = fwd_analytical;
        }
        if m > 0 {
            rates[m][m - 1] = rev_analytical;
        }
    }
    rates
}

fn scaled_generator(rates: &[Vec<f64>], t: f64) -> Vec<Vec<f64>> {
    let n = rates.len();
    let mut q = vec![vec![0.0; n]; n];
    for i in 0..n {
        let mut row_sum = 0.0;
        for j in 0..n {
            if i != j {
                let entry = rates[i][j] * t;
                q[i][j] = entry;
                row_sum += entry;
            }
        }
        q[i][i] = -row_sum;
    }
    q
}

#[test]
fn t1_reversible_dimerization_matches_exact_ctmc_stationary_and_transient_oracles() {
    let (model, init) = dimerization_model();
    let rates = build_dimerization_rates(&model);
    let n_states = rates.len();

    let t_transient = 0.5_f64;
    let t_stationary = 16.0_f64;
    let pi = solve_stationary_ctmc(&rates);
    let p_transient = mat_exp(&scaled_generator(&rates, t_transient))[0].clone();
    let p_eq = mat_exp(&scaled_generator(&rates, t_stationary))[0].clone();

    // Verify relaxation and genuine kinetic separation between transient and stationary regimes.
    let max_eq_diff = p_eq
        .iter()
        .zip(&pi)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        max_eq_diff < 1e-6,
        "t={t_stationary} should relax to stationary distribution; max diff={max_eq_diff:e}"
    );
    assert!(
        (p_transient[0] - pi[0]).abs() > 0.15,
        "transient t={t_transient} should differ strongly from stationary distribution"
    );

    let grid = SampleGrid::windowed(t_transient, t_stationary, 2).unwrap();
    let replicas = 20_000_u64;
    for (idx, p) in p_transient.iter().enumerate() {
        assert!(
            p * (replicas as f64) >= 5.0,
            "transient expected count for macrostate {idx} is too small for chi2: {}",
            p * (replicas as f64)
        );
    }
    for (idx, p) in pi.iter().enumerate() {
        assert!(
            p * (replicas as f64) >= 5.0,
            "stationary expected count for macrostate {idx} is too small for chi2: {}",
            p * (replicas as f64)
        );
    }

    let direct_samples = run_ensemble(10_001, replicas, |ctx| {
        let mut sim =
            Simulation::with_rng(model.clone(), init.clone(), ctx.rng(StreamRole::Dynamics));
        let traj = sample_trajectory(&mut sim, &grid, None, AbsorptionPolicy::Error, |m, s| {
            m.check_conservation(&init, s).unwrap();
            s.counts()[1] as usize
        })
        .unwrap();
        [traj.values[0], traj.values[1]]
    });

    let inc_samples = run_ensemble(20_001, replicas, |ctx| {
        let mut sim = IncrementalSimulation::with_rng(
            model.clone(),
            init.clone(),
            ctx.rng(StreamRole::Dynamics),
        )
        .unwrap();
        let traj = sample_trajectory(&mut sim, &grid, None, AbsorptionPolicy::Error, |m, s| {
            m.check_conservation(&init, s).unwrap();
            s.counts()[1] as usize
        })
        .unwrap();
        [traj.values[0], traj.values[1]]
    });

    let crit = chi2_upper_quantile(n_states - 1, 1e-4);
    for (label, samples) in [("direct", direct_samples), ("incremental", inc_samples)] {
        let mut obs_transient = vec![0_u64; n_states];
        let mut obs_stationary = vec![0_u64; n_states];
        for [m_trans, m_stat] in samples {
            obs_transient[m_trans] += 1;
            obs_stationary[m_stat] += 1;
        }
        let chi2_trans = chi2_statistic(&obs_transient, &p_transient);
        assert!(
            chi2_trans <= crit,
            "{label} transient chi2={chi2_trans:.4} > crit={crit:.4}, obs={obs_transient:?}, p={p_transient:?}"
        );
        let chi2_stat = chi2_statistic(&obs_stationary, &pi);
        assert!(
            chi2_stat <= crit,
            "{label} stationary chi2={chi2_stat:.4} > crit={crit:.4}, obs={obs_stationary:?}, pi={pi:?}"
        );
    }
}

#[test]
fn t2_catalytic_michaelis_menten_direct_vs_incremental_parity_and_conservation() {
    let (model, init) = michaelis_menten_model();

    let mut direct = Simulation::new(model.clone(), init.clone(), 0x51a7_0001);
    let mut inc = IncrementalSimulation::new(model.clone(), init.clone(), 0x51a7_0001).unwrap();

    let mut direct_events = Vec::new();
    let mut inc_events = Vec::new();
    let mut step_idx = 0_usize;

    loop {
        let before = inc.state().clone();

        let mut enum_rates = vec![0.0; model.channel_count()];
        model.enumerate_events(&before, &mut |ReactionEvent(ch), r| {
            assert_eq!(enum_rates[ch], 0.0);
            enum_rates[ch] = r;
        });
        for (ch, &er) in enum_rates.iter().enumerate() {
            assert_eq!(model.slot_rate(&before, ch).to_bits(), er.to_bits());
        }

        let inc_step = inc.step().unwrap();
        let direct_step = direct.step().unwrap();
        match (inc_step, direct_step) {
            (
                Step::Fired {
                    event: ie,
                    dt: i_dt,
                },
                Step::Fired {
                    event: de,
                    dt: d_dt,
                },
            ) => {
                assert_eq!(ie, de, "event mismatch at step {step_idx}");
                assert!(
                    (i_dt - d_dt).abs() <= 1e-13 * i_dt.max(1.0),
                    "dt mismatch at step {step_idx}: direct={d_dt}, inc={i_dt}"
                );
                model.check_conservation(&init, inc.state()).unwrap();
                model.check_conservation(&init, direct.state()).unwrap();
                verify_incremental_step_parity(&model, &before, inc.state(), &ie, inc.tree(), 0.0)
                    .unwrap_or_else(|e| panic!("parity failed at step {step_idx}: {e}"));
                inc_events.push(ie.0);
                direct_events.push(de.0);
                step_idx += 1;
                assert!(step_idx < 10_000, "unexpected non-terminating trajectory");
            }
            (Step::Absorbed, Step::Absorbed) => break,
            (inc_s, dir_s) => {
                panic!("absorption mismatch at step {step_idx}: {inc_s:?} vs {dir_s:?}")
            }
        }
    }

    assert_eq!(inc.state().counts(), &[12, 0, 0, 42]);
    assert_eq!(direct.state().counts(), &[12, 0, 0, 42]);
    assert_eq!(inc.state().fired_counts()[2], 42);
    assert_eq!(
        inc.state().total_firings(),
        inc.state()
            .fired_counts()
            .iter()
            .copied()
            .map(u128::from)
            .sum::<u128>()
    );
    assert_eq!(inc_events, direct_events);
    assert_eq!(
        inc.state().fired_counts()[0] + 2,
        inc.state().fired_counts()[1] + inc.state().fired_counts()[2]
    );

    let n_runs = 400_u64;
    let direct_times = run_ensemble(70_001, n_runs, |ctx| {
        let mut sim =
            Simulation::with_rng(model.clone(), init.clone(), ctx.rng(StreamRole::Dynamics));
        while matches!(sim.step().unwrap(), Step::Fired { .. }) {}
        assert_eq!(sim.state().counts(), &[12, 0, 0, 42]);
        sim.time()
    });
    let inc_times = run_ensemble(80_001, n_runs, |ctx| {
        let mut sim = IncrementalSimulation::with_rng(
            model.clone(),
            init.clone(),
            ctx.rng(StreamRole::Dynamics),
        )
        .unwrap();
        while matches!(sim.step().unwrap(), Step::Fired { .. }) {}
        assert_eq!(sim.state().counts(), &[12, 0, 0, 42]);
        sim.time()
    });
    let mean = |xs: &[f64]| xs.iter().sum::<f64>() / (xs.len() as f64);
    let se = |xs: &[f64], m: f64| {
        (xs.iter().map(|x| (x - m).powi(2)).sum::<f64>()
            / ((xs.len() - 1) as f64)
            / (xs.len() as f64))
            .sqrt()
    };
    let md = mean(&direct_times);
    let mi = mean(&inc_times);
    let se_diff = (se(&direct_times, md).powi(2) + se(&inc_times, mi).powi(2)).sqrt();
    assert!(
        (md - mi).abs() <= 4.0 * se_diff,
        "direct mean completion time {md} differs from incremental {mi} (se_diff={se_diff})"
    );
}

#[test]
fn t3_reversible_undo_checkpoint_roundtrip_and_validation_errors() {
    let valid_ch = ReactionChannel::new(vec![(0, 1)], vec![(1, 1)], 1.0);
    assert_eq!(
        ReactionNetwork::new(0, vec![valid_ch.clone()]).unwrap_err(),
        ReactionError::ZeroSpecies
    );
    assert_eq!(
        ReactionNetwork::new(2, vec![]).unwrap_err(),
        ReactionError::EmptyNetwork
    );
    assert_eq!(
        ReactionNetwork::new(2, vec![ReactionChannel::new(vec![(2, 1)], vec![], 1.0)]).unwrap_err(),
        ReactionError::SpeciesOutOfRange {
            channel: 0,
            species: 2,
            species_count: 2,
        }
    );
    assert_eq!(
        ReactionNetwork::new(2, vec![ReactionChannel::new(vec![], vec![(2, 1)], 1.0)]).unwrap_err(),
        ReactionError::SpeciesOutOfRange {
            channel: 0,
            species: 2,
            species_count: 2,
        }
    );
    assert_eq!(
        ReactionNetwork::new(2, vec![ReactionChannel::new(vec![(0, 0)], vec![], 1.0)]).unwrap_err(),
        ReactionError::ZeroStoichiometry {
            channel: 0,
            role: StoichiometryRole::Reactant,
            species: 0,
        }
    );
    assert_eq!(
        ReactionNetwork::new(2, vec![ReactionChannel::new(vec![], vec![(1, 0)], 1.0)]).unwrap_err(),
        ReactionError::ZeroStoichiometry {
            channel: 0,
            role: StoichiometryRole::Product,
            species: 1,
        }
    );
    assert_eq!(
        ReactionNetwork::new(
            2,
            vec![ReactionChannel::new(vec![(0, 1), (0, 2)], vec![], 1.0)],
        )
        .unwrap_err(),
        ReactionError::DuplicateSpecies {
            channel: 0,
            role: StoichiometryRole::Reactant,
            species: 0,
        }
    );
    assert_eq!(
        ReactionNetwork::new(
            2,
            vec![ReactionChannel::new(vec![], vec![(1, 1), (1, 1)], 1.0)],
        )
        .unwrap_err(),
        ReactionError::DuplicateSpecies {
            channel: 0,
            role: StoichiometryRole::Product,
            species: 1,
        }
    );
    assert!(matches!(
        ReactionNetwork::new(2, vec![ReactionChannel::new(vec![(0, 1)], vec![], -0.5)])
            .unwrap_err(),
        ReactionError::InvalidRateConstant { channel: 0, .. }
    ));
    assert!(matches!(
        ReactionNetwork::new(
            2,
            vec![ReactionChannel::new(vec![(0, 1)], vec![], f64::NAN)],
        )
        .unwrap_err(),
        ReactionError::InvalidRateConstant { channel: 0, .. }
    ));
    assert_eq!(
        ReactionNetwork::new(2, vec![valid_ch.clone().with_reverse(1)]).unwrap_err(),
        ReactionError::InvalidReverseChannel {
            channel: 0,
            reverse_channel: 1,
        }
    );
    assert_eq!(
        ReactionNetwork::new(
            2,
            vec![
                valid_ch.clone().with_reverse(1),
                ReactionChannel::new(vec![(1, 1)], vec![(0, 1)], 0.5),
            ],
        )
        .unwrap_err(),
        ReactionError::InvalidReverseChannel {
            channel: 0,
            reverse_channel: 1,
        }
    );
    assert_eq!(
        ReactionNetwork::new(
            2,
            vec![
                valid_ch.clone().with_reverse(1),
                ReactionChannel::new(vec![(1, 2)], vec![(0, 1)], 0.5).with_reverse(0),
            ],
        )
        .unwrap_err(),
        ReactionError::ReversePairMismatch {
            channel: 0,
            reverse_channel: 1,
        }
    );

    let base = ReactionNetwork::new(2, vec![valid_ch]).unwrap();
    assert_eq!(
        base.clone()
            .with_conservation_laws(vec![vec![1, 1, 0]])
            .unwrap_err(),
        ReactionError::ConservationLawWidth {
            law: 0,
            expected: 2,
            found: 3,
        }
    );
    assert_eq!(
        base.clone()
            .with_conservation_laws(vec![vec![1, 2]])
            .unwrap_err(),
        ReactionError::ConservationStoichiometryMismatch {
            law: 0,
            channel: 0,
            net_delta: 1,
        }
    );

    let (model, init) = michaelis_menten_model();
    assert_eq!(model.species_count(), 4);
    assert_eq!(model.channel_count(), 3);
    assert_eq!(
        model.initial_state(vec![1, 2]).unwrap_err(),
        ReactionError::StateSpeciesMismatch {
            expected: 4,
            found: 2,
        }
    );
    let mut bad_state = ReactionState {
        counts: vec![1, 2, 0, 0],
        fired_counts: vec![0, 0],
    };
    assert_eq!(
        model.apply(&mut bad_state, &ReactionEvent(0)).unwrap_err(),
        ReactionError::StateChannelMismatch {
            expected: 3,
            found: 2,
        }
    );
    assert_eq!(
        model
            .apply(&mut init.clone(), &ReactionEvent(9))
            .unwrap_err(),
        ReactionError::ChannelOutOfRange {
            channel: 9,
            channel_count: 3,
        }
    );

    let mut empty_es = model.initial_state(vec![5, 5, 0, 0]).unwrap();
    let snapshot_es = empty_es.clone();
    assert_eq!(
        model
            .apply_reversible(&mut empty_es, &ReactionEvent(2))
            .unwrap_err(),
        ReactionError::InsufficientReactants {
            channel: 2,
            species: 2,
            available: 0,
            required: 1,
        }
    );
    assert_eq!(
        empty_es, snapshot_es,
        "failed apply_reversible must not mutate state"
    );
    assert_eq!(
        model.check_conservation(&init, &empty_es).unwrap_err(),
        ReactionError::ConservationViolation {
            law: 0,
            initial: 12,
            current: 5,
        }
    );

    let mut work = init.clone();
    for ch in 0..model.channel_count() {
        let before = work.clone();
        let undo = model
            .apply_reversible(&mut work, &ReactionEvent(ch))
            .unwrap();
        assert_ne!(work, before);
        model.check_conservation(&init, &work).unwrap();
        model.undo(&mut work, undo);
        assert_eq!(work, before);
    }

    let seed = 0xcafe_babe;
    let total_steps = 40;
    let split = 17;

    let mut full_sim = Simulation::new(model.clone(), init.clone(), seed);
    let mut split_sim = Simulation::new(model.clone(), init.clone(), seed);
    let mut full_inc = IncrementalSimulation::new(model.clone(), init.clone(), seed).unwrap();
    let mut split_inc = IncrementalSimulation::new(model.clone(), init.clone(), seed).unwrap();

    for step in 0..total_steps {
        let s_full = full_sim.step().unwrap();
        let i_full = full_inc.step().unwrap();
        assert!(matches!(s_full, Step::Fired { .. }));
        assert!(matches!(i_full, Step::Fired { .. }));
        if step < split {
            split_sim.step().unwrap();
            split_inc.step().unwrap();
        }
    }

    let ckpt_sim = split_sim.checkpoint();
    let ckpt_inc = split_inc.checkpoint();
    let mut resumed_sim = Simulation::resume(model.clone(), &ckpt_sim).unwrap();
    let mut resumed_inc = IncrementalSimulation::resume(model.clone(), &ckpt_inc).unwrap();

    for _ in split..total_steps {
        resumed_sim.step().unwrap();
        resumed_inc.step().unwrap();
    }

    assert_eq!(resumed_sim.state(), full_sim.state());
    assert_eq!(resumed_sim.time().to_bits(), full_sim.time().to_bits());
    assert_eq!(resumed_sim.steps(), full_sim.steps());
    assert_eq!(resumed_sim.checkpoint(), full_sim.checkpoint());

    assert_eq!(resumed_inc.state(), full_inc.state());
    assert_eq!(resumed_inc.time().to_bits(), full_inc.time().to_bits());
    assert_eq!(resumed_inc.steps(), full_inc.steps());
    assert_eq!(resumed_inc.checkpoint(), full_inc.checkpoint());

    let (other_model, _) = dimerization_model();
    assert!(matches!(
        Simulation::resume(other_model, &ckpt_sim),
        Err(CheckpointError::ParamsMismatch)
    ));
    assert!(matches!(
        Simulation::resume(model.clone(), &ckpt_sim[..ckpt_sim.len() - 1]),
        Err(CheckpointError::Truncated | CheckpointError::InvalidState(_))
    ));
    let mut bad_state_bytes = Vec::new();
    model.encode_state(full_sim.state(), &mut bad_state_bytes);
    assert!(
        model
            .decode_state(&bad_state_bytes[..bad_state_bytes.len() - 1])
            .is_err()
    );
}

#[test]
fn t4_extreme_scale_propensities_large_species_counters_and_checkpoint_dimensions() {
    // C(1000, 20) * 1e-40 overflows u128 (~3.4e41 > u128::MAX) while true propensity is ~33.948.
    let high_order = ReactionNetwork::new(
        2,
        vec![ReactionChannel::new(vec![(0, 20)], vec![(1, 1)], 1e-40)],
    )
    .unwrap();
    let s_high = high_order.initial_state(vec![1_000, 0]).unwrap();
    let p_high = high_order.propensity(&s_high, 0);
    let mut expected_c = 1.0_f64;
    for i in 1..=20_u32 {
        expected_c *= f64::from(1_000 - 20 + i) / f64::from(i);
    }
    let expected_p = expected_c * 1e-40;
    assert!(
        p_high.is_finite(),
        "expected finite propensity, got {p_high}"
    );
    assert!(
        (p_high - expected_p).abs() <= 1e-12 * expected_p,
        "got {p_high}, expected {expected_p}"
    );
    let mut sim_high = Simulation::new(high_order, s_high, 77);
    assert!(matches!(sim_high.step().unwrap(), Step::Fired { dt, .. } if dt.is_finite()));

    // Product of 26 species at 2^40 copies reaches 2^1040 > f64::MAX before rate_constant = 2^-1000.
    let n_many = 26;
    let reactants: Vec<(usize, u32)> = (0..n_many).map(|sp| (sp, 1)).collect();
    let rescaled = ReactionNetwork::new(
        n_many + 1,
        vec![ReactionChannel::new(
            reactants,
            vec![(n_many, 1)],
            2.0_f64.powi(-1000),
        )],
    )
    .unwrap();
    let mut many_counts = vec![1_u64 << 40; n_many];
    many_counts.push(0);
    let s_many = rescaled.initial_state(many_counts).unwrap();
    let p_many = rescaled.propensity(&s_many, 0);
    assert_eq!(p_many, 2.0_f64.powi(40));

    // Zero-reactant impossible huge order with finite or overflowing other terms stays 0.0, never NaN/inf.
    let extreme_order = ReactionNetwork::new(
        2,
        vec![ReactionChannel::new(
            vec![(0, 3_000), (1, 1)],
            vec![],
            1e-10,
        )],
    )
    .unwrap();
    assert_eq!(
        extreme_order.propensity(&extreme_order.initial_state(vec![6_000, 1]).unwrap(), 0),
        f64::INFINITY
    );
    assert_eq!(
        extreme_order.propensity(&extreme_order.initial_state(vec![6_000, 0]).unwrap(), 0),
        0.0
    );

    // Sparse species index space up to usize::MAX constructs without capacity panic.
    let sparse = ReactionNetwork::new(
        usize::MAX,
        vec![ReactionChannel::new(vec![(0, 1)], vec![(1, 1)], 1.0)],
    )
    .unwrap();
    assert_eq!(sparse.species_count(), usize::MAX);

    // total_firings represents sums above u64::MAX cleanly without wrapping or panicking.
    let two_ch = ReactionNetwork::new(
        2,
        vec![
            ReactionChannel::new(vec![(0, 1)], vec![(1, 1)], 0.0),
            ReactionChannel::new(vec![(0, 1)], vec![(1, 1)], 1.0),
        ],
    )
    .unwrap();
    let mut big_counters = ReactionState {
        counts: vec![5, 0],
        fired_counts: vec![u64::MAX, 0],
    };
    two_ch.apply(&mut big_counters, &ReactionEvent(1)).unwrap();
    assert_eq!(big_counters.total_firings(), u128::from(u64::MAX) + 1);

    // Mismatched ReactionState dimensions with matching total word count must fail checkpoint decode/resume.
    let dim_model = ReactionNetwork::new(
        2,
        vec![ReactionChannel::new(vec![(0, 1)], vec![(1, 1)], 1.0)],
    )
    .unwrap();
    let malformed = ReactionState {
        counts: vec![4],
        fired_counts: vec![7, 9],
    };
    let sim_bad = Simulation::new(dim_model.clone(), malformed.clone(), 9);
    let bad_ckpt = sim_bad.checkpoint();
    assert!(matches!(
        Simulation::resume(dim_model.clone(), &bad_ckpt),
        Err(CheckpointError::InvalidState(_))
    ));
    let mut encoded_bad = Vec::new();
    dim_model.encode_state(&malformed, &mut encoded_bad);
    assert!(dim_model.decode_state(&encoded_bad).is_err());
}
