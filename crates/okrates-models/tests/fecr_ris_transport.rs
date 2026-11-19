//! Integration tests for single-defect [`FeCrRis`] transport.

use okrates_core::{
    DirectCatalog, Engine, KmcRng, SampleGrid, SamplingError, Simulation, Step, SumTreeCatalog,
};
use okrates_models::fecr::{FeCrModel, Species};
use okrates_models::fecr_diffusion::{
    FitMethod, TransportConfig, estimate as estimate_vacancy, jackknife_ratio_mean_se, mean_se,
    run_transport, sample_fecr_run, summarize as summarize_vacancy,
};
use okrates_models::fecr_martinez::{Martinez2012, params as m12_params};
use okrates_models::fecr_ris::{
    Cell, DumbbellParams, FeCrRis, FeCrRisEvent, FeCrRisParams, Record,
};
use okrates_models::fecr_ris_transport::{
    DefectKind, RisDisplacements, RisObservables, RisReplicaEstimate, RisReplicaRun,
    RisTransportConfig, RisTransportError, estimate_ris, random_single_defect_state,
    run_ris_transport, sample_ris_transport, summarize_ris_transport,
};

const A_FE: f64 = 2.8665e-10;
const Z_GATE: f64 = 4.5;

fn bulk_model(cells: [usize; 3], temperature_k: f64) -> FeCrRis {
    FeCrRis::new(FeCrRisParams {
        cells,
        sink_planes: Vec::new(),
        temperature_k,
        k0_dpa_per_s: 0.0,
        capacity: 1,
        dumbbell: DumbbellParams::senninger2016_r1(),
    })
    .expect("valid bulk RIS model")
}

fn check_pure_fe_defect(defect: DefectKind, runs: u64, target_hops: f64, seed_base: u64) {
    let model = bulk_model([6, 6, 6], 650.0);
    let mut rng = KmcRng::from_seed(seed_base);
    let probe = random_single_defect_state(&model, defect, 0, &mut rng, seed_base).unwrap();
    let total_rate: f64 = (0..8).map(|j| model.rate(&probe, 0, j)).sum();
    assert!(total_rate > 0.0);
    // All 8 vacancy jumps (or 8 C-site rotational dumbbell slots) share identical pure-Fe rate.
    let single_rate = total_rate / 8.0;
    let d_analytic = A_FE * A_FE * single_rate;

    let t_end = target_hops / total_rate;
    let config = RisTransportConfig::windowed(0.1 * t_end, t_end, 10, A_FE);
    let results = run_ris_transport(&model, defect, 0, seed_base, runs, &config).unwrap();
    let summary = summarize_ris_transport(&results, defect, model.lattice().site_count(), 0)
        .expect("summary succeeds");

    let expected_fe_ratio = match defect {
        DefectKind::Vacancy => -1.0,
        DefectKind::Dumbbell => 1.0,
    };
    assert!(
        (summary.l_fe_d_over_l_fe_fe.0 - expected_fe_ratio).abs() < 1e-12,
        "{}: L_FeD/L_FeFe = {} expected {}",
        defect.name(),
        summary.l_fe_d_over_l_fe_fe.0,
        expected_fe_ratio
    );
    assert!(summary.l_cr_d_over_l_cr_cr.0.is_nan());
    assert!(summary.d_cr_over_d_fe.0.is_nan());
    assert!((summary.mean_initial_sro_alpha1).abs() < 1e-12);
    assert!((summary.mean_final_sro_alpha1).abs() < 1e-12);

    let (d_mean, d_se) = summary.defect;
    let z = (d_mean - d_analytic) / d_se;
    assert!(
        z.abs() <= Z_GATE,
        "{} pure-Fe D mismatch: got {d_mean:e} +/- {d_se:e}, analytic {d_analytic:e} (z={z:.2})",
        defect.name()
    );
}

#[test]
fn t1_pure_fe_analytic_smoke() {
    check_pure_fe_defect(DefectKind::Vacancy, 80, 300.0, 2026_1001);
    check_pure_fe_defect(DefectKind::Dumbbell, 80, 300.0, 2026_2001);
}

#[test]
#[ignore = "full tier: statistical"]
fn t1_pure_fe_analytic() {
    check_pure_fe_defect(DefectKind::Vacancy, 500, 1500.0, 2026_3001);
    check_pure_fe_defect(DefectKind::Dumbbell, 500, 1500.0, 2026_4001);
}

#[test]
fn t2_exact_integer_conservation() {
    let model = bulk_model([8, 8, 8], 650.0);
    let cr_count = 102; // ~10 % Cr on 1024 sites

    for (defect, sign, seed) in [
        (DefectKind::Vacancy, -1_i64, 91_001),
        (DefectKind::Dumbbell, 1_i64, 92_001),
    ] {
        let mut init_rng = KmcRng::from_seed(seed);
        let init = random_single_defect_state(&model, defect, cr_count, &mut init_rng, seed)
            .expect("valid single defect");
        let total_rate: f64 = (0..8).map(|j| model.rate(&init, 0, j)).sum();
        let grid = SampleGrid::from_origin(250.0 / total_rate, 32).unwrap();
        let mut engine =
            Engine::<FeCrRis, DirectCatalog<FeCrRisEvent>>::new(model.clone(), init, seed + 1)
                .unwrap();

        let traj = sample_ris_transport(&mut engine, &grid, A_FE, None).unwrap();
        assert!(traj.fired_steps > 50, "too few steps: {}", traj.fired_steps);

        let d = traj.final_displacements;
        for axis in 0..3 {
            assert_eq!(
                d.fe[axis] + d.cr[axis],
                sign * d.defect[axis],
                "{} conservation violated on axis {axis}: fe={:?} cr={:?} defect={:?}",
                defect.name(),
                d.fe,
                d.cr,
                d.defect
            );
        }

        for obs in &traj.values {
            let term_scale = obs.fe_fe + 2.0 * obs.fe_cr.abs() + obs.cr_cr + obs.defect_msd;
            let tol = 1e-11 * term_scale;
            let sum_spec = obs.fe_fe + 2.0 * obs.fe_cr + obs.cr_cr;
            assert!(
                (sum_spec - obs.defect_msd).abs() <= tol,
                "{} quadratic identity mismatch: sum={sum_spec} vs defect={}",
                defect.name(),
                obs.defect_msd
            );
            assert!(
                (obs.fe_defect - (sign as f64) * (obs.fe_fe + obs.fe_cr)).abs() <= tol,
                "{} cross identity mismatch",
                defect.name()
            );
        }
    }

    // Strict crossing of `t_end` must report pre-jump endpoint displacements, SRO, and floored count.
    let mut probe_rng = KmcRng::from_seed(94_001);
    let init_cross =
        random_single_defect_state(&model, DefectKind::Dumbbell, 102, &mut probe_rng, 94_001)
            .unwrap();
    let mut step_eng = Engine::<FeCrRis, DirectCatalog<FeCrRisEvent>>::new(
        model.clone(),
        init_cross.clone(),
        94_002,
    )
    .unwrap();
    let first_dt = match step_eng.step().unwrap() {
        Step::Fired { dt, .. } => dt,
        Step::Absorbed => panic!("expected fired first jump"),
    };
    assert!(first_dt > 0.0);
    let mut cross_eng =
        Engine::<FeCrRis, DirectCatalog<FeCrRisEvent>>::new(model.clone(), init_cross, 94_002)
            .unwrap();
    let half_grid = SampleGrid::from_origin(0.5 * first_dt, 2).unwrap();
    let cross_traj = sample_ris_transport(&mut cross_eng, &half_grid, A_FE, None).unwrap();
    assert_eq!(cross_traj.final_displacements, RisDisplacements::default());
    assert_eq!(
        cross_traj.final_sro_alpha1.to_bits(),
        cross_traj.initial_sro_alpha1.to_bits()
    );
    assert_eq!(cross_traj.floored_jumps, 0);

    // Cross-check dumbbell jumper inference directly against `FeCrRis::jump_result`.
    let mut init_rng = KmcRng::from_seed(93_001);
    let init = random_single_defect_state(&model, DefectKind::Dumbbell, 300, &mut init_rng, 93_001)
        .unwrap();
    let mut engine =
        Engine::<FeCrRis, DirectCatalog<FeCrRisEvent>>::new(model.clone(), init, 93_002).unwrap();
    for _ in 0..120 {
        let before = engine.state().clone();
        let Record::Dumbbell {
            site: o,
            plus,
            minus,
            ..
        } = before.records()[0]
        else {
            panic!("expected dumbbell");
        };
        let Step::Fired {
            event: FeCrRisEvent::DumbbellJump { rec: 0, jump },
            ..
        } = engine.step().unwrap()
        else {
            panic!("expected DumbbellJump on rec 0");
        };
        let (_, expected_rest) = model.jump_result(&before, 0, jump).unwrap();
        let expected_jumper = if expected_rest == plus { minus } else { plus };
        let inferred_jumper = if plus == minus {
            plus
        } else {
            match engine.state().cells()[o] {
                Cell::Fe => Species::Cr,
                Cell::Cr => Species::Fe,
                _ => panic!("expected rest atom on site {o}"),
            }
        };
        assert_eq!(inferred_jumper, expected_jumper);
    }
}

fn check_direct_and_sum_tree_parity(target_hops: f64) {
    let model = bulk_model([8, 8, 8], 650.0);
    let cr_count = 102;

    for (defect, seed) in [(DefectKind::Vacancy, 7101), (DefectKind::Dumbbell, 7201)] {
        let mut init_rng = KmcRng::from_seed(seed);
        let init = random_single_defect_state(&model, defect, cr_count, &mut init_rng, seed)
            .expect("valid single defect");
        let rate0: f64 = (0..8).map(|j| model.rate(&init, 0, j)).sum();
        let t_end = target_hops / rate0;
        let grid = SampleGrid::windowed(0.1 * t_end, t_end, 16).unwrap();

        let mut direct = Engine::<FeCrRis, DirectCatalog<FeCrRisEvent>>::new(
            model.clone(),
            init.clone(),
            seed + 10,
        )
        .unwrap();
        let mut sumtree =
            Engine::<FeCrRis, SumTreeCatalog>::new(model.clone(), init, seed + 10).unwrap();

        let traj_d = sample_ris_transport(&mut direct, &grid, A_FE, None).unwrap();
        let traj_s = sample_ris_transport(&mut sumtree, &grid, A_FE, None).unwrap();

        assert_eq!(traj_d.fired_steps, traj_s.fired_steps);
        assert_eq!(traj_d.final_displacements, traj_s.final_displacements);
        assert_eq!(traj_d.floored_jumps, traj_s.floored_jumps);
        assert_eq!(
            traj_d.initial_sro_alpha1.to_bits(),
            traj_s.initial_sro_alpha1.to_bits()
        );
        assert_eq!(
            traj_d.final_sro_alpha1.to_bits(),
            traj_s.final_sro_alpha1.to_bits()
        );
        assert_eq!(traj_d.times.len(), traj_s.times.len());
        for (a, b) in traj_d.values.iter().zip(&traj_s.values) {
            assert_eq!(a.defect_msd.to_bits(), b.defect_msd.to_bits());
            assert_eq!(a.fe_fe.to_bits(), b.fe_fe.to_bits());
            assert_eq!(a.fe_cr.to_bits(), b.fe_cr.to_bits());
            assert_eq!(a.cr_cr.to_bits(), b.cr_cr.to_bits());
            assert_eq!(a.fe_defect.to_bits(), b.fe_defect.to_bits());
            assert_eq!(a.cr_defect.to_bits(), b.cr_defect.to_bits());
        }
    }
}

#[test]
fn t3_direct_and_sum_tree_bitwise_parity() {
    check_direct_and_sum_tree_parity(120.0);
}

#[test]
#[ignore = "full tier: equivalence"]
fn t3_direct_and_sum_tree_bitwise_parity_full() {
    check_direct_and_sum_tree_parity(5000.0);
}

fn check_vacancy_parity(runs: u64, ensemble_hops: f64) {
    let cells = [6, 6, 6];
    let temp = 800.0;
    let cr_count = 43; // ~10 % Cr on 432 sites

    let ris_model = bulk_model(cells, temp);
    let m12 = Martinez2012::new(temp).unwrap();
    let m12_model =
        FeCrModel::new(ris_model.lattice().clone(), m12, m12_params::NU_FE, temp).unwrap();

    let mut init_rng = KmcRng::from_seed(55_001);
    let ris_state = random_single_defect_state(
        &ris_model,
        DefectKind::Vacancy,
        cr_count,
        &mut init_rng,
        55_001,
    )
    .unwrap();
    let occ: Vec<Species> = ris_state
        .cells()
        .iter()
        .map(|c| match c {
            Cell::Fe => Species::Fe,
            Cell::Cr => Species::Cr,
            Cell::Vacancy => Species::Vacancy,
            Cell::Dumbbell => panic!("expected vacancy occupancy"),
        })
        .collect();
    let m12_state = m12_model.state(occ).unwrap();

    let total_rate: f64 = (0..8).map(|d| ris_model.rate(&ris_state, 0, d)).sum();
    let grid = SampleGrid::windowed(20.0 / total_rate, 160.0 / total_rate, 10).unwrap();

    let mut ris_engine =
        Engine::<FeCrRis, DirectCatalog<FeCrRisEvent>>::new(ris_model.clone(), ris_state, 55_002)
            .unwrap();
    let mut m12_sim = Simulation::new(m12_model.clone(), m12_state, 55_002);

    let ris_traj = sample_ris_transport(&mut ris_engine, &grid, A_FE, None).unwrap();
    let m12_traj = sample_fecr_run(&mut m12_sim, &grid, A_FE, None).unwrap();

    assert_eq!(ris_traj.fired_steps, m12_traj.fired_steps);
    assert_eq!(ris_traj.times, m12_traj.times);
    for (r, m) in ris_traj.values.iter().zip(&m12_traj.values) {
        assert!(
            (r.defect_msd - m.vacancy).abs() <= 1e-12 * m.vacancy.max(1e-30),
            "MSD mismatch: {} vs {}",
            r.defect_msd,
            m.vacancy
        );
        assert!((r.fe_fe - m.collective[0]).abs() <= 1e-12 * m.collective[0].max(1e-30));
        assert!((r.fe_cr - m.collective[1]).abs() <= 1e-12 * m.vacancy.max(1e-30));
        assert!((r.cr_cr - m.collective[2]).abs() <= 1e-12 * m.vacancy.max(1e-30));
    }

    let t_end = ensemble_hops / total_rate;
    let ris_cfg = RisTransportConfig::windowed(0.1 * t_end, t_end, 10, A_FE);
    let m12_cfg = TransportConfig::windowed(0.1 * t_end, t_end, 10, A_FE);

    let ris_runs = run_ris_transport(
        &ris_model,
        DefectKind::Vacancy,
        cr_count,
        8801,
        runs,
        &ris_cfg,
    )
    .unwrap();
    let ris_sum = summarize_ris_transport(
        &ris_runs,
        DefectKind::Vacancy,
        ris_model.lattice().site_count(),
        cr_count,
    )
    .unwrap();
    let m12_runs = run_transport(&m12_model, cr_count, 8801, runs, &m12_cfg, 0).unwrap();
    let m12_sum = summarize_vacancy(&m12_runs, cr_count, 431);

    let check_se = |name: &str, a: (f64, f64), b: (f64, f64)| {
        let comb_se = (a.1 * a.1 + b.1 * b.1).sqrt();
        let z = (a.0 - b.0) / comb_se;
        assert!(
            z.abs() <= 3.5,
            "{name} ensemble parity failed: {} +/- {} vs {} +/- {} (z={z:.2})",
            a.0,
            a.1,
            b.0,
            b.1
        );
    };
    check_se("D_v", ris_sum.defect, m12_sum.vacancy);
    check_se("Lambda_FeFe", ris_sum.fe_fe, m12_sum.collective[0]);
    check_se("Lambda_FeCr", ris_sum.fe_cr, m12_sum.collective[1]);
    check_se("Lambda_CrCr", ris_sum.cr_cr, m12_sum.collective[2]);

    let m12_est = estimate_vacancy(
        &m12_traj.times,
        &m12_traj.values,
        okrates_models::fecr_diffusion::ols_slope,
    );
    assert!(ris_sum.fe_defect.0 < 0.0 && ris_sum.cr_defect.0 < 0.0);
    assert!(m12_est.vacancy > 0.0);
}

#[test]
fn t4_vacancy_parity_with_fecr_diffusion() {
    check_vacancy_parity(8, 120.0);
}

#[test]
#[ignore = "full tier: statistical"]
fn t4_vacancy_parity_with_fecr_diffusion_full() {
    check_vacancy_parity(64, 1500.0);
}

#[test]
fn t5_jackknife_ratio_mean_se_and_edge_cases() {
    let (det_ratio, det_se) = jackknife_ratio_mean_se(&[1.0, 3.0], &[1.0, 2.0], 1.0);
    assert!((det_ratio - 4.0 / 3.0).abs() < 1e-12);
    assert!((det_se - 0.25).abs() < 1e-12);

    let (zero_ratio, zero_se) = jackknife_ratio_mean_se(&[0.0], &[0.0], 1.0);
    assert!(zero_ratio.is_nan() && zero_se.is_nan());

    let mut rng = KmcRng::from_seed(42_424);
    let n = 1000;
    let mut runs = Vec::with_capacity(n);
    for _ in 0..n {
        let l_fe_i = -rng.open01().ln();
        let l_cr_i = -0.35 * rng.open01().ln();
        runs.push(RisReplicaRun {
            estimate: RisReplicaEstimate {
                defect: l_fe_i + l_cr_i,
                fe_fe: l_fe_i,
                fe_cr: 0.0,
                cr_cr: l_cr_i,
                fe_defect: l_fe_i,
                cr_defect: l_cr_i,
            },
            fired: 100,
            floored_jumps: 0,
            initial_sro_alpha1: 0.0,
            final_sro_alpha1: 0.0,
            final_displacements: RisDisplacements::default(),
        });
    }

    let summary = summarize_ris_transport(&runs, DefectKind::Dumbbell, 1024, 100).unwrap();
    let (ratio, se) = summary.l_cr_d_over_l_fe_d;
    assert!(((ratio - 0.35) / se).abs() <= 4.0, "ratio={ratio} se={se}");

    // Partial diffusivity scale: (n_fe / n_cr) with n_atoms = 1024 + 1 = 1025, n_fe = 925, n_cr = 100.
    let expected_d_ratio = ratio * (925.0 / 100.0);
    assert!((summary.d_cr_over_d_fe.0 - expected_d_ratio).abs() < 1e-12);

    // Per-replica ratio mean exhibits positive Jensen bias from 1/L_FeI.
    let per_replica: Vec<f64> = runs
        .iter()
        .map(|r| r.estimate.cr_defect / r.estimate.fe_defect)
        .collect();
    let (naive_mean, _) = mean_se(&per_replica);
    assert!(
        naive_mean > 0.6,
        "expected strong Jensen bias, got {naive_mean}"
    );

    // Zero Cr composition masks d_cr/d_fe and l_cr_d/l_cr_cr with NaN instead of dividing by zero.
    let pure_fe_sum = summarize_ris_transport(&runs[..4], DefectKind::Vacancy, 1024, 0).unwrap();
    assert!(pure_fe_sum.l_cr_d_over_l_cr_cr.0.is_nan());
    assert!(pure_fe_sum.d_cr_over_d_fe.0.is_nan());
}

#[test]
fn t6_validation_and_error_guards() {
    let base_params = FeCrRisParams {
        cells: [6, 6, 6],
        sink_planes: Vec::new(),
        temperature_k: 650.0,
        k0_dpa_per_s: 0.0,
        capacity: 1,
        dumbbell: DumbbellParams::senninger2016_r1(),
    };

    let mut k0_params = base_params.clone();
    k0_params.k0_dpa_per_s = 1e-4;
    let m_k0 = FeCrRis::new(k0_params).unwrap();
    let cfg = RisTransportConfig::windowed(1e-12, 1e-11, 4, A_FE).with_max_steps(Some(10));
    assert!(matches!(
        run_ris_transport(&m_k0, DefectKind::Vacancy, 10, 1, 2, &cfg),
        Err(RisTransportError::NonZeroCreationRate(_))
    ));

    let mut sink_params = base_params.clone();
    sink_params.sink_planes = vec![0];
    let m_sink = FeCrRis::new(sink_params).unwrap();
    assert!(matches!(
        run_ris_transport(&m_sink, DefectKind::Dumbbell, 10, 1, 2, &cfg),
        Err(RisTransportError::NonEmptySinkPlanes(1))
    ));

    let mut cap_params = base_params.clone();
    cap_params.capacity = 2;
    let m_cap = FeCrRis::new(cap_params).unwrap();
    assert!(matches!(
        run_ris_transport(&m_cap, DefectKind::Vacancy, 10, 1, 2, &cfg),
        Err(RisTransportError::InvalidCapacity { capacity: 2 })
    ));

    let m_ok = FeCrRis::new(base_params).unwrap();
    let empty_state = m_ok.random_state(10, 1, 2).unwrap();
    let mut eng_empty =
        Engine::<FeCrRis, DirectCatalog<FeCrRisEvent>>::new(m_ok.clone(), empty_state, 3).unwrap();
    let grid = SampleGrid::windowed(0.1, 1.0, 4).unwrap();
    assert!(matches!(
        sample_ris_transport(&mut eng_empty, &grid, A_FE, None),
        Err(RisTransportError::ExpectedSingleDefect {
            vacancies: 0,
            dumbbells: 0
        })
    ));

    assert!(matches!(
        run_ris_transport(&m_ok, DefectKind::Vacancy, 10, 1, 1, &cfg),
        Err(RisTransportError::InsufficientReplicas(1))
    ));
    for bad_a in [0.0, 1e200, 1e-200] {
        assert!(matches!(
            run_ris_transport(
                &m_ok,
                DefectKind::Vacancy,
                10,
                1,
                2,
                &RisTransportConfig::windowed(0.1, 1.0, 4, bad_a)
            ),
            Err(RisTransportError::InvalidLatticeParameter(_))
        ));
    }
    assert!(matches!(
        run_ris_transport(
            &m_ok,
            DefectKind::Vacancy,
            10,
            1,
            2,
            &RisTransportConfig::windowed(0.1, 1.0, usize::MAX, A_FE)
        ),
        Err(RisTransportError::SampleAllocation(usize::MAX))
    ));
    let n = m_ok.lattice().site_count();
    assert!(matches!(
        run_ris_transport(&m_ok, DefectKind::Vacancy, n, 1, 2, &cfg),
        Err(RisTransportError::InvalidCrCount {
            defect: DefectKind::Vacancy,
            ..
        })
    ));
    assert!(matches!(
        run_ris_transport(&m_ok, DefectKind::Dumbbell, n + 2, 1, 2, &cfg),
        Err(RisTransportError::InvalidCrCount {
            defect: DefectKind::Dumbbell,
            ..
        })
    ));
    assert!(matches!(
        run_ris_transport(
            &m_ok,
            DefectKind::Vacancy,
            10,
            1,
            2,
            &RisTransportConfig::windowed(0.1, 1.0, 1, A_FE)
        ),
        Err(RisTransportError::InsufficientSamples {
            fit_method: FitMethod::WindowedOls,
            sample_count: 1
        })
    ));

    // Tiny time window avoids 0/0 underflow and produces exact 0.0; duplicate times return typed error.
    let zero_obs = RisObservables {
        defect_msd: 0.0,
        fe_fe: 0.0,
        fe_cr: 0.0,
        cr_cr: 0.0,
        fe_defect: 0.0,
        cr_defect: 0.0,
    };
    let tiny_est = estimate_ris(
        &[0.0, 1e-200],
        &[zero_obs, zero_obs],
        FitMethod::WindowedOls,
    )
    .expect("normalized OLS avoids underflow");
    assert_eq!(tiny_est.defect, 0.0);
    assert!(matches!(
        estimate_ris(&[1.0, 1.0], &[zero_obs, zero_obs], FitMethod::WindowedOls),
        Err(RisTransportError::NonFiniteEstimate)
    ));

    let mut rng = KmcRng::from_seed(99);
    let init = random_single_defect_state(&m_ok, DefectKind::Vacancy, 10, &mut rng, 99).unwrap();
    let rate0: f64 = (0..8).map(|d| m_ok.rate(&init, 0, d)).sum();

    let origin_cfg = RisTransportConfig::from_origin(20.0 / rate0, 1, A_FE);
    let origin_runs =
        run_ris_transport(&m_ok, DefectKind::Vacancy, 10, 99, 2, &origin_cfg).unwrap();
    assert_eq!(origin_runs.len(), 2);
    assert!(origin_runs.iter().all(|r| r.estimate.defect.is_finite()));

    let mut eng_cap = Engine::<FeCrRis, DirectCatalog<FeCrRisEvent>>::new(m_ok, init, 100).unwrap();
    let long_grid = SampleGrid::from_origin(10_000.0 / rate0, 4).unwrap();
    assert!(matches!(
        sample_ris_transport(&mut eng_cap, &long_grid, A_FE, Some(5)),
        Err(RisTransportError::Sampling(
            SamplingError::MaxStepsExceeded { max_steps: 5, .. }
        ))
    ));
}
