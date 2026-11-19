//! Unit and integration tests for Fe–Cr grain-boundary RIS observables, continuous-time
//! planar defect tallies, relaxation fits, validation guards, and ensemble determinism.

use okrates_core::{Clock, IncrementalSimulation, KmcRng, Step, StreamRole};
use okrates_models::fecr::Species;
use okrates_models::fecr_ris::{Defect, DumbbellParams, FeCrRis, FeCrRisError, FeCrRisParams};
use okrates_models::fecr_ris_segregation::{
    DEFAULT_LATTICE_A_M, PlaneCounts, PlaneDefectTally, RisSegregationConfig, RisSegregationError,
    compute_segregation_metrics, default_bulk_min_distance_planes, fit_exponential_relaxation,
    place_random_cr, run_ris_segregation, summarize_ris_segregation,
};

fn test_model(cells: [usize; 3], sink_planes: Vec<i64>, k0: f64, capacity: usize) -> FeCrRis {
    FeCrRis::new(FeCrRisParams {
        cells,
        sink_planes,
        temperature_k: 773.0,
        k0_dpa_per_s: k0,
        capacity,
        dumbbell: DumbbellParams::senninger2016_r1(),
    })
    .unwrap()
}

fn plane_x(cells: [usize; 3], site: usize) -> usize {
    let [lx, ly, lz] = cells;
    let stride = ly * lz;
    let half = lx * stride;
    let (sub, within) = if site < half {
        (0, site)
    } else {
        (1, site - half)
    };
    2 * (within / stride) + sub
}

#[test]
fn t1_plane_counts_and_segregation_metrics_analytic() {
    let cells = [6, 6, 6];
    let model = test_model(cells, vec![0], 1.0e-3, 16);
    let n = model.lattice().site_count();
    let mut occ = vec![Species::Fe; n];

    // Group sites by (100) atomic plane (nx = 12 planes, 36 sites/plane).
    let mut by_plane = vec![Vec::new(); 12];
    for site in 0..n {
        by_plane[plane_x(cells, site)].push(site);
    }

    // Set Cr atoms per plane: sink x=0 has 12 Cr (C=1/3, Cr/Fe=0.5), x=1 has 9 Cr (Cr/Fe=1/3),
    // plane x=2 gets a Fe-Cr dumbbell (+1 Fe, +1 Cr over 35 remaining sites with 5 Cr),
    // plane x=10 gets a vacancy with 6 Cr + 29 Fe (outside RC_OFFSETS from x=2), and all remaining
    // planes (including bulk x=4..=8) have 6 Cr + 30 Fe (C_bulk = 6/36 = 1/6, bulk Cr/Fe = 0.2).
    for &s in &by_plane[0][..12] {
        occ[s] = Species::Cr;
    }
    for &s in &by_plane[1][..9] {
        occ[s] = Species::Cr;
    }
    for &s in &by_plane[2][1..6] {
        occ[s] = Species::Cr;
    }
    for (idx, plane) in by_plane.iter().enumerate().skip(3) {
        if idx == 10 {
            for &s in &plane[1..7] {
                occ[s] = Species::Cr;
            }
        } else {
            for &s in &plane[..6] {
                occ[s] = Species::Cr;
            }
        }
    }

    let dumbbell_site = by_plane[2][0];
    let vac_site = by_plane[10][0];
    let state = model
        .state_with_defects(
            occ,
            &[
                Defect::Dumbbell {
                    site: dumbbell_site,
                    axis: 0,
                    plus: Species::Fe,
                    minus: Species::Cr,
                },
                Defect::Vacancy { site: vac_site },
            ],
            1,
        )
        .unwrap();

    let counts = PlaneCounts::from_state(&model, &state).unwrap();
    assert_eq!(counts.plane_count(), 12);
    assert_eq!(counts.sites_per_plane, 36);
    assert_eq!(counts.fe[0], 24);
    assert_eq!(counts.cr[0], 12);
    assert_eq!(counts.fe[2], 31);
    assert_eq!(counts.cr[2], 6);
    assert_eq!(counts.dumbbells[2], 1);
    assert_eq!(counts.fe[10], 29);
    assert_eq!(counts.cr[10], 6);
    assert_eq!(counts.vacancies[10], 1);

    let total_fe: u64 = counts.fe.iter().sum();
    let total_cr: u64 = counts.cr.iter().sum();
    let atoms = state.atom_counts();
    assert_eq!(total_fe, atoms[0] + atoms[2] + atoms[4]);
    assert_eq!(total_cr, atoms[1] + atoms[3] + atoms[5]);

    // Bulk window min_distance = 4 selects planes 4..=8, all having Cr=6, Fe=30 (ratio = 0.2).
    let a_m = 2.8665e-10;
    let dx_nm = 0.5 * a_m * 1.0e9;
    let metrics = compute_segregation_metrics(&counts, 0, 4, a_m).unwrap();
    assert!((metrics.c_cr_gb - (12.0 / 36.0)).abs() < 1e-14);
    assert!((metrics.c_cr_bulk - (6.0 / 36.0)).abs() < 1e-14);

    // Non-bulk planes with ratios != 0.2 are x=0 (0.5), x=1 (1/3), x=2 (6/31), x=10 (6/29).
    let expected_excess_ratio =
        (12.0 / 24.0 - 0.2) + (9.0 / 27.0 - 0.2) + (6.0 / 31.0 - 0.2) + (6.0 / 29.0 - 0.2);
    let expected_s_tot = dx_nm * expected_excess_ratio;
    assert!(
        (metrics.s_cr_tot_nm - expected_s_tot).abs() < 1e-12,
        "got {}, expected {}",
        metrics.s_cr_tot_nm,
        expected_s_tot
    );

    let expected_excess_linear = (12.0 / 36.0 - 1.0 / 6.0)
        + (9.0 / 36.0 - 1.0 / 6.0)
        + (6.0 / 37.0 - 1.0 / 6.0)
        + (6.0 / 35.0 - 1.0 / 6.0);
    assert!((metrics.s_cr_linear_nm - dx_nm * expected_excess_linear).abs() < 1e-12);
}

#[test]
fn t2_zero_bias_random_alloy_and_bulk_window() {
    assert_eq!(default_bulk_min_distance_planes([128, 32, 32]), 96);
    assert_eq!(default_bulk_min_distance_planes([180, 32, 32]), 135);

    let cells = [12, 8, 8];
    let model = test_model(cells, vec![0], 1.0e-3, 8);
    let sites = model.lattice().site_count();
    let cr_count = (0.10 * sites as f64).round() as usize;
    let bulk_min = default_bulk_min_distance_planes(cells);

    let n_draw = 64_usize;
    let mut tot_vals = Vec::with_capacity(n_draw);
    let mut lin_vals = Vec::with_capacity(n_draw);
    for id in 0..n_draw {
        let mut rng = KmcRng::for_replica(77, id as u64, StreamRole::InitialState);
        let occ = place_random_cr(sites, cr_count, &mut rng);
        let state = model.state_with_defects(occ, &[], id as u64).unwrap();
        let counts = PlaneCounts::from_state(&model, &state).unwrap();
        let m = compute_segregation_metrics(&counts, 0, bulk_min, DEFAULT_LATTICE_A_M).unwrap();
        tot_vals.push(m.s_cr_tot_nm);
        lin_vals.push(m.s_cr_linear_nm);
    }

    let mean_tot = tot_vals.iter().sum::<f64>() / n_draw as f64;
    let se_tot = (tot_vals
        .iter()
        .map(|&v| (v - mean_tot).powi(2))
        .sum::<f64>()
        / ((n_draw - 1) * n_draw) as f64)
        .sqrt();
    let mean_lin = lin_vals.iter().sum::<f64>() / n_draw as f64;
    assert!(
        mean_tot.abs() <= 3.0 * se_tot,
        "random alloy S_Cr^tot bias: mean={mean_tot}, se={se_tot}"
    );
    assert!(mean_lin.abs() < 0.05, "mean linear excess {mean_lin}");
}

#[test]
fn t3_continuous_time_plane_defect_tally_and_strict_crossing() {
    let cells = [6, 6, 6];
    let model = test_model(cells, vec![5], 1.0, 16);
    let sites_per_plane = 36_usize;
    let n = model.lattice().site_count();

    let mut v_plane2 = 0;
    let mut d_plane8 = 0;
    for s in 0..n {
        if plane_x(cells, s) == 2 {
            v_plane2 = s;
            break;
        }
    }
    for s in 0..n {
        if plane_x(cells, s) == 8 {
            d_plane8 = s;
            break;
        }
    }
    let final_state = model
        .state_with_defects(
            vec![Species::Fe; n],
            &[
                Defect::Vacancy { site: v_plane2 },
                Defect::Dumbbell {
                    site: d_plane8,
                    axis: 0,
                    plus: Species::Fe,
                    minus: Species::Fe,
                },
            ],
            1,
        )
        .unwrap();
    let counts = PlaneCounts::from_state(&model, &final_state).unwrap();

    let mut tally = PlaneDefectTally::new(cells).unwrap();
    // Pre-steady [0, 0.5] -> zero overlap with [1.0, 4.0].
    tally.update_active(&[(2, true), (8, false)], 0.0);
    // Straddling start [0.5, 2.0] -> overlap 1.0 s.
    tally.update_active(&[(2, true), (8, false)], 1.0);
    // Interior [2.0, 3.0] -> overlap 1.0 s (vacancy moves to plane 3, two dumbbells at 8).
    tally.update_active(&[(3, true), (8, false), (8, false)], 1.0);
    // Straddling end [3.0, 5.0] -> overlap 1.0 s.
    tally.update_active(&[(2, true)], 1.0);

    let profile = tally
        .into_profile(&counts, 5, 4, DEFAULT_LATTICE_A_M)
        .unwrap();
    assert!((profile.steady_elapsed_s - 3.0).abs() < 1e-14);
    assert_eq!(profile.d_planes, (-6..6).collect::<Vec<_>>());
    let dx_nm = 0.5 * DEFAULT_LATTICE_A_M * 1.0e9;
    assert_eq!(profile.distance_nm[6], 0.0);
    assert!((profile.distance_nm[0] + 6.0 * dx_nm).abs() < 1e-15);

    // Sink at plane 5 -> plane 2 is d = -3 (idx 3), plane 3 is d = -2 (idx 4), plane 8 is d = +3 (idx 9).
    let expected_v2 = (2.0 / 3.0) / sites_per_plane as f64;
    let expected_v3 = (1.0 / 3.0) / sites_per_plane as f64;
    let expected_i8 = (3.0 / 3.0) / sites_per_plane as f64;
    assert!((profile.c_v_mean[3] - expected_v2).abs() < 1e-14);
    assert!((profile.c_v_mean[4] - expected_v3).abs() < 1e-14);
    assert!((profile.c_i_mean[9] - expected_i8).abs() < 1e-14);
    assert!((profile.v_mean_total - 1.0).abs() < 1e-14);
    assert!((profile.i_mean_total - 1.0).abs() < 1e-14);

    // Also verify run_ris_segregation matches manual Clock-ordered trajectory sampling bit-for-bit.
    let m_sim = test_model([6, 6, 6], vec![0], 1.0e5, 32);
    let config = RisSegregationConfig::uniform(40, 2.0e-3, 4, 5.0e-4, 4, 1, 99).unwrap();
    let rep = &run_ris_segregation(&m_sim, &config).unwrap()[0];

    let mut init_rng = KmcRng::for_replica(99, 0, StreamRole::InitialState);
    let occ = place_random_cr(m_sim.lattice().site_count(), 40, &mut init_rng);
    let s0 = m_sim.state_with_defects(occ, &[], 99).unwrap();
    let mut sim = IncrementalSimulation::with_rng(
        m_sim.clone(),
        s0,
        KmcRng::for_replica(99, 0, StreamRole::Dynamics),
    )
    .unwrap();
    let t_samples: Vec<f64> = config
        .sample_doses
        .iter()
        .map(|&d| d / m_sim.params().k0_dpa_per_s)
        .collect();
    let mut manual_gb = Vec::new();
    let mut idx = 0;
    while idx < t_samples.len() {
        let t_target = t_samples[idx];
        let mut pre_counts = None;
        match sim
            .step_observing(|m, state_before, next_clock| {
                if next_clock.cmp_f64(t_target).is_gt() {
                    pre_counts = Some(PlaneCounts::from_state(m, state_before).unwrap());
                }
            })
            .unwrap()
        {
            Step::Fired { .. } => {
                if let Some(counts) = pre_counts {
                    let met =
                        compute_segregation_metrics(&counts, 0, 4, DEFAULT_LATTICE_A_M).unwrap();
                    while idx < t_samples.len() && sim.clock().cmp_f64(t_samples[idx]).is_gt() {
                        manual_gb.push(met.c_cr_gb);
                        idx += 1;
                    }
                }
            }
            Step::Absorbed => panic!("unexpected absorption"),
        }
    }
    assert_eq!(sim.steps(), rep.steps);
    for (s, &expected_gb) in rep.samples.iter().zip(&manual_gb) {
        assert_eq!(s.metrics.c_cr_gb.to_bits(), expected_gb.to_bits());
    }
    let mut check_clk = Clock::new();
    check_clk.advance(1.0).unwrap();
    assert_eq!(rep.samples.len(), 4);
}

#[test]
fn t4_exponential_relaxation_fit_recovers_synthetic_curves() {
    let doses = [0.005, 0.015, 0.03, 0.06, 0.12, 0.24, 0.35];

    let y_gb: Vec<f64> = doses
        .iter()
        .map(|&d| 0.132 + (0.10 - 0.132) * (-d / 0.023_f64).exp())
        .collect();
    let fit_gb = fit_exponential_relaxation(&doses, &y_gb, 0.10).unwrap();
    assert!((fit_gb.phi_rel - 0.023).abs() < 1e-10);
    assert!((fit_gb.y_inf - 0.132).abs() < 1e-12);
    assert!(fit_gb.rmse < 1e-12 && fit_gb.r2 > 0.999_999_999);

    let y_s: Vec<f64> = doses
        .iter()
        .map(|&d| 0.58 * (1.0 - (-d / 0.107_f64).exp()))
        .collect();
    let fit_s = fit_exponential_relaxation(&doses, &y_s, 0.0).unwrap();
    assert!((fit_s.phi_rel - 0.107).abs() < 1e-10);
    assert!((fit_s.y_inf - 0.58).abs() < 1e-12);

    let y_dep: Vec<f64> = doses
        .iter()
        .map(|&d| 0.075 + (0.10 - 0.075) * (-d / 0.045_f64).exp())
        .collect();
    let fit_dep = fit_exponential_relaxation(&doses, &y_dep, 0.10).unwrap();
    assert!((fit_dep.phi_rel - 0.045).abs() < 1e-10);
    assert!((fit_dep.y_inf - 0.075).abs() < 1e-12);

    assert!(fit_exponential_relaxation(&[0.1, 0.2], &[0.1, 0.2], 0.0).is_none());
    assert!(fit_exponential_relaxation(&[0.2, 0.1, 0.3], &[0.1, 0.2, 0.3], 0.0).is_none());
    assert!(fit_exponential_relaxation(&[0.0, 0.1, 0.2], &[0.1, 0.2, 0.3], 0.0).is_none());
    assert!(fit_exponential_relaxation(&doses, &[0.1; 7], 0.0).is_none());
    assert!(fit_exponential_relaxation(&[0.1, 0.2, f64::NAN], &[0.1, 0.2, 0.3], 0.0).is_none());
}

#[test]
fn t5_validation_and_error_guards() {
    let m_ok = test_model([6, 6, 6], vec![0], 1.0e5, 32);
    let m_no_sink = test_model([6, 6, 6], vec![], 1.0e5, 32);
    let m_two_sinks = test_model([6, 6, 6], vec![0, 6], 1.0e5, 32);
    let m_no_k0 = test_model([6, 6, 6], vec![0], 0.0, 32);

    let cfg = RisSegregationConfig::uniform(40, 1e-3, 2, 0.0, 4, 1, 7).unwrap();
    assert!(matches!(
        run_ris_segregation(&m_no_sink, &cfg),
        Err(RisSegregationError::InvalidSinkPlanes(0))
    ));
    assert!(matches!(
        run_ris_segregation(&m_two_sinks, &cfg),
        Err(RisSegregationError::InvalidSinkPlanes(2))
    ));
    let fe_state = m_ok
        .state_with_defects(vec![Species::Fe; m_ok.lattice().site_count()], &[], 1)
        .unwrap();
    let fe_counts = PlaneCounts::from_state(&m_ok, &fe_state).unwrap();
    assert!(matches!(
        compute_segregation_metrics(&fe_counts, 12, 4, DEFAULT_LATTICE_A_M),
        Err(RisSegregationError::SinkPlaneOutOfRange {
            sink_plane: 12,
            planes: 12
        })
    ));
    assert!(matches!(
        run_ris_segregation(&m_no_k0, &cfg),
        Err(RisSegregationError::NonPositiveCreationRate(_))
    ));

    let mut bad_cr = cfg.clone();
    bad_cr.cr_count = m_ok.lattice().site_count() + 1;
    assert!(matches!(
        run_ris_segregation(&m_ok, &bad_cr),
        Err(RisSegregationError::InvalidCrCount { .. })
    ));

    let mut bad_bulk = cfg.clone();
    bad_bulk.bulk_min_distance_planes = 0;
    assert!(matches!(
        run_ris_segregation(&m_ok, &bad_bulk),
        Err(RisSegregationError::InvalidBulkWindow { .. })
    ));
    bad_bulk.bulk_min_distance_planes = 7;
    assert!(matches!(
        run_ris_segregation(&m_ok, &bad_bulk),
        Err(RisSegregationError::InvalidBulkWindow { .. })
    ));

    let mut bad_steady = cfg.clone();
    bad_steady.steady_start_dpa = 1e-3;
    assert!(matches!(
        run_ris_segregation(&m_ok, &bad_steady),
        Err(RisSegregationError::InvalidSteadyStart { .. })
    ));
    bad_steady.steady_start_dpa = -0.1;
    assert!(matches!(
        run_ris_segregation(&m_ok, &bad_steady),
        Err(RisSegregationError::InvalidSteadyStart { .. })
    ));

    let mut bad_grid = cfg.clone();
    bad_grid.sample_doses = vec![2e-3, 1e-3];
    assert!(matches!(
        run_ris_segregation(&m_ok, &bad_grid),
        Err(RisSegregationError::InvalidDoseGrid(_))
    ));
    bad_grid.sample_doses = vec![];
    assert!(matches!(
        run_ris_segregation(&m_ok, &bad_grid),
        Err(RisSegregationError::InvalidDoseGrid(_))
    ));

    let mut bad_runs = cfg.clone();
    bad_runs.runs = 0;
    assert!(matches!(
        run_ris_segregation(&m_ok, &bad_runs),
        Err(RisSegregationError::InsufficientReplicas(0))
    ));

    let bad_lat = cfg.clone().with_lattice_a_m(0.0);
    assert!(matches!(
        run_ris_segregation(&m_ok, &bad_lat),
        Err(RisSegregationError::InvalidLatticeParameter(_))
    ));

    let tight_steps = RisSegregationConfig::uniform(40, 1.0, 2, 0.0, 4, 1, 7)
        .unwrap()
        .with_max_steps(Some(1));
    assert!(matches!(
        run_ris_segregation(&m_ok, &tight_steps),
        Err(RisSegregationError::MaxStepsExceeded { max_steps: 1, .. })
    ));

    let m_tiny_cap = test_model([6, 6, 6], vec![0], 1.0e8, 2);
    let cap_cfg = RisSegregationConfig::uniform(40, 1.0, 2, 0.0, 4, 1, 7).unwrap();
    assert!(matches!(
        run_ris_segregation(&m_tiny_cap, &cap_cfg),
        Err(RisSegregationError::Sim {
            source: okrates_core::SimError::Model(FeCrRisError::PoolExhausted { .. }),
            ..
        })
    ));

    let pure_cr = m_ok
        .state_with_defects(vec![Species::Cr; m_ok.lattice().site_count()], &[], 1)
        .unwrap();
    let cr_counts = PlaneCounts::from_state(&m_ok, &pure_cr).unwrap();
    assert!(matches!(
        compute_segregation_metrics(&cr_counts, 0, 4, DEFAULT_LATTICE_A_M),
        Err(RisSegregationError::ZeroFePlane { plane: 0 })
    ));

    assert!(matches!(
        summarize_ris_segregation(&[]),
        Err(RisSegregationError::InsufficientReplicas(0))
    ));
}

#[test]
fn t6_determinism_and_conservation_smoke() {
    let model = test_model([8, 6, 6], vec![0], 1.0e3, 32);
    let config = RisSegregationConfig::uniform(43, 2.0e-3, 4, 5.0e-4, 4, 3, u64::MAX).unwrap();

    let run_a = run_ris_segregation(&model, &config).unwrap();
    let run_b = run_ris_segregation(&model, &config).unwrap();
    assert_eq!(run_a.len(), 3);
    for (idx, (ra, rb)) in run_a.iter().zip(&run_b).enumerate() {
        assert_eq!(ra.replica_id, idx as u64);
        assert_eq!(ra.steps, rb.steps);
        assert_eq!(ra.final_time_s.to_bits(), rb.final_time_s.to_bits());
        assert_eq!(ra.samples, rb.samples);
        assert_eq!(ra.profile, rb.profile);
    }

    let sum_a = summarize_ris_segregation(&run_a).unwrap();
    let sum_b = summarize_ris_segregation(&run_b).unwrap();
    assert_eq!(sum_a, sum_b);
    assert_eq!(sum_a.runs, 3);
    assert_eq!(sum_a.doses.len(), 4);
    assert_eq!(sum_a.d_planes.len(), 16);
}
