use okrates_core::testing::{chi2_statistic, chi2_upper_quantile, mat_exp, solve_stationary_ctmc};
use okrates_core::{
    AbsorptionPolicy, CheckpointError, Checkpointable, IncrementalModel, IncrementalSimulation,
    KmcRng, Model, ReversibleModel, SampleGrid, SimError, Simulation, Step, StreamRole,
    run_ensemble, sample_trajectory, verify_incremental_step_parity,
};
use okrates_models::lattice_hopping::{
    BasisSite, HopStencil, IonHop, LatticeHopping, LatticeHoppingError, PathwayParams,
    PeriodicLattice,
};
use okrates_models::vacancy::BOLTZMANN_EV_PER_K;

fn mean_and_se(values: &[f64]) -> (f64, f64) {
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    let var = values.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
    (mean, (var / n).sqrt())
}

#[test]
fn t1_non_bcc_single_ion_analytic_tracer_diffusion_sc_fcc_and_square2d() {
    let rate = 500.0_f64;
    let t_end = 5e-3_f64;
    let grid = SampleGrid::from_origin(t_end, 1).unwrap();
    let runs = 2_000_u64;

    // 1. 3D Simple Cubic: z = 6, jump length = a => MSD(t) = 6 * rate * a^2 * t
    let a_sc = 3.0e-10_f64;
    let sc = PeriodicLattice::simple_cubic([6, 6, 6], a_sc).unwrap();
    assert_eq!(sc.spatial_dimension(), 3);
    assert_eq!(sc.max_degree(), 6);
    let m_sc = LatticeHopping::homogeneous(sc, rate).unwrap();
    let s_sc = m_sc.initial_state(&[0]).unwrap();
    let msds_sc = run_ensemble(11_001, runs, |ctx| {
        let mut sim = IncrementalSimulation::with_rng(
            m_sc.clone(),
            s_sc.clone(),
            ctx.rng(StreamRole::Dynamics),
        )
        .unwrap();
        sample_trajectory(&mut sim, &grid, None, AbsorptionPolicy::Error, |m, st| {
            m.check_invariants(st).unwrap();
            st.mean_tracer_msd_m2(m.lattice())
        })
        .unwrap()
        .values[0]
    });
    let (mean_sc, se_sc) = mean_and_se(&msds_sc);
    let exact_sc = 6.0 * rate * a_sc * a_sc * t_end;
    assert!(
        (mean_sc - exact_sc).abs() <= 3.5 * se_sc,
        "SC MSD {mean_sc:e} +/- {se_sc:e} vs exact {exact_sc:e}"
    );

    // 2. 3D FCC: z = 12, jump length = a / sqrt(2) => MSD(t) = 12 * rate * (a^2 / 2) * t = 6 * rate * a^2 * t
    let a_fcc = 4.0e-10_f64;
    let fcc = PeriodicLattice::fcc([4, 4, 4], a_fcc).unwrap();
    assert_eq!(fcc.spatial_dimension(), 3);
    assert_eq!(fcc.site_count(), 256);
    assert_eq!(fcc.max_degree(), 12);
    for site in 0..fcc.site_count() {
        assert_eq!(fcc.degree(site), 12);
        for hop in fcc.hops(site) {
            let l2 = fcc.squared_displacement_m2(hop.delta_ticks);
            assert!((l2 - 0.5 * a_fcc * a_fcc).abs() < 1e-12 * a_fcc * a_fcc);
        }
    }
    let m_fcc = LatticeHopping::homogeneous(fcc, rate).unwrap();
    let s_fcc = m_fcc.initial_state(&[0]).unwrap();
    let msds_fcc = run_ensemble(12_001, runs, |ctx| {
        let mut sim = IncrementalSimulation::with_rng(
            m_fcc.clone(),
            s_fcc.clone(),
            ctx.rng(StreamRole::Dynamics),
        )
        .unwrap();
        sample_trajectory(&mut sim, &grid, None, AbsorptionPolicy::Error, |m, st| {
            m.check_invariants(st).unwrap();
            assert!(
                (st.collective_msd_m2(m.lattice()) - st.mean_tracer_msd_m2(m.lattice())).abs()
                    < 1e-30
            );
            st.mean_tracer_msd_m2(m.lattice())
        })
        .unwrap()
        .values[0]
    });
    let (mean_fcc, se_fcc) = mean_and_se(&msds_fcc);
    let exact_fcc = 6.0 * rate * a_fcc * a_fcc * t_end;
    assert!(
        (mean_fcc - exact_fcc).abs() <= 3.5 * se_fcc,
        "FCC MSD {mean_fcc:e} +/- {se_fcc:e} vs exact {exact_fcc:e}"
    );

    // 3. 2D Square: z = 4, jump length = a => MSD(t) = 4 * rate * a^2 * t, z-displacement == 0
    let a_sq = 3.5e-10_f64;
    let sq = PeriodicLattice::square_2d([8, 8], a_sq).unwrap();
    assert_eq!(sq.spatial_dimension(), 2);
    assert_eq!(sq.site_count(), 64);
    assert_eq!(sq.max_degree(), 4);
    let m_sq = LatticeHopping::homogeneous(sq, rate).unwrap();
    let s_sq = m_sq.initial_state(&[0]).unwrap();
    let msds_sq = run_ensemble(13_001, runs, |ctx| {
        let mut sim = IncrementalSimulation::with_rng(
            m_sq.clone(),
            s_sq.clone(),
            ctx.rng(StreamRole::Dynamics),
        )
        .unwrap();
        sample_trajectory(&mut sim, &grid, None, AbsorptionPolicy::Error, |m, st| {
            m.check_invariants(st).unwrap();
            assert_eq!(st.displacements_ticks()[0][2], 0);
            assert_eq!(st.ion_displacement_m(m.lattice(), 0).unwrap()[2], 0.0);
            st.mean_tracer_msd_m2(m.lattice())
        })
        .unwrap()
        .values[0]
    });
    let (mean_sq, se_sq) = mean_and_se(&msds_sq);
    let exact_sq = 4.0 * rate * a_sq * a_sq * t_end;
    assert!(
        (mean_sq - exact_sq).abs() <= 3.5 * se_sq,
        "2D Square MSD {mean_sq:e} +/- {se_sq:e} vs exact {exact_sq:e}"
    );
}

#[test]
fn t2_multi_sublattice_energetics_obeys_exact_ctmc_stationary_boltzmann_and_transient_oracles() {
    // 2D bipartite crystal with 2 inequivalent basis sites:
    //   basis 0 (type 0, degree 4): bridges along x to basis 1 (pathway 0) and hops directly along y to basis 0 (pathway 1)
    //   basis 1 (type 1, degree 2): bridges along x to basis 0 (pathway 0)
    let a = 3.0e-10_f64;
    let basis = [
        BasisSite {
            offset_ticks: [0, 0, 0],
            site_type: 0,
        },
        BasisSite {
            offset_ticks: [1, 0, 0],
            site_type: 1,
        },
    ];
    let stencils = [
        HopStencil {
            from_basis: 0,
            delta_ticks: [1, 0, 0],
            pathway: 0,
        },
        HopStencil {
            from_basis: 0,
            delta_ticks: [-1, 0, 0],
            pathway: 0,
        },
        HopStencil {
            from_basis: 0,
            delta_ticks: [0, 2, 0],
            pathway: 1,
        },
        HopStencil {
            from_basis: 0,
            delta_ticks: [0, -2, 0],
            pathway: 1,
        },
        HopStencil {
            from_basis: 1,
            delta_ticks: [1, 0, 0],
            pathway: 0,
        },
        HopStencil {
            from_basis: 1,
            delta_ticks: [-1, 0, 0],
            pathway: 0,
        },
    ];
    let metric = [[0.5 * a, 0.0, 0.0], [0.0, 0.5 * a, 0.0], [0.0, 0.0, 0.0]];
    let lattice = PeriodicLattice::from_unit_cell(
        [2, 2, 1],
        [2, 2, 1],
        [true, true, false],
        metric,
        &basis,
        &stencils,
    )
    .unwrap();
    let n_sites = lattice.site_count();
    assert_eq!(n_sites, 8);
    assert_eq!(lattice.max_degree(), 4);

    let site_energies_ev = vec![0.0_f64, 0.16_f64];
    let pathways = vec![
        PathwayParams {
            attempt_frequency_hz: 1e12,
            migration_barrier_ev: 0.05,
        },
        PathwayParams {
            attempt_frequency_hz: 2e12,
            migration_barrier_ev: 0.25,
        },
    ];
    let temperature_k = 600.0_f64;
    let model = LatticeHopping::with_energetics(
        lattice,
        site_energies_ev.clone(),
        pathways.clone(),
        temperature_k,
    )
    .unwrap();

    // Independent analytical generator oracle activating SymmetricFloor on pathway 0 (1->0 clamps -0.03 to 0; 0->1 clamps 0.13 to 0.16).
    let kbt = BOLTZMANN_EV_PER_K * temperature_k;
    let mut rates = vec![vec![0.0_f64; n_sites]; n_sites];
    for (i, row) in rates.iter_mut().enumerate() {
        let e_from = site_energies_ev[model.lattice().site_type(i).unwrap()];
        for hop in model.lattice().hops(i) {
            let e_to = site_energies_ev[model.lattice().site_type(hop.target).unwrap()];
            let delta_e = e_to - e_from;
            let pw = pathways[hop.pathway];
            let act_ev = (pw.migration_barrier_ev + 0.5 * delta_e)
                .max(0.0)
                .max(delta_e);
            row[hop.target] += pw.attempt_frequency_hz * (-act_ev / kbt).exp();
        }
        let st_i = model.initial_state(&[i]).unwrap();
        let mut model_row = vec![0.0_f64; n_sites];
        model.enumerate_events(&st_i, &mut |IonHop { branch, .. }, r| {
            let target = model.lattice().hops(i)[branch].target;
            model_row[target] += r;
        });
        for (target, (&oracle_r, &model_r)) in row.iter().zip(&model_row).enumerate() {
            assert!(
                (model_r - oracle_r).abs() <= 1e-13 * oracle_r.max(1.0),
                "site {i} -> {target}: model={model_r} vs oracle={oracle_r}"
            );
        }
    }

    let pi = solve_stationary_ctmc(&rates);
    let mut exact_boltzmann: Vec<f64> = (0..n_sites)
        .map(|s| {
            let e = site_energies_ev[model.lattice().site_type(s).unwrap()];
            (-e / kbt).exp()
        })
        .collect();
    let z: f64 = exact_boltzmann.iter().sum();
    for p in &mut exact_boltzmann {
        *p /= z;
    }
    for i in 0..n_sites {
        assert!(
            (pi[i] - exact_boltzmann[i]).abs() < 1e-13,
            "site {i}: pi={} vs Boltzmann={}",
            pi[i],
            exact_boltzmann[i]
        );
        for j in 0..n_sites {
            assert!(
                (pi[i] * rates[i][j] - pi[j] * rates[j][i]).abs()
                    <= 1e-12 * (pi[i] * rates[i][j]).max(1.0),
                "detailed balance violated ({i}, {j})"
            );
        }
    }

    let scaled_q = |t: f64| -> Vec<Vec<f64>> {
        let mut q = vec![vec![0.0; n_sites]; n_sites];
        for i in 0..n_sites {
            let mut sum = 0.0;
            for j in 0..n_sites {
                if i != j {
                    let v = rates[i][j] * t;
                    q[i][j] = v;
                    sum += v;
                }
            }
            q[i][i] = -sum;
        }
        q
    };

    let t_transient = 2.0e-11_f64;
    let t_stationary = 1.0e-9_f64;
    let p_transient = mat_exp(&scaled_q(t_transient))[0].clone();
    let p_eq = mat_exp(&scaled_q(t_stationary))[0].clone();
    let max_eq_err = p_eq
        .iter()
        .zip(&pi)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        max_eq_err < 1e-6,
        "stationary relaxation error={max_eq_err:e}"
    );
    assert!(
        (p_transient[0] - pi[0]).abs() > 0.10,
        "transient p0={} vs pi0={}",
        p_transient[0],
        pi[0]
    );

    let grid = SampleGrid::windowed(t_transient, t_stationary, 2).unwrap();
    let replicas = 20_000_u64;
    for (idx, p) in p_transient.iter().enumerate() {
        assert!(
            p * (replicas as f64) >= 5.0,
            "transient bin {idx} too sparse"
        );
    }
    let init = model.initial_state(&[0]).unwrap();
    let direct_obs = run_ensemble(21_001, replicas, |ctx| {
        let mut sim =
            Simulation::with_rng(model.clone(), init.clone(), ctx.rng(StreamRole::Dynamics));
        let traj = sample_trajectory(&mut sim, &grid, None, AbsorptionPolicy::Error, |_, st| {
            st.sites()[0]
        })
        .unwrap();
        [traj.values[0], traj.values[1]]
    });
    let inc_obs = run_ensemble(22_001, replicas, |ctx| {
        let mut sim = IncrementalSimulation::with_rng(
            model.clone(),
            init.clone(),
            ctx.rng(StreamRole::Dynamics),
        )
        .unwrap();
        let traj = sample_trajectory(&mut sim, &grid, None, AbsorptionPolicy::Error, |_, st| {
            st.sites()[0]
        })
        .unwrap();
        [traj.values[0], traj.values[1]]
    });

    let crit = chi2_upper_quantile(n_sites - 1, 1e-4);
    for (label, obs) in [("direct", direct_obs), ("incremental", inc_obs)] {
        let mut trans_counts = vec![0_u64; n_sites];
        let mut eq_counts = vec![0_u64; n_sites];
        for [s_t, s_e] in obs {
            trans_counts[s_t] += 1;
            eq_counts[s_e] += 1;
        }
        let chi2_t = chi2_statistic(&trans_counts, &p_transient);
        assert!(
            chi2_t <= crit,
            "{label} transient chi2={chi2_t:.3} > {crit:.3}"
        );
        let chi2_e = chi2_statistic(&eq_counts, &pi);
        assert!(
            chi2_e <= crit,
            "{label} stationary chi2={chi2_e:.3} > {crit:.3}"
        );
    }
}

#[test]
fn t3_crowded_multi_ion_exclusion_direct_vs_incremental_parity_and_invariants() {
    // Part A: Crowded 3D FCC (32 sites, 12 ions -> theta = 0.375)
    let a = 3.8e-10_f64;
    let fcc = PeriodicLattice::fcc([2, 2, 2], a).unwrap();
    let m_fcc = LatticeHopping::homogeneous(fcc, 1_000.0).unwrap();
    let mut init_rng = KmcRng::from_seed(999);
    let s_fcc = m_fcc.random_state_with_rng(12, &mut init_rng).unwrap();

    // Part B: Non-periodic finite 1D/2D strip with boundary truncation (variable degree < max_degree)
    let chain = PeriodicLattice::from_unit_cell(
        [5, 1, 1],
        [1, 1, 1],
        [false, false, false],
        [[a, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        &[BasisSite {
            offset_ticks: [0, 0, 0],
            site_type: 0,
        }],
        &[
            HopStencil {
                from_basis: 0,
                delta_ticks: [-1, 0, 0],
                pathway: 0,
            },
            HopStencil {
                from_basis: 0,
                delta_ticks: [1, 0, 0],
                pathway: 0,
            },
        ],
    )
    .unwrap();
    assert_eq!(chain.degree(0), 1);
    assert_eq!(chain.degree(2), 2);
    assert_eq!(chain.degree(4), 1);
    assert_eq!(chain.max_degree(), 2);
    let m_chain = LatticeHopping::homogeneous(chain, 250.0).unwrap();
    let s_chain = m_chain.initial_state(&[0, 2, 4]).unwrap();

    for (label, model, init, steps) in [
        ("crowded_fcc", m_fcc.clone(), s_fcc.clone(), 400_usize),
        ("bounded_chain", m_chain, s_chain, 150_usize),
    ] {
        let mut direct = Simulation::new(model.clone(), init.clone(), 0xface_1234);
        let mut inc = IncrementalSimulation::new(model.clone(), init, 0xface_1234).unwrap();
        let stride = model.lattice().max_degree();

        for step in 0..steps {
            let before = inc.state().clone();

            let mut enum_rates = vec![0.0_f64; model.slot_count(&before)];
            model.enumerate_events(&before, &mut |IonHop { ion, branch }, r| {
                let slot = ion * stride + branch;
                assert_eq!(enum_rates[slot], 0.0);
                enum_rates[slot] = r;
            });
            for (slot, &er) in enum_rates.iter().enumerate() {
                assert_eq!(
                    model.slot_rate(&before, slot).to_bits(),
                    er.to_bits(),
                    "{label} step {step} slot {slot}"
                );
            }

            let (
                Step::Fired {
                    event: ie,
                    dt: i_dt,
                },
                Step::Fired {
                    event: de,
                    dt: d_dt,
                },
            ) = (inc.step().unwrap(), direct.step().unwrap())
            else {
                panic!("{label} unexpected absorption at step {step}");
            };
            assert_eq!(ie, de, "{label} step {step} event mismatch");
            assert!(
                (i_dt - d_dt).abs() <= 1e-13 * i_dt.max(1.0),
                "{label} step {step} dt mismatch"
            );
            assert_eq!(
                inc.state(),
                direct.state(),
                "{label} state divergence at step {step}"
            );
            model.check_invariants(inc.state()).unwrap();
            verify_incremental_step_parity(&model, &before, inc.state(), &ie, inc.tree(), 0.0)
                .unwrap_or_else(|e| panic!("{label} parity failed at step {step}: {e}"));
        }
    }

    // Ensemble check on crowded FCC: Direct and Incremental agree on both tracer and collective MSD,
    // and site exclusion reduces tracer MSD relative to collective MSD (Haven correlation).
    let grid = SampleGrid::from_origin(5e-3, 1).unwrap();
    let runs = 200_u64;
    let direct_vals = run_ensemble(31_001, runs, |ctx| {
        let mut sim =
            Simulation::with_rng(m_fcc.clone(), s_fcc.clone(), ctx.rng(StreamRole::Dynamics));
        sample_trajectory(&mut sim, &grid, None, AbsorptionPolicy::Error, |m, st| {
            (
                st.mean_tracer_msd_m2(m.lattice()),
                st.collective_msd_m2(m.lattice()),
            )
        })
        .unwrap()
        .values[0]
    });
    let inc_vals = run_ensemble(32_001, runs, |ctx| {
        let mut sim = IncrementalSimulation::with_rng(
            m_fcc.clone(),
            s_fcc.clone(),
            ctx.rng(StreamRole::Dynamics),
        )
        .unwrap();
        sample_trajectory(&mut sim, &grid, None, AbsorptionPolicy::Error, |m, st| {
            (
                st.mean_tracer_msd_m2(m.lattice()),
                st.collective_msd_m2(m.lattice()),
            )
        })
        .unwrap()
        .values[0]
    });
    let (d_tr, d_tr_se) = mean_and_se(&direct_vals.iter().map(|v| v.0).collect::<Vec<_>>());
    let (i_tr, i_tr_se) = mean_and_se(&inc_vals.iter().map(|v| v.0).collect::<Vec<_>>());
    let (d_co, d_co_se) = mean_and_se(&direct_vals.iter().map(|v| v.1).collect::<Vec<_>>());
    let (i_co, i_co_se) = mean_and_se(&inc_vals.iter().map(|v| v.1).collect::<Vec<_>>());
    assert!((d_tr - i_tr).abs() <= 4.0 * (d_tr_se.powi(2) + i_tr_se.powi(2)).sqrt());
    assert!((d_co - i_co).abs() <= 4.0 * (d_co_se.powi(2) + i_co_se.powi(2)).sqrt());
    assert!(
        d_tr < d_co,
        "expected tracer correlation under exclusion: tracer={d_tr:e} vs coll={d_co:e}"
    );
}

#[test]
fn t4_reversible_undo_checkpoint_roundtrip_and_validation_errors() {
    let sc = PeriodicLattice::simple_cubic([3, 3, 3], 3.0e-10).unwrap();
    let basis0 = [BasisSite {
        offset_ticks: [0, 0, 0],
        site_type: 0,
    }];
    let st_x = [
        HopStencil {
            from_basis: 0,
            delta_ticks: [-1, 0, 0],
            pathway: 0,
        },
        HopStencil {
            from_basis: 0,
            delta_ticks: [1, 0, 0],
            pathway: 0,
        },
    ];
    let eye = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

    assert!(matches!(
        PeriodicLattice::simple_cubic([1, 2, 2], 3.0e-10).unwrap_err(),
        LatticeHoppingError::InvalidCells { .. }
    ));
    assert_eq!(
        PeriodicLattice::from_unit_cell([2, 2, 2], [0, 1, 1], [true; 3], eye, &basis0, &st_x)
            .unwrap_err(),
        LatticeHoppingError::InvalidSubTicks([0, 1, 1])
    );
    assert_eq!(
        PeriodicLattice::from_unit_cell(
            [2000, 2000, 2000],
            [1, 1, 1],
            [true; 3],
            eye,
            &basis0,
            &st_x
        )
        .unwrap_err(),
        LatticeHoppingError::LatticeTooLarge {
            cells: [2000, 2000, 2000],
        }
    );
    assert_eq!(
        PeriodicLattice::simple_cubic([2, 2, 2], 0.0).unwrap_err(),
        LatticeHoppingError::InvalidMetric
    );
    assert_eq!(
        PeriodicLattice::from_unit_cell([2, 2, 2], [1, 1, 1], [true; 3], eye, &[], &st_x)
            .unwrap_err(),
        LatticeHoppingError::EmptyBasis
    );
    assert_eq!(
        PeriodicLattice::from_unit_cell(
            [2, 2, 2],
            [1, 1, 1],
            [true; 3],
            eye,
            &[BasisSite {
                offset_ticks: [1, 0, 0],
                site_type: 0,
            }],
            &st_x,
        )
        .unwrap_err(),
        LatticeHoppingError::BasisOffsetOutOfRange {
            basis: 0,
            offset: [1, 0, 0],
            sub_ticks: [1, 1, 1],
        }
    );
    assert_eq!(
        PeriodicLattice::from_unit_cell(
            [2, 2, 2],
            [2, 1, 1],
            [true; 3],
            eye,
            &[basis0[0], basis0[0]],
            &st_x,
        )
        .unwrap_err(),
        LatticeHoppingError::DuplicateBasisOffset([0, 0, 0])
    );
    assert_eq!(
        PeriodicLattice::from_unit_cell([2, 2, 2], [1, 1, 1], [true; 3], eye, &basis0, &[])
            .unwrap_err(),
        LatticeHoppingError::EmptyStencils
    );
    assert_eq!(
        PeriodicLattice::from_unit_cell(
            [2, 2, 2],
            [1, 1, 1],
            [true; 3],
            eye,
            &basis0,
            &[HopStencil {
                from_basis: 1,
                delta_ticks: [1, 0, 0],
                pathway: 0,
            }],
        )
        .unwrap_err(),
        LatticeHoppingError::StencilBasisOutOfRange {
            basis: 1,
            basis_count: 1,
        }
    );
    assert_eq!(
        PeriodicLattice::from_unit_cell(
            [2, 2, 2],
            [1, 1, 1],
            [true; 3],
            eye,
            &basis0,
            &[HopStencil {
                from_basis: 0,
                delta_ticks: [0, 0, 0],
                pathway: 0,
            }],
        )
        .unwrap_err(),
        LatticeHoppingError::SelfLoopStencil {
            basis: 0,
            delta_ticks: [0, 0, 0],
        }
    );
    assert_eq!(
        PeriodicLattice::from_unit_cell(
            [2, 2, 2],
            [2, 1, 1],
            [true; 3],
            eye,
            &basis0,
            &[HopStencil {
                from_basis: 0,
                delta_ticks: [1, 0, 0],
                pathway: 0,
            }],
        )
        .unwrap_err(),
        LatticeHoppingError::UnmatchedStencilTarget {
            from_basis: 0,
            delta_ticks: [1, 0, 0],
        }
    );
    assert_eq!(
        PeriodicLattice::from_unit_cell(
            [2, 2, 2],
            [1, 1, 1],
            [true; 3],
            eye,
            &basis0,
            &[st_x[0], st_x[0]],
        )
        .unwrap_err(),
        LatticeHoppingError::DuplicateDirectedHop {
            site: 0,
            target: 4,
            delta_ticks: [-1, 0, 0],
        }
    );
    assert_eq!(
        PeriodicLattice::from_unit_cell(
            [2, 2, 2],
            [1, 1, 1],
            [true; 3],
            eye,
            &basis0,
            &[HopStencil {
                from_basis: 0,
                delta_ticks: [1, 0, 0],
                pathway: 0,
            }],
        )
        .unwrap_err(),
        LatticeHoppingError::MissingReciprocalHop {
            site: 0,
            target: 4,
            delta_ticks: [1, 0, 0],
        }
    );

    assert!(matches!(
        LatticeHopping::with_energetics(
            sc.clone(),
            vec![f64::NAN],
            vec![PathwayParams {
                attempt_frequency_hz: 1e12,
                migration_barrier_ev: 0.3,
            }],
            500.0,
        )
        .unwrap_err(),
        LatticeHoppingError::InvalidSiteEnergy { site_type: 0, .. }
    ));
    assert_eq!(
        LatticeHopping::with_energetics(
            sc.clone(),
            vec![0.0],
            vec![PathwayParams {
                attempt_frequency_hz: 1e12,
                migration_barrier_ev: 0.3,
            }],
            0.0,
        )
        .unwrap_err(),
        LatticeHoppingError::InvalidTemperature(0.0)
    );
    assert_eq!(
        LatticeHopping::with_energetics(
            sc.clone(),
            vec![0.0],
            vec![PathwayParams {
                attempt_frequency_hz: -1.0,
                migration_barrier_ev: 0.3,
            }],
            500.0,
        )
        .unwrap_err(),
        LatticeHoppingError::InvalidAttemptFrequency {
            pathway: 0,
            frequency_hz: -1.0,
        }
    );
    assert_eq!(
        LatticeHopping::with_energetics(
            sc.clone(),
            vec![0.0],
            vec![PathwayParams {
                attempt_frequency_hz: 1e12,
                migration_barrier_ev: -0.1,
            }],
            500.0,
        )
        .unwrap_err(),
        LatticeHoppingError::InvalidMigrationBarrier {
            pathway: 0,
            barrier_ev: -0.1,
        }
    );
    assert!(matches!(
        LatticeHopping::homogeneous(sc.clone(), f64::NAN).unwrap_err(),
        LatticeHoppingError::InvalidJumpRate(_)
    ));

    let model = LatticeHopping::homogeneous(sc.clone(), 100.0).unwrap();
    assert_eq!(
        model.initial_state(&[]).unwrap_err(),
        LatticeHoppingError::ZeroIons
    );
    assert_eq!(
        model
            .initial_state(&(0..28).collect::<Vec<_>>())
            .unwrap_err(),
        LatticeHoppingError::TooManyIons {
            ions: 28,
            sites: 27,
        }
    );
    assert_eq!(
        model.initial_state(&[27]).unwrap_err(),
        LatticeHoppingError::SiteOutOfRange {
            site: 27,
            sites: 27
        }
    );
    assert_eq!(
        model.initial_state(&[4, 4]).unwrap_err(),
        LatticeHoppingError::DuplicateIonSite(4)
    );

    // Ion 0 at site 0, ion 1 at sc.hops(0)[0].target.
    let neighbor0 = sc.hops(0)[0].target;
    let init = model.initial_state(&[0, neighbor0]).unwrap();
    let mut blocked = init.clone();
    assert_eq!(
        model
            .apply_reversible(&mut blocked, &IonHop { ion: 0, branch: 0 })
            .unwrap_err(),
        LatticeHoppingError::TargetOccupied {
            ion: 0,
            target: neighbor0,
        }
    );
    assert_eq!(
        blocked, init,
        "failed apply_reversible must not mutate state"
    );
    assert_eq!(
        model
            .apply(&mut blocked, &IonHop { ion: 2, branch: 0 })
            .unwrap_err(),
        LatticeHoppingError::IonOutOfRange { ion: 2, ions: 2 }
    );
    assert_eq!(
        model
            .apply(&mut blocked, &IonHop { ion: 0, branch: 6 })
            .unwrap_err(),
        LatticeHoppingError::BranchOutOfRange {
            ion: 0,
            branch: 6,
            degree: 6,
        }
    );

    // Bitwise reversible undo across multi-ion hops.
    let mut work = model.initial_state(&[0, 13]).unwrap();
    let free_branch = (0..sc.degree(0))
        .find(|&b| sc.hops(0)[b].target != 13)
        .unwrap();
    let before = work.clone();
    let undo = model
        .apply_reversible(
            &mut work,
            &IonHop {
                ion: 0,
                branch: free_branch,
            },
        )
        .unwrap();
    assert_ne!(work, before);
    model.check_invariants(&work).unwrap();
    model.undo(&mut work, undo);
    assert_eq!(work, before);

    // Checkpoint/resume bitwise determinism and rejection of corrupt payloads / mismatched models.
    let seed = 0xdead_beef;
    let total_steps = 45;
    let split = 19;
    let mut full_sim = Simulation::new(model.clone(), init.clone(), seed);
    let mut split_sim = Simulation::new(model.clone(), init.clone(), seed);
    let mut full_inc = IncrementalSimulation::new(model.clone(), init.clone(), seed).unwrap();
    let mut split_inc = IncrementalSimulation::new(model.clone(), init, seed).unwrap();

    for step in 0..total_steps {
        assert!(matches!(full_sim.step().unwrap(), Step::Fired { .. }));
        assert!(matches!(full_inc.step().unwrap(), Step::Fired { .. }));
        if step < split {
            split_sim.step().unwrap();
            split_inc.step().unwrap();
        }
    }
    let ckpt_sim = split_sim.checkpoint();
    let ckpt_inc = split_inc.checkpoint();
    let mut res_sim = Simulation::resume(model.clone(), &ckpt_sim).unwrap();
    let mut res_inc = IncrementalSimulation::resume(model.clone(), &ckpt_inc).unwrap();
    for _ in split..total_steps {
        res_sim.step().unwrap();
        res_inc.step().unwrap();
    }
    assert_eq!(res_sim.state(), full_sim.state());
    assert_eq!(res_sim.time().to_bits(), full_sim.time().to_bits());
    assert_eq!(res_sim.checkpoint(), full_sim.checkpoint());
    assert_eq!(res_inc.state(), full_inc.state());
    assert_eq!(res_inc.time().to_bits(), full_inc.time().to_bits());
    assert_eq!(res_inc.checkpoint(), full_inc.checkpoint());

    let other_model =
        LatticeHopping::homogeneous(PeriodicLattice::fcc([2, 2, 2], 4.0e-10).unwrap(), 100.0)
            .unwrap();
    assert!(matches!(
        Simulation::resume(other_model, &ckpt_sim),
        Err(CheckpointError::ParamsMismatch)
    ));
    let mut raw_state = Vec::new();
    model.encode_state(full_sim.state(), &mut raw_state);
    assert!(
        model
            .decode_state(&raw_state[..raw_state.len() - 1])
            .is_err()
    );
    assert!(
        model
            .decode_state(&raw_state[..raw_state.len() - 8])
            .is_err()
    );

    // total_hops crosses u64::MAX without wrapping or panicking when encoded/decoded and stepped.
    let base_sites = [0_usize, 13];
    let st_max_prep = model.initial_state(&base_sites).unwrap();
    let mut max_bytes = Vec::new();
    model.encode_state(&st_max_prep, &mut max_bytes);
    // Word offsets: [n, geometry_hash, moved_ion, moved_site, s0, o0, dx0, dy0, dz0, h0, s1, o1, dx1, dy1, dz1, h1]
    // h0 is word 9 (bytes 72..80).
    max_bytes[72..80].copy_from_slice(&u64::MAX.to_le_bytes());
    let mut st_high = model.decode_state(&max_bytes).unwrap();
    assert_eq!(
        model
            .apply(&mut st_high, &IonHop { ion: 0, branch: 0 })
            .unwrap_err(),
        LatticeHoppingError::HopCountOverflow { ion: 0 }
    );
    model
        .apply(&mut st_high, &IonHop { ion: 1, branch: 0 })
        .unwrap();
    assert_eq!(st_high.total_hops(), u128::from(u64::MAX) + 1);

    // Checked accessors return None for out-of-range indices and Some for valid ones.
    assert_eq!(sc.site_type(sc.site_count()), None);
    assert_eq!(sc.site_coord_ticks(sc.site_count()), None);
    assert!(sc.site_coord_ticks(0).is_some());
    assert_eq!(
        st_max_prep.ion_displacement_m(&sc, st_max_prep.ion_count()),
        None
    );
    assert!(st_max_prep.ion_displacement_m(&sc, 0).is_some());
    assert_eq!(model.transition_rate(0, 0, 0), Some(100.0));
    assert_eq!(model.transition_rate(1, 0, 0), None);
    assert_eq!(model.transition_rate(0, 1, 0), None);
    assert_eq!(model.transition_rate(0, 0, 1), None);

    // Sub-1e-90 metric scale stays valid without cross-product squaring underflow.
    let tiny_sq = PeriodicLattice::square_2d([2, 2], 1e-90).unwrap();
    assert_eq!(tiny_sq.site_count(), 4);
    assert_eq!(tiny_sq.squared_displacement_m2([1, 0, 0]), 1e-180);

    // Sparse site_type label in homogeneous model does not allocate label^2 storage and distinguishes param checkpoints.
    let sparse_lat = PeriodicLattice::from_unit_cell(
        [2, 2, 1],
        [1, 1, 1],
        [true, true, false],
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
        &[BasisSite {
            offset_ticks: [0, 0, 0],
            site_type: 1_000_000,
        }],
        &st_x,
    )
    .unwrap();
    let m_sparse_a = LatticeHopping::homogeneous(sparse_lat.clone(), 12.0).unwrap();
    let m_sparse_b = LatticeHopping::homogeneous(sparse_lat, 24.0).unwrap();
    assert_eq!(
        m_sparse_a.transition_rate(1_000_000, 1_000_000, 0),
        Some(12.0)
    );
    assert_eq!(m_sparse_a.transition_rate(1_000_001, 0, 0), None);
    let s_sparse = m_sparse_a.initial_state(&[0]).unwrap();
    let ckpt_sparse_a = Simulation::new(m_sparse_a, s_sparse, 1);
    let mut pa = Vec::new();
    let mut pb = Vec::new();
    ckpt_sparse_a.model().encode_params(&mut pa);
    m_sparse_b.encode_params(&mut pb);
    assert_ne!(pa, pb);

    // Log-space Arrhenius fallback preserves finite positive rates when exp(-barrier / kbt) underflows f64
    // (both exact-zero exponents <= -746 and subnormal factors like exp(-720)).
    let t_underflow = 1.0_f64;
    let barrier_750 = 750.0 * BOLTZMANN_EV_PER_K * t_underflow;
    let m_underflow = LatticeHopping::with_energetics(
        sc.clone(),
        vec![0.0],
        vec![PathwayParams {
            attempt_frequency_hz: 1e30,
            migration_barrier_ev: barrier_750,
        }],
        t_underflow,
    )
    .unwrap();
    let r_underflow = m_underflow.transition_rate(0, 0, 0).unwrap();
    let expected_750 = 10_f64.powf(30.0 - 750.0 / 10_f64.ln());
    assert!(
        r_underflow > 0.0
            && r_underflow.is_finite()
            && (r_underflow - expected_750).abs() / expected_750 <= 1e-12,
        "r_underflow={r_underflow:e} vs expected={expected_750:e}"
    );
    let subnormal_factor = (-720.0_f64).exp();
    assert!(subnormal_factor > 0.0 && subnormal_factor < f64::MIN_POSITIVE);
    let barrier_720 = 720.0 * BOLTZMANN_EV_PER_K * t_underflow;
    let m_subnormal = LatticeHopping::with_energetics(
        sc.clone(),
        vec![0.0],
        vec![PathwayParams {
            attempt_frequency_hz: 1e30,
            migration_barrier_ev: barrier_720,
        }],
        t_underflow,
    )
    .unwrap();
    let r_subnormal = m_subnormal.transition_rate(0, 0, 0).unwrap();
    let expected_720 = 10_f64.powf(30.0 - 720.0 / 10_f64.ln());
    assert!(
        r_subnormal > 0.0
            && r_subnormal.is_finite()
            && (r_subnormal - expected_720).abs() / expected_720 <= 1e-12,
        "r_subnormal={r_subnormal:e} vs expected={expected_720:e}"
    );

    // Zero-hop non-zero periodic-period displacement is rejected cleanly by check_invariants and decode_state.
    let mut spoof_bytes = Vec::new();
    model.encode_state(&st_max_prep, &mut spoof_bytes);
    // Word 6 is dx0 (bytes 48..56); set to full period 3 while hop_counts[0] == 0.
    spoof_bytes[48..56].copy_from_slice(&3_i64.to_le_bytes());
    assert!(model.decode_state(&spoof_bytes).is_err());

    // Mismatched smaller state in larger model's enumerate_events returns InvalidRate cleanly without panicking.
    let big_model = LatticeHopping::homogeneous(
        PeriodicLattice::simple_cubic([4, 4, 4], 3.0e-10).unwrap(),
        10.0,
    )
    .unwrap();
    let small_state = model.initial_state(&[0]).unwrap();
    let mut mismatched_sim = Simulation::new(big_model, small_state, 5);
    assert!(matches!(
        mismatched_sim.step(),
        Err(SimError::InvalidRate { .. })
    ));

    // Rank-deficient and nearly coplanar metric matrices (det ~ 5.55e-17 or 1e-13 < RANK_TOL) are rejected.
    for bad_metric in [
        [[1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [1.0, 2.0, 3.0]],
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 1e-13]],
    ] {
        assert_eq!(
            PeriodicLattice::from_unit_cell(
                [2, 2, 2],
                [1, 1, 1],
                [true, true, true],
                bad_metric,
                &[BasisSite {
                    offset_ticks: [0, 0, 0],
                    site_type: 0,
                }],
                &st_x,
            )
            .unwrap_err(),
            LatticeHoppingError::InvalidMetric
        );
    }

    // Equal site/ion counts on distinct geometries ([2, 3] vs [3, 2]) fail compatibility across direct,
    // incremental, reversible, and checkpoint seams, while distinct models sharing [2, 3] accept the state.
    let lat_2x3 = PeriodicLattice::square_2d([2, 3], 3.0e-10).unwrap();
    let lat_3x2 = PeriodicLattice::square_2d([3, 2], 3.0e-10).unwrap();
    let m_2x3 = LatticeHopping::homogeneous(lat_2x3.clone(), 10.0).unwrap();
    let m_2x3_alt = LatticeHopping::homogeneous(lat_2x3, 50.0).unwrap();
    let m_3x2 = LatticeHopping::homogeneous(lat_3x2, 10.0).unwrap();
    let mut st_2x3 = m_2x3.initial_state(&[0]).unwrap();
    m_2x3
        .apply(&mut st_2x3, &IonHop { ion: 0, branch: 0 })
        .unwrap();
    assert_eq!(m_2x3_alt.check_invariants(&st_2x3), Ok(()));
    assert!(
        Simulation::new(m_2x3_alt.clone(), st_2x3.clone(), 7)
            .step()
            .is_ok()
    );

    assert!(matches!(
        m_3x2.check_invariants(&st_2x3),
        Err(LatticeHoppingError::StateMismatch(_))
    ));
    assert!(matches!(
        Simulation::new(m_3x2.clone(), st_2x3.clone(), 7).step(),
        Err(SimError::InvalidRate { .. })
    ));
    assert!(matches!(
        IncrementalSimulation::new(m_3x2.clone(), st_2x3.clone(), 7),
        Err(SimError::InvalidRate { .. })
    ));
    let mut rev_buf = st_2x3.clone();
    assert!(matches!(
        m_3x2.apply_reversible(&mut rev_buf, &IonHop { ion: 0, branch: 0 }),
        Err(LatticeHoppingError::StateMismatch(_))
    ));
    let mut bytes_2x3 = Vec::new();
    m_2x3.encode_state(&st_2x3, &mut bytes_2x3);
    assert!(m_3x2.decode_state(&bytes_2x3).is_err());
    assert_eq!(m_2x3_alt.decode_state(&bytes_2x3).unwrap(), st_2x3);

    // Even when max_degree == 0 (no exits on open 2-cell chain with step +/-2 ticks),
    // a valid state absorbs cleanly while an incompatible state still fails in both Direct and Incremental engines.
    let hop2 = [
        HopStencil {
            from_basis: 0,
            delta_ticks: [2, 0, 0],
            pathway: 0,
        },
        HopStencil {
            from_basis: 0,
            delta_ticks: [-2, 0, 0],
            pathway: 0,
        },
    ];
    let lat_zero_hop = PeriodicLattice::from_unit_cell(
        [2, 1, 1],
        [1, 1, 1],
        [false, false, false],
        [
            [3.0e-10, 0.0, 0.0],
            [0.0, 3.0e-10, 0.0],
            [0.0, 0.0, 3.0e-10],
        ],
        &[BasisSite {
            offset_ticks: [0, 0, 0],
            site_type: 0,
        }],
        &hop2,
    )
    .unwrap();
    assert_eq!(lat_zero_hop.max_degree(), 0);
    let m_zero_hop = LatticeHopping::homogeneous(lat_zero_hop, 10.0).unwrap();
    let st_zero_ok = m_zero_hop.initial_state(&[0]).unwrap();
    assert!(matches!(
        Simulation::new(m_zero_hop.clone(), st_zero_ok.clone(), 9)
            .step()
            .unwrap(),
        Step::Absorbed
    ));
    assert!(matches!(
        IncrementalSimulation::new(m_zero_hop.clone(), st_zero_ok, 9)
            .unwrap()
            .step()
            .unwrap(),
        Step::Absorbed
    ));
    let lat_four = PeriodicLattice::from_unit_cell(
        [4, 1, 1],
        [1, 1, 1],
        [false, false, false],
        [
            [3.0e-10, 0.0, 0.0],
            [0.0, 3.0e-10, 0.0],
            [0.0, 0.0, 3.0e-10],
        ],
        &[BasisSite {
            offset_ticks: [0, 0, 0],
            site_type: 0,
        }],
        &hop2,
    )
    .unwrap();
    let st_four = LatticeHopping::homogeneous(lat_four, 10.0)
        .unwrap()
        .initial_state(&[0])
        .unwrap();
    assert!(matches!(
        Simulation::new(m_zero_hop.clone(), st_four.clone(), 9).step(),
        Err(SimError::InvalidRate { .. })
    ));
    assert!(matches!(
        IncrementalSimulation::new(m_zero_hop, st_four, 9),
        Err(SimError::InvalidRate { .. })
    ));
}
