use std::collections::HashMap;

use okrates_core::testing::{chi2_statistic, chi2_upper_quantile, mat_exp, solve_stationary_ctmc};
use okrates_core::{
    AbsorptionPolicy, CheckpointError, Checkpointable, IncrementalModel, IncrementalSimulation,
    Model, ReversibleModel, SampleGrid, SimError, Simulation, Step, StreamRole, run_ensemble,
    sample_trajectory, verify_incremental_step_parity,
};
use okrates_models::lattice_gas::{
    LatticeGas, LatticeGasEnergetics, LatticeGasError, LatticeGasEvent, LatticeGasState, PairRule,
    SingleSiteRule,
};
use okrates_models::lattice_hopping::{BasisSite, HopStencil, PeriodicLattice};
use okrates_models::vacancy::BOLTZMANN_EV_PER_K;

fn mean_and_se(values: &[f64]) -> (f64, f64) {
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    let var = values.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
    (mean, (var / n).sqrt())
}

fn assert_incremental_parity(
    label: &str,
    model: &LatticeGas,
    init: LatticeGasState,
    seed: u64,
    steps: usize,
) {
    let mut direct = Simulation::new(model.clone(), init.clone(), seed);
    let mut inc = IncrementalSimulation::new(model.clone(), init, seed).unwrap();
    let stride =
        model.single_rules().len() + model.lattice().max_degree() * model.pair_rules().len();
    let single_len = model.single_rules().len();
    let pair_len = model.pair_rules().len();

    for step in 0..steps {
        let before = inc.state().clone();
        let mut enum_rates = vec![0.0_f64; model.slot_count(&before)];
        model.enumerate_events(&before, &mut |ev, r| {
            let slot = match ev {
                LatticeGasEvent::SingleSite { site, rule } => site * stride + rule,
                LatticeGasEvent::Pair { site, branch, rule } => {
                    site * stride + single_len + branch * pair_len + rule
                }
            };
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
        verify_incremental_step_parity(model, &before, inc.state(), &ie, inc.tree(), 0.0)
            .unwrap_or_else(|e| panic!("{label} parity failed at step {step}: {e}"));
    }
}

#[test]
fn t1_competitive_langmuir_adsorption_desorption_matches_analytic_isotherm_and_transient_ode() {
    let lat = PeriodicLattice::square_2d([6, 6], 3.0e-10).unwrap();
    let p_a = 0.4_f64;
    let k_ads_a_per_bar = 15.0_f64;
    let k_des_a_base = 4.0_f64;
    let p_b = 1.2_f64;
    let k_ads_b_per_bar = 8.0_f64;
    let k_des_b_base = 6.0_f64;
    let t_k = 450.0_f64;
    let e_a = -0.10_f64;
    let e_b = -0.04_f64;

    let single_rules = vec![
        SingleSiteRule::adsorption(0, p_a, k_ads_a_per_bar).unwrap(),
        SingleSiteRule::desorption(0, k_des_a_base).unwrap(),
        SingleSiteRule::adsorption(1, p_b, k_ads_b_per_bar).unwrap(),
        SingleSiteRule::desorption(1, k_des_b_base).unwrap(),
    ];
    let pair_rules = vec![
        PairRule::diffusion(0, 20.0).unwrap(),
        PairRule::diffusion(1, 15.0).unwrap(),
    ];
    let ener = LatticeGasEnergetics {
        temperature_k: t_k,
        site_energies_ev: vec![e_a, e_b],
        pair_interactions_ev: vec![0.0; 4],
    };
    let model =
        LatticeGas::new(lat.clone(), 2, single_rules.clone(), pair_rules, Some(ener)).unwrap();

    // Independent analytical calculation of effective site rates and Langmuir equilibrium coverage.
    let two_kbt = 2.0 * BOLTZMANN_EV_PER_K * t_k;
    let kbt = BOLTZMANN_EV_PER_K * t_k;
    let r_ads_a = (p_a * k_ads_a_per_bar) * (-e_a / two_kbt).exp();
    let r_des_a = k_des_a_base * (e_a / two_kbt).exp();
    let r_ads_b = (p_b * k_ads_b_per_bar) * (-e_b / two_kbt).exp();
    let r_des_b = k_des_b_base * (e_b / two_kbt).exp();
    let k_a = (p_a * k_ads_a_per_bar / k_des_a_base) * (-e_a / kbt).exp();
    let k_b = (p_b * k_ads_b_per_bar / k_des_b_base) * (-e_b / kbt).exp();
    let denom = 1.0 + k_a + k_b;
    let eq_expected = [k_a / denom, k_b / denom];

    let eq_api = model.langmuir_equilibrium_coverage(0).unwrap();
    assert!((eq_api[0] - eq_expected[0]).abs() <= 1e-13);
    assert!((eq_api[1] - eq_expected[1]).abs() <= 1e-13);

    // Exact 3-state site CTMC generator (0 = A, 1 = B, 2 = empty), starting from empty state 2.
    let q_site = [
        [-r_des_a, 0.0, r_des_a],
        [0.0, -r_des_b, r_des_b],
        [r_ads_a, r_ads_b, -(r_ads_a + r_ads_b)],
    ];
    let grid = SampleGrid::explicit(vec![0.05, 0.25, 2.5]).unwrap();
    let p_t0 = mat_exp(
        &q_site
            .iter()
            .map(|row| row.iter().map(|&v| v * 0.05).collect())
            .collect::<Vec<_>>(),
    );
    let p_t1 = mat_exp(
        &q_site
            .iter()
            .map(|row| row.iter().map(|&v| v * 0.25).collect())
            .collect::<Vec<_>>(),
    );
    let p_t2 = mat_exp(
        &q_site
            .iter()
            .map(|row| row.iter().map(|&v| v * 2.5).collect())
            .collect::<Vec<_>>(),
    );

    let init = model.empty_state();
    let runs = 800_u64;
    let samples = run_ensemble(41_001, runs, |ctx| {
        let mut sim = IncrementalSimulation::with_rng(
            model.clone(),
            init.clone(),
            ctx.rng(StreamRole::Dynamics),
        )
        .unwrap();
        let traj = sample_trajectory(&mut sim, &grid, None, AbsorptionPolicy::Error, |m, st| {
            m.check_invariants(st).unwrap();
            [
                st.coverage(0).unwrap(),
                st.coverage(1).unwrap(),
                st.empty_fraction(),
            ]
        })
        .unwrap();
        traj.values
    });

    let exact_rows = [&p_t0[2], &p_t1[2], &p_t2[2]];
    for (ti, exact) in exact_rows.iter().enumerate() {
        for sp_idx in 0..3 {
            let vals: Vec<f64> = samples.iter().map(|run| run[ti][sp_idx]).collect();
            let (mean, se) = mean_and_se(&vals);
            assert!(
                (mean - exact[sp_idx]).abs() <= 3.5 * se,
                "t_idx={ti} sp_idx={sp_idx}: mean={mean:.6} +/- {se:.6} vs exact={:.6}",
                exact[sp_idx]
            );
        }
    }
    // At t = 2.5 s the chain has relaxed cleanly onto the analytical Langmuir equilibrium isotherm.
    let eq_a_vals: Vec<f64> = samples.iter().map(|run| run[2][0]).collect();
    let eq_b_vals: Vec<f64> = samples.iter().map(|run| run[2][1]).collect();
    let (mean_a, se_a) = mean_and_se(&eq_a_vals);
    let (mean_b, se_b) = mean_and_se(&eq_b_vals);
    assert!((mean_a - eq_expected[0]).abs() <= 3.5 * se_a);
    assert!((mean_b - eq_expected[1]).abs() <= 3.5 * se_b);
    assert!(
        samples
            .iter()
            .all(|run| ((run[2][0] + run[2][1] + run[2][2]) - 1.0).abs() <= 1e-14)
    );

    // langmuir_equilibrium_coverage returns None outside its domain.
    assert_eq!(model.langmuir_equilibrium_coverage(1), None);
    let no_des = LatticeGas::new(
        lat.clone(),
        1,
        vec![SingleSiteRule::adsorption(0, 1.0, 5.0).unwrap()],
        vec![],
        None,
    )
    .unwrap();
    assert_eq!(no_des.langmuir_equilibrium_coverage(0), None);
    let with_conv = LatticeGas::new(
        lat.clone(),
        2,
        vec![
            SingleSiteRule::adsorption(0, 1.0, 5.0).unwrap(),
            SingleSiteRule::desorption(0, 2.0).unwrap(),
            SingleSiteRule::conversion(0, 1, 1.0).unwrap(),
            SingleSiteRule::desorption(1, 2.0).unwrap(),
        ],
        vec![],
        None,
    )
    .unwrap();
    assert_eq!(with_conv.langmuir_equilibrium_coverage(0), None);
    let with_pairs = LatticeGas::new(
        lat,
        1,
        vec![single_rules[0], single_rules[1]],
        vec![],
        Some(LatticeGasEnergetics {
            temperature_k: t_k,
            site_energies_ev: vec![e_a],
            pair_interactions_ev: vec![0.05],
        }),
    )
    .unwrap();
    assert_eq!(with_pairs.langmuir_equilibrium_coverage(0), None);
}

#[test]
fn t2_interacting_lattice_gas_obeys_exact_grand_canonical_boltzmann_ctmc_oracle() {
    // Small 2x2 periodic square lattice => 4 sites, where periodic wraps create 2 bonds per orthogonal neighbor
    // (m_ij = 2 along x and y!), testing exact local_delta_e multiplicity and detailed balance.
    let lat = PeriodicLattice::square_2d([2, 2], 3.0e-10).unwrap();
    let t_k = 500.0_f64;
    let kbt = BOLTZMANN_EV_PER_K * t_k;

    let lambda_a = 1.6_f64;
    let lambda_b = 0.75_f64;
    let k_des_a = 3.0_f64;
    let k_des_b = 4.0_f64;
    let k_ads_a = lambda_a * k_des_a;
    let k_ads_b = lambda_b * k_des_b;
    let k_assoc_ab = 2.0_f64;
    let k_dissoc_ab = lambda_a * lambda_b * k_assoc_ab;

    let single_rules = vec![
        SingleSiteRule::adsorption(0, 1.0, k_ads_a).unwrap(),
        SingleSiteRule::desorption(0, k_des_a).unwrap(),
        SingleSiteRule::adsorption(1, 1.0, k_ads_b).unwrap(),
        SingleSiteRule::desorption(1, k_des_b).unwrap(),
    ];
    let pair_rules = vec![
        PairRule::diffusion(0, 5.0).unwrap(),
        PairRule::diffusion(1, 3.5).unwrap(),
        PairRule::dissociative_adsorption(0, 1, 1.0, k_dissoc_ab).unwrap(),
        PairRule::associative_desorption(0, 1, k_assoc_ab).unwrap(),
    ];
    // Repulsive A-A (+0.045), attractive A-B (-0.055), mild repulsive B-B (+0.030).
    let ener = LatticeGasEnergetics {
        temperature_k: t_k,
        site_energies_ev: vec![-0.02, 0.03],
        pair_interactions_ev: vec![0.045, -0.055, -0.055, 0.030],
    };
    let model = LatticeGas::new(lat, 2, single_rules, pair_rules, Some(ener)).unwrap();

    // Enumerate all 3^4 = 81 lattice configurations and verify exact transition rates, energy deltas,
    // detailed balance wrt grand-canonical Boltzmann weights, and exact CTMC transition matrix.
    let mut configs = Vec::with_capacity(81);
    let mut idx_map = HashMap::with_capacity(81);
    let decode_code = |mut code: usize| -> Vec<Option<usize>> {
        let mut occ = vec![None; 4];
        for slot in &mut occ {
            *slot = match code % 3 {
                0 => None,
                1 => Some(0),
                _ => Some(1),
            };
            code /= 3;
        }
        occ
    };
    let encode_occ = |occ: &[Option<usize>]| -> usize {
        let mut acc = 0_usize;
        for (i, &v) in occ.iter().enumerate() {
            let d = match v {
                None => 0,
                Some(0) => 1,
                Some(1) => 2,
                _ => unreachable!(),
            };
            acc += d * 3_usize.pow(i as u32);
        }
        acc
    };

    let mut q = vec![vec![0.0_f64; 81]; 81];
    let mut unnorm_pi = vec![0.0_f64; 81];
    for (code, slot) in unnorm_pi.iter_mut().enumerate() {
        let occ = decode_code(code);
        let st = model.state_from_occupants(&occ).unwrap();
        let e = model.total_energy_ev(&st).unwrap();
        let n_a = st.species_count_on_surface(0).unwrap() as i32;
        let n_b = st.species_count_on_surface(1).unwrap() as i32;
        *slot = lambda_a.powi(n_a) * lambda_b.powi(n_b) * (-e / kbt).exp();
        idx_map.insert(occ, code);
        configs.push(st);
    }
    let z_sum: f64 = unnorm_pi.iter().sum();
    let pi_exact: Vec<f64> = unnorm_pi.iter().map(|w| w / z_sum).collect();

    for (u, st_u) in configs.iter().enumerate() {
        let e_u = model.total_energy_ev(st_u).unwrap();
        let mut row_sum = 0.0_f64;
        model.enumerate_events(st_u, &mut |ev, rate| {
            let mut next = st_u.clone();
            model.apply(&mut next, &ev).unwrap();
            let v = encode_occ(next.occupants());
            let e_v = model.total_energy_ev(&next).unwrap();
            let de = e_v - e_u;
            let expected_rate = match ev {
                LatticeGasEvent::SingleSite { rule, .. } => {
                    model.single_rules()[rule].base_rate_hz * (-de / (2.0 * kbt)).exp()
                }
                LatticeGasEvent::Pair { rule, .. } => {
                    model.pair_rules()[rule].base_rate_hz * (-de / (2.0 * kbt)).exp()
                }
            };
            assert!(
                (rate - expected_rate).abs() <= 1e-12 * expected_rate.max(1.0),
                "u={u} ev={ev:?}: emitted {rate:.12e} vs exact global {expected_rate:.12e}"
            );
            q[u][v] += rate;
            row_sum += rate;
        });
        q[u][u] = -row_sum;
    }

    // Check detailed balance pi(u) * Q(u, v) == pi(v) * Q(v, u) for all u != v.
    for u in 0..81 {
        for v in (u + 1)..81 {
            let lhs = pi_exact[u] * q[u][v];
            let rhs = pi_exact[v] * q[v][u];
            assert!(
                (lhs - rhs).abs() <= 1e-12 * (lhs + rhs + 1.0),
                "detailed balance violated for ({u}, {v}): lhs={lhs:e} vs rhs={rhs:e}"
            );
        }
    }

    let pi_solver = solve_stationary_ctmc(&q);
    for s in 0..81 {
        assert!(
            (pi_solver[s] - pi_exact[s]).abs() <= 1e-12,
            "state {s}: solver={:.12} vs grand_canonical={:.12}",
            pi_solver[s],
            pi_exact[s]
        );
    }

    // Verify Direct and Incremental simulations match the exact transient CTMC matrix exponential p(t) = e^{Q t}.
    let t_obs = 0.08_f64;
    let qt: Vec<Vec<f64>> = q
        .iter()
        .map(|row| row.iter().map(|&v| v * t_obs).collect())
        .collect();
    let p_t = mat_exp(&qt)[0].clone();
    let grid = SampleGrid::from_origin(t_obs, 1).unwrap();
    let init = configs[0].clone();
    let n_runs = 4_000_u64;

    let obs_direct = run_ensemble(77_001, n_runs, |ctx| {
        let mut sim =
            Simulation::with_rng(model.clone(), init.clone(), ctx.rng(StreamRole::Dynamics));
        sample_trajectory(&mut sim, &grid, None, AbsorptionPolicy::Error, |_, st| {
            encode_occ(st.occupants())
        })
        .unwrap()
        .values[0]
    });
    let obs_inc = run_ensemble(77_001, n_runs, |ctx| {
        let mut sim = IncrementalSimulation::with_rng(
            model.clone(),
            init.clone(),
            ctx.rng(StreamRole::Dynamics),
        )
        .unwrap();
        sample_trajectory(&mut sim, &grid, None, AbsorptionPolicy::Error, |_, st| {
            encode_occ(st.occupants())
        })
        .unwrap()
        .values[0]
    });
    assert_eq!(
        obs_direct, obs_inc,
        "Direct vs Incremental trajectories must match bitwise"
    );

    // Pool any tiny-probability states so expected counts >= 5 before chi2 evaluation.
    let mut raw_counts = [0_u64; 81];
    for &st_idx in &obs_inc {
        raw_counts[st_idx] += 1;
    }
    let mut pooled_obs = Vec::new();
    let mut pooled_probs = Vec::new();
    let mut tail_obs = 0_u64;
    let mut tail_prob = 0.0_f64;
    for s in 0..81 {
        let e_cnt = (n_runs as f64) * p_t[s];
        if e_cnt >= 5.0 {
            pooled_obs.push(raw_counts[s]);
            pooled_probs.push(p_t[s]);
        } else {
            tail_obs += raw_counts[s];
            tail_prob += p_t[s];
        }
    }
    if tail_prob > 0.0 {
        pooled_obs.push(tail_obs);
        pooled_probs.push(tail_prob);
    }
    let chi2 = chi2_statistic(&pooled_obs, &pooled_probs);
    let dof = pooled_obs.len() - 1;
    let bound = chi2_upper_quantile(dof, 1e-4);
    assert!(
        chi2 <= bound,
        "chi2={chi2:.3} exceeds upper quantile {bound:.3} for dof={dof}"
    );
    assert_incremental_parity("interacting_2x2", &model, init, 990, 200);
}

#[test]
fn t3_co_oxidation_catalysis_direct_vs_incremental_parity_and_turnover_frequency() {
    // CO oxidation reaction network (species 0 = CO*, 1 = O*):
    // single rules: [0] CO ads, [1] CO des
    // pair rules:   [0] O2 dissoc ads, [1] O2 assoc des, [2] CO diff, [3] O diff, [4] LH CO* + O* -> 2* (+ CO2 gas)
    let single_rules = vec![
        SingleSiteRule::adsorption(0, 1.0, 12.0).unwrap(),
        SingleSiteRule::desorption(0, 3.5).unwrap(),
    ];
    let pair_rules = vec![
        PairRule::dissociative_adsorption(1, 1, 1.0, 18.0).unwrap(),
        PairRule::associative_desorption(1, 1, 1.5).unwrap(),
        PairRule::diffusion(0, 25.0).unwrap(),
        PairRule::diffusion(1, 8.0).unwrap(),
        PairRule::reaction((Some(0), Some(1)), (None, None), 40.0).unwrap(),
    ];
    let ener = LatticeGasEnergetics {
        temperature_k: 550.0,
        site_energies_ev: vec![-0.05, -0.02],
        pair_interactions_ev: vec![0.04, -0.02, -0.02, 0.06],
    };

    // 1. Periodic 4x4 square catalyst surface.
    let lat_periodic = PeriodicLattice::square_2d([4, 4], 2.8e-10).unwrap();
    let m_periodic = LatticeGas::new(
        lat_periodic,
        2,
        single_rules.clone(),
        pair_rules.clone(),
        Some(ener.clone()),
    )
    .unwrap();
    let st_periodic = m_periodic.empty_state();
    assert_incremental_parity(
        "periodic_co_oxidation",
        &m_periodic,
        st_periodic.clone(),
        991,
        300,
    );

    // 2. Open-boundary 3x4 rectangular patch with mixed corner/edge/interior degrees (2, 3, 4).
    let hop_open = [
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
        HopStencil {
            from_basis: 0,
            delta_ticks: [0, -1, 0],
            pathway: 0,
        },
        HopStencil {
            from_basis: 0,
            delta_ticks: [0, 1, 0],
            pathway: 0,
        },
    ];
    let lat_open = PeriodicLattice::from_unit_cell(
        [3, 4, 1],
        [1, 1, 1],
        [false, false, false],
        [
            [2.8e-10, 0.0, 0.0],
            [0.0, 2.8e-10, 0.0],
            [0.0, 0.0, 2.8e-10],
        ],
        &[BasisSite {
            offset_ticks: [0, 0, 0],
            site_type: 0,
        }],
        &hop_open,
    )
    .unwrap();
    assert_eq!(lat_open.degree(0), 2);
    assert_eq!(lat_open.degree(1), 3);
    assert_eq!(lat_open.degree(5), 4);
    let m_open = LatticeGas::new(lat_open, 2, single_rules, pair_rules, Some(ener)).unwrap();
    let st_open = m_open.empty_state();
    assert_incremental_parity("open_co_oxidation", &m_open, st_open, 992, 300);

    // Bitwise simulation comparison and exact stoichiometry conservation over 300 events.
    let mut sim_d = Simulation::new(m_periodic.clone(), st_periodic.clone(), 12_345);
    let mut sim_i = IncrementalSimulation::new(m_periodic.clone(), st_periodic, 12_345).unwrap();
    for _ in 0..300 {
        let (
            Step::Fired {
                event: ed,
                dt: d_dt,
            },
            Step::Fired {
                event: ei,
                dt: i_dt,
            },
        ) = (sim_d.step().unwrap(), sim_i.step().unwrap())
        else {
            panic!("unexpected absorption in CO oxidation steady drive");
        };
        assert_eq!(ed, ei);
        assert!((d_dt - i_dt).abs() <= 1e-13 * i_dt.max(1.0));
        assert_eq!(sim_d.state(), sim_i.state());
        let st = sim_i.state();
        m_periodic.check_invariants(st).unwrap();

        let n_co = st.species_count_on_surface(0).unwrap();
        let n_o = st.species_count_on_surface(1).unwrap();
        assert_eq!(n_co + n_o + st.empty_site_count(), st.site_count());

        let co_ads = st.single_rule_count(0).unwrap();
        let co_des = st.single_rule_count(1).unwrap();
        let o2_ads = st.pair_rule_count(0).unwrap();
        let o2_des = st.pair_rule_count(1).unwrap();
        let co2_lh = st.pair_rule_count(4).unwrap();
        assert_eq!(co_ads, (n_co as u64) + co_des + co2_lh);
        assert_eq!(2 * o2_ads, (n_o as u64) + 2 * o2_des + co2_lh);
    }

    let elapsed = sim_i.time();
    let co2_count = sim_i.state().pair_rule_count(4).unwrap();
    assert!(co2_count > 0);
    let expected_tof = (co2_count as f64) / ((sim_i.state().site_count() as f64) * elapsed);
    let api_tof = sim_i
        .state()
        .turnover_frequency_per_site_hz(elapsed, &[], &[4])
        .unwrap();
    assert!((api_tof - expected_tof).abs() <= 1e-15 * expected_tof);
    assert_eq!(
        sim_i.state().turnover_frequency_per_site_hz(0.0, &[], &[4]),
        None
    );
    assert_eq!(
        sim_i
            .state()
            .turnover_frequency_per_site_hz(-1.0, &[], &[4]),
        None
    );
    assert_eq!(
        sim_i
            .state()
            .turnover_frequency_per_site_hz(f64::NAN, &[], &[4]),
        None
    );
    assert_eq!(
        sim_i
            .state()
            .turnover_frequency_per_site_hz(elapsed, &[99], &[]),
        None
    );
    assert_eq!(
        sim_i
            .state()
            .turnover_frequency_per_site_hz(elapsed, &[], &[99]),
        None
    );
}

#[test]
fn t4_reversible_undo_checkpoint_roundtrip_and_validation_errors() {
    let lat = PeriodicLattice::square_2d([2, 3], 3.0e-10).unwrap();

    // Constructor validation errors.
    assert_eq!(
        LatticeGas::new(
            lat.clone(),
            0,
            vec![SingleSiteRule::desorption(0, 1.0).unwrap()],
            vec![],
            None,
        )
        .unwrap_err(),
        LatticeGasError::ZeroSpecies
    );
    assert_eq!(
        LatticeGas::new(lat.clone(), 2, vec![], vec![], None).unwrap_err(),
        LatticeGasError::NoRules
    );
    assert_eq!(
        SingleSiteRule::conversion(1, 1, 1.0).unwrap_err(),
        LatticeGasError::NoOpSingleRule { rule: 0 }
    );
    assert_eq!(
        PairRule::reaction((Some(0), None), (Some(0), None), 1.0).unwrap_err(),
        LatticeGasError::NoOpPairRule { rule: 0 }
    );
    assert_eq!(
        LatticeGas::new(
            lat.clone(),
            1,
            vec![SingleSiteRule {
                from: Some(0),
                to: Some(0),
                site_type: None,
                base_rate_hz: 1.0,
            }],
            vec![],
            None,
        )
        .unwrap_err(),
        LatticeGasError::NoOpSingleRule { rule: 0 }
    );
    assert_eq!(
        LatticeGas::new(
            lat.clone(),
            1,
            vec![],
            vec![PairRule {
                from: (None, Some(0)),
                to: (None, Some(0)),
                pathway: None,
                base_rate_hz: 1.0,
            }],
            None,
        )
        .unwrap_err(),
        LatticeGasError::NoOpPairRule { rule: 0 }
    );
    assert_eq!(
        LatticeGas::new(
            lat.clone(),
            1,
            vec![SingleSiteRule::adsorption(1, 1.0, 2.0).unwrap()],
            vec![],
            None,
        )
        .unwrap_err(),
        LatticeGasError::SpeciesOutOfRange {
            species: 1,
            species_count: 1,
        }
    );
    assert_eq!(
        LatticeGas::new(
            lat.clone(),
            1,
            vec![
                SingleSiteRule::adsorption(0, 1.0, 2.0)
                    .unwrap()
                    .with_site_type(2)
            ],
            vec![],
            None,
        )
        .unwrap_err(),
        LatticeGasError::SiteTypeOutOfRange {
            site_type: 2,
            type_count: 1,
        }
    );
    assert_eq!(
        LatticeGas::new(
            lat.clone(),
            1,
            vec![],
            vec![PairRule::diffusion(0, 2.0).unwrap().with_pathway(3)],
            None,
        )
        .unwrap_err(),
        LatticeGasError::PathwayOutOfRange {
            pathway: 3,
            pathway_count: 1,
        }
    );
    assert_eq!(
        SingleSiteRule::adsorption(0, -0.5, 1.0).unwrap_err(),
        LatticeGasError::InvalidPressure { pressure_bar: -0.5 }
    );
    assert_eq!(
        SingleSiteRule::desorption(0, -1.0).unwrap_err(),
        LatticeGasError::InvalidRate { rate_hz: -1.0 }
    );
    assert!(matches!(
        SingleSiteRule::desorption(0, f64::NAN),
        Err(LatticeGasError::InvalidRate { .. })
    ));
    assert_eq!(
        LatticeGas::new(
            lat.clone(),
            2,
            vec![SingleSiteRule::adsorption(0, 1.0, 2.0).unwrap()],
            vec![],
            Some(LatticeGasEnergetics {
                temperature_k: -10.0,
                site_energies_ev: vec![0.0, 0.0],
                pair_interactions_ev: vec![0.0; 4],
            }),
        )
        .unwrap_err(),
        LatticeGasError::InvalidTemperature(-10.0)
    );
    assert_eq!(
        LatticeGas::new(
            lat.clone(),
            2,
            vec![SingleSiteRule::adsorption(0, 1.0, 2.0).unwrap()],
            vec![],
            Some(LatticeGasEnergetics {
                temperature_k: 300.0,
                site_energies_ev: vec![0.0],
                pair_interactions_ev: vec![0.0; 4],
            }),
        )
        .unwrap_err(),
        LatticeGasError::InvalidSiteEnergies
    );
    assert_eq!(
        LatticeGas::new(
            lat.clone(),
            2,
            vec![SingleSiteRule::adsorption(0, 1.0, 2.0).unwrap()],
            vec![],
            Some(LatticeGasEnergetics {
                temperature_k: 300.0,
                site_energies_ev: vec![0.0, 0.0],
                pair_interactions_ev: vec![0.0, 0.1, -0.1, 0.0],
            }),
        )
        .unwrap_err(),
        LatticeGasError::InvalidPairInteractions
    );

    // Rich multi-rule model for runtime apply_reversible, undo, checkpoints, and error guards.
    let single_rules = vec![
        SingleSiteRule::adsorption(0, 1.0, 10.0).unwrap(),
        SingleSiteRule::desorption(0, 3.0).unwrap(),
        SingleSiteRule::conversion(0, 1, 5.0).unwrap(),
    ];
    let pair_rules = vec![
        PairRule::dissociative_adsorption(1, 0, 1.0, 8.0).unwrap(),
        PairRule::diffusion(0, 6.0).unwrap(),
        PairRule::associative_desorption(1, 0, 4.0).unwrap(),
        PairRule::reaction((Some(0), Some(1)), (Some(1), Some(0)), 2.5).unwrap(),
    ];
    let ener = LatticeGasEnergetics {
        temperature_k: 500.0,
        site_energies_ev: vec![-0.05, 0.02],
        pair_interactions_ev: vec![0.04, -0.03, -0.03, 0.01],
    };
    let model = LatticeGas::new(lat.clone(), 2, single_rules, pair_rules, Some(ener)).unwrap();
    let mut work = model.empty_state();

    // Runtime validation errors do not mutate state.
    let before_err = work.clone();
    assert_eq!(
        model
            .apply_reversible(&mut work, &LatticeGasEvent::SingleSite { site: 6, rule: 0 })
            .unwrap_err(),
        LatticeGasError::SiteOutOfRange {
            site: 6,
            site_count: 6
        }
    );
    assert_eq!(
        model
            .apply_reversible(&mut work, &LatticeGasEvent::SingleSite { site: 0, rule: 3 })
            .unwrap_err(),
        LatticeGasError::SingleRuleOutOfRange {
            rule: 3,
            rule_count: 3
        }
    );
    assert_eq!(
        model
            .apply_reversible(
                &mut work,
                &LatticeGasEvent::Pair {
                    site: 0,
                    branch: 4,
                    rule: 0
                }
            )
            .unwrap_err(),
        LatticeGasError::BranchOutOfRange {
            site: 0,
            branch: 4,
            degree: 4
        }
    );
    assert_eq!(
        model
            .apply_reversible(
                &mut work,
                &LatticeGasEvent::Pair {
                    site: 0,
                    branch: 0,
                    rule: 4
                }
            )
            .unwrap_err(),
        LatticeGasError::PairRuleOutOfRange {
            rule: 4,
            rule_count: 4
        }
    );
    assert_eq!(
        model
            .apply_reversible(&mut work, &LatticeGasEvent::SingleSite { site: 0, rule: 1 })
            .unwrap_err(),
        LatticeGasError::RulePreconditionMismatch
    );
    assert_eq!(work, before_err);

    // Reversible undo restores exact bitwise state for every transition class.
    let evs = [
        LatticeGasEvent::SingleSite { site: 0, rule: 0 }, // 0: None -> A
        LatticeGasEvent::SingleSite { site: 0, rule: 2 }, // 0: A -> B
        LatticeGasEvent::Pair {
            site: 1,
            branch: 0,
            rule: 0,
        }, // (1, target): (None, None) -> (B, A)
        LatticeGasEvent::Pair {
            site: 1,
            branch: 0,
            rule: 2,
        }, // (1, target): (B, A) -> (None, None)
        LatticeGasEvent::SingleSite { site: 0, rule: 1 }, // precondition check later
    ];
    let u0 = model.apply_reversible(&mut work, &evs[0]).unwrap();
    model.check_invariants(&work).unwrap();
    let u1 = model.apply_reversible(&mut work, &evs[1]).unwrap();
    let target_from_1 = lat.hops(1)[0].target;
    let u2 = model.apply_reversible(&mut work, &evs[2]).unwrap();
    assert_eq!(work.occupant(1), Some(Some(1)));
    assert_eq!(work.occupant(target_from_1), Some(Some(0)));
    assert_eq!(work.total_events(), 3);
    model.undo(&mut work, u2);
    model.undo(&mut work, u1);
    model.undo(&mut work, u0);
    assert_eq!(work, before_err);
    assert_eq!(work.total_events(), 0);

    // Checkpoint/resume bitwise determinism + rejection of wrong geometry/params/corrupt bytes.
    let mut full_sim = Simulation::new(model.clone(), model.empty_state(), 0xface_b00c);
    let mut split_sim = Simulation::new(model.clone(), model.empty_state(), 0xface_b00c);
    let mut full_inc =
        IncrementalSimulation::new(model.clone(), model.empty_state(), 0xface_b00c).unwrap();
    let mut split_inc =
        IncrementalSimulation::new(model.clone(), model.empty_state(), 0xface_b00c).unwrap();
    for step in 0..40 {
        full_sim.step().unwrap();
        full_inc.step().unwrap();
        if step < 17 {
            split_sim.step().unwrap();
            split_inc.step().unwrap();
        }
    }
    let ckpt_d = split_sim.checkpoint();
    let ckpt_i = split_inc.checkpoint();
    let mut res_d = Simulation::resume(model.clone(), &ckpt_d).unwrap();
    let mut res_i = IncrementalSimulation::resume(model.clone(), &ckpt_i).unwrap();
    for _ in 17..40 {
        res_d.step().unwrap();
        res_i.step().unwrap();
    }
    assert_eq!(res_d.state(), full_sim.state());
    assert_eq!(res_d.time().to_bits(), full_sim.time().to_bits());
    assert_eq!(res_i.state(), full_inc.state());
    assert_eq!(res_i.time().to_bits(), full_inc.time().to_bits());
    assert_eq!(res_i.state(), res_d.state());

    // Distinct model params fail resume with ParamsMismatch.
    let diff_rate_model = LatticeGas::new(
        lat.clone(),
        2,
        vec![
            SingleSiteRule::adsorption(0, 2.0, 10.0).unwrap(),
            SingleSiteRule::desorption(0, 3.0).unwrap(),
            SingleSiteRule::conversion(0, 1, 5.0).unwrap(),
        ],
        model.pair_rules().to_vec(),
        None,
    )
    .unwrap();
    assert!(matches!(
        Simulation::resume(diff_rate_model, &ckpt_d),
        Err(CheckpointError::ParamsMismatch)
    ));

    // Equal-site-count cross-geometry [2, 3] vs [3, 2] fails all direct/incremental/reversible/checkpoint seams.
    let lat_3x2 = PeriodicLattice::square_2d([3, 2], 3.0e-10).unwrap();
    let m_3x2 = LatticeGas::new(
        lat_3x2,
        2,
        model.single_rules().to_vec(),
        model.pair_rules().to_vec(),
        None,
    )
    .unwrap();
    assert!(matches!(
        m_3x2.check_invariants(full_sim.state()),
        Err(LatticeGasError::StateMismatch(_))
    ));
    assert!(matches!(
        Simulation::new(m_3x2.clone(), full_sim.state().clone(), 7).step(),
        Err(SimError::InvalidRate { .. })
    ));
    assert!(matches!(
        IncrementalSimulation::new(m_3x2.clone(), full_sim.state().clone(), 7),
        Err(SimError::InvalidRate { .. })
    ));
    let mut bad_buf = full_sim.state().clone();
    assert!(matches!(
        m_3x2.apply_reversible(
            &mut bad_buf,
            &LatticeGasEvent::SingleSite { site: 0, rule: 0 }
        ),
        Err(LatticeGasError::StateMismatch(_))
    ));
    let mut raw = Vec::new();
    model.encode_state(full_sim.state(), &mut raw);
    assert!(m_3x2.decode_state(&raw).is_err());
    assert!(model.decode_state(&raw[..raw.len() - 1]).is_err());
    assert!(model.decode_state(&raw[..raw.len() - 8]).is_err());

    // Word 5 is occupants[0] (after 5-word header); inject invalid species 2 => error.
    let mut bad_sp = raw.clone();
    bad_sp[40..48].copy_from_slice(&2_u64.to_le_bytes());
    assert!(model.decode_state(&bad_sp).is_err());

    // CounterOverflow on u64::MAX fired single rule count.
    let mut overflow_raw = Vec::new();
    model.encode_state(&model.empty_state(), &mut overflow_raw);
    // Word offsets: 5 header + 6 occupants = word 11 is single_rule_counts[0] (bytes 88..96).
    overflow_raw[88..96].copy_from_slice(&u64::MAX.to_le_bytes());
    let mut st_overflow = model.decode_state(&overflow_raw).unwrap();
    assert_eq!(
        model
            .apply_reversible(
                &mut st_overflow,
                &LatticeGasEvent::SingleSite { site: 0, rule: 0 }
            )
            .unwrap_err(),
        LatticeGasError::CounterOverflow
    );

    // Zero-slot open 2-cell chain (max_degree == 0, pair-only rule => site_stride == 0):
    // valid 2-site state absorbs cleanly; mismatched 4-site state fails with InvalidRate in BOTH engines.
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
    let lat_zero = PeriodicLattice::from_unit_cell(
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
    let m_zero = LatticeGas::new(
        lat_zero,
        1,
        vec![],
        vec![PairRule::diffusion(0, 5.0).unwrap()],
        None,
    )
    .unwrap();
    let st_zero_ok = m_zero.state_from_occupants(&[Some(0), None]).unwrap();
    assert!(matches!(
        Simulation::new(m_zero.clone(), st_zero_ok.clone(), 9)
            .step()
            .unwrap(),
        Step::Absorbed
    ));
    assert!(matches!(
        IncrementalSimulation::new(m_zero.clone(), st_zero_ok, 9)
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
    let st_four = LatticeGas::new(
        lat_four,
        1,
        vec![],
        vec![PairRule::diffusion(0, 5.0).unwrap()],
        None,
    )
    .unwrap()
    .empty_state();
    assert!(matches!(
        Simulation::new(m_zero.clone(), st_four.clone(), 9).step(),
        Err(SimError::InvalidRate { .. })
    ));
    assert!(matches!(
        IncrementalSimulation::new(m_zero, st_four, 9),
        Err(SimError::InvalidRate { .. })
    ));

    // Log-space Boltzmann modulation handles subnormal, underflow, and factor-overflow cancellation accurately.
    let k_t = 1.0_f64;
    // Note modulation computes exp(-delta_e / (2 * k_B * T)).
    let e_subnormal = 720.0 * 2.0 * BOLTZMANN_EV_PER_K * k_t;
    let e_underflow = 750.0 * 2.0 * BOLTZMANN_EV_PER_K * k_t;
    let e_overflow_factor = -500.0 * 2.0 * BOLTZMANN_EV_PER_K * k_t;
    for (base, energy_ev, log_exp) in [
        (1e30_f64, e_subnormal, -720.0_f64),
        (1e30_f64, e_underflow, -750.0_f64),
        (1e-200_f64, e_overflow_factor, 500.0_f64),
    ] {
        let m_scale = LatticeGas::new(
            lat.clone(),
            1,
            vec![SingleSiteRule::adsorption(0, 1.0, base).unwrap()],
            vec![],
            Some(LatticeGasEnergetics {
                temperature_k: k_t,
                site_energies_ev: vec![energy_ev],
                pair_interactions_ev: vec![0.0],
            }),
        )
        .unwrap();
        let st_e = m_scale.empty_state();
        let r = m_scale.evaluate_single_rate(&st_e, 0, 0).unwrap();
        let expected = 10_f64.powf(base.log10() + log_exp * std::f64::consts::LOG10_E);
        assert!(
            r > 0.0 && r.is_finite() && (r - expected).abs() / expected <= 1e-12,
            "rate {r:e} vs expected {expected:e}"
        );
    }
}
