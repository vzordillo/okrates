//! Analytic CTMC conformance corpus: non-reversible cycles, detailed balance,
//! phase-type absorption times, CoupledModel parity, and lazy trajectory sampling.

use std::cell::Cell;

use okrates_core::ensemble::run_ensemble;
use okrates_core::reference::{CoupledGraphWalk, EdgeRateTable, GraphRandomWalk, GraphTopology};
use okrates_core::testing::{
    absorbing_escape_cdf, chi2_statistic, chi2_upper_quantile, ks_pvalue, ks_statistic,
    solve_stationary_ctmc,
};
use okrates_core::{
    AbsorptionPolicy, IncrementalSimulation, SampleGrid, SamplingError, Simulation, Step,
    StreamRole, sample_trajectory,
};

#[test]
fn nonreversible_cyclic_ctmc_matches_stationary_generator_solution() {
    // 4-state driven cycle with asymmetric chord (non-zero stationary probability current).
    let rates = vec![
        vec![0.0, 2.0, 1.0, 0.2],
        vec![0.4, 0.0, 3.0, 0.0],
        vec![0.5, 0.3, 0.0, 5.0],
        vec![7.0, 0.1, 0.4, 0.0],
    ];
    let pi = solve_stationary_ctmc(&rates);
    assert!((pi.iter().sum::<f64>() - 1.0).abs() < 1e-14);
    // Confirm genuine irreversibility: pi_0 q_01 != pi_1 q_10.
    assert!((pi[0] * rates[0][1] - pi[1] * rates[1][0]).abs() > 0.1);

    let adj: Vec<Vec<(usize, f64)>> = rates
        .iter()
        .enumerate()
        .map(|(i, row)| {
            row.iter()
                .enumerate()
                .filter(|&(j, &r)| i != j && r > 0.0)
                .map(|(j, &r)| (j, r))
                .collect()
        })
        .collect();
    let walk = GraphRandomWalk::new(adj.clone()).unwrap();

    let targets: Vec<Vec<usize>> = adj
        .iter()
        .map(|edges| edges.iter().map(|&(to, _)| to).collect())
        .collect();
    let flat_rates: Vec<f64> = adj
        .iter()
        .flat_map(|edges| edges.iter().map(|&(_, r)| r))
        .collect();
    let topo = GraphTopology::new(targets).unwrap();
    let rate_table = EdgeRateTable::new(&topo, flat_rates).unwrap();
    let coupled = CoupledGraphWalk::new(topo, rate_table);

    let grid = SampleGrid::from_origin(12.0, 1).unwrap();
    let replicas = 4_000_u64;

    // Test both DirectCatalog (on GraphRandomWalk) and SumTreeCatalog (on CoupledGraphWalk).
    let direct_states = run_ensemble(9001, replicas, |ctx| {
        let mut sim = Simulation::with_rng(walk.clone(), 0, ctx.rng(StreamRole::Dynamics));
        sample_trajectory(&mut sim, &grid, None, AbsorptionPolicy::Error, |_, &s| s)
            .unwrap()
            .values[0]
    });
    let coupled_inc_states = run_ensemble(9002, replicas, |ctx| {
        let mut sim =
            IncrementalSimulation::with_rng(coupled.clone(), 0, ctx.rng(StreamRole::Dynamics))
                .unwrap();
        sample_trajectory(&mut sim, &grid, None, AbsorptionPolicy::Error, |_, &s| s)
            .unwrap()
            .values[0]
    });

    for (label, states) in [
        ("direct", direct_states),
        ("coupled_inc", coupled_inc_states),
    ] {
        let mut counts = [0_u64; 4];
        for s in states {
            counts[s] += 1;
        }
        let chi2 = chi2_statistic(&counts, &pi);
        let crit = chi2_upper_quantile(3, 1e-5);
        assert!(
            chi2 <= crit,
            "{label}: chi2={chi2:.4} > crit={crit:.4}, counts={counts:?}, pi={pi:?}"
        );
    }
}

#[test]
fn reversible_birth_death_ladder_obeys_detailed_balance() {
    // 5-state asymmetric ladder with exact detailed balance pi_i f_i = pi_{i+1} b_{i+1}.
    let forward = [3.2, 1.8, 2.5, 0.9];
    let backward = [1.1, 2.7, 1.4, 3.6];
    let mut rates = vec![vec![0.0; 5]; 5];
    for i in 0..4 {
        rates[i][i + 1] = forward[i];
        rates[i + 1][i] = backward[i];
    }
    let pi = solve_stationary_ctmc(&rates);
    for i in 0..4 {
        assert!((pi[i] * forward[i] - pi[i + 1] * backward[i]).abs() < 1e-14);
    }

    let adj: Vec<Vec<(usize, f64)>> = (0..5)
        .map(|i| {
            let mut edges = Vec::new();
            if i > 0 {
                edges.push((i - 1, backward[i - 1]));
            }
            if i + 1 < 5 {
                edges.push((i + 1, forward[i]));
            }
            edges
        })
        .collect();
    let model = GraphRandomWalk::new(adj).unwrap();
    let grid = SampleGrid::from_origin(20.0, 1).unwrap();
    let replicas = 3_500_u64;
    let states = run_ensemble(777, replicas, |ctx| {
        let mut sim =
            IncrementalSimulation::with_rng(model.clone(), 2, ctx.rng(StreamRole::Dynamics))
                .unwrap();
        sample_trajectory(&mut sim, &grid, None, AbsorptionPolicy::Error, |_, &s| s)
            .unwrap()
            .values[0]
    });
    let mut counts = [0_u64; 5];
    for s in states {
        counts[s] += 1;
    }
    let chi2 = chi2_statistic(&counts, &pi);
    let crit = chi2_upper_quantile(4, 1e-5);
    assert!(chi2 <= crit, "chi2={chi2:.4} > crit={crit:.4}");
}

#[test]
fn phase_type_absorbing_escape_times_match_matrix_exponential_cdf() {
    // 3 transient states {0, 1, 2} + absorbing state 3.
    let model = GraphRandomWalk::new(vec![
        vec![(1, 2.5), (3, 0.5)],
        vec![(0, 1.2), (2, 1.8), (3, 0.8)],
        vec![(1, 1.5), (3, 2.2)],
        vec![],
    ])
    .unwrap();
    let transient = vec![
        vec![0.0, 2.5, 0.0],
        vec![1.2, 0.0, 1.8],
        vec![0.0, 1.5, 0.0],
    ];
    let exit = [0.5, 0.8, 2.2];
    let init = [1.0, 0.0, 0.0];

    let n = 2_500_usize;
    let escape_times = run_ensemble(4242, n as u64, |ctx| {
        let mut sim = Simulation::with_rng(model.clone(), 0, ctx.rng(StreamRole::Dynamics));
        loop {
            if sim.step().unwrap() == Step::Absorbed {
                return sim.time();
            }
        }
    });

    let d = ks_statistic(&escape_times, |t| {
        absorbing_escape_cdf(&transient, &exit, &init, t)
    });
    let pval = ks_pvalue(d, n);
    assert!(
        pval > 1e-4,
        "KS test rejected exact phase-type CDF: D={d:.5}, p={pval:.5e}"
    );
}

#[test]
fn lazy_sampler_evaluates_observer_only_on_grid_crossings_and_handles_policies() {
    let model = GraphRandomWalk::new(vec![vec![(1, 5.0)], vec![(0, 5.0)]]).unwrap();
    let grid = SampleGrid::windowed(2.0, 10.0, 5).unwrap();

    // Dense reference run evaluating state on every step.
    let mut ref_sim = Simulation::new(model.clone(), 0, 123);
    let mut dense = Vec::new();
    while dense.len() < grid.len() {
        let held = *ref_sim.state();
        ref_sim.step().unwrap();
        let t_after = ref_sim.time();
        while dense.len() < grid.len() && grid.times()[dense.len()] < t_after {
            dense.push(held);
        }
    }

    let calls = Cell::new(0_usize);
    let mut lazy_sim = Simulation::new(model, 0, 123);
    let sampled = sample_trajectory(
        &mut lazy_sim,
        &grid,
        None,
        AbsorptionPolicy::Error,
        |_, &s| {
            calls.set(calls.get() + 1);
            s
        },
    )
    .unwrap();

    assert_eq!(sampled.values, dense);
    assert!(sampled.fired_steps > 20);
    assert!(
        calls.get() <= grid.len(),
        "observer called {} times for {} grid points",
        calls.get(),
        grid.len()
    );

    // Absorbing state behavior under Error vs HoldFinal, plus MaxStepsExceeded.
    let absorbing = GraphRandomWalk::new(vec![vec![(1, 2.0)], vec![]]).unwrap();
    let abs_grid = SampleGrid::from_origin(50.0, 4).unwrap();

    let mut s_err = Simulation::new(absorbing.clone(), 0, 9);
    assert!(matches!(
        sample_trajectory(
            &mut s_err,
            &abs_grid,
            None,
            AbsorptionPolicy::Error,
            |_, &s| s
        ),
        Err(SamplingError::Absorbed { steps: 1, .. })
    ));

    let mut s_hold = Simulation::new(absorbing.clone(), 0, 9);
    let held = sample_trajectory(
        &mut s_hold,
        &abs_grid,
        None,
        AbsorptionPolicy::HoldFinal,
        |_, &s| s,
    )
    .unwrap();
    assert_eq!(held.values.len(), 4);
    assert_eq!(*held.values.last().unwrap(), 1);

    let fast = GraphRandomWalk::new(vec![vec![(0, 100.0)]]).unwrap();
    let mut s_cap = Simulation::new(fast, 0, 9);
    assert!(matches!(
        sample_trajectory(
            &mut s_cap,
            &abs_grid,
            Some(5),
            AbsorptionPolicy::Error,
            |_, &s| s
        ),
        Err(SamplingError::MaxStepsExceeded { max_steps: 5, .. })
    ));
}
