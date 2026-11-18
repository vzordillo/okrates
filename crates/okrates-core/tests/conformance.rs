//! Analytic CTMC conformance corpus: non-reversible cycles, detailed balance,
//! phase-type absorption times, CoupledModel parity, and lazy trajectory sampling.

use std::cell::Cell;
use std::error::Error;
use std::fmt;

use okrates_core::ensemble::run_ensemble;
use okrates_core::reference::{CoupledGraphWalk, EdgeRateTable, GraphRandomWalk, GraphTopology};
use okrates_core::testing::{
    absorbing_escape_cdf, chi2_statistic, chi2_upper_quantile, ks_pvalue, ks_statistic, mat_exp,
    solve_stationary_ctmc,
};
use okrates_core::{
    AbsorptionPolicy, Clock, CoupledModel, EnumeratedTopology, IncrementalModel,
    IncrementalSimulation, Model, RateModel, SampleGrid, SamplingError, SimError, Simulation,
    SlottedTopology, Step, StreamRole, SumTree, TrajectorySource, TransitionTopology,
    sample_trajectory,
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
        for _ in 0..100_000 {
            if sim.step().unwrap() == Step::Absorbed {
                return sim.time();
            }
        }
        panic!("absorbing CTMC exceeded 100_000 steps without absorption");
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
    let huge_grid = SampleGrid::from_origin(1e308, 4).unwrap();
    assert!(huge_grid.times().iter().all(|t| t.is_finite()));
    assert_eq!(*huge_grid.times().last().unwrap(), 1e308);

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

    let lo = f64::from_bits(0.5_f64.to_bits() + 1);
    let hi = f64::from_bits(lo.to_bits() + 3);
    for count in 2..=12 {
        let g = SampleGrid::windowed(lo, hi, count).unwrap();
        assert_eq!(g.times()[0], lo);
        assert_eq!(*g.times().last().unwrap(), hi);
        assert!(g.times().windows(2).all(|w| w[0] <= w[1]));
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ConformanceFault;

impl fmt::Display for ConformanceFault {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "injected conformance fault")
    }
}

impl Error for ConformanceFault {}

#[derive(Clone, Copy, Debug)]
struct ScriptedClockSource {
    state: usize,
    clock: Clock,
    jumps: &'static [(usize, f64)],
    cursor: usize,
    fail_at: Option<usize>,
}

struct DummyModel;

impl Model for DummyModel {
    type State = usize;
    type Event = usize;
    type Error = ConformanceFault;

    fn enumerate_events(&self, _state: &usize, _emit: &mut dyn FnMut(usize, f64)) {}

    fn apply(&self, _state: &mut usize, _event: &usize) -> Result<(), ConformanceFault> {
        Ok(())
    }
}

impl TrajectorySource for ScriptedClockSource {
    type Model = DummyModel;

    fn model(&self) -> &DummyModel {
        &DummyModel
    }

    fn state(&self) -> &usize {
        &self.state
    }

    fn clock(&self) -> Clock {
        self.clock
    }

    fn is_absorbed(&mut self) -> Result<bool, SimError<ConformanceFault>> {
        Ok(self.cursor >= self.jumps.len())
    }

    fn step_observing<F>(
        &mut self,
        before_apply: F,
    ) -> Result<Step<usize>, SimError<ConformanceFault>>
    where
        F: FnOnce(&DummyModel, &usize, Clock),
    {
        if self.cursor >= self.jumps.len() {
            return Ok(Step::Absorbed);
        }
        let (next_state, dt) = self.jumps[self.cursor];
        let mut next_clock = self.clock;
        next_clock.advance(dt).map_err(SimError::Clock)?;
        before_apply(&DummyModel, &self.state, next_clock);
        if self.fail_at == Some(self.cursor) {
            return Err(SimError::Model(ConformanceFault));
        }
        self.state = next_state;
        self.clock = next_clock;
        self.cursor += 1;
        Ok(Step::Fired {
            event: next_state,
            dt,
        })
    }
}

#[test]
fn sampler_handles_exact_ties_compensated_clock_limits_and_mid_run_errors() {
    let g_one = SampleGrid::explicit(vec![1.0]).unwrap();

    let mut tie_cap = ScriptedClockSource {
        state: 0,
        clock: Clock::new(),
        jumps: &[(7, 1.0), (8, 1.0)],
        cursor: 0,
        fail_at: None,
    };
    let out = sample_trajectory(
        &mut tie_cap,
        &g_one,
        Some(1),
        AbsorptionPolicy::Error,
        |_, &s| s,
    )
    .unwrap();
    assert_eq!(out.values, vec![7]);
    assert_eq!(out.fired_steps, 1);

    let mut tie_abs = ScriptedClockSource {
        state: 0,
        clock: Clock::new(),
        jumps: &[(9, 1.0)],
        cursor: 0,
        fail_at: None,
    };
    let out = sample_trajectory(
        &mut tie_abs,
        &g_one,
        None,
        AbsorptionPolicy::Error,
        |_, &s| s,
    )
    .unwrap();
    assert_eq!(out.values, vec![9]);
    assert_eq!(out.fired_steps, 1);

    let g_zero = SampleGrid::explicit(vec![0.0]).unwrap();
    let mut tie_origin = ScriptedClockSource {
        state: 42,
        clock: Clock::new(),
        jumps: &[],
        cursor: 0,
        fail_at: None,
    };
    let out = sample_trajectory(
        &mut tie_origin,
        &g_zero,
        None,
        AbsorptionPolicy::Error,
        |_, &s| s,
    )
    .unwrap();
    assert_eq!(out.values, vec![42]);
    assert_eq!(out.fired_steps, 0);

    let g_ten = SampleGrid::explicit(vec![10.0]).unwrap();
    let mut hold_at_cap = ScriptedClockSource {
        state: 0,
        clock: Clock::new(),
        jumps: &[(5, 0.5)],
        cursor: 0,
        fail_at: None,
    };
    let out = sample_trajectory(
        &mut hold_at_cap,
        &g_ten,
        Some(1),
        AbsorptionPolicy::HoldFinal,
        |_, &s| s,
    )
    .unwrap();
    assert_eq!(out.values, vec![5]);
    assert_eq!(out.fired_steps, 1);

    let mut past_clock = ScriptedClockSource {
        state: 0,
        clock: Clock::from_parts(1.0, 2.0_f64.powi(-54)).unwrap(),
        jumps: &[(1, 1.0)],
        cursor: 0,
        fail_at: None,
    };
    assert!(matches!(
        sample_trajectory(
            &mut past_clock,
            &g_one,
            None,
            AbsorptionPolicy::Error,
            |_, &s| s
        ),
        Err(SamplingError::GridBeforeSimTime { grid_start, .. }) if grid_start == 1.0
    ));

    let mut sub_ulp_cross = ScriptedClockSource {
        state: 3,
        clock: Clock::from_parts(1.0, -2.0_f64.powi(-54)).unwrap(),
        jumps: &[(4, 0.5 * f64::EPSILON)],
        cursor: 0,
        fail_at: None,
    };
    let out = sample_trajectory(
        &mut sub_ulp_cross,
        &g_one,
        None,
        AbsorptionPolicy::Error,
        |_, &s| s,
    )
    .unwrap();
    assert_eq!(out.values, vec![3]);
    assert_eq!(out.fired_steps, 1);

    let mut mid_err = ScriptedClockSource {
        state: 0,
        clock: Clock::new(),
        jumps: &[(1, 1.0), (2, 1.0), (3, 1.0)],
        cursor: 0,
        fail_at: Some(2),
    };
    assert!(matches!(
        sample_trajectory(
            &mut mid_err,
            &g_ten,
            None,
            AbsorptionPolicy::Error,
            |_, &s| s
        ),
        Err(SamplingError::Sim {
            steps: 2,
            source: SimError::Model(ConformanceFault),
            ..
        })
    ));
    assert_eq!(mid_err.state, 2);
    assert_eq!(mid_err.clock.time(), 2.0);
}

#[derive(Clone, Copy, Debug)]
struct RingPhaseTopology {
    fail_node: Option<usize>,
}

impl TransitionTopology for RingPhaseTopology {
    type State = [u8; 4];
    type Event = (usize, u8);
    type Undo = (usize, u8);
    type Error = ConformanceFault;

    fn apply(
        &self,
        state: &mut [u8; 4],
        &(node, delta): &(usize, u8),
    ) -> Result<(usize, u8), ConformanceFault> {
        if self.fail_node == Some(node) {
            return Err(ConformanceFault);
        }
        let prev = state[node];
        state[node] = (prev + delta) % 4;
        Ok((node, prev))
    }

    fn undo(&self, state: &mut [u8; 4], (node, prev): (usize, u8)) {
        state[node] = prev;
    }
}

impl EnumeratedTopology for RingPhaseTopology {
    fn enumerate_candidates(&self, state: &[u8; 4], emit: &mut dyn FnMut((usize, u8))) {
        for slot in 0..8 {
            if self.is_slot_active(state, slot) {
                emit(self.slot_event(state, slot));
            }
        }
    }
}

impl SlottedTopology for RingPhaseTopology {
    fn slot_count(&self, _state: &[u8; 4]) -> usize {
        8
    }

    fn slot_event(&self, _state: &[u8; 4], slot: usize) -> (usize, u8) {
        (slot / 2, 1 + (slot % 2) as u8)
    }

    fn is_slot_active(&self, state: &[u8; 4], slot: usize) -> bool {
        let node = slot / 2;
        state[(node + 2) % 4] != 3
    }

    fn affected_slots(&self, _state: &[u8; 4], &(node, _): &(usize, u8), out: &mut Vec<usize>) {
        for dep in [node, (node + 3) % 4, (node + 2) % 4] {
            out.extend(2 * dep..2 * dep + 2);
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct StateDependentRingRates;

impl RateModel<RingPhaseTopology> for StateDependentRingRates {
    fn rate(
        &self,
        _topo: &RingPhaseTopology,
        state: &[u8; 4],
        &(node, delta): &(usize, u8),
    ) -> f64 {
        let neighbor = state[(node + 1) % 4] as usize;
        0.5 * (1 + (state[node] as usize + 2 * neighbor + delta as usize) % 5) as f64
    }
}

#[test]
fn coupled_model_state_dependent_rates_match_across_backends_and_roll_back_on_error() {
    let coupled = CoupledModel::new(
        RingPhaseTopology { fail_node: None },
        StateDependentRingRates,
    );
    let init = [0_u8, 1, 2, 0];
    let mut direct = Simulation::new(coupled, init, 314);
    let mut inc = IncrementalSimulation::new(coupled, init, 314).unwrap();
    for _ in 0..200 {
        let step_d = direct.step().unwrap();
        let step_i = inc.step().unwrap();
        assert_eq!(step_d, step_i);
        assert_eq!(direct.state(), inc.state());
        assert_eq!(direct.time().to_bits(), inc.time().to_bits());
        let expected_rates: Vec<f64> = (0..8).map(|s| coupled.slot_rate(inc.state(), s)).collect();
        assert_eq!(*inc.tree(), SumTree::new(&expected_rates));
    }

    let failing = CoupledModel::new(
        RingPhaseTopology { fail_node: Some(0) },
        StateDependentRingRates,
    );
    let mut fail_inc = IncrementalSimulation::new(failing, [3, 3, 0, 3], 11).unwrap();
    let tree_before = fail_inc.tree().clone();
    assert!(matches!(
        fail_inc.step(),
        Err(SimError::Model(ConformanceFault))
    ));
    assert_eq!(*fail_inc.state(), [3, 3, 0, 3]);
    assert_eq!(fail_inc.steps(), 0);
    assert_eq!(*fail_inc.tree(), tree_before);

    let mut fail_direct = Simulation::new(failing, [3, 3, 0, 3], 11);
    assert!(matches!(
        fail_direct.step(),
        Err(SimError::Model(ConformanceFault))
    ));
    assert_eq!(*fail_direct.state(), [3, 3, 0, 3]);
    assert_eq!(fail_direct.steps(), 0);

    let topo_a = GraphTopology::new(vec![vec![1], vec![0]]).unwrap();
    let topo_b = GraphTopology::new(vec![vec![1, 1], vec![]]).unwrap();
    let table_a = EdgeRateTable::new(&topo_a, vec![1.0, 2.0]).unwrap();
    let mismatched = CoupledGraphWalk::new(topo_b, table_a);
    assert!(matches!(
        IncrementalSimulation::new(mismatched.clone(), 0, 1),
        Err(SimError::InvalidRate { index: 0, .. })
    ));
    assert!(matches!(
        Simulation::new(mismatched, 0, 1).step(),
        Err(SimError::InvalidRate { index: 0, .. })
    ));
    assert_eq!(topo_a.slot_event(&0, 99).from, usize::MAX);
}

#[test]
fn ctmc_solvers_and_statistics_handle_extreme_scales_and_stiff_generators() {
    assert!((ks_pvalue(1e-4, 16) - 1.0).abs() < 1e-15);
    assert_eq!(chi2_statistic(&[10, 0], &[1.0, 0.0]), 0.0);
    assert!(chi2_statistic(&[9, 1], &[1.0, 0.0]).is_infinite());

    assert_eq!(solve_stationary_ctmc(&[vec![0.0]]), vec![1.0]);
    for exp in [-300_i32, -14, 0, 20, 300] {
        let c = 10.0_f64.powi(exp);
        let pi = solve_stationary_ctmc(&[vec![0.0, c], vec![3.0 * c, 0.0]]);
        assert!((pi[0] - 0.75).abs() < 1e-14, "exp={exp}, pi={pi:?}");
        assert!((pi[1] - 0.25).abs() < 1e-14, "exp={exp}, pi={pi:?}");
    }

    let huge = mat_exp(&[vec![-1e308, 1e308], vec![0.0, 0.0]]);
    assert_eq!(huge, vec![vec![0.0, 1.0], vec![0.0, 1.0]]);
    let stiff_sym = mat_exp(&[vec![-1e20, 1e20], vec![1e20, -1e20]]);
    for row in &stiff_sym {
        for &v in row {
            assert!((v - 0.5).abs() < 1e-14, "got {v}");
        }
    }

    let r54 = (1_u64 << 54) as f64;
    let tr = vec![vec![0.0, r54], vec![r54, 0.0]];
    for (ex, t) in [
        (1.0, 1.0),
        (2.0, 1.0),
        (1.0, 2.0_f64.powi(-12)),
        (2.0, 2.0_f64.powi(-12)),
    ] {
        let got = absorbing_escape_cdf(&tr, &[ex, ex], &[1.0, 0.0], t);
        let exact = -(-ex * t).exp_m1();
        assert!(
            (got - exact).abs() < 1e-14,
            "ex={ex}, t={t}, got={got}, exact={exact}"
        );
    }
    let stiff_diag = absorbing_escape_cdf(
        &[vec![0.0, 0.0], vec![0.0, 0.0]],
        &[1e-4, (1_u64 << 44) as f64],
        &[1.0, 0.0],
        1.0,
    );
    assert!((stiff_diag - (-(-1e-4_f64).exp_m1())).abs() < 1e-16);

    assert!(std::panic::catch_unwind(|| mat_exp(&[vec![1.0, 2.0], vec![3.0]])).is_err());
    assert!(std::panic::catch_unwind(|| mat_exp(&[vec![f64::NAN]])).is_err());
    assert!(
        std::panic::catch_unwind(|| solve_stationary_ctmc(&[vec![0.0, 0.0], vec![1.0, 0.0]]))
            .is_err()
    );
}
