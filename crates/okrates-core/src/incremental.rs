//! Incremental event catalog with a binary sum-tree selector.
//!
//! An alternative scheduler beside the direct-method
//! [`Simulation`](crate::simulation::Simulation): events live in stable slots
//! whose rates are updated locally after each step.

use crate::checkpoint::{self, CheckpointError, Checkpointable};
use crate::clock::Clock;
use crate::model::Model;
use crate::rate::Rate;
use crate::rng::KmcRng;
use crate::simulation::{RunSummary, SimError, Step};

/// A complete binary tree of partial rate sums over a fixed number of slots.
///
/// Every internal node is exactly the sum of its two children, recomputed
/// from the children on each update, so the tree depends only on the current
/// slot rates and not on the order of updates.
#[derive(Clone, Debug, PartialEq)]
pub struct SumTree {
    len: usize,
    cap: usize,
    nodes: Vec<f64>,
}

impl SumTree {
    /// Builds a tree whose slot `i` has rate `rates[i]`.
    ///
    /// Callers pass validated non-negative finite rates.
    pub fn new(rates: &[f64]) -> Self {
        debug_assert!(rates.iter().all(|r| r.is_finite() && *r >= 0.0));
        let len = rates.len();
        let cap = len.max(1).next_power_of_two();
        let mut nodes = vec![0.0; 2 * cap];
        nodes[cap..cap + len].copy_from_slice(rates);
        for k in (1..cap).rev() {
            nodes[k] = nodes[2 * k] + nodes[2 * k + 1];
        }
        Self { len, cap, nodes }
    }

    /// Returns the number of slots.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns whether the tree has no slots.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns the sum of all slot rates.
    pub fn total(&self) -> f64 {
        self.nodes[1]
    }

    /// Returns the rate of `slot`.
    ///
    /// # Panics
    ///
    /// Panics if `slot >= len()`.
    pub fn rate(&self, slot: usize) -> f64 {
        assert!(
            slot < self.len,
            "slot {slot} out of range for {} slots",
            self.len
        );
        self.nodes[self.cap + slot]
    }

    /// Returns the raw node array; index 0 is unused and index 1 is the root.
    pub fn nodes(&self) -> &[f64] {
        &self.nodes
    }

    /// Sets the rate of `slot` and recomputes its ancestors from their children.
    ///
    /// # Panics
    ///
    /// Panics if `slot >= len()`.
    pub fn set(&mut self, slot: usize, rate: f64) {
        assert!(
            slot < self.len,
            "slot {slot} out of range for {} slots",
            self.len
        );
        debug_assert!(rate.is_finite() && rate >= 0.0);
        self.nodes[self.cap + slot] = rate;
        let mut k = (self.cap + slot) / 2;
        while k >= 1 {
            self.nodes[k] = self.nodes[2 * k] + self.nodes[2 * k + 1];
            k /= 2;
        }
    }

    /// Returns the slot selected by `u`, which must lie in the open interval (0, 1).
    ///
    /// Requires `total() > 0`. The descent goes left when the target is below
    /// the left sum, otherwise right with the left sum subtracted. The
    /// `r == 0.0` guard means the descent never enters a zero-sum subtree, so
    /// the selected slot always has a positive rate, even if rounding puts the
    /// target at or above the left sum.
    pub fn select(&self, u: f64) -> usize {
        debug_assert!(self.total() > 0.0);
        debug_assert!(u > 0.0 && u < 1.0);
        let mut target = u * self.total();
        let mut k = 1;
        while k < self.cap {
            let (l, r) = (self.nodes[2 * k], self.nodes[2 * k + 1]);
            if target < l || r == 0.0 {
                k *= 2;
            } else {
                target -= l;
                k = 2 * k + 1;
            }
        }
        k - self.cap
    }
}

/// A model whose events live in stable slots that can be updated locally.
pub trait IncrementalModel: Model {
    /// The number of slots. Must not change while the model runs.
    fn slot_count(&self, state: &Self::State) -> usize;
    /// The current rate of `slot`; 0 disables it. Must equal the rate that `enumerate_events` emits for the same event.
    fn slot_rate(&self, state: &Self::State, slot: usize) -> f64;
    /// The event stored in `slot`.
    fn slot_event(&self, state: &Self::State, slot: usize) -> Self::Event;
    /// After `event` has been applied to give `state`, push every slot whose rate may have changed (duplicates allowed).
    fn affected_slots(&self, state: &Self::State, event: &Self::Event, out: &mut Vec<usize>);
}

/// A serial kinetic Monte Carlo simulation using an incremental sum-tree catalog.
pub struct IncrementalSimulation<M: IncrementalModel> {
    model: M,
    state: M::State,
    clock: Clock,
    rng: KmcRng,
    steps: u64,
    tree: SumTree,
    scratch: Vec<usize>,
}

fn build_tree<M: IncrementalModel>(
    model: &M,
    state: &M::State,
) -> Result<SumTree, (usize, crate::rate::RateError)> {
    let count = model.slot_count(state);
    let mut rates = Vec::with_capacity(count);
    for slot in 0..count {
        match Rate::new(model.slot_rate(state, slot)) {
            Ok(rate) => rates.push(rate.get()),
            Err(err) => return Err((slot, err)),
        }
    }
    Ok(SumTree::new(&rates))
}

impl<M: IncrementalModel> IncrementalSimulation<M> {
    /// Creates a simulation with a seeded random stream, building the catalog
    /// from every slot's current rate.
    pub fn new(model: M, state: M::State, seed: u64) -> Result<Self, SimError<M::Error>> {
        let tree = build_tree(&model, &state)
            .map_err(|(index, source)| SimError::InvalidRate { index, source })?;
        Ok(Self {
            model,
            state,
            clock: Clock::new(),
            rng: KmcRng::from_seed(seed),
            steps: 0,
            tree,
            scratch: Vec::new(),
        })
    }

    /// Returns the model.
    pub fn model(&self) -> &M {
        &self.model
    }

    /// Returns the current model state.
    pub fn state(&self) -> &M::State {
        &self.state
    }

    /// Returns the current simulation time.
    pub fn time(&self) -> f64 {
        self.clock.time()
    }

    /// Returns the total number of events fired by this simulation.
    pub fn steps(&self) -> u64 {
        self.steps
    }

    /// Returns the sum tree holding the current slot rates.
    pub fn tree(&self) -> &SumTree {
        &self.tree
    }

    /// Fires one event, or reports that the state is absorbing.
    ///
    /// Uses the same two random draws per step as ADR-0002 (selection, then
    /// waiting time). Selection is the sum-tree descent, so trajectories are
    /// not bitwise equal to those of `Simulation`.
    pub fn step(&mut self) -> Result<Step<M::Event>, SimError<M::Error>> {
        let total = self.tree.total();
        if !total.is_finite() {
            return Err(SimError::TotalRateNotFinite);
        }
        if total == 0.0 {
            return Ok(Step::Absorbed);
        }

        let u1 = self.rng.open01();
        let slot = self.tree.select(u1);
        let u2 = self.rng.open01();
        let dt = -u2.ln() / total;

        let mut advanced_clock = self.clock;
        advanced_clock.advance(dt).map_err(SimError::Clock)?;
        let event = self.model.slot_event(&self.state, slot);
        self.model
            .apply(&mut self.state, &event)
            .map_err(SimError::Model)?;
        self.clock = advanced_clock;
        self.steps += 1;

        self.scratch.clear();
        self.model
            .affected_slots(&self.state, &event, &mut self.scratch);
        for &slot in &self.scratch {
            let rate = Rate::new(self.model.slot_rate(&self.state, slot)).map_err(|source| {
                SimError::InvalidRate {
                    index: slot,
                    source,
                }
            })?;
            self.tree.set(slot, rate.get());
        }
        Ok(Step::Fired { event, dt })
    }

    /// Runs up to max_steps events, stopping early if the state is absorbing.
    pub fn run(&mut self, max_steps: u64) -> Result<RunSummary, SimError<M::Error>> {
        let mut fired = 0;
        let mut absorbed = false;
        for _ in 0..max_steps {
            match self.step()? {
                Step::Fired { .. } => fired += 1,
                Step::Absorbed => {
                    absorbed = true;
                    break;
                }
            }
        }
        Ok(RunSummary {
            steps: fired,
            time: self.clock.time(),
            absorbed,
        })
    }
}

impl<M: IncrementalModel + Checkpointable> IncrementalSimulation<M> {
    /// Encodes the simulation as a checkpoint in schema version 1.
    ///
    /// The bytes equal those a direct `Simulation` writes for the same
    /// steps, time, random stream and state.
    pub fn checkpoint(&self) -> Vec<u8> {
        checkpoint::encode(self.steps, &self.clock, &self.rng, &self.model, &self.state)
    }

    /// Rebuilds a simulation from `bytes`, validating them against `model`
    /// and rebuilding the catalog from every slot's rate.
    pub fn resume(model: M, bytes: &[u8]) -> Result<Self, CheckpointError> {
        let (steps, clock, rng, state) = checkpoint::decode(&model, bytes)?;
        let tree = build_tree(&model, &state).map_err(|(slot, err)| {
            CheckpointError::InvalidState(format!("invalid rate in slot {slot}: {err}").into())
        })?;
        Ok(Self {
            model,
            state,
            clock,
            rng,
            steps,
            tree,
            scratch: Vec::new(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::SumTree;

    #[test]
    fn select_matches_linear_scan() {
        let rates = [3.0, 0.0, 1.0, 4.0, 0.0, 2.0, 6.0, 0.0, 0.0];
        let tree = SumTree::new(&rates);
        assert_eq!(tree.total(), 16.0);
        for half in 1..32 {
            let t = half as f64 * 0.5;
            let mut acc = 0.0;
            let expected = rates
                .iter()
                .position(|&r| {
                    acc += r;
                    t < acc
                })
                .unwrap();
            let got = tree.select(t / 16.0);
            assert_eq!(got, expected, "t={t}");
            assert!(rates[got] > 0.0, "t={t}");
        }
    }

    #[test]
    fn select_boundaries_skip_zero_rate_slots() {
        let tree = SumTree::new(&[0.0, 2.0, 5.0, 0.0, 0.0, 0.0]);
        assert_eq!(tree.select(1.0 - f64::EPSILON / 2.0), 2);
        assert_eq!(tree.select(f64::EPSILON / 2.0), 1);
        let tree = SumTree::new(&[0.1, 0.2, 0.0, 0.0]);
        assert_eq!(tree.select(1.0 - f64::EPSILON / 2.0), 1);
        // Here `target - (0.1 + 1/3)` rounds up to exactly 3.3, the left sum of
        // the right subtree, whose right child is 0; only the zero guard keeps
        // the descent out of slot 3.
        let tree = SumTree::new(&[0.1, 1.0 / 3.0, 3.3, 0.0]);
        assert_eq!(tree.select(1.0 - f64::EPSILON / 2.0), 2);
    }

    #[test]
    fn tree_is_history_independent() {
        let rates: Vec<f64> = (0..37).map(|i| 0.1 * ((i * 7) % 11) as f64).collect();
        let a = SumTree::new(&rates);
        let mut b = SumTree::new(&[0.0; 37]);
        for j in 0..37 {
            let i = (j * 17) % 37;
            b.set(i, rates[i]);
        }
        b.set(5, 9.9);
        b.set(5, rates[5]);
        assert_eq!(a, b);
    }

    #[test]
    fn single_slot_tree() {
        let tree = SumTree::new(&[2.0]);
        assert_eq!(tree.select(0.3), 0);
        assert_eq!(tree.total(), 2.0);
        assert_eq!(tree.len(), 1);
    }
}
