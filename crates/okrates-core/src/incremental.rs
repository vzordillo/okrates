//! Incremental event catalog with a binary sum-tree selector.

use crate::catalog::{Engine, SumTreeCatalog};
use crate::checkpoint::{CheckpointError, Checkpointable};
use crate::clock::Clock;
use crate::model::ReversibleModel;
use crate::rate::Rate;
use crate::rng::KmcRng;
use crate::simulation::{RunSummary, SimError, Step};
use crate::topology::SlotError;

/// Complete binary tree of partial rate sums over a fixed slot count.
///
/// Ancestors are recomputed directly from children on every update, making the tree
/// independent of update history.
#[derive(Clone, Debug, PartialEq)]
pub struct SumTree {
    len: usize,
    cap: usize,
    nodes: Vec<f64>,
}

impl SumTree {
    /// Checks `len` against power-of-two capacity and addressable `f64` tree buffer limits.
    pub fn check_len(len: usize) -> Result<(usize, usize), SlotError> {
        let cap = len
            .max(1)
            .checked_next_power_of_two()
            .ok_or(SlotError::TooLarge { slot_count: len })?;
        let node_count = cap
            .checked_mul(2)
            .ok_or(SlotError::TooLarge { slot_count: len })?;
        node_count
            .checked_mul(std::mem::size_of::<f64>())
            .filter(|&bytes| bytes <= isize::MAX as usize)
            .ok_or(SlotError::TooLarge { slot_count: len })?;
        Ok((cap, node_count))
    }

    pub fn try_with_len(len: usize) -> Result<Self, SlotError> {
        let (cap, node_count) = Self::check_len(len)?;
        let mut nodes = Vec::new();
        nodes
            .try_reserve_exact(node_count)
            .map_err(|_| SlotError::TooLarge { slot_count: len })?;
        nodes.resize(node_count, 0.0);
        Ok(Self { len, cap, nodes })
    }

    pub fn try_new(rates: &[f64]) -> Result<Self, SlotError> {
        let mut tree = Self::try_with_len(rates.len())?;
        for (slot, &raw) in rates.iter().enumerate() {
            let rate = Rate::new(raw).map_err(|source| SlotError::InvalidRate { slot, source })?;
            tree.nodes[tree.cap + slot] = rate.get();
        }
        tree.recompute_internal_sums();
        if !tree.total().is_finite() {
            return Err(SlotError::TotalNotFinite);
        }
        Ok(tree)
    }

    /// Builds a tree from finite non-negative slot `rates`.
    pub fn new(rates: &[f64]) -> Self {
        Self::try_new(rates).unwrap_or_else(|e| panic!("{e}"))
    }

    fn recompute_internal_sums(&mut self) {
        for k in (1..self.cap).rev() {
            self.nodes[k] = self.nodes[2 * k] + self.nodes[2 * k + 1];
        }
    }

    pub(crate) fn init_from_fn<E>(
        &mut self,
        mut f: impl FnMut(usize) -> Result<f64, E>,
    ) -> Result<(), E> {
        for slot in 0..self.len {
            self.nodes[self.cap + slot] = f(slot)?;
        }
        self.recompute_internal_sums();
        Ok(())
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn total(&self) -> f64 {
        self.nodes[1]
    }

    pub fn try_rate(&self, slot: usize) -> Result<f64, SlotError> {
        if slot >= self.len {
            return Err(SlotError::OutOfRange {
                slot,
                slot_count: self.len,
            });
        }
        Ok(self.nodes[self.cap + slot])
    }

    pub fn rate(&self, slot: usize) -> f64 {
        self.try_rate(slot).unwrap_or_else(|e| panic!("{e}"))
    }

    pub fn nodes(&self) -> &[f64] {
        &self.nodes
    }

    fn set_unchecked(&mut self, slot: usize, rate: f64) {
        self.nodes[self.cap + slot] = rate;
        let mut k = (self.cap + slot) / 2;
        while k >= 1 {
            self.nodes[k] = self.nodes[2 * k] + self.nodes[2 * k + 1];
            k /= 2;
        }
    }

    pub fn try_set(&mut self, slot: usize, rate: f64) -> Result<(), SlotError> {
        if slot >= self.len {
            return Err(SlotError::OutOfRange {
                slot,
                slot_count: self.len,
            });
        }
        let rate = Rate::new(rate)
            .map_err(|source| SlotError::InvalidRate { slot, source })?
            .get();
        let prev = self.nodes[self.cap + slot];
        self.set_unchecked(slot, rate);
        if !self.total().is_finite() {
            self.set_unchecked(slot, prev);
            return Err(SlotError::TotalNotFinite);
        }
        Ok(())
    }

    pub fn set(&mut self, slot: usize, rate: f64) {
        self.try_set(slot, rate).unwrap_or_else(|e| panic!("{e}"));
    }

    /// Selects a positive-rate slot from `u` in `(0, 1)`; requires `total() > 0`.
    ///
    /// The `r == 0.0` guard prevents rounding from descending into zero-sum subtrees.
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

/// Reversible model with stable integer event slots for local `O(k log M)` catalog updates.
pub trait IncrementalModel: ReversibleModel {
    /// Fixed slot count across the lifetime of a simulation.
    fn slot_count(&self, state: &Self::State) -> usize;

    /// Non-negative rate of `slot` (`0.0` disables the slot).
    fn slot_rate(&self, state: &Self::State, slot: usize) -> f64;

    fn slot_event(&self, state: &Self::State, slot: usize) -> Self::Event;

    /// Appends every slot whose rate may have changed after `event` produced `state` (duplicates allowed).
    fn affected_slots(&self, state: &Self::State, event: &Self::Event, out: &mut Vec<usize>);
}

/// Serial KMC simulation backed by [`Engine`] and [`SumTreeCatalog`].
pub struct IncrementalSimulation<M: IncrementalModel> {
    engine: Engine<M, SumTreeCatalog>,
}

impl<M: IncrementalModel> IncrementalSimulation<M> {
    pub fn new(model: M, state: M::State, seed: u64) -> Result<Self, SimError<M::Error>> {
        Ok(Self {
            engine: Engine::new(model, state, seed)?,
        })
    }

    pub fn with_rng(model: M, state: M::State, rng: KmcRng) -> Result<Self, SimError<M::Error>> {
        Ok(Self {
            engine: Engine::with_rng(model, state, rng)?,
        })
    }

    pub fn engine(&self) -> &Engine<M, SumTreeCatalog> {
        &self.engine
    }

    pub fn model(&self) -> &M {
        self.engine.model()
    }

    pub fn state(&self) -> &M::State {
        self.engine.state()
    }

    pub fn clock(&self) -> Clock {
        self.engine.clock()
    }

    pub fn time(&self) -> f64 {
        self.engine.time()
    }

    pub fn steps(&self) -> u64 {
        self.engine.steps()
    }

    pub fn tree(&self) -> &SumTree {
        self.engine.catalog().tree()
    }

    /// Validates active rates without advancing RNG, clock, or step count; returns `Ok(true)` if absorbed.
    pub fn is_absorbed(&mut self) -> Result<bool, SimError<M::Error>> {
        self.engine.is_absorbed()
    }

    /// Fires one transition atomically; on `Err`, simulation state, sum-tree, clock, steps, and RNG are unchanged.
    pub fn step(&mut self) -> Result<Step<M::Event>, SimError<M::Error>> {
        self.engine.step()
    }

    /// Invokes `before_apply(&model, &state_before, next_clock)` after clock validation and before
    /// transition application; external side effects in `before_apply` are not rolled back on `Err`.
    pub fn step_observing<F>(
        &mut self,
        before_apply: F,
    ) -> Result<Step<M::Event>, SimError<M::Error>>
    where
        F: FnOnce(&M, &M::State, Clock),
    {
        self.engine.step_observing(before_apply)
    }

    pub fn run(&mut self, max_steps: u64) -> Result<RunSummary, SimError<M::Error>> {
        self.engine.run(max_steps)
    }
}

impl<M: IncrementalModel + Checkpointable> IncrementalSimulation<M> {
    pub fn checkpoint(&self) -> Vec<u8> {
        self.engine.checkpoint()
    }

    pub fn resume(model: M, bytes: &[u8]) -> Result<Self, CheckpointError> {
        Ok(Self {
            engine: Engine::resume(model, bytes)?,
        })
    }
}

/// Verifies that [`IncrementalModel::affected_slots`] and `tree` match a fresh slot-by-slot evaluation of `after`.
///
/// Checks that every slot whose rate changed between `before` and `after` was reported by
/// `model.affected_slots(after, event, &mut affected)`, and that every slot rate and `tree.total()`
/// match `model.slot_rate(after, ..)` within `tol`.
pub fn verify_incremental_step_parity<M: IncrementalModel>(
    model: &M,
    before: &M::State,
    after: &M::State,
    event: &M::Event,
    tree: &SumTree,
    tol: f64,
) -> Result<(), String> {
    if !tol.is_finite() || tol < 0.0 {
        return Err(format!("invalid tolerance {tol}"));
    }
    let slot_count = model.slot_count(after);
    if slot_count != tree.len() {
        return Err(format!(
            "slot count mismatch: model has {slot_count}, tree has {}",
            tree.len()
        ));
    }
    let mut affected = Vec::new();
    model.affected_slots(after, event, &mut affected);
    affected.sort_unstable();
    affected.dedup();

    let mut after_rates = Vec::with_capacity(slot_count);
    for slot in 0..slot_count {
        let before_rate = Rate::new(model.slot_rate(before, slot))
            .map_err(|e| format!("invalid rate before step at slot {slot}: {e}"))?
            .get();
        let after_rate = Rate::new(model.slot_rate(after, slot))
            .map_err(|e| format!("invalid rate after step at slot {slot}: {e}"))?
            .get();
        if before_rate != after_rate && affected.binary_search(&slot).is_err() {
            return Err(format!(
                "affected_slots omitted slot {slot} whose rate changed from {before_rate} to {after_rate}"
            ));
        }
        let tree_rate = tree
            .try_rate(slot)
            .map_err(|e| format!("tree read failed at slot {slot}: {e}"))?;
        if (tree_rate - after_rate).abs() > tol {
            return Err(format!(
                "slot {slot} rate mismatch: tree={tree_rate}, expected={after_rate}"
            ));
        }
        after_rates.push(after_rate);
    }
    let expected_tree =
        SumTree::try_new(&after_rates).map_err(|e| format!("fresh sum-tree build failed: {e}"))?;
    if (tree.total() - expected_tree.total()).abs() > tol {
        return Err(format!(
            "total rate mismatch: tree={}, expected={}",
            tree.total(),
            expected_tree.total()
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::SumTree;
    use crate::topology::SlotError;

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
        // `target - (0.1 + 1/3)` rounds up to 3.3, so only `r == 0.0` avoids slot 3.
        let tree = SumTree::new(&[0.1, 1.0 / 3.0, 3.3, 0.0]);
        assert_eq!(tree.select(1.0 - f64::EPSILON / 2.0), 2);
    }

    #[test]
    fn tree_is_history_independent_and_bounds_checked() {
        let rates: Vec<f64> = (0..37).map(|i| 0.1 * ((i * 7) % 11) as f64).collect();
        let a = SumTree::new(&rates);
        let mut b = SumTree::new(&[0.0; 37]);
        for j in 0..37 {
            let i = (j * 17) % 37;
            b.try_set(i, rates[i]).unwrap();
        }
        b.try_set(5, 9.9).unwrap();
        b.try_set(5, rates[5]).unwrap();
        assert_eq!(a, b);
        assert_eq!(
            b.try_set(37, 1.0),
            Err(SlotError::OutOfRange {
                slot: 37,
                slot_count: 37
            })
        );
        assert_eq!(
            b.try_rate(37),
            Err(SlotError::OutOfRange {
                slot: 37,
                slot_count: 37
            })
        );
    }

    #[test]
    fn single_slot_tree() {
        let tree = SumTree::new(&[2.0]);
        assert_eq!(tree.select(0.3), 0);
        assert_eq!(tree.total(), 2.0);
        assert_eq!(tree.len(), 1);
    }
}
