//! Unified catalog backends and transactional KMC stepping engine.
//!
//! Residence-time (BKL) selection follows Bortz, Kalos, and Lebowitz,
//! *J. Comput. Phys.* **17**, 10–18 (1975), doi:10.1016/0021-9991(75)90060-1.

use crate::checkpoint::{self, CheckpointError, Checkpointable};
use crate::clock::Clock;
use crate::incremental::{IncrementalModel, SumTree};
use crate::model::Model;
use crate::rate::Rate;
use crate::rng::KmcRng;
use crate::simulation::{RunSummary, SimError, Step};
use crate::topology::SlotError;

/// Event-catalog strategy backing [`Engine`].
pub trait CatalogBackend<M: Model>: Sized {
    type Selection: Copy;

    fn build(model: &M, state: &M::State) -> Result<Self, SimError<M::Error>>;

    /// Validates active rates and returns `Ok(None)` when `state` is absorbing.
    fn prepare_total(
        &mut self,
        model: &M,
        state: &M::State,
    ) -> Result<Option<f64>, SimError<M::Error>>;

    /// Selects a candidate from `u` in `(0, 1)` given a positive finite total rate.
    fn select(&self, u: f64, total: f64) -> Self::Selection;

    /// Applies `selection` and updates catalog state atomically, rolling back `state` on error.
    fn apply_and_update(
        &mut self,
        model: &M,
        state: &mut M::State,
        selection: Self::Selection,
    ) -> Result<M::Event, SimError<M::Error>>;
}

/// Linear-scan event catalog rebuilt on each step (`O(M)` per step).
#[derive(Clone, Debug)]
pub struct DirectCatalog<E> {
    events: Vec<E>,
    rates: Vec<Rate>,
}

impl<E> Default for DirectCatalog<E> {
    fn default() -> Self {
        Self::new()
    }
}

impl<E> DirectCatalog<E> {
    pub fn new() -> Self {
        Self {
            events: Vec::new(),
            rates: Vec::new(),
        }
    }

    pub fn len(&self) -> usize {
        self.events.len()
    }

    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }

    pub fn allocated_bytes(&self) -> usize {
        self.events.capacity() * std::mem::size_of::<E>()
            + self.rates.capacity() * std::mem::size_of::<Rate>()
    }
}

pub(crate) fn direct_select(rates: &[Rate], total: f64, u: f64) -> usize {
    let target = u * total;
    let mut acc = 0.0;
    for (i, rate) in rates.iter().enumerate() {
        acc += rate.get();
        if target < acc {
            return i;
        }
    }
    // Identical summation order guarantees `acc == total`; fallback handles `target == total` rounding.
    rates.len() - 1
}

impl<M: Model> CatalogBackend<M> for DirectCatalog<M::Event> {
    type Selection = usize;

    fn build(_model: &M, _state: &M::State) -> Result<Self, SimError<M::Error>> {
        Ok(Self::new())
    }

    fn prepare_total(
        &mut self,
        model: &M,
        state: &M::State,
    ) -> Result<Option<f64>, SimError<M::Error>> {
        self.events.clear();
        self.rates.clear();

        let mut validation_error = None;
        let mut index = 0;
        let (events, rates) = (&mut self.events, &mut self.rates);
        model.enumerate_events(state, &mut |event, value| {
            let event_index = index;
            index += 1;
            match Rate::new(value) {
                Ok(rate) if !rate.is_zero() => {
                    events.push(event);
                    rates.push(rate);
                }
                Ok(_) => {}
                Err(source) => {
                    if validation_error.is_none() {
                        validation_error = Some(SimError::InvalidRate {
                            index: event_index,
                            source,
                        });
                    }
                }
            }
        });
        if let Some(error) = validation_error {
            self.events.clear();
            self.rates.clear();
            return Err(error);
        }

        let mut total = 0.0;
        for rate in &self.rates {
            total += rate.get();
        }
        if !total.is_finite() {
            self.events.clear();
            self.rates.clear();
            return Err(SimError::TotalRateNotFinite);
        }
        if self.events.is_empty() {
            return Ok(None);
        }
        Ok(Some(total))
    }

    fn select(&self, u: f64, total: f64) -> Self::Selection {
        direct_select(&self.rates, total, u)
    }

    fn apply_and_update(
        &mut self,
        model: &M,
        state: &mut M::State,
        selection: Self::Selection,
    ) -> Result<M::Event, SimError<M::Error>> {
        let slot_count = self.events.len();
        let event_ref = self.events.get(selection).ok_or(SimError::InvalidSlot {
            slot: selection,
            slot_count,
        })?;
        model.apply(state, event_ref).map_err(SimError::Model)?;
        self.rates.swap_remove(selection);
        Ok(self.events.swap_remove(selection))
    }
}

/// Incremental binary sum-tree catalog updated locally on affected slots (`O(k log M)` per step).
#[derive(Clone, Debug)]
pub struct SumTreeCatalog {
    tree: SumTree,
    scratch: Vec<usize>,
    staged: Vec<(usize, f64, f64)>,
}

impl SumTreeCatalog {
    pub fn tree(&self) -> &SumTree {
        &self.tree
    }

    pub fn allocated_bytes(&self) -> usize {
        std::mem::size_of_val(self.tree.nodes())
    }
}

fn map_slot_error<ME>(err: SlotError) -> SimError<ME> {
    match err {
        SlotError::OutOfRange { slot, slot_count } => SimError::InvalidSlot { slot, slot_count },
        SlotError::TooLarge { slot_count } => SimError::CatalogTooLarge { slot_count },
        SlotError::InvalidRate { slot, source } => SimError::InvalidRate {
            index: slot,
            source,
        },
        SlotError::TotalNotFinite => SimError::TotalRateNotFinite,
    }
}

impl<M: IncrementalModel> CatalogBackend<M> for SumTreeCatalog {
    type Selection = usize;

    fn build(model: &M, state: &M::State) -> Result<Self, SimError<M::Error>> {
        let count = model.slot_count(state);
        let mut tree = SumTree::try_with_len(count)
            .map_err(|_| SimError::CatalogTooLarge { slot_count: count })?;
        tree.init_from_fn(|slot| {
            Rate::new(model.slot_rate(state, slot))
                .map(Rate::get)
                .map_err(|source| SimError::InvalidRate {
                    index: slot,
                    source,
                })
        })?;
        Ok(Self {
            tree,
            scratch: Vec::new(),
            staged: Vec::new(),
        })
    }

    fn prepare_total(
        &mut self,
        _model: &M,
        _state: &M::State,
    ) -> Result<Option<f64>, SimError<M::Error>> {
        let total = self.tree.total();
        if !total.is_finite() {
            return Err(SimError::TotalRateNotFinite);
        }
        if total == 0.0 {
            return Ok(None);
        }
        Ok(Some(total))
    }

    fn select(&self, u: f64, _total: f64) -> Self::Selection {
        self.tree.select(u)
    }

    fn apply_and_update(
        &mut self,
        model: &M,
        state: &mut M::State,
        slot: Self::Selection,
    ) -> Result<M::Event, SimError<M::Error>> {
        let expected_slots = self.tree.len();
        if slot >= expected_slots {
            return Err(SimError::InvalidSlot {
                slot,
                slot_count: expected_slots,
            });
        }
        let event = model.slot_event(state, slot);
        let undo = model
            .apply_reversible(state, &event)
            .map_err(SimError::Model)?;

        let found_slots = model.slot_count(state);
        if found_slots != expected_slots {
            model.undo(state, undo);
            return Err(SimError::SlotCountChanged {
                expected: expected_slots,
                found: found_slots,
            });
        }

        self.scratch.clear();
        self.staged.clear();
        model.affected_slots(state, &event, &mut self.scratch);
        for &affected in &self.scratch {
            let prev = match self.tree.try_rate(affected) {
                Ok(prev) => prev,
                Err(err) => {
                    model.undo(state, undo);
                    return Err(map_slot_error(err));
                }
            };
            match Rate::new(model.slot_rate(state, affected)) {
                Ok(rate) => self.staged.push((affected, prev, rate.get())),
                Err(source) => {
                    model.undo(state, undo);
                    return Err(SimError::InvalidRate {
                        index: affected,
                        source,
                    });
                }
            }
        }

        for (applied, &(affected, _, rate)) in self.staged.iter().enumerate() {
            if let Err(err) = self.tree.try_set(affected, rate) {
                for &(prev_slot, prev_rate, _) in self.staged[..applied].iter().rev() {
                    let _ = self.tree.try_set(prev_slot, prev_rate);
                }
                model.undo(state, undo);
                return Err(map_slot_error(err));
            }
        }
        Ok(event)
    }
}

/// Unified rejection-free KMC simulation engine parameterized by [`CatalogBackend`].
pub struct Engine<M: Model, C> {
    model: M,
    state: M::State,
    clock: Clock,
    rng: KmcRng,
    steps: u64,
    catalog: C,
}

impl<M: Model, C: CatalogBackend<M>> Engine<M, C> {
    pub fn new(model: M, state: M::State, seed: u64) -> Result<Self, SimError<M::Error>> {
        Self::with_rng(model, state, KmcRng::from_seed(seed))
    }

    pub fn with_rng(model: M, state: M::State, rng: KmcRng) -> Result<Self, SimError<M::Error>> {
        let catalog = C::build(&model, &state)?;
        Ok(Self::from_parts(
            model,
            state,
            Clock::new(),
            rng,
            0,
            catalog,
        ))
    }

    pub(crate) fn from_parts(
        model: M,
        state: M::State,
        clock: Clock,
        rng: KmcRng,
        steps: u64,
        catalog: C,
    ) -> Self {
        Self {
            model,
            state,
            clock,
            rng,
            steps,
            catalog,
        }
    }

    pub fn model(&self) -> &M {
        &self.model
    }

    pub fn state(&self) -> &M::State {
        &self.state
    }

    pub fn clock(&self) -> Clock {
        self.clock
    }

    pub fn time(&self) -> f64 {
        self.clock.time()
    }

    pub fn steps(&self) -> u64 {
        self.steps
    }

    pub fn catalog(&self) -> &C {
        &self.catalog
    }

    /// Validates active rates without advancing RNG, clock, or step count; returns `Ok(true)` if absorbed.
    pub fn is_absorbed(&mut self) -> Result<bool, SimError<M::Error>> {
        Ok(self
            .catalog
            .prepare_total(&self.model, &self.state)?
            .is_none())
    }

    /// Fires one transition atomically; on `Err`, `state`, `clock`, `steps`, `rng`, and
    /// sum-tree weights are unchanged (`DirectCatalog` scratch buffers may be cleared on validation error).
    pub fn step(&mut self) -> Result<Step<M::Event>, SimError<M::Error>> {
        self.step_observing(|_, _, _| {})
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
        let Some(total) = self.catalog.prepare_total(&self.model, &self.state)? else {
            return Ok(Step::Absorbed);
        };
        let next_steps = self
            .steps
            .checked_add(1)
            .ok_or(SimError::StepCounterOverflow)?;

        let mut next_rng = self.rng.clone();
        let u1 = next_rng.open01();
        let selection = self.catalog.select(u1, total);
        let u2 = next_rng.open01();
        let dt = -u2.ln() / total;

        let mut next_clock = self.clock;
        next_clock.advance(dt).map_err(SimError::Clock)?;

        before_apply(&self.model, &self.state, next_clock);
        let event = self
            .catalog
            .apply_and_update(&self.model, &mut self.state, selection)?;

        self.rng = next_rng;
        self.clock = next_clock;
        self.steps = next_steps;
        Ok(Step::Fired { event, dt })
    }

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

impl<M: Checkpointable, C: CatalogBackend<M>> Engine<M, C> {
    pub fn checkpoint(&self) -> Vec<u8> {
        checkpoint::encode(self.steps, &self.clock, &self.rng, &self.model, &self.state)
    }

    pub fn resume(model: M, bytes: &[u8]) -> Result<Self, CheckpointError> {
        let (steps, clock, rng, state) = checkpoint::decode(&model, bytes)?;
        let catalog = C::build(&model, &state).map_err(|err| match err {
            SimError::InvalidRate { index, source } => CheckpointError::InvalidState(
                format!("invalid rate in slot {index}: {source}").into(),
            ),
            other => CheckpointError::InvalidState(format!("catalog build failed: {other}").into()),
        })?;
        Ok(Self::from_parts(model, state, clock, rng, steps, catalog))
    }
}
