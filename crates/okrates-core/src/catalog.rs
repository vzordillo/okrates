//! Unified catalog backends and transactional KMC stepping engine.

use crate::checkpoint::{self, CheckpointError, Checkpointable};
use crate::clock::Clock;
use crate::incremental::{IncrementalModel, SumTree};
use crate::model::Model;
use crate::rate::Rate;
use crate::rng::KmcRng;
use crate::simulation::{RunSummary, SimError, Step};

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
            return Err(error);
        }

        let mut total = 0.0;
        for rate in &self.rates {
            total += rate.get();
        }
        if !total.is_finite() {
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
        let event = self.events.swap_remove(selection);
        model.apply(state, &event).map_err(SimError::Model)?;
        Ok(event)
    }
}

/// Incremental binary sum-tree catalog updated locally on affected slots (`O(k log M)` per step).
#[derive(Clone, Debug)]
pub struct SumTreeCatalog {
    tree: SumTree,
    scratch: Vec<usize>,
    staged: Vec<(usize, f64)>,
}

impl SumTreeCatalog {
    pub fn tree(&self) -> &SumTree {
        &self.tree
    }
}

impl<M: IncrementalModel> CatalogBackend<M> for SumTreeCatalog {
    type Selection = usize;

    fn build(model: &M, state: &M::State) -> Result<Self, SimError<M::Error>> {
        let count = model.slot_count(state);
        let mut rates = Vec::with_capacity(count);
        for slot in 0..count {
            let rate = Rate::new(model.slot_rate(state, slot)).map_err(|source| {
                SimError::InvalidRate {
                    index: slot,
                    source,
                }
            })?;
            rates.push(rate.get());
        }
        Ok(Self {
            tree: SumTree::new(&rates),
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
            if affected >= expected_slots {
                model.undo(state, undo);
                return Err(SimError::InvalidSlot {
                    slot: affected,
                    slot_count: expected_slots,
                });
            }
            match Rate::new(model.slot_rate(state, affected)) {
                Ok(rate) => self.staged.push((affected, rate.get())),
                Err(source) => {
                    model.undo(state, undo);
                    return Err(SimError::InvalidRate {
                        index: affected,
                        source,
                    });
                }
            }
        }

        for &(affected, rate) in &self.staged {
            let _ = self.tree.try_set(affected, rate);
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

    /// Fires one transition atomically; on `Err`, `state`, `catalog`, `clock`, `steps`, and `rng` are unchanged.
    pub fn step(&mut self) -> Result<Step<M::Event>, SimError<M::Error>> {
        self.step_observing(|_, _, _| {})
    }

    /// Like [`Engine::step`], invoking `before_apply(&model, &state_before, next_clock)` once `dt` is validated
    /// and before applying the transition (if `apply_and_update` later errors, engine state still rolls back).
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
        self.steps += 1;
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
