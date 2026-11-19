//! Direct-method kinetic Monte Carlo simulation and shared step/error types.
//!
//! Residence-time (BKL) selection follows Bortz, Kalos, and Lebowitz,
//! *J. Comput. Phys.* **17**, 10–18 (1975), doi:10.1016/0021-9991(75)90060-1.

use std::error::Error;
use std::fmt;

use crate::catalog::{DirectCatalog, Engine};
use crate::checkpoint::{CheckpointError, Checkpointable};
use crate::clock::{Clock, ClockError};
use crate::model::Model;
use crate::rate::RateError;
use crate::rng::KmcRng;

#[derive(Clone, Debug, PartialEq)]
pub enum Step<E> {
    Fired { event: E, dt: f64 },
    Absorbed,
}

#[derive(Debug)]
pub enum SimError<ME> {
    InvalidRate { index: usize, source: RateError },
    TotalRateNotFinite,
    InvalidSlot { slot: usize, slot_count: usize },
    CatalogTooLarge { slot_count: usize },
    SlotCountChanged { expected: usize, found: usize },
    StepCounterOverflow,
    Clock(ClockError),
    Model(ME),
}

impl<ME: Error> fmt::Display for SimError<ME> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidRate { index, source } => {
                write!(f, "invalid rate for event {index}: {source}")
            }
            Self::TotalRateNotFinite => write!(f, "total event rate is not finite"),
            Self::InvalidSlot { slot, slot_count } => {
                write!(f, "slot {slot} out of range for {slot_count} slots")
            }
            Self::CatalogTooLarge { slot_count } => {
                write!(f, "catalog of {slot_count} slots exceeds capacity")
            }
            Self::SlotCountChanged { expected, found } => {
                write!(f, "slot count changed from {expected} to {found}")
            }
            Self::StepCounterOverflow => write!(f, "step counter overflowed u64::MAX"),
            Self::Clock(source) => write!(f, "clock error: {source}"),
            Self::Model(source) => write!(f, "model error: {source}"),
        }
    }
}

impl<ME: Error + 'static> Error for SimError<ME> {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::InvalidRate { source, .. } => Some(source),
            Self::TotalRateNotFinite
            | Self::InvalidSlot { .. }
            | Self::CatalogTooLarge { .. }
            | Self::SlotCountChanged { .. }
            | Self::StepCounterOverflow => None,
            Self::Clock(source) => Some(source),
            Self::Model(source) => Some(source),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RunSummary {
    pub steps: u64,
    pub time: f64,
    pub absorbed: bool,
}

/// Serial direct-method KMC simulation backed by [`Engine`] and [`DirectCatalog`].
pub struct Simulation<M: Model> {
    engine: Engine<M, DirectCatalog<M::Event>>,
}

impl<M: Model> Simulation<M> {
    pub fn new(model: M, state: M::State, seed: u64) -> Self {
        Self::with_rng(model, state, KmcRng::from_seed(seed))
    }

    pub fn with_rng(model: M, state: M::State, rng: KmcRng) -> Self {
        Self {
            engine: Engine::from_parts(model, state, Clock::new(), rng, 0, DirectCatalog::new()),
        }
    }

    pub fn engine(&self) -> &Engine<M, DirectCatalog<M::Event>> {
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

    /// Validates active rates without advancing RNG, clock, or step count; returns `Ok(true)` if absorbed.
    pub fn is_absorbed(&mut self) -> Result<bool, SimError<M::Error>> {
        self.engine.is_absorbed()
    }

    /// Fires one transition atomically; on `Err`, simulation state, catalog, clock, steps, and RNG are unchanged.
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

impl<M: Checkpointable> Simulation<M> {
    pub fn checkpoint(&self) -> Vec<u8> {
        self.engine.checkpoint()
    }

    pub fn resume(model: M, bytes: &[u8]) -> Result<Self, CheckpointError> {
        Ok(Self {
            engine: Engine::resume(model, bytes)?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::{Simulation, Step};
    use crate::catalog::direct_select;
    use crate::rate::Rate;
    use crate::reference::GraphRandomWalk;
    use crate::rng::KmcRng;

    fn rates(values: &[f64]) -> Vec<Rate> {
        values
            .iter()
            .map(|&value| Rate::new(value).unwrap())
            .collect()
    }

    #[test]
    fn select_handles_two_equal_rates_and_the_boundary() {
        let values = rates(&[1.0, 1.0]);
        assert_eq!(direct_select(&values, 2.0, 0.5), 1);
        assert_eq!(direct_select(&values, 2.0, 0.49999999999999994), 0);
        assert_eq!(direct_select(&values, 2.0, 0.25), 0);
    }

    #[test]
    fn select_handles_three_rates() {
        let values = rates(&[1.0, 2.0, 3.0]);
        assert_eq!(direct_select(&values, 6.0, 0.1), 0);
        assert_eq!(direct_select(&values, 6.0, 0.4), 1);
        assert_eq!(direct_select(&values, 6.0, 0.9), 2);
    }

    #[test]
    fn select_falls_back_to_the_last_rate_when_target_equals_total() {
        assert_eq!(direct_select(&rates(&[3.0]), 3.0, 1.0), 0);
        assert_eq!(direct_select(&rates(&[1.0, 1.0]), 2.0, 1.0), 1);
    }

    #[test]
    fn absorbing_step_draws_no_random_numbers() {
        let model = GraphRandomWalk::new(vec![vec![]]).unwrap();
        let mut sim = Simulation::with_rng(model.clone(), 0, KmcRng::from_seed(99));
        for _ in 0..3 {
            assert_eq!(sim.step().unwrap(), Step::Absorbed);
        }
        let fresh = Simulation::new(model, 0, 99);
        assert_eq!(sim.checkpoint(), fresh.checkpoint());
    }
}
