//! Direct-method kinetic Monte Carlo simulation engine.

use std::error::Error;
use std::fmt;

use crate::checkpoint::{CheckpointError, Checkpointable, MAGIC, Reader, SCHEMA_VERSION};
use crate::clock::{Clock, ClockError};
use crate::model::Model;
use crate::rate::{Rate, RateError};
use crate::rng::KmcRng;

/// The result of one simulation step.
#[derive(Clone, Debug, PartialEq)]
pub enum Step<E> {
    /// An event fired after the stated waiting time.
    Fired {
        /// The event that fired.
        event: E,
        /// The elapsed time before the event fired.
        dt: f64,
    },
    /// No positive-rate event was available.
    Absorbed,
}

/// An error while validating rates, advancing time, or applying an event.
#[derive(Debug)]
pub enum SimError<ME> {
    /// An enumerated event had an invalid rate.
    InvalidRate {
        /// The event's zero-based position in enumeration order.
        index: usize,
        /// The invalid rate error.
        source: RateError,
    },
    /// The finite event rates had a non-finite total.
    TotalRateNotFinite,
    /// The clock could not accept the sampled waiting time.
    Clock(ClockError),
    /// The model could not apply the selected event.
    Model(ME),
}

impl<ME: Error> fmt::Display for SimError<ME> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidRate { index, source } => {
                write!(f, "invalid rate for event {index}: {source}")
            }
            Self::TotalRateNotFinite => write!(f, "total event rate is not finite"),
            Self::Clock(source) => write!(f, "clock error: {source}"),
            Self::Model(source) => write!(f, "model error: {source}"),
        }
    }
}

impl<ME: Error + 'static> Error for SimError<ME> {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::InvalidRate { source, .. } => Some(source),
            Self::TotalRateNotFinite => None,
            Self::Clock(source) => Some(source),
            Self::Model(source) => Some(source),
        }
    }
}

/// The result of running a simulation for a bounded number of steps.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RunSummary {
    /// The number of events fired during this run call.
    pub steps: u64,
    /// The simulation time after this run call.
    pub time: f64,
    /// Whether the simulation stopped in an absorbing state.
    pub absorbed: bool,
}

/// A serial direct-method kinetic Monte Carlo simulation.
pub struct Simulation<M: Model> {
    model: M,
    state: M::State,
    clock: Clock,
    rng: KmcRng,
    steps: u64,
    events: Vec<M::Event>,
    rates: Vec<Rate>,
}

impl<M: Model> Simulation<M> {
    /// Creates a simulation with a seeded random stream.
    pub fn new(model: M, state: M::State, seed: u64) -> Self {
        Self {
            model,
            state,
            clock: Clock::new(),
            rng: KmcRng::from_seed(seed),
            steps: 0,
            events: Vec::new(),
            rates: Vec::new(),
        }
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

    /// Fires one event, or reports that the state is absorbing.
    pub fn step(&mut self) -> Result<Step<M::Event>, SimError<M::Error>> {
        self.events.clear();
        self.rates.clear();

        let mut validation_error = None;
        let mut index = 0;
        let (model, state, events, rates) =
            (&self.model, &self.state, &mut self.events, &mut self.rates);
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
            return Ok(Step::Absorbed);
        }

        let u1 = self.rng.open01();
        let idx = select(&self.rates, total, u1);
        let u2 = self.rng.open01();
        let dt = -u2.ln() / total;

        let mut advanced_clock = self.clock;
        advanced_clock.advance(dt).map_err(SimError::Clock)?;
        let event = self.events.swap_remove(idx);
        self.model
            .apply(&mut self.state, &event)
            .map_err(SimError::Model)?;
        self.clock = advanced_clock;
        self.steps += 1;
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

impl<M: Checkpointable> Simulation<M> {
    /// Encodes the simulation as a checkpoint in schema version 1.
    pub fn checkpoint(&self) -> Vec<u8> {
        let (seed, stream, word_pos) = self.rng.export_state();
        let mut out = Vec::new();
        out.extend_from_slice(MAGIC);
        out.extend_from_slice(&SCHEMA_VERSION.to_le_bytes());
        out.extend_from_slice(&self.steps.to_le_bytes());
        out.extend_from_slice(&self.clock.time().to_bits().to_le_bytes());
        out.extend_from_slice(&seed);
        out.extend_from_slice(&stream.to_le_bytes());
        out.extend_from_slice(&word_pos.to_le_bytes());
        let id = M::MODEL_ID.as_bytes();
        out.extend_from_slice(&(id.len() as u32).to_le_bytes());
        out.extend_from_slice(id);
        let mut params = Vec::new();
        self.model.encode_params(&mut params);
        out.extend_from_slice(&(params.len() as u64).to_le_bytes());
        out.extend_from_slice(&params);
        let mut state = Vec::new();
        self.model.encode_state(&self.state, &mut state);
        out.extend_from_slice(&(state.len() as u64).to_le_bytes());
        out.extend_from_slice(&state);
        out
    }

    /// Rebuilds a simulation from `bytes`, validating them against `model`.
    pub fn resume(model: M, bytes: &[u8]) -> Result<Self, CheckpointError> {
        let mut reader = Reader::new(bytes);
        if reader.take(MAGIC.len())? != MAGIC {
            return Err(CheckpointError::BadMagic);
        }
        let version = reader.u32()?;
        if version != SCHEMA_VERSION {
            return Err(CheckpointError::UnsupportedVersion(version));
        }
        let steps = reader.u64()?;
        let time = f64::from_bits(reader.u64()?);
        let seed = reader.array::<32>()?;
        let stream = reader.u64()?;
        let word_pos = reader.u128()?;
        let id_len = usize::try_from(reader.u32()?).map_err(|_| CheckpointError::Truncated)?;
        let id = reader.take(id_len)?;
        let params = reader.section()?;
        let state_bytes = reader.section()?;
        if !reader.is_empty() {
            return Err(CheckpointError::TrailingBytes);
        }
        // A negative sign bit also rejects -0.0, which Clock::new cannot restore bitwise.
        if !time.is_finite() || time.is_sign_negative() {
            return Err(CheckpointError::InvalidTime(time));
        }
        if id != M::MODEL_ID.as_bytes() {
            return Err(CheckpointError::ModelMismatch);
        }
        let mut expected = Vec::new();
        model.encode_params(&mut expected);
        if params != expected {
            return Err(CheckpointError::ParamsMismatch);
        }
        let state = model
            .decode_state(state_bytes)
            .map_err(CheckpointError::InvalidState)?;

        // 0.0 + time is exact, so advancing a new clock restores the stored bits.
        let mut clock = Clock::new();
        if time > 0.0 {
            clock
                .advance(time)
                .map_err(|_| CheckpointError::InvalidTime(time))?;
        }
        Ok(Self {
            model,
            state,
            clock,
            rng: KmcRng::from_state(seed, stream, word_pos),
            steps,
            events: Vec::new(),
            rates: Vec::new(),
        })
    }
}

fn select(rates: &[Rate], total: f64, u: f64) -> usize {
    let target = u * total;
    let mut acc = 0.0;
    for (i, rate) in rates.iter().enumerate() {
        acc += rate.get();
        if target < acc {
            return i;
        }
    }
    // The running sum reproduces total bitwise in the same order, so this
    // fallback selects the final event when rounding puts target at total.
    rates.len() - 1
}

#[cfg(test)]
mod tests {
    use super::{Simulation, Step, select};
    use crate::rate::Rate;
    use crate::reference::GraphRandomWalk;

    fn rates(values: &[f64]) -> Vec<Rate> {
        values
            .iter()
            .map(|&value| Rate::new(value).unwrap())
            .collect()
    }

    #[test]
    fn select_handles_two_equal_rates_and_the_boundary() {
        let values = rates(&[1.0, 1.0]);
        assert_eq!(select(&values, 2.0, 0.5), 1);
        assert_eq!(select(&values, 2.0, 0.49999999999999994), 0);
        assert_eq!(select(&values, 2.0, 0.25), 0);
    }

    #[test]
    fn select_handles_three_rates() {
        let values = rates(&[1.0, 2.0, 3.0]);
        assert_eq!(select(&values, 6.0, 0.1), 0);
        assert_eq!(select(&values, 6.0, 0.4), 1);
        assert_eq!(select(&values, 6.0, 0.9), 2);
    }

    #[test]
    fn select_falls_back_to_the_last_rate_when_target_equals_total() {
        assert_eq!(select(&rates(&[3.0]), 3.0, 1.0), 0);
        assert_eq!(select(&rates(&[1.0, 1.0]), 2.0, 1.0), 1);
    }

    #[test]
    fn absorbing_step_draws_no_random_numbers() {
        let model = GraphRandomWalk::new(vec![vec![]]).unwrap();
        let mut sim = Simulation::new(model, 0, 99);
        for _ in 0..3 {
            assert_eq!(sim.step().unwrap(), Step::Absorbed);
        }
        assert_eq!(
            sim.rng.open01().to_bits(),
            crate::rng::KmcRng::from_seed(99).open01().to_bits()
        );
    }
}
