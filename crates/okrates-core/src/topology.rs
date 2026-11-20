//! Decoupled transition topology and rate evaluation contracts.

use std::error::Error;
use std::fmt;

use crate::incremental::IncrementalModel;
use crate::model::{Model, ReversibleModel};
use crate::rate::RateError;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SlotError {
    OutOfRange { slot: usize, slot_count: usize },
    TooLarge { slot_count: usize },
    InvalidRate { slot: usize, source: RateError },
    TotalNotFinite,
}

impl fmt::Display for SlotError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::OutOfRange { slot, slot_count } => {
                write!(f, "slot {slot} out of range for {slot_count} slots")
            }
            Self::TooLarge { slot_count } => {
                write!(f, "slot count {slot_count} exceeds sum-tree capacity")
            }
            Self::InvalidRate { slot, source } => {
                write!(f, "invalid rate for slot {slot}: {source}")
            }
            Self::TotalNotFinite => write!(f, "total slot rate overflowed to non-finite"),
        }
    }
}

impl Error for SlotError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::InvalidRate { source, .. } => Some(source),
            Self::OutOfRange { .. } | Self::TooLarge { .. } | Self::TotalNotFinite => None,
        }
    }
}

/// Reversible state-transition geometry independent of transition rates.
pub trait TransitionTopology {
    type State;
    type Event;
    type Undo;
    type Error: Error;

    /// On `Err`, `state` must be unchanged.
    fn apply(
        &self,
        state: &mut Self::State,
        event: &Self::Event,
    ) -> Result<Self::Undo, Self::Error>;

    fn undo(&self, state: &mut Self::State, undo: Self::Undo);
}

/// Topology capable of enumerating candidate transitions from `state`.
pub trait EnumeratedTopology: TransitionTopology {
    fn enumerate_candidates(&self, state: &Self::State, emit: &mut dyn FnMut(Self::Event));
}

/// Topology whose candidate transitions occupy stable integer slots `0..slot_count(state)`.
///
/// [`EnumeratedTopology::enumerate_candidates`] must emit the exact events of active slots
/// (`is_slot_active(state, slot) == true`) so [`CoupledModel`] matches across direct and incremental catalogs.
pub trait SlottedTopology: EnumeratedTopology {
    fn slot_count(&self, state: &Self::State) -> usize;

    fn slot_event(&self, state: &Self::State, slot: usize) -> Self::Event;

    fn is_slot_active(&self, _state: &Self::State, _slot: usize) -> bool {
        true
    }

    /// Appends every slot whose `is_slot_active` or [`RateModel::rate`] depends on state modified by `event`.
    fn affected_slots(&self, state: &Self::State, event: &Self::Event, out: &mut Vec<usize>);
}

/// Evaluates non-negative transition rates (inverse time units) on a [`TransitionTopology`].
pub trait RateModel<T: TransitionTopology> {
    fn rate(&self, topology: &T, state: &T::State, event: &T::Event) -> f64;
}

/// Declares rate-dependent affected slots beyond a topology's purely geometric dependencies.
pub trait RateDependencies<T: SlottedTopology> {
    /// Appends slots whose [`RateModel::rate`] may change after `event` produces `state`.
    fn affected_rate_slots(
        &self,
        topology: &T,
        state: &T::State,
        event: &T::Event,
        out: &mut Vec<usize>,
    );
}

/// Default dependency policy assuming [`SlottedTopology::affected_slots`] already covers all rate dependencies.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TopologyOnlyDependencies;

impl<T: SlottedTopology> RateDependencies<T> for TopologyOnlyDependencies {
    fn affected_rate_slots(
        &self,
        _topology: &T,
        _state: &T::State,
        _event: &T::Event,
        _out: &mut Vec<usize>,
    ) {
    }
}

/// Composes a [`TransitionTopology`] with a [`RateModel`] (and optional [`RateDependencies`]) into a full KMC model.
///
/// Incremental updates invalidate both `T::affected_slots` and `D::affected_rate_slots`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CoupledModel<T, R, D = TopologyOnlyDependencies> {
    pub topology: T,
    pub rates: R,
    pub dependencies: D,
}

impl<T, R> CoupledModel<T, R> {
    pub fn new(topology: T, rates: R) -> Self {
        Self {
            topology,
            rates,
            dependencies: TopologyOnlyDependencies,
        }
    }
}

impl<T, R, D> CoupledModel<T, R, D> {
    /// Composes `topology`, `rates`, and an explicit [`RateDependencies`] provider.
    pub fn with_dependencies(topology: T, rates: R, dependencies: D) -> Self {
        Self {
            topology,
            rates,
            dependencies,
        }
    }
}

impl<T, R, D> Model for CoupledModel<T, R, D>
where
    T: EnumeratedTopology,
    R: RateModel<T>,
{
    type State = T::State;
    type Event = T::Event;
    type Error = T::Error;

    fn enumerate_events(&self, state: &Self::State, emit: &mut dyn FnMut(Self::Event, f64)) {
        self.topology.enumerate_candidates(state, &mut |event| {
            let rate = self.rates.rate(&self.topology, state, &event);
            emit(event, rate);
        });
    }

    fn apply(&self, state: &mut Self::State, event: &Self::Event) -> Result<(), Self::Error> {
        self.topology.apply(state, event).map(|_| ())
    }
}

impl<T, R, D> ReversibleModel for CoupledModel<T, R, D>
where
    T: EnumeratedTopology,
    R: RateModel<T>,
{
    type Undo = T::Undo;

    fn apply_reversible(
        &self,
        state: &mut Self::State,
        event: &Self::Event,
    ) -> Result<Self::Undo, Self::Error> {
        self.topology.apply(state, event)
    }

    fn undo(&self, state: &mut Self::State, undo: Self::Undo) {
        self.topology.undo(state, undo);
    }
}

impl<T, R, D> IncrementalModel for CoupledModel<T, R, D>
where
    T: SlottedTopology,
    R: RateModel<T>,
    D: RateDependencies<T>,
{
    fn slot_count(&self, state: &Self::State) -> usize {
        self.topology.slot_count(state)
    }

    fn slot_rate(&self, state: &Self::State, slot: usize) -> f64 {
        if slot >= self.topology.slot_count(state) {
            return f64::NAN;
        }
        if self.topology.is_slot_active(state, slot) {
            let event = self.topology.slot_event(state, slot);
            self.rates.rate(&self.topology, state, &event)
        } else {
            0.0
        }
    }

    fn slot_event(&self, state: &Self::State, slot: usize) -> Self::Event {
        self.topology.slot_event(state, slot)
    }

    fn affected_slots(&self, state: &Self::State, event: &Self::Event, out: &mut Vec<usize>) {
        let start = out.len();
        self.topology.affected_slots(state, event, out);
        let before_deps = out.len();
        self.dependencies
            .affected_rate_slots(&self.topology, state, event, out);
        if out.len() > before_deps && out.len() - start > 1 {
            out[start..].sort_unstable();
            let slice = &mut out[start..];
            let mut write = 1;
            for read in 1..slice.len() {
                if slice[read] != slice[write - 1] {
                    slice[write] = slice[read];
                    write += 1;
                }
            }
            out.truncate(start + write);
        }
    }
}
