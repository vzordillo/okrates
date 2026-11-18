//! Decoupled transition topology and rate evaluation contracts.

use std::error::Error;
use std::fmt;

use crate::incremental::IncrementalModel;
use crate::model::{Model, ReversibleModel};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SlotError {
    OutOfRange { slot: usize, slot_count: usize },
}

impl fmt::Display for SlotError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::OutOfRange { slot, slot_count } => {
                write!(f, "slot {slot} out of range for {slot_count} slots")
            }
        }
    }
}

impl Error for SlotError {}

/// Reversible state-transition geometry independent of transition rates.
pub trait TransitionTopology {
    type State;
    type Event;
    type Undo;
    type Error: Error;

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

/// Composes a [`TransitionTopology`] with a [`RateModel`] into a full KMC model.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CoupledModel<T, R> {
    pub topology: T,
    pub rates: R,
}

impl<T, R> CoupledModel<T, R> {
    pub fn new(topology: T, rates: R) -> Self {
        Self { topology, rates }
    }
}

impl<T, R> Model for CoupledModel<T, R>
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

impl<T, R> ReversibleModel for CoupledModel<T, R>
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

impl<T, R> IncrementalModel for CoupledModel<T, R>
where
    T: SlottedTopology,
    R: RateModel<T>,
{
    fn slot_count(&self, state: &Self::State) -> usize {
        self.topology.slot_count(state)
    }

    fn slot_rate(&self, state: &Self::State, slot: usize) -> f64 {
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
        self.topology.affected_slots(state, event, out);
    }
}
