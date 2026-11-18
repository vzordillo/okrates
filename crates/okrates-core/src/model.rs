//! Core model and reversible transition contracts.

use std::error::Error;

/// Baseline KMC model contract: enumerates `(event, rate)` pairs and applies transitions.
///
/// Rates are in inverse time units; zero-rate events are ignored by the engine.
/// Implementations must leave `state` logically unchanged when `apply` returns `Err`.
pub trait Model {
    type State;
    type Event;
    type Error: Error;

    fn enumerate_events(&self, state: &Self::State, emit: &mut dyn FnMut(Self::Event, f64));

    fn apply(&self, state: &mut Self::State, event: &Self::Event) -> Result<(), Self::Error>;
}

/// Extension of [`Model`] with a rollback token so post-transition catalog validation
/// failures restore `state` unchanged; `apply_reversible` must also leave `state` unchanged on `Err`.
pub trait ReversibleModel: Model {
    type Undo;

    fn apply_reversible(
        &self,
        state: &mut Self::State,
        event: &Self::Event,
    ) -> Result<Self::Undo, Self::Error>;

    fn undo(&self, state: &mut Self::State, undo: Self::Undo);
}

/// Adapter supplying [`ReversibleModel`] for any [`Model`] whose `State: Clone` (`O(|State|)` per step).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CloneRollback<M>(pub M);

impl<M: Model> Model for CloneRollback<M> {
    type State = M::State;
    type Event = M::Event;
    type Error = M::Error;

    fn enumerate_events(&self, state: &Self::State, emit: &mut dyn FnMut(Self::Event, f64)) {
        self.0.enumerate_events(state, emit);
    }

    fn apply(&self, state: &mut Self::State, event: &Self::Event) -> Result<(), Self::Error> {
        self.0.apply(state, event)
    }
}

impl<M> ReversibleModel for CloneRollback<M>
where
    M: Model,
    M::State: Clone,
{
    type Undo = M::State;

    fn apply_reversible(
        &self,
        state: &mut Self::State,
        event: &Self::Event,
    ) -> Result<Self::Undo, Self::Error> {
        let snapshot = state.clone();
        if let Err(err) = self.0.apply(state, event) {
            *state = snapshot;
            return Err(err);
        }
        Ok(snapshot)
    }

    fn undo(&self, state: &mut Self::State, undo: Self::Undo) {
        *state = undo;
    }
}
