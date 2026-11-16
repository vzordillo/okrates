//! The model contract for state, rates, and transitions.

/// A kinetic Monte Carlo model.
///
/// The engine owns sampling, the clock, and the lifecycle; the model owns
/// state, legality, rates, and transitions. The event catalog is rebuilt
/// from scratch every step as the baseline; incremental updates come later.
pub trait Model {
    /// The model's mutable simulation state.
    type State;
    /// An event that can be applied to the state.
    type Event;
    /// An error returned when applying an event fails.
    type Error: std::error::Error;

    /// Emit every currently possible event with its rate (inverse time units).
    ///
    /// The engine validates each rate; zero-rate events are ignored.
    fn enumerate_events(&self, state: &Self::State, emit: &mut dyn FnMut(Self::Event, f64));

    /// Execute `event` on `state`.
    fn apply(&self, state: &mut Self::State, event: &Self::Event) -> Result<(), Self::Error>;
}
