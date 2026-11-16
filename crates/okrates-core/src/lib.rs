//! Serial kinetic Monte Carlo primitives and direct-method simulation.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod checkpoint;
pub mod clock;
pub mod model;
pub mod rate;
pub mod reference;
pub mod rng;
pub mod simulation;

pub use checkpoint::{CheckpointError, Checkpointable, SCHEMA_VERSION};
pub use clock::{Clock, ClockError};
pub use model::Model;
pub use rate::{Rate, RateError};
pub use rng::KmcRng;
pub use simulation::{RunSummary, SimError, Simulation, Step};
