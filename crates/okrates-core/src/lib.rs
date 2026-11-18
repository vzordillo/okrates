//! Serial kinetic Monte Carlo primitives, catalog engines, and trajectory sampling.

#![forbid(unsafe_code)]

pub mod catalog;
pub mod checkpoint;
pub mod clock;
pub mod ensemble;
pub mod incremental;
pub mod model;
pub mod rate;
pub mod reference;
pub mod rng;
pub mod sampler;
pub mod simulation;
pub mod testing;
pub mod topology;

pub use catalog::{CatalogBackend, DirectCatalog, Engine, SumTreeCatalog};
pub use checkpoint::{CheckpointError, Checkpointable, SCHEMA_VERSION};
pub use clock::{Clock, ClockError};
pub use ensemble::{EnsembleError, Replica, ReplicaContext, run_ensemble, run_replicas};
pub use incremental::{IncrementalModel, IncrementalSimulation, SumTree};
pub use model::{CloneRollback, IndependentSnapshot, Model, ReversibleModel};
pub use rate::{Rate, RateError};
pub use rng::{KmcRng, StreamRole};
pub use sampler::{
    AbsorptionPolicy, GridError, Observe, SampleGrid, SampledTrajectory, SamplingError,
    TrajectorySource, sample_trajectory, sample_trajectory_with,
};
pub use simulation::{RunSummary, SimError, Simulation, Step};
pub use topology::{
    CoupledModel, EnumeratedTopology, RateModel, SlotError, SlottedTopology, TransitionTopology,
};
