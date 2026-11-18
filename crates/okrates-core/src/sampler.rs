//! Piecewise-constant right-continuous CTMC trajectory sampling.

use std::error::Error;
use std::fmt;

use crate::catalog::{CatalogBackend, Engine};
use crate::clock::Clock;
use crate::incremental::{IncrementalModel, IncrementalSimulation};
use crate::model::Model;
use crate::simulation::{SimError, Simulation, Step};

/// Behavior when a trajectory enters an absorbing state (`total_rate == 0`) before `t_end`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AbsorptionPolicy {
    Error,
    HoldFinal,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum GridError {
    NonFiniteBounds { t_start: f64, t_end: f64 },
    InvalidWindow { t_start: f64, t_end: f64 },
    EmptyCount,
    NegativeTime { index: usize, found: f64 },
    NonMonotone { index: usize, prev: f64, found: f64 },
}

impl fmt::Display for GridError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonFiniteBounds { t_start, t_end } => {
                write!(f, "sample window [{t_start}, {t_end}] must be finite")
            }
            Self::InvalidWindow { t_start, t_end } => {
                write!(
                    f,
                    "sample window requires 0 <= t_start < t_end, got [{t_start}, {t_end}]"
                )
            }
            Self::EmptyCount => write!(f, "sample grid requires at least one point"),
            Self::NegativeTime { index, found } => {
                write!(
                    f,
                    "sample time at index {index} must be non-negative, got {found}"
                )
            }
            Self::NonMonotone { index, prev, found } => {
                write!(
                    f,
                    "sample times must be non-decreasing: index {index} has {found} < {prev}"
                )
            }
        }
    }
}

impl Error for GridError {}

/// Validated monotone observation times `0 <= t_0 <= ... <= t_{K-1}`.
#[derive(Clone, Debug, PartialEq)]
pub struct SampleGrid {
    times: Vec<f64>,
}

impl SampleGrid {
    /// Uniform grid `t_k = k * t_end / count` for `k = 1..=count`.
    pub fn from_origin(t_end: f64, count: usize) -> Result<Self, GridError> {
        Self::windowed(0.0, t_end, count)
    }

    /// Uniform grid over `[t_equil, t_end]`.
    ///
    /// When `t_equil == 0.0`, emits `k * t_end / count` (`k = 1..=count`).
    /// When `t_equil > 0.0` and `count >= 2`, spans `[t_equil, t_end]` inclusive (`[t_end]` if `count == 1`).
    pub fn windowed(t_equil: f64, t_end: f64, count: usize) -> Result<Self, GridError> {
        if !t_equil.is_finite() || !t_end.is_finite() {
            return Err(GridError::NonFiniteBounds {
                t_start: t_equil,
                t_end,
            });
        }
        if t_equil < 0.0 || t_equil >= t_end {
            return Err(GridError::InvalidWindow {
                t_start: t_equil,
                t_end,
            });
        }
        if count == 0 {
            return Err(GridError::EmptyCount);
        }
        let mut times = if t_equil == 0.0 {
            (1..=count)
                .map(|k| {
                    if k == count {
                        t_end
                    } else {
                        let prod = k as f64 * t_end;
                        let v = if prod.is_finite() {
                            prod / count as f64
                        } else {
                            (k as f64 / count as f64) * t_end
                        };
                        v.clamp(0.0, t_end)
                    }
                })
                .collect()
        } else if count == 1 {
            vec![t_end]
        } else {
            let span = t_end - t_equil;
            let denom = (count - 1) as f64;
            (0..count)
                .map(|i| {
                    if i == 0 {
                        t_equil
                    } else if i + 1 == count {
                        t_end
                    } else {
                        let w = i as f64 / denom;
                        let raw = if span.is_finite() {
                            t_equil + w * span
                        } else {
                            (1.0 - w) * t_equil + w * t_end
                        };
                        raw.clamp(t_equil, t_end)
                    }
                })
                .collect()
        };
        for i in 1..times.len() {
            if times[i] < times[i - 1] {
                times[i] = times[i - 1];
            }
        }
        Ok(Self { times })
    }

    pub fn explicit(times: Vec<f64>) -> Result<Self, GridError> {
        if times.is_empty() {
            return Err(GridError::EmptyCount);
        }
        for (i, &t) in times.iter().enumerate() {
            if !t.is_finite() {
                return Err(GridError::NonFiniteBounds {
                    t_start: times[0],
                    t_end: t,
                });
            }
            if t < 0.0 || t.is_sign_negative() {
                return Err(GridError::NegativeTime { index: i, found: t });
            }
            if i > 0 && t < times[i - 1] {
                return Err(GridError::NonMonotone {
                    index: i,
                    prev: times[i - 1],
                    found: t,
                });
            }
        }
        Ok(Self { times })
    }

    pub fn times(&self) -> &[f64] {
        &self.times
    }

    pub fn len(&self) -> usize {
        self.times.len()
    }

    pub fn is_empty(&self) -> bool {
        self.times.is_empty()
    }
}

#[derive(Debug)]
pub enum SamplingError<ME> {
    Grid(GridError),
    GridBeforeSimTime {
        grid_start: f64,
        sim_time: f64,
    },
    Absorbed {
        steps: u64,
        time: f64,
    },
    MaxStepsExceeded {
        max_steps: u64,
        time: f64,
    },
    Sim {
        steps: u64,
        time: f64,
        source: SimError<ME>,
    },
}

impl<ME: Error> fmt::Display for SamplingError<ME> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Grid(err) => write!(f, "{err}"),
            Self::GridBeforeSimTime {
                grid_start,
                sim_time,
            } => write!(
                f,
                "grid starts at t={grid_start} before simulation time t={sim_time}"
            ),
            Self::Absorbed { steps, time } => {
                write!(f, "trajectory absorbed after {steps} steps at t={time:e}")
            }
            Self::MaxStepsExceeded { max_steps, time } => write!(
                f,
                "exceeded max_steps={max_steps} before grid end (reached t={time:e})"
            ),
            Self::Sim {
                steps,
                time,
                source,
            } => write!(f, "step error after {steps} steps at t={time:e}: {source}"),
        }
    }
}

impl<ME: Error + 'static> Error for SamplingError<ME> {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Grid(err) => Some(err),
            Self::Sim { source, .. } => Some(source),
            Self::GridBeforeSimTime { .. }
            | Self::Absorbed { .. }
            | Self::MaxStepsExceeded { .. } => None,
        }
    }
}

impl<ME> From<GridError> for SamplingError<ME> {
    fn from(err: GridError) -> Self {
        Self::Grid(err)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct SampledTrajectory<T> {
    pub times: Vec<f64>,
    pub values: Vec<T>,
    pub fired_steps: u64,
}

pub type ModelStepResult<M> = Result<Step<<M as Model>::Event>, SimError<<M as Model>::Error>>;

/// Stepping interface shared by [`Engine`], [`Simulation`], and [`IncrementalSimulation`].
pub trait TrajectorySource {
    type Model: Model;

    fn model(&self) -> &Self::Model;
    fn state(&self) -> &<Self::Model as Model>::State;
    fn clock(&self) -> Clock;

    /// Validates active rates without advancing RNG, clock, or step count; returns `Ok(true)` if absorbed.
    fn is_absorbed(&mut self) -> Result<bool, SimError<<Self::Model as Model>::Error>>;

    /// Invokes `before_apply(&model, &state_before, next_clock)` after clock validation and before
    /// transition application; external side effects in `before_apply` are not rolled back on `Err`.
    fn step_observing<F>(&mut self, before_apply: F) -> ModelStepResult<Self::Model>
    where
        F: FnOnce(&Self::Model, &<Self::Model as Model>::State, Clock);
}

impl<M: Model, C: CatalogBackend<M>> TrajectorySource for Engine<M, C> {
    type Model = M;

    fn model(&self) -> &M {
        self.model()
    }

    fn state(&self) -> &M::State {
        self.state()
    }

    fn clock(&self) -> Clock {
        self.clock()
    }

    fn is_absorbed(&mut self) -> Result<bool, SimError<M::Error>> {
        Engine::is_absorbed(self)
    }

    fn step_observing<F>(&mut self, before_apply: F) -> ModelStepResult<M>
    where
        F: FnOnce(&M, &M::State, Clock),
    {
        Engine::step_observing(self, before_apply)
    }
}

impl<M: Model> TrajectorySource for Simulation<M> {
    type Model = M;

    fn model(&self) -> &M {
        self.model()
    }

    fn state(&self) -> &M::State {
        self.state()
    }

    fn clock(&self) -> Clock {
        self.clock()
    }

    fn is_absorbed(&mut self) -> Result<bool, SimError<M::Error>> {
        Simulation::is_absorbed(self)
    }

    fn step_observing<F>(&mut self, before_apply: F) -> ModelStepResult<M>
    where
        F: FnOnce(&M, &M::State, Clock),
    {
        Simulation::step_observing(self, before_apply)
    }
}

impl<M: IncrementalModel> TrajectorySource for IncrementalSimulation<M> {
    type Model = M;

    fn model(&self) -> &M {
        self.model()
    }

    fn state(&self) -> &M::State {
        self.state()
    }

    fn clock(&self) -> Clock {
        self.clock()
    }

    fn is_absorbed(&mut self) -> Result<bool, SimError<M::Error>> {
        IncrementalSimulation::is_absorbed(self)
    }

    fn step_observing<F>(&mut self, before_apply: F) -> ModelStepResult<M>
    where
        F: FnOnce(&M, &M::State, Clock),
    {
        IncrementalSimulation::step_observing(self, before_apply)
    }
}

/// Samples a right-continuous CTMC trajectory on `grid` using exact compensated [`Clock`] ordering.
///
/// `observe` runs lazily on steps crossing a grid target (before `apply`, so caller side effects in
/// `observe` are not rolled back if `apply` errors); mid-trajectory errors leave `sim` at the last completed step.
pub fn sample_trajectory<S, T, F>(
    sim: &mut S,
    grid: &SampleGrid,
    max_steps: Option<u64>,
    absorption: AbsorptionPolicy,
    observe: F,
) -> Result<SampledTrajectory<T>, SamplingError<<S::Model as Model>::Error>>
where
    S: TrajectorySource,
    T: Clone,
    F: FnMut(&S::Model, &<S::Model as Model>::State) -> T,
{
    sample_trajectory_with(sim, grid, max_steps, absorption, FnObserver(observe))
}

/// Maps a model state to the recorded value `T`; the named-type alternative to the closure taken by [`sample_trajectory`].
pub trait Observe<M: Model, T> {
    fn observe(&mut self, model: &M, state: &M::State) -> T;
}

/// Adapter that lets a closure act as an [`Observe`].
struct FnObserver<F>(F);

impl<M: Model, T, F: FnMut(&M, &M::State) -> T> Observe<M, T> for FnObserver<F> {
    fn observe(&mut self, model: &M, state: &M::State) -> T {
        (self.0)(model, state)
    }
}

/// [`sample_trajectory`] with a named [`Observe`] implementation in place of a closure.
///
/// Callers that are generic over the model and would otherwise pass a closure whose argument type is
/// a projection (`<M as Model>::State`) use this to avoid a rustc 1.100.0 codegen ICE
/// (`SignatureMismatch` on the closure's `FnMut` impl) in release builds.
pub fn sample_trajectory_with<S, T, O>(
    sim: &mut S,
    grid: &SampleGrid,
    max_steps: Option<u64>,
    absorption: AbsorptionPolicy,
    mut observer: O,
) -> Result<SampledTrajectory<T>, SamplingError<<S::Model as Model>::Error>>
where
    S: TrajectorySource,
    T: Clone,
    O: Observe<S::Model, T>,
{
    let times = grid.times();
    if sim.clock().cmp_f64(times[0]).is_gt() {
        return Err(SamplingError::GridBeforeSimTime {
            grid_start: times[0],
            sim_time: sim.clock().time(),
        });
    }

    let mut values = Vec::with_capacity(times.len());
    let mut fired_steps = 0_u64;

    while values.len() < times.len() {
        if sim.clock().cmp_f64(times[values.len()]).is_eq() {
            let current = observer.observe(sim.model(), sim.state());
            while values.len() < times.len() && sim.clock().cmp_f64(times[values.len()]).is_eq() {
                values.push(current.clone());
            }
            if values.len() >= times.len() {
                break;
            }
        }

        if let Some(limit) = max_steps
            && fired_steps >= limit
        {
            if sim.is_absorbed().map_err(|source| SamplingError::Sim {
                steps: fired_steps,
                time: sim.clock().time(),
                source,
            })? {
                match absorption {
                    AbsorptionPolicy::Error => {
                        return Err(SamplingError::Absorbed {
                            steps: fired_steps,
                            time: sim.clock().time(),
                        });
                    }
                    AbsorptionPolicy::HoldFinal => {
                        let held = observer.observe(sim.model(), sim.state());
                        values.resize(times.len(), held);
                        break;
                    }
                }
            }
            return Err(SamplingError::MaxStepsExceeded {
                max_steps: limit,
                time: sim.clock().time(),
            });
        }

        let next_target = times[values.len()];
        let mut crossed: Option<(T, Clock)> = None;
        let step = sim
            .step_observing(|model, state_before, next_clock| {
                if next_clock.cmp_f64(next_target).is_gt() {
                    crossed = Some((observer.observe(model, state_before), next_clock));
                }
            })
            .map_err(|source| SamplingError::Sim {
                steps: fired_steps,
                time: sim.clock().time(),
                source,
            })?;

        match step {
            Step::Fired { .. } => {
                fired_steps += 1;
                if let Some((held, clock_after)) = crossed {
                    while values.len() < times.len()
                        && clock_after.cmp_f64(times[values.len()]).is_gt()
                    {
                        values.push(held.clone());
                    }
                }
            }
            Step::Absorbed => match absorption {
                AbsorptionPolicy::Error => {
                    return Err(SamplingError::Absorbed {
                        steps: fired_steps,
                        time: sim.clock().time(),
                    });
                }
                AbsorptionPolicy::HoldFinal => {
                    let held = observer.observe(sim.model(), sim.state());
                    values.resize(times.len(), held);
                }
            },
        }
    }

    Ok(SampledTrajectory {
        times: times.to_vec(),
        values,
        fired_steps,
    })
}
