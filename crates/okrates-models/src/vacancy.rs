//! A single vacancy exchanging with its eight nearest neighbours in BCC iron.

use std::convert::Infallible;
use std::error::Error;
use std::fmt;

use okrates_core::{Checkpointable, Model};

use crate::bcc::{BccLattice, DISPLACEMENTS};

/// Boltzmann's constant in eV/K, from exact SI-defined constants.
///
/// This is the exact SI value of k_B in J/K divided by the elementary charge in C.
pub const BOLTZMANN_EV_PER_K: f64 = 1.380649e-23 / 1.602176634e-19;

/// Computes an Arrhenius jump rate from attempt frequency, migration energy, and temperature.
///
/// The inputs are in s⁻¹, eV, and K respectively, and the result is in s⁻¹.
pub fn jump_rate(
    attempt_frequency: f64,
    migration_energy_ev: f64,
    temperature_k: f64,
) -> Result<f64, RateParameterError> {
    if !attempt_frequency.is_finite() || attempt_frequency <= 0.0 {
        return Err(RateParameterError::AttemptFrequency(attempt_frequency));
    }
    if !migration_energy_ev.is_finite() || migration_energy_ev < 0.0 {
        return Err(RateParameterError::MigrationEnergy(migration_energy_ev));
    }
    if !temperature_k.is_finite() || temperature_k <= 0.0 {
        return Err(RateParameterError::Temperature(temperature_k));
    }

    let rate =
        attempt_frequency * (-migration_energy_ev / (BOLTZMANN_EV_PER_K * temperature_k)).exp();
    if !rate.is_finite() {
        return Err(RateParameterError::Overflow);
    }
    if rate.partial_cmp(&0.0) != Some(std::cmp::Ordering::Greater) {
        return Err(RateParameterError::Underflow);
    }
    Ok(rate)
}

/// An error computing an Arrhenius rate.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum RateParameterError {
    /// The attempt frequency was not finite and positive.
    AttemptFrequency(f64),
    /// The migration energy was not finite and non-negative.
    MigrationEnergy(f64),
    /// The temperature was not finite and positive.
    Temperature(f64),
    /// The rate underflowed to a non-positive value.
    Underflow,
    /// The rate was not finite.
    Overflow,
}

impl fmt::Display for RateParameterError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::AttemptFrequency(value) => {
                write!(
                    f,
                    "attempt frequency must be finite and positive, got {value}"
                )
            }
            Self::MigrationEnergy(value) => {
                write!(
                    f,
                    "migration energy must be finite and non-negative, got {value}"
                )
            }
            Self::Temperature(value) => {
                write!(f, "temperature must be finite and positive, got {value}")
            }
            Self::Underflow => write!(f, "jump rate underflowed to a non-positive value"),
            Self::Overflow => write!(f, "jump rate is not finite"),
        }
    }
}

impl Error for RateParameterError {}

/// The species occupying a lattice site.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Species {
    /// An iron atom.
    Fe,
    /// The single vacancy.
    Vacancy,
}

/// The occupancy and unwrapped position of a single vacancy.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VacancyState {
    occupancy: Vec<Species>,
    vacancy: usize,
    origin: usize,
    displacement: [i64; 3],
}

impl VacancyState {
    /// Returns the species at every lattice site.
    pub fn occupancy(&self) -> &[Species] {
        &self.occupancy
    }

    /// Returns the vacancy's wrapped lattice site.
    pub fn vacancy(&self) -> usize {
        self.vacancy
    }

    /// Returns the vacancy's origin lattice site.
    pub fn origin(&self) -> usize {
        self.origin
    }

    /// Returns the vacancy's unwrapped displacement in half-lattice units.
    pub fn displacement(&self) -> [i64; 3] {
        self.displacement
    }

    /// Returns squared displacement in the physical length units of `lattice_parameter`.
    pub fn squared_displacement(&self, lattice_parameter: f64) -> f64 {
        let integer_squared = self
            .displacement
            .iter()
            .map(|&component| component * component)
            .sum::<i64>();
        integer_squared as f64 * lattice_parameter * lattice_parameter / 4.0
    }
}

/// A vacancy jump along one of the eight BCC nearest-neighbour directions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VacancyJump {
    /// The index into [`DISPLACEMENTS`](crate::bcc::DISPLACEMENTS).
    pub direction: usize,
}

/// A single-vacancy exchange model on a periodic BCC lattice.
///
/// This is the vacancy diffusivity model, not atom tracer diffusion. For this model,
/// `D_v = r a²`.
#[derive(Clone, Debug)]
pub struct VacancyDiffusion {
    lattice: BccLattice,
    jump_rate: f64,
}

impl VacancyDiffusion {
    /// Creates a model with the supplied lattice and per-direction jump rate.
    pub fn new(lattice: BccLattice, jump_rate: f64) -> Self {
        Self { lattice, jump_rate }
    }

    /// Returns the model's lattice.
    pub fn lattice(&self) -> &BccLattice {
        &self.lattice
    }

    /// Returns the per-direction jump rate.
    pub fn jump_rate(&self) -> f64 {
        self.jump_rate
    }

    /// Creates an all-iron state with a vacancy at `vacancy_site`.
    pub fn initial_state(&self, vacancy_site: usize) -> Result<VacancyState, InvariantViolation> {
        let sites = self.lattice.site_count();
        if vacancy_site >= sites {
            return Err(InvariantViolation::SiteOutOfRange {
                site: vacancy_site,
                sites,
            });
        }

        let mut occupancy = vec![Species::Fe; sites];
        occupancy[vacancy_site] = Species::Vacancy;
        Ok(VacancyState {
            occupancy,
            vacancy: vacancy_site,
            origin: vacancy_site,
            displacement: [0; 3],
        })
    }

    /// Checks occupancy, vacancy location, and consistency with unwrapped displacement.
    pub fn check_invariants(&self, state: &VacancyState) -> Result<(), InvariantViolation> {
        let expected = self.lattice.site_count();
        if state.occupancy.len() != expected {
            return Err(InvariantViolation::SiteCount {
                expected,
                found: state.occupancy.len(),
            });
        }

        let vacancy_count = state
            .occupancy
            .iter()
            .filter(|&&species| species == Species::Vacancy)
            .count();
        if vacancy_count != 1 {
            return Err(InvariantViolation::VacancyCount(vacancy_count));
        }

        if state.vacancy >= state.occupancy.len() {
            return Err(InvariantViolation::SiteOutOfRange {
                site: state.vacancy,
                sites: state.occupancy.len(),
            });
        }
        if state.origin >= state.occupancy.len() {
            return Err(InvariantViolation::SiteOutOfRange {
                site: state.origin,
                sites: state.occupancy.len(),
            });
        }
        if state.occupancy[state.vacancy] != Species::Vacancy {
            return Err(InvariantViolation::VacancyMismatch {
                site: state.vacancy,
            });
        }

        let origin = self.lattice.position(state.origin);
        let unwrapped = std::array::from_fn(|axis| origin[axis] + state.displacement[axis]);
        if self.lattice.site_at(unwrapped) != Some(state.vacancy) {
            return Err(InvariantViolation::UnwrappedMismatch);
        }
        Ok(())
    }
}

impl Model for VacancyDiffusion {
    type State = VacancyState;
    type Event = VacancyJump;
    type Error = Infallible;

    fn enumerate_events(&self, _state: &Self::State, emit: &mut dyn FnMut(Self::Event, f64)) {
        for direction in 0..8 {
            emit(VacancyJump { direction }, self.jump_rate);
        }
    }

    fn apply(&self, state: &mut Self::State, event: &Self::Event) -> Result<(), Self::Error> {
        let target = self.lattice.neighbor(state.vacancy, event.direction);
        state.occupancy.swap(state.vacancy, target);
        state.vacancy = target;
        for axis in 0..3 {
            state.displacement[axis] += DISPLACEMENTS[event.direction][axis];
        }
        Ok(())
    }
}

impl Checkpointable for VacancyDiffusion {
    const MODEL_ID: &'static str = "okrates.bcc_single_vacancy";

    fn encode_params(&self, out: &mut Vec<u8>) {
        for cells in self.lattice.cells() {
            out.extend_from_slice(&(cells as u64).to_le_bytes());
        }
        out.extend_from_slice(&self.jump_rate.to_bits().to_le_bytes());
    }

    fn encode_state(&self, state: &Self::State, out: &mut Vec<u8>) {
        out.extend_from_slice(&(state.vacancy as u64).to_le_bytes());
        out.extend_from_slice(&(state.origin as u64).to_le_bytes());
        for component in state.displacement {
            out.extend_from_slice(&component.to_le_bytes());
        }
    }

    fn decode_state(&self, bytes: &[u8]) -> Result<Self::State, Box<dyn Error + Send + Sync>> {
        if bytes.len() != 40 {
            return Err(format!("state must be 40 bytes, got {}", bytes.len()).into());
        }
        let word = |i: usize| -> [u8; 8] { bytes[8 * i..8 * i + 8].try_into().unwrap() };
        let sites = self.lattice.site_count();
        let site = |i: usize| -> Result<usize, InvariantViolation> {
            let raw = u64::from_le_bytes(word(i));
            match usize::try_from(raw) {
                Ok(site) if site < sites => Ok(site),
                _ => Err(InvariantViolation::SiteOutOfRange {
                    site: usize::try_from(raw).unwrap_or(usize::MAX),
                    sites,
                }),
            }
        };
        let vacancy = site(0)?;
        let origin = site(1)?;
        let mut state = self.initial_state(vacancy)?;
        state.origin = origin;
        state.displacement = std::array::from_fn(|axis| i64::from_le_bytes(word(2 + axis)));
        self.check_invariants(&state)?;
        Ok(state)
    }
}

/// A violation of a single-vacancy state invariant.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InvariantViolation {
    /// The occupancy length differs from the lattice site count.
    SiteCount {
        /// The lattice site count.
        expected: usize,
        /// The occupancy vector length.
        found: usize,
    },
    /// The occupancy contains a number of vacancies other than one.
    VacancyCount(usize),
    /// A referenced lattice site is out of range.
    SiteOutOfRange {
        /// The invalid site index.
        site: usize,
        /// The number of sites in the lattice.
        sites: usize,
    },
    /// The vacancy field does not point to the occupied vacancy.
    VacancyMismatch {
        /// The site stored in the vacancy field.
        site: usize,
    },
    /// The unwrapped displacement does not lead to the wrapped vacancy site.
    UnwrappedMismatch,
}

impl fmt::Display for InvariantViolation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SiteCount { expected, found } => {
                write!(f, "occupancy has {found} sites, expected {expected}")
            }
            Self::VacancyCount(count) => write!(f, "occupancy has {count} vacancies, expected 1"),
            Self::SiteOutOfRange { site, sites } => {
                write!(f, "site {site} is out of range for {sites} sites")
            }
            Self::VacancyMismatch { site } => {
                write!(f, "vacancy field points to non-vacancy site {site}")
            }
            Self::UnwrappedMismatch => {
                write!(f, "unwrapped displacement does not match vacancy site")
            }
        }
    }
}

impl Error for InvariantViolation {}

#[cfg(test)]
mod tests {
    use super::{
        BOLTZMANN_EV_PER_K, InvariantViolation, RateParameterError, Species, VacancyDiffusion,
        VacancyJump, jump_rate,
    };
    use crate::bcc::BccLattice;
    use okrates_core::Model;

    #[test]
    fn jump_rate_value() {
        let rate = jump_rate(1e13, 0.65, 800.0).unwrap();
        assert!((rate / 803_859_640.301_204_7 - 1.0).abs() < 1e-12);
        assert!((BOLTZMANN_EV_PER_K / 8.617_333_262_145_179e-5 - 1.0).abs() < 1e-15);
    }

    #[test]
    fn jump_rate_errors() {
        for attempt_frequency in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert!(matches!(
                jump_rate(attempt_frequency, 0.1, 300.0),
                Err(RateParameterError::AttemptFrequency(_))
            ));
        }
        for migration_energy in [-0.1, f64::NAN, f64::INFINITY] {
            assert!(matches!(
                jump_rate(1.0, migration_energy, 300.0),
                Err(RateParameterError::MigrationEnergy(_))
            ));
        }
        for temperature in [0.0, -5.0, f64::NAN, f64::INFINITY] {
            assert!(matches!(
                jump_rate(1.0, 0.1, temperature),
                Err(RateParameterError::Temperature(_))
            ));
        }
        assert_eq!(
            jump_rate(1e13, 0.65, 1.0),
            Err(RateParameterError::Underflow)
        );
        assert_eq!(jump_rate(f64::MAX, 0.0, 300.0), Ok(f64::MAX));
    }

    #[test]
    fn enumerates_eight_equal_rate_jumps_to_distinct_neighbours() {
        let lattice = BccLattice::new([3, 3, 3]).unwrap();
        let model = VacancyDiffusion::new(lattice, 2.5);
        let state = model.initial_state(7).unwrap();
        let mut events = Vec::new();
        model.enumerate_events(&state, &mut |event, rate| events.push((event, rate)));

        assert_eq!(events.len(), 8);
        assert_eq!(
            events
                .iter()
                .map(|(event, _)| event.direction)
                .collect::<Vec<_>>(),
            (0..8).collect::<Vec<_>>()
        );
        assert!(events.iter().all(|(_, rate)| *rate == 2.5));
        let targets = events
            .iter()
            .map(|(event, _)| model.lattice().neighbor(state.vacancy(), event.direction))
            .collect::<Vec<_>>();
        for index in 0..targets.len() {
            assert!(!targets[..index].contains(&targets[index]));
        }
    }

    #[test]
    fn invariants_detect_corruption() {
        let model = VacancyDiffusion::new(BccLattice::new([2, 2, 2]).unwrap(), 1.0);
        let valid = model.initial_state(3).unwrap();

        let mut extra_vacancy = valid.clone();
        extra_vacancy.occupancy[5] = Species::Vacancy;
        assert_eq!(
            model.check_invariants(&extra_vacancy),
            Err(InvariantViolation::VacancyCount(2))
        );

        let mut truncated = valid.clone();
        truncated.occupancy.pop();
        assert_eq!(
            model.check_invariants(&truncated),
            Err(InvariantViolation::SiteCount {
                expected: 16,
                found: 15
            })
        );

        let mut mismatched = valid.clone();
        mismatched.vacancy = 4;
        assert_eq!(
            model.check_invariants(&mismatched),
            Err(InvariantViolation::VacancyMismatch { site: 4 })
        );

        let mut unwrapped_mismatch = valid.clone();
        unwrapped_mismatch.displacement = [1, 1, 1];
        assert_eq!(
            model.check_invariants(&unwrapped_mismatch),
            Err(InvariantViolation::UnwrappedMismatch)
        );

        assert_eq!(
            model.initial_state(16),
            Err(InvariantViolation::SiteOutOfRange {
                site: 16,
                sites: 16
            })
        );
    }

    #[test]
    fn invariant_bounds_are_checked() {
        let model = VacancyDiffusion::new(BccLattice::new([2, 2, 2]).unwrap(), 1.0);
        let valid = model.initial_state(3).unwrap();
        let mut bad_vacancy = valid.clone();
        bad_vacancy.vacancy = 16;
        assert_eq!(
            model.check_invariants(&bad_vacancy),
            Err(InvariantViolation::SiteOutOfRange {
                site: 16,
                sites: 16
            })
        );
        let mut bad_origin = valid;
        bad_origin.origin = 16;
        assert_eq!(
            model.check_invariants(&bad_origin),
            Err(InvariantViolation::SiteOutOfRange {
                site: 16,
                sites: 16
            })
        );
    }

    #[test]
    fn apply_swaps_the_vacancy_and_tracks_unwrapped_displacement() {
        let lattice = BccLattice::new([2, 2, 2]).unwrap();
        let model = VacancyDiffusion::new(lattice, 1.0);
        let mut state = model.initial_state(0).unwrap();
        model
            .apply(&mut state, &VacancyJump { direction: 0 })
            .unwrap();
        assert_eq!(state.occupancy()[0], Species::Fe);
        assert_eq!(state.occupancy()[state.vacancy()], Species::Vacancy);
        assert_eq!(state.displacement(), [-1, -1, -1]);
        assert_eq!(model.check_invariants(&state), Ok(()));
    }
}
