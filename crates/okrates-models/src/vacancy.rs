//! Single-vacancy nearest-neighbour diffusion on a periodic BCC lattice (`D_v = r a^2`).

use std::error::Error;
use std::fmt;

use okrates_core::{Checkpointable, IncrementalModel, IndependentSnapshot, Model, ReversibleModel};

use crate::bcc::{BccLattice, DISPLACEMENTS};

/// Exact SI Boltzmann constant `k_B / e` in `eV/K`.
pub const BOLTZMANN_EV_PER_K: f64 = 1.380649e-23 / 1.602176634e-19;

/// Arrhenius jump rate `nu * exp(-E_m / (k_B T))` in `s^-1` (`attempt_frequency` in `s^-1`, `migration_energy_ev` in `eV`, `temperature_k` in `K`).
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

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum RateParameterError {
    AttemptFrequency(f64),
    MigrationEnergy(f64),
    Temperature(f64),
    Underflow,
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Species {
    Fe,
    Vacancy,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VacancyState {
    occupancy: Vec<Species>,
    vacancy: usize,
    origin: usize,
    displacement: [i64; 3],
}

impl IndependentSnapshot for VacancyState {
    fn snapshot(&self) -> Self {
        self.clone()
    }
}

impl VacancyState {
    pub fn occupancy(&self) -> &[Species] {
        &self.occupancy
    }

    pub fn vacancy(&self) -> usize {
        self.vacancy
    }

    pub fn origin(&self) -> usize {
        self.origin
    }

    /// Unwrapped displacement in `a/2` units.
    pub fn displacement(&self) -> [i64; 3] {
        self.displacement
    }

    /// Squared displacement in the squared length units of `lattice_parameter`.
    pub fn squared_displacement(&self, lattice_parameter: f64) -> f64 {
        let sq_sum = self
            .displacement
            .iter()
            .map(|&component| (component as f64) * (component as f64))
            .sum::<f64>();
        sq_sum * lattice_parameter * lattice_parameter / 4.0
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VacancyJump {
    pub direction: usize,
}

/// Isotropic single-vacancy BCC random walk (`D_v = r a^2`, distinct from tracer diffusivity).
#[derive(Clone, Debug)]
pub struct VacancyDiffusion {
    lattice: BccLattice,
    jump_rate: f64,
}

impl VacancyDiffusion {
    pub fn new(lattice: BccLattice, jump_rate: f64) -> Self {
        Self { lattice, jump_rate }
    }

    pub fn lattice(&self) -> &BccLattice {
        &self.lattice
    }

    pub fn jump_rate(&self) -> f64 {
        self.jump_rate
    }

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
        let wrapped = std::array::from_fn(|axis| {
            let period = 2 * self.lattice.cells()[axis] as i128;
            (origin[axis] as i128 + state.displacement[axis] as i128).rem_euclid(period) as i64
        });
        if self.lattice.site_at(wrapped) != Some(state.vacancy) {
            return Err(InvariantViolation::UnwrappedMismatch);
        }
        Ok(())
    }
}

impl Model for VacancyDiffusion {
    type State = VacancyState;
    type Event = VacancyJump;
    type Error = InvariantViolation;

    fn enumerate_events(&self, _state: &Self::State, emit: &mut dyn FnMut(Self::Event, f64)) {
        for direction in 0..8 {
            emit(VacancyJump { direction }, self.jump_rate);
        }
    }

    fn apply(&self, state: &mut Self::State, event: &Self::Event) -> Result<(), Self::Error> {
        let Some(&shift) = DISPLACEMENTS.get(event.direction) else {
            return Err(InvariantViolation::InvalidDirection(event.direction));
        };
        let mut next_disp = [0_i64; 3];
        for axis in 0..3 {
            next_disp[axis] = state.displacement[axis]
                .checked_add(shift[axis])
                .ok_or(InvariantViolation::DisplacementOverflow)?;
        }
        let target = self.lattice.neighbor(state.vacancy, event.direction);
        state.occupancy.swap(state.vacancy, target);
        state.vacancy = target;
        state.displacement = next_disp;
        Ok(())
    }
}

impl ReversibleModel for VacancyDiffusion {
    type Undo = usize;

    fn apply_reversible(
        &self,
        state: &mut Self::State,
        event: &Self::Event,
    ) -> Result<Self::Undo, Self::Error> {
        self.apply(state, event)?;
        Ok(7 - event.direction)
    }

    fn undo(&self, state: &mut Self::State, opposite: Self::Undo) {
        let _ = self.apply(
            state,
            &VacancyJump {
                direction: opposite,
            },
        );
    }
}

impl IncrementalModel for VacancyDiffusion {
    fn slot_count(&self, _state: &Self::State) -> usize {
        8
    }

    fn slot_rate(&self, _state: &Self::State, slot: usize) -> f64 {
        if slot < 8 { self.jump_rate } else { f64::NAN }
    }

    fn slot_event(&self, _state: &Self::State, slot: usize) -> Self::Event {
        VacancyJump { direction: slot }
    }

    fn affected_slots(&self, _state: &Self::State, _event: &Self::Event, out: &mut Vec<usize>) {
        out.extend(0..8);
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InvariantViolation {
    SiteCount { expected: usize, found: usize },
    VacancyCount(usize),
    SiteOutOfRange { site: usize, sites: usize },
    VacancyMismatch { site: usize },
    UnwrappedMismatch,
    DisplacementOverflow,
    InvalidDirection(usize),
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
            Self::DisplacementOverflow => {
                write!(f, "vacancy unwrapped displacement overflowed i64")
            }
            Self::InvalidDirection(dir) => {
                write!(f, "vacancy jump direction {dir} is out of 0..8")
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
    use okrates_core::{Checkpointable, Model, ReversibleModel};

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
        for temperature in [0.0, -10.0, f64::NAN, f64::INFINITY] {
            assert!(matches!(
                jump_rate(1.0, 0.1, temperature),
                Err(RateParameterError::Temperature(_))
            ));
        }
        assert_eq!(
            jump_rate(1.0, 1000.0, 1.0),
            Err(RateParameterError::Underflow)
        );
        assert_eq!(jump_rate(f64::MAX, 0.0, 1.0), Ok(f64::MAX));
    }

    #[test]
    fn reversible_undo_restores_exact_state() {
        let lattice = BccLattice::new([3, 3, 3]).unwrap();
        let model = VacancyDiffusion::new(lattice, 1.0);
        let mut state = model.initial_state(0).unwrap();
        let before = state.clone();
        let undo = model
            .apply_reversible(&mut state, &VacancyJump { direction: 5 })
            .unwrap();
        assert_ne!(state, before);
        model.undo(&mut state, undo);
        assert_eq!(state, before);
    }

    #[test]
    fn check_invariants_rejects_corrupted_states() {
        let lattice = BccLattice::new([2, 2, 2]).unwrap();
        let model = VacancyDiffusion::new(lattice, 1.0);
        let base = model.initial_state(0).unwrap();

        let mut wrong_len = base.clone();
        wrong_len.occupancy.pop();
        assert!(matches!(
            model.check_invariants(&wrong_len),
            Err(InvariantViolation::SiteCount { .. })
        ));

        let mut two_vacancies = base.clone();
        two_vacancies.occupancy[1] = Species::Vacancy;
        assert_eq!(
            model.check_invariants(&two_vacancies),
            Err(InvariantViolation::VacancyCount(2))
        );

        let mut bad_disp = base;
        bad_disp.displacement = [1, 1, 1];
        assert_eq!(
            model.check_invariants(&bad_disp),
            Err(InvariantViolation::UnwrappedMismatch)
        );
    }

    #[test]
    fn enumerates_eight_equal_rates() {
        let model = VacancyDiffusion::new(BccLattice::new([2, 2, 2]).unwrap(), 2.5);
        let state = model.initial_state(0).unwrap();
        let mut out = Vec::new();
        model.enumerate_events(&state, &mut |ev, r| out.push((ev.direction, r)));
        assert_eq!(out.len(), 8);
        assert!(
            out.iter()
                .enumerate()
                .all(|(i, &(d, r))| d == i && r == 2.5)
        );
    }

    #[test]
    fn displacement_overflow_is_atomic_and_large_squared_displacement_stays_finite() {
        let model = VacancyDiffusion::new(BccLattice::new([2, 2, 2]).unwrap(), 1.0);
        let near_max = i64::MAX - 3;
        let mut raw = Vec::new();
        model.encode_state(&model.initial_state(0).unwrap(), &mut raw);
        raw[16..24].copy_from_slice(&near_max.to_le_bytes());
        let mut state = model.decode_state(&raw).unwrap();
        assert!(state.squared_displacement(1.0).is_finite());

        for _ in 0..3 {
            model
                .apply(&mut state, &VacancyJump { direction: 7 })
                .unwrap();
        }
        let snapshot = state.clone();
        assert_eq!(
            model.apply(&mut state, &VacancyJump { direction: 7 }),
            Err(InvariantViolation::DisplacementOverflow)
        );
        assert_eq!(state, snapshot);
        assert_eq!(
            model.apply(&mut state, &VacancyJump { direction: 8 }),
            Err(InvariantViolation::InvalidDirection(8))
        );
        assert_eq!(state, snapshot);
    }
}
