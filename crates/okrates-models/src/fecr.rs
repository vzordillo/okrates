//! A single vacancy in a BCC Fe–Cr alloy with pluggable energetics and KRA barriers.
//!
//! The vacancy exchanges with one of its eight nearest neighbours. The activation
//! energy of a jump comes from [`Energetics::barrier`]. For [`PairEnergetics`] it is
//! the kinetically resolved activation energy (KRA) of the migrating atom plus half
//! the energy change of the exchange, so that forward and reverse rates satisfy
//! detailed balance exactly.
//!
//! The energetics are supplied through the [`Energetics`] trait. [`PairEnergetics`]
//! is a nearest-neighbour pair model. The values returned by
//! [`PairEnergetics::illustrative`] are ILLUSTRATIVE: round numbers with no source,
//! not DFT or literature values. They exist to exercise the model, not to describe
//! any real alloy.

use std::convert::Infallible;
use std::error::Error;
use std::fmt;

use okrates_core::{KmcRng, Model};

use crate::bcc::{BccLattice, DISPLACEMENTS};
use crate::vacancy::BOLTZMANN_EV_PER_K;

/// The species occupying a lattice site.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Species {
    /// An iron atom.
    Fe,
    /// A chromium atom.
    Cr,
    /// The single vacancy.
    Vacancy,
}

impl Species {
    /// Returns the species' index: Fe 0, Cr 1, Vacancy 2.
    pub fn index(self) -> usize {
        match self {
            Self::Fe => 0,
            Self::Cr => 1,
            Self::Vacancy => 2,
        }
    }
}

/// Energetics of a single-vacancy Fe–Cr configuration.
pub trait Energetics {
    /// Total configuration energy in eV.
    fn total_energy(&self, lattice: &BccLattice, occupancy: &[Species]) -> f64;
    /// E_after − E_before in eV for exchanging the vacancy at `vacancy_site` with the atom at its neighbour `atom_site`, computed locally.
    fn exchange_delta(
        &self,
        lattice: &BccLattice,
        occupancy: &[Species],
        vacancy_site: usize,
        atom_site: usize,
    ) -> f64;
    /// E_after − E_before in eV for swapping the atoms on sites `i` and `j`, neither of which is the vacancy.
    ///
    /// The default recomputes [`total_energy`](Self::total_energy) before and after the swap.
    fn swap_delta(&self, lattice: &BccLattice, occupancy: &[Species], i: usize, j: usize) -> f64 {
        let before = self.total_energy(lattice, occupancy);
        let mut after = occupancy.to_vec();
        after.swap(i, j);
        self.total_energy(lattice, &after) - before
    }
    /// Activation energy in eV of the atom on `atom_site` jumping into the vacancy at `vacancy_site`.
    fn barrier(
        &self,
        lattice: &BccLattice,
        occupancy: &[Species],
        vacancy_site: usize,
        atom_site: usize,
    ) -> f64;
    /// A per-species attempt frequency (prefactor) in s⁻¹; `None` means the model's attempt frequency is used.
    fn attempt_frequency(&self, atom: Species) -> Option<f64> {
        let _ = atom;
        None
    }
    /// Checks the energetics against the model's lattice and temperature; called by [`FeCrModel::new`].
    fn validate(&self, lattice: &BccLattice, temperature_k: f64) -> Result<(), FeCrError> {
        let _ = (lattice, temperature_k);
        Ok(())
    }
}

/// A nearest-neighbour pair model with per-species KRA values.
#[derive(Clone, Debug, PartialEq)]
pub struct PairEnergetics {
    pair: [[f64; 3]; 3],
    kra_fe: f64,
    kra_cr: f64,
}

impl PairEnergetics {
    /// Kinetically resolved activation energy of `atom` (Fe or Cr) in eV.
    pub fn kra(&self, atom: Species) -> f64 {
        match atom {
            Species::Fe => self.kra_fe,
            _ => self.kra_cr,
        }
    }

    /// A bound B with |exchange_delta| ≤ B for every single-vacancy configuration in which `atom` jumps.
    pub fn delta_bound(&self, atom: Species) -> f64 {
        let values = [self.d(atom, Species::Fe), self.d(atom, Species::Cr)];
        let max = values[0].max(values[1]);
        let min = values[0].min(values[1]);
        7.0 * (max - min)
    }

    /// Creates pair energetics from a symmetric pair-energy matrix (eV, indexed by
    /// [`Species::index`]) and the Fe and Cr KRA values (eV).
    pub fn new(pair: [[f64; 3]; 3], kra_fe: f64, kra_cr: f64) -> Result<Self, FeCrError> {
        if pair.iter().flatten().any(|v| !v.is_finite())
            || !kra_fe.is_finite()
            || !kra_cr.is_finite()
        {
            return Err(FeCrError::NonFinite);
        }
        for a in 0..3 {
            for b in 0..a {
                if pair[a][b] != pair[b][a] {
                    return Err(FeCrError::Asymmetric);
                }
            }
        }
        if kra_fe < 0.0 {
            return Err(FeCrError::NegativeKra(Species::Fe));
        }
        if kra_cr < 0.0 {
            return Err(FeCrError::NegativeKra(Species::Cr));
        }
        Ok(Self {
            pair,
            kra_fe,
            kra_cr,
        })
    }

    /// Returns ILLUSTRATIVE energetics: round numbers with no source, not DFT or
    /// literature values.
    ///
    /// Pair energies in eV: FeFe −0.60, CrCr −0.55, FeCr −0.56, FeV −0.20,
    /// CrV −0.25, VV 0.0. KRA in eV: Fe 0.60, Cr 0.55.
    pub fn illustrative() -> Self {
        let (fe, cr, v) = (0, 1, 2);
        let mut pair = [[0.0; 3]; 3];
        let mut set = |a: usize, b: usize, e: f64| {
            pair[a][b] = e;
            pair[b][a] = e;
        };
        set(fe, fe, -0.60);
        set(cr, cr, -0.55);
        set(fe, cr, -0.56);
        set(fe, v, -0.20);
        set(cr, v, -0.25);
        set(v, v, 0.0);
        Self::new(pair, 0.60, 0.55).expect("illustrative energetics are valid")
    }

    /// Returns the pair-energy matrix indexed by [`Species::index`].
    pub fn pair(&self) -> &[[f64; 3]; 3] {
        &self.pair
    }

    fn d(&self, atom: Species, other: Species) -> f64 {
        self.pair[atom.index()][other.index()] - self.pair[Species::Vacancy.index()][other.index()]
    }
}

impl Energetics for PairEnergetics {
    fn total_energy(&self, lattice: &BccLattice, occupancy: &[Species]) -> f64 {
        let mut sum = 0.0;
        for (site, &species) in occupancy.iter().enumerate() {
            for direction in 0..8 {
                let other = occupancy[lattice.neighbor(site, direction)];
                sum += self.pair[species.index()][other.index()];
            }
        }
        0.5 * sum
    }

    fn exchange_delta(
        &self,
        lattice: &BccLattice,
        occupancy: &[Species],
        vacancy_site: usize,
        atom_site: usize,
    ) -> f64 {
        let atom = occupancy[atom_site];
        let gained: f64 = lattice
            .neighbors(vacancy_site)
            .iter()
            .filter(|&&k| k != atom_site)
            .map(|&k| self.d(atom, occupancy[k]))
            .sum();
        let lost: f64 = lattice
            .neighbors(atom_site)
            .iter()
            .filter(|&&k| k != vacancy_site)
            .map(|&k| self.d(atom, occupancy[k]))
            .sum();
        gained - lost
    }

    fn swap_delta(&self, lattice: &BccLattice, occupancy: &[Species], i: usize, j: usize) -> f64 {
        let a = occupancy[i].index();
        let b = occupancy[j].index();
        let around_i: f64 = lattice
            .neighbors(i)
            .iter()
            .filter(|&&k| k != j)
            .map(|&k| {
                let o = occupancy[k].index();
                self.pair[b][o] - self.pair[a][o]
            })
            .sum();
        let around_j: f64 = lattice
            .neighbors(j)
            .iter()
            .filter(|&&k| k != i)
            .map(|&k| {
                let o = occupancy[k].index();
                self.pair[a][o] - self.pair[b][o]
            })
            .sum();
        around_i + around_j
    }

    fn barrier(
        &self,
        lattice: &BccLattice,
        occupancy: &[Species],
        vacancy_site: usize,
        atom_site: usize,
    ) -> f64 {
        self.kra(occupancy[atom_site])
            + self.exchange_delta(lattice, occupancy, vacancy_site, atom_site) / 2.0
    }

    fn validate(&self, _lattice: &BccLattice, _temperature_k: f64) -> Result<(), FeCrError> {
        for atom in [Species::Fe, Species::Cr] {
            let kra = self.kra(atom);
            let bound = self.delta_bound(atom);
            if kra < bound / 2.0 {
                return Err(FeCrError::NegativeBarrier { atom, kra, bound });
            }
        }
        Ok(())
    }
}

/// A vacancy jump along one of the eight BCC nearest-neighbour directions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FeCrJump {
    /// The index into [`DISPLACEMENTS`](crate::bcc::DISPLACEMENTS).
    pub direction: usize,
}

/// The occupancy of every site of an Fe–Cr lattice with one vacancy.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FeCrState {
    occupancy: Vec<Species>,
    vacancy: usize,
    /// Entity id on each site. Ids equal the initial site index.
    atom_at: Vec<usize>,
    /// Initial site of each entity, indexed by entity id.
    origin: Vec<usize>,
    /// Unwrapped displacement of each entity in half-lattice units, indexed by entity id.
    disp: Vec<[i64; 3]>,
}

impl FeCrState {
    /// Returns the species at every lattice site.
    pub fn occupancy(&self) -> &[Species] {
        &self.occupancy
    }

    /// Returns the vacancy's lattice site.
    pub fn vacancy(&self) -> usize {
        self.vacancy
    }

    /// Returns the counts of Fe, Cr and vacancy sites, in that order.
    pub fn counts(&self) -> [usize; 3] {
        let mut counts = [0; 3];
        for species in &self.occupancy {
            counts[species.index()] += 1;
        }
        counts
    }

    /// Returns the unwrapped displacement of the vacancy in half-lattice units (a/2).
    pub fn vacancy_displacement(&self) -> [i64; 3] {
        self.disp[self.atom_at[self.vacancy]]
    }

    /// Returns the unwrapped displacement (a/2 units) of the entity currently on `site`.
    pub fn displacement_of_site(&self, site: usize) -> [i64; 3] {
        self.disp[self.atom_at[site]]
    }

    /// Returns the species and unwrapped displacement (a/2 units) of every atom, in site order.
    pub fn atom_displacements(&self) -> impl Iterator<Item = (Species, [i64; 3])> + '_ {
        self.occupancy
            .iter()
            .enumerate()
            .filter(|&(_, &species)| species != Species::Vacancy)
            .map(|(site, &species)| (species, self.displacement_of_site(site)))
    }

    /// Returns the summed unwrapped displacement (a/2 units) of all atoms of `species`.
    pub fn collective_displacement(&self, species: Species) -> [i64; 3] {
        let mut sum = [0; 3];
        for (s, d) in self.atom_displacements() {
            if s == species {
                for axis in 0..3 {
                    sum[axis] += d[axis];
                }
            }
        }
        sum
    }

    /// Returns the sum of squared unwrapped displacements (a/2 units squared) of all atoms of `species`.
    pub fn tracer_squared_sum(&self, species: Species) -> i64 {
        self.atom_displacements()
            .filter(|&(s, _)| s == species)
            .map(|(_, d)| d.iter().map(|c| c * c).sum::<i64>())
            .sum()
    }
}

/// A single-vacancy Fe–Cr exchange model on a periodic BCC lattice.
#[derive(Clone, Debug)]
pub struct FeCrModel<E: Energetics> {
    lattice: BccLattice,
    energetics: E,
    attempt_frequency: f64,
    temperature_k: f64,
}

impl<E: Energetics> FeCrModel<E> {
    /// Creates a model; rejects non-positive attempt frequency or temperature and
    /// energetics that fail [`Energetics::validate`] (for [`PairEnergetics`], KRA
    /// values too small to keep every barrier non-negative).
    pub fn new(
        lattice: BccLattice,
        energetics: E,
        attempt_frequency: f64,
        temperature_k: f64,
    ) -> Result<Self, FeCrError> {
        if !attempt_frequency.is_finite() || attempt_frequency <= 0.0 {
            return Err(FeCrError::AttemptFrequency(attempt_frequency));
        }
        if !temperature_k.is_finite() || temperature_k <= 0.0 {
            return Err(FeCrError::Temperature(temperature_k));
        }
        energetics.validate(&lattice, temperature_k)?;
        Ok(Self {
            lattice,
            energetics,
            attempt_frequency,
            temperature_k,
        })
    }

    /// Returns the model's lattice.
    pub fn lattice(&self) -> &BccLattice {
        &self.lattice
    }

    /// Returns the model's energetics.
    pub fn energetics(&self) -> &E {
        &self.energetics
    }

    /// Returns k_B T in eV.
    pub fn kt(&self) -> f64 {
        BOLTZMANN_EV_PER_K * self.temperature_k
    }

    /// Creates a state from an occupancy vector with exactly one vacancy.
    pub fn state(&self, occupancy: Vec<Species>) -> Result<FeCrState, FeCrError> {
        let expected = self.lattice.site_count();
        if occupancy.len() != expected {
            return Err(FeCrError::SiteCount {
                expected,
                found: occupancy.len(),
            });
        }
        let vacancies: Vec<usize> = occupancy
            .iter()
            .enumerate()
            .filter(|&(_, &s)| s == Species::Vacancy)
            .map(|(i, _)| i)
            .collect();
        if vacancies.len() != 1 {
            return Err(FeCrError::VacancyCount(vacancies.len()));
        }
        Ok(FeCrState {
            vacancy: vacancies[0],
            atom_at: (0..expected).collect(),
            origin: (0..expected).collect(),
            disp: vec![[0; 3]; expected],
            occupancy,
        })
    }

    /// Creates a reproducible random state with one vacancy and `cr_count` Cr atoms.
    pub fn random_state(&self, cr_count: usize, seed: u64) -> Result<FeCrState, FeCrError> {
        let n = self.lattice.site_count();
        if cr_count + 1 > n {
            return Err(FeCrError::TooManyAtoms);
        }
        let mut rng = KmcRng::from_seed(seed);
        let mut order: Vec<usize> = (0..n).collect();
        for i in 0..cr_count + 1 {
            let j = i + ((rng.open01() * (n - i) as f64) as usize).min(n - i - 1);
            order.swap(i, j);
        }
        let mut occupancy = vec![Species::Fe; n];
        occupancy[order[0]] = Species::Vacancy;
        for &site in &order[1..cr_count + 1] {
            occupancy[site] = Species::Cr;
        }
        self.state(occupancy)
    }

    /// Returns the activation energy in eV of the jump along `direction`.
    pub fn barrier(&self, state: &FeCrState, direction: usize) -> f64 {
        let target = self.lattice.neighbor(state.vacancy, direction);
        self.energetics
            .barrier(&self.lattice, &state.occupancy, state.vacancy, target)
    }

    /// Returns the jump rate in s⁻¹ along `direction`.
    ///
    /// The model's attempt frequency is ignored for species where the energetics supply one.
    pub fn rate(&self, state: &FeCrState, direction: usize) -> f64 {
        let target = self.lattice.neighbor(state.vacancy, direction);
        let atom = state.occupancy[target];
        let nu = self
            .energetics
            .attempt_frequency(atom)
            .unwrap_or(self.attempt_frequency);
        nu * (-self.barrier(state, direction) / self.kt()).exp()
    }

    /// Checks occupancy length, vacancy count, the vacancy field, that displacements
    /// sum to zero and that each entity's displacement matches its current site.
    pub fn check_invariants(&self, state: &FeCrState) -> Result<(), FeCrError> {
        let expected = self.lattice.site_count();
        if state.occupancy.len() != expected {
            return Err(FeCrError::SiteCount {
                expected,
                found: state.occupancy.len(),
            });
        }
        let vacancies = state.counts()[Species::Vacancy.index()];
        if vacancies != 1 {
            return Err(FeCrError::VacancyCount(vacancies));
        }
        if state.vacancy >= expected || state.occupancy[state.vacancy] != Species::Vacancy {
            return Err(FeCrError::VacancyMismatch);
        }
        for len in [state.atom_at.len(), state.origin.len(), state.disp.len()] {
            if len != expected {
                return Err(FeCrError::SiteCount {
                    expected,
                    found: len,
                });
            }
        }
        let mut total = [0_i64; 3];
        for d in &state.disp {
            for axis in 0..3 {
                total[axis] += d[axis];
            }
        }
        if total != [0; 3] {
            return Err(FeCrError::DisplacementSum);
        }
        let period: [i64; 3] = std::array::from_fn(|axis| 2 * self.lattice.cells()[axis] as i64);
        for site in 0..expected {
            let entity = state.atom_at[site];
            if entity >= expected || state.origin[entity] >= expected {
                return Err(FeCrError::UnwrappedMismatch { site });
            }
            let here = self.lattice.position(site);
            let start = self.lattice.position(state.origin[entity]);
            let d = state.disp[entity];
            if (0..3).any(|axis| (here[axis] - start[axis] - d[axis]).rem_euclid(period[axis]) != 0)
            {
                return Err(FeCrError::UnwrappedMismatch { site });
            }
        }
        Ok(())
    }
}

impl<E: Energetics> Model for FeCrModel<E> {
    type State = FeCrState;
    type Event = FeCrJump;
    type Error = Infallible;

    fn enumerate_events(&self, state: &Self::State, emit: &mut dyn FnMut(Self::Event, f64)) {
        for direction in 0..8 {
            emit(FeCrJump { direction }, self.rate(state, direction));
        }
    }

    fn apply(&self, state: &mut Self::State, event: &Self::Event) -> Result<(), Self::Error> {
        let v = state.vacancy;
        let target = self.lattice.neighbor(v, event.direction);
        let shift = DISPLACEMENTS[event.direction];
        let vacancy_id = state.atom_at[v];
        let atom_id = state.atom_at[target];
        for axis in 0..3 {
            state.disp[vacancy_id][axis] += shift[axis];
            state.disp[atom_id][axis] -= shift[axis];
        }
        state.occupancy.swap(v, target);
        state.atom_at.swap(v, target);
        state.vacancy = target;
        Ok(())
    }
}

/// An error constructing or validating Fe–Cr energetics, models or states.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum FeCrError {
    /// A pair energy or KRA value was not finite.
    NonFinite,
    /// The pair-energy matrix was not symmetric.
    Asymmetric,
    /// A KRA value was negative.
    NegativeKra(Species),
    /// The attempt frequency was not finite and positive.
    AttemptFrequency(f64),
    /// The temperature was not finite and positive.
    Temperature(f64),
    /// The KRA is below half the exchange-energy bound, so a barrier could be negative.
    NegativeBarrier {
        /// The migrating species.
        atom: Species,
        /// Its KRA in eV.
        kra: f64,
        /// The exchange-energy bound in eV.
        bound: f64,
    },
    /// The occupancy length differs from the lattice site count.
    SiteCount {
        /// The lattice site count.
        expected: usize,
        /// The occupancy vector length.
        found: usize,
    },
    /// The occupancy contains a number of vacancies other than one.
    VacancyCount(usize),
    /// The vacancy field does not point to the vacancy.
    VacancyMismatch,
    /// More Cr atoms and vacancies were requested than there are sites.
    TooManyAtoms,
    /// The unwrapped displacements of all entities do not sum to zero.
    DisplacementSum,
    /// An entity's unwrapped displacement does not map its origin to its current site.
    UnwrappedMismatch {
        /// The site holding the inconsistent entity.
        site: usize,
    },
    /// The lattice has fewer than `min_cells` conventional cells along some axis.
    LatticeTooSmall {
        /// The smallest supported number of cells per axis.
        min_cells: usize,
    },
    /// The model temperature differs from the temperature the energetics were built for.
    TemperatureMismatch {
        /// The model temperature in K.
        model: f64,
        /// The energetics temperature in K.
        energetics: f64,
    },
}

impl fmt::Display for FeCrError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonFinite => write!(f, "pair energies and KRA values must be finite"),
            Self::Asymmetric => write!(f, "pair-energy matrix must be symmetric"),
            Self::NegativeKra(atom) => write!(f, "KRA of {atom:?} must be non-negative"),
            Self::AttemptFrequency(v) => {
                write!(f, "attempt frequency must be finite and positive, got {v}")
            }
            Self::Temperature(v) => {
                write!(f, "temperature must be finite and positive, got {v}")
            }
            Self::NegativeBarrier { atom, kra, bound } => write!(
                f,
                "KRA {kra} eV of {atom:?} is below half the exchange bound {bound} eV"
            ),
            Self::SiteCount { expected, found } => {
                write!(f, "occupancy has {found} sites, expected {expected}")
            }
            Self::VacancyCount(n) => write!(f, "occupancy has {n} vacancies, expected 1"),
            Self::VacancyMismatch => write!(f, "vacancy field does not point to the vacancy"),
            Self::TooManyAtoms => write!(f, "more Cr atoms and vacancies than sites"),
            Self::DisplacementSum => write!(f, "displacements do not sum to zero"),
            Self::UnwrappedMismatch { site } => write!(
                f,
                "entity on site {site} has an unwrapped displacement inconsistent with its site"
            ),
            Self::LatticeTooSmall { min_cells } => {
                write!(
                    f,
                    "lattice needs at least {min_cells} cells along every axis"
                )
            }
            Self::TemperatureMismatch { model, energetics } => write!(
                f,
                "model temperature {model} K differs from energetics temperature {energetics} K"
            ),
        }
    }
}

impl Error for FeCrError {}

#[cfg(test)]
mod tests {
    use super::{FeCrError, PairEnergetics, Species};

    #[test]
    fn illustrative_passes_new() {
        let e = PairEnergetics::illustrative();
        assert_eq!(PairEnergetics::new(*e.pair(), 0.60, 0.55), Ok(e));
    }

    #[test]
    fn rejects_asymmetric_and_non_finite_pairs() {
        let mut pair = *PairEnergetics::illustrative().pair();
        pair[0][1] += 0.01;
        assert_eq!(
            PairEnergetics::new(pair, 0.6, 0.55),
            Err(FeCrError::Asymmetric)
        );
        let mut pair = *PairEnergetics::illustrative().pair();
        pair[2][2] = f64::NAN;
        assert_eq!(
            PairEnergetics::new(pair, 0.6, 0.55),
            Err(FeCrError::NonFinite)
        );
        let pair = *PairEnergetics::illustrative().pair();
        assert_eq!(
            PairEnergetics::new(pair, -0.1, 0.55),
            Err(FeCrError::NegativeKra(Species::Fe))
        );
    }

    #[test]
    fn illustrative_delta_bounds() {
        let e = PairEnergetics::illustrative();
        assert!((e.delta_bound(Species::Fe) - 7.0 * 0.09).abs() < 1e-12);
        assert!((e.delta_bound(Species::Cr) - 7.0 * 0.06).abs() < 1e-12);
    }
}
