//! Many vacancies exchanging with their nearest neighbours in BCC iron.
//!
//! A vacancy cannot exchange with another vacancy, so the jump rate of a
//! direction that points at a vacancy is zero. The model implements both
//! [`Model`] and [`IncrementalModel`].

use std::error::Error;
use std::fmt;

use okrates_core::{Checkpointable, IncrementalModel, KmcRng, Model};

use crate::bcc::{BccLattice, DISPLACEMENTS};

/// The occupancy and unwrapped positions of all vacancies.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MultiVacancyState {
    /// Per site: `None` is iron, `Some(id)` is vacancy `id`.
    site_vacancy: Vec<Option<usize>>,
    /// Per vacancy id: current site.
    sites: Vec<usize>,
    /// Per vacancy id: initial site.
    origins: Vec<usize>,
    /// Per vacancy id: unwrapped displacement in half-lattice units.
    displacements: Vec<[i64; 3]>,
}

impl MultiVacancyState {
    /// Returns the number of vacancies.
    pub fn vacancy_count(&self) -> usize {
        self.sites.len()
    }

    /// Returns the current site of every vacancy, indexed by vacancy id.
    pub fn sites(&self) -> &[usize] {
        &self.sites
    }

    /// Returns, for every lattice site, the id of the vacancy on it, if any.
    pub fn site_vacancy(&self) -> &[Option<usize>] {
        &self.site_vacancy
    }

    /// Returns every vacancy's unwrapped displacement in half-lattice units.
    pub fn displacements(&self) -> &[[i64; 3]] {
        &self.displacements
    }

    /// Returns the squared displacement of vacancy `id` in the physical length
    /// units of `lattice_parameter`.
    ///
    /// # Panics
    ///
    /// Panics if `id` is not a vacancy id.
    pub fn squared_displacement(&self, id: usize, lattice_parameter: f64) -> f64 {
        let integer_squared = self.displacements[id]
            .iter()
            .map(|&component| component * component)
            .sum::<i64>();
        integer_squared as f64 * lattice_parameter * lattice_parameter / 4.0
    }

    /// Returns the number of unordered nearest-neighbour pairs of sites that
    /// both hold a vacancy.
    pub fn vv_neighbor_pairs(&self, lattice: &BccLattice) -> usize {
        let mut ordered = 0;
        for &site in &self.sites {
            for direction in 0..8 {
                if self.site_vacancy[lattice.neighbor(site, direction)].is_some() {
                    ordered += 1;
                }
            }
        }
        ordered / 2
    }
}

/// A jump of one vacancy along one of the eight nearest-neighbour directions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MultiVacancyJump {
    /// The vacancy id.
    pub vacancy: usize,
    /// The index into [`DISPLACEMENTS`](crate::bcc::DISPLACEMENTS).
    pub direction: usize,
}

/// A many-vacancy exchange model on a periodic BCC lattice.
#[derive(Clone, Debug)]
pub struct MultiVacancyDiffusion {
    lattice: BccLattice,
    jump_rate: f64,
}

impl MultiVacancyDiffusion {
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

    /// Creates a state with one vacancy per entry of `sites`; the vacancy id is
    /// the index in `sites`.
    pub fn initial_state(&self, sites: &[usize]) -> Result<MultiVacancyState, MultiVacancyError> {
        let site_count = self.lattice.site_count();
        if sites.is_empty() {
            return Err(MultiVacancyError::NoVacancies);
        }
        let mut site_vacancy = vec![None; site_count];
        for (id, &site) in sites.iter().enumerate() {
            if site >= site_count {
                return Err(MultiVacancyError::SiteOutOfRange {
                    site,
                    sites: site_count,
                });
            }
            if site_vacancy[site].is_some() {
                return Err(MultiVacancyError::DuplicateSite(site));
            }
            site_vacancy[site] = Some(id);
        }
        Ok(MultiVacancyState {
            site_vacancy,
            sites: sites.to_vec(),
            origins: sites.to_vec(),
            displacements: vec![[0; 3]; sites.len()],
        })
    }

    /// Draws `count` distinct sites with a partial Fisher-Yates shuffle driven
    /// by `seed`.
    pub fn random_sites(&self, count: usize, seed: u64) -> Result<Vec<usize>, MultiVacancyError> {
        let n = self.lattice.site_count();
        if count == 0 {
            return Err(MultiVacancyError::NoVacancies);
        }
        if count > n {
            return Err(MultiVacancyError::TooManyVacancies { count, sites: n });
        }
        let mut rng = KmcRng::from_seed(seed);
        let mut all: Vec<usize> = (0..n).collect();
        for i in 0..count {
            let j = i + ((rng.open01() * (n - i) as f64) as usize).min(n - i - 1);
            all.swap(i, j);
        }
        all.truncate(count);
        Ok(all)
    }

    /// Checks occupancy, vacancy locations, and consistency with the unwrapped
    /// displacements.
    pub fn check_invariants(&self, state: &MultiVacancyState) -> Result<(), MultiVacancyError> {
        let site_count = self.lattice.site_count();
        if state.site_vacancy.len() != site_count {
            return Err(MultiVacancyError::SiteCount {
                expected: site_count,
                found: state.site_vacancy.len(),
            });
        }
        let n = state.sites.len();
        if state.origins.len() != n || state.displacements.len() != n {
            return Err(MultiVacancyError::LengthMismatch);
        }
        for v in 0..n {
            for site in [state.sites[v], state.origins[v]] {
                if site >= site_count {
                    return Err(MultiVacancyError::SiteOutOfRange {
                        site,
                        sites: site_count,
                    });
                }
            }
            if state.site_vacancy[state.sites[v]] != Some(v) {
                return Err(MultiVacancyError::VacancyMismatch { vacancy: v });
            }
        }
        let found = state.site_vacancy.iter().filter(|s| s.is_some()).count();
        if found != n {
            return Err(MultiVacancyError::VacancyCount { expected: n, found });
        }
        for v in 0..n {
            let origin = self.lattice.position(state.origins[v]);
            let unwrapped = std::array::from_fn(|axis| origin[axis] + state.displacements[v][axis]);
            if self.lattice.site_at(unwrapped) != Some(state.sites[v]) {
                return Err(MultiVacancyError::UnwrappedMismatch { vacancy: v });
            }
        }
        Ok(())
    }

    fn rate_of(&self, state: &MultiVacancyState, vacancy: usize, direction: usize) -> f64 {
        let target = self.lattice.neighbor(state.sites[vacancy], direction);
        if state.site_vacancy[target].is_none() {
            self.jump_rate
        } else {
            0.0
        }
    }
}

impl Model for MultiVacancyDiffusion {
    type State = MultiVacancyState;
    type Event = MultiVacancyJump;
    type Error = MultiVacancyError;

    fn enumerate_events(&self, state: &Self::State, emit: &mut dyn FnMut(Self::Event, f64)) {
        for vacancy in 0..state.sites.len() {
            for direction in 0..8 {
                emit(
                    MultiVacancyJump { vacancy, direction },
                    self.rate_of(state, vacancy, direction),
                );
            }
        }
    }

    fn apply(&self, state: &mut Self::State, event: &Self::Event) -> Result<(), Self::Error> {
        let MultiVacancyJump { vacancy, direction } = *event;
        let old = state.sites[vacancy];
        let target = self.lattice.neighbor(old, direction);
        if state.site_vacancy[target].is_some() {
            return Err(MultiVacancyError::VacancyExchange { vacancy, direction });
        }
        state.site_vacancy[old] = None;
        state.site_vacancy[target] = Some(vacancy);
        state.sites[vacancy] = target;
        for axis in 0..3 {
            state.displacements[vacancy][axis] += DISPLACEMENTS[direction][axis];
        }
        Ok(())
    }
}

impl IncrementalModel for MultiVacancyDiffusion {
    fn slot_count(&self, state: &Self::State) -> usize {
        8 * state.sites.len()
    }

    fn slot_rate(&self, state: &Self::State, slot: usize) -> f64 {
        self.rate_of(state, slot / 8, slot % 8)
    }

    fn slot_event(&self, _state: &Self::State, slot: usize) -> Self::Event {
        MultiVacancyJump {
            vacancy: slot / 8,
            direction: slot % 8,
        }
    }

    fn affected_slots(&self, state: &Self::State, event: &Self::Event, out: &mut Vec<usize>) {
        out.extend((0..8).map(|d| event.vacancy * 8 + d));
        let new = state.sites[event.vacancy];
        let old = self.lattice.neighbor(new, 7 - event.direction);
        for site in [old, new] {
            for &neighbor in self.lattice.neighbors(site) {
                if let Some(w) = state.site_vacancy[neighbor]
                    && w != event.vacancy
                {
                    out.extend((0..8).map(|d| w * 8 + d));
                }
            }
        }
    }
}

impl Checkpointable for MultiVacancyDiffusion {
    const MODEL_ID: &'static str = "okrates.bcc_multi_vacancy";

    fn encode_params(&self, out: &mut Vec<u8>) {
        for cells in self.lattice.cells() {
            out.extend_from_slice(&(cells as u64).to_le_bytes());
        }
        out.extend_from_slice(&self.jump_rate.to_bits().to_le_bytes());
    }

    fn encode_state(&self, state: &Self::State, out: &mut Vec<u8>) {
        out.extend_from_slice(&(state.sites.len() as u64).to_le_bytes());
        for v in 0..state.sites.len() {
            out.extend_from_slice(&(state.sites[v] as u64).to_le_bytes());
            out.extend_from_slice(&(state.origins[v] as u64).to_le_bytes());
            for component in state.displacements[v] {
                out.extend_from_slice(&component.to_le_bytes());
            }
        }
    }

    fn decode_state(&self, bytes: &[u8]) -> Result<Self::State, Box<dyn Error + Send + Sync>> {
        if bytes.len() < 8 {
            return Err(format!("state must have at least 8 bytes, got {}", bytes.len()).into());
        }
        let word = |offset: usize| -> [u8; 8] { bytes[offset..offset + 8].try_into().unwrap() };
        let raw_count = u64::from_le_bytes(word(0));
        let expected = usize::try_from(raw_count)
            .ok()
            .and_then(|n| n.checked_mul(40))
            .and_then(|b| b.checked_add(8));
        let n = match expected {
            Some(len) if len == bytes.len() && raw_count >= 1 => raw_count as usize,
            _ => {
                return Err(format!(
                    "state of {} bytes does not match vacancy count {raw_count} (8 + 40 n bytes, n >= 1)",
                    bytes.len()
                )
                .into());
            }
        };
        let site_count = self.lattice.site_count();
        let site = |offset: usize| -> Result<usize, MultiVacancyError> {
            let raw = u64::from_le_bytes(word(offset));
            match usize::try_from(raw) {
                Ok(site) if site < site_count => Ok(site),
                _ => Err(MultiVacancyError::SiteOutOfRange {
                    site: usize::try_from(raw).unwrap_or(usize::MAX),
                    sites: site_count,
                }),
            }
        };
        let mut sites = Vec::with_capacity(n);
        let mut origins = Vec::with_capacity(n);
        let mut displacements = Vec::with_capacity(n);
        for v in 0..n {
            let base = 8 + 40 * v;
            sites.push(site(base)?);
            origins.push(site(base + 8)?);
            displacements.push(std::array::from_fn(|axis| {
                i64::from_le_bytes(word(base + 16 + 8 * axis))
            }));
        }
        let mut state = self.initial_state(&sites)?;
        state.origins = origins;
        state.displacements = displacements;
        self.check_invariants(&state)?;
        Ok(state)
    }
}

/// An error building, checking, or advancing a many-vacancy state.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MultiVacancyError {
    /// A referenced lattice site is out of range.
    SiteOutOfRange {
        /// The invalid site index.
        site: usize,
        /// The number of sites in the lattice.
        sites: usize,
    },
    /// Two vacancies were placed on the same site.
    DuplicateSite(usize),
    /// At least one vacancy is required.
    NoVacancies,
    /// More vacancies were requested than the lattice has sites.
    TooManyVacancies {
        /// The requested number of vacancies.
        count: usize,
        /// The number of sites in the lattice.
        sites: usize,
    },
    /// The occupancy length differs from the lattice site count.
    SiteCount {
        /// The lattice site count.
        expected: usize,
        /// The occupancy vector length.
        found: usize,
    },
    /// The per-vacancy vectors have different lengths.
    LengthMismatch,
    /// The occupancy of a vacancy's site does not name that vacancy.
    VacancyMismatch {
        /// The vacancy id.
        vacancy: usize,
    },
    /// The occupancy holds a number of vacancies other than the vacancy count.
    VacancyCount {
        /// The vacancy count.
        expected: usize,
        /// The number of occupied-by-vacancy sites.
        found: usize,
    },
    /// The unwrapped displacement does not lead to the vacancy's site.
    UnwrappedMismatch {
        /// The vacancy id.
        vacancy: usize,
    },
    /// A jump targeted a site that already holds a vacancy.
    VacancyExchange {
        /// The vacancy id.
        vacancy: usize,
        /// The jump direction.
        direction: usize,
    },
}

impl fmt::Display for MultiVacancyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SiteOutOfRange { site, sites } => {
                write!(f, "site {site} is out of range for {sites} sites")
            }
            Self::DuplicateSite(site) => write!(f, "site {site} holds more than one vacancy"),
            Self::NoVacancies => write!(f, "at least one vacancy is required"),
            Self::TooManyVacancies { count, sites } => {
                write!(f, "{count} vacancies do not fit on {sites} sites")
            }
            Self::SiteCount { expected, found } => {
                write!(f, "occupancy has {found} sites, expected {expected}")
            }
            Self::LengthMismatch => write!(f, "per-vacancy vectors have different lengths"),
            Self::VacancyMismatch { vacancy } => {
                write!(f, "occupancy of vacancy {vacancy}'s site does not name it")
            }
            Self::VacancyCount { expected, found } => {
                write!(f, "occupancy has {found} vacancies, expected {expected}")
            }
            Self::UnwrappedMismatch { vacancy } => write!(
                f,
                "unwrapped displacement of vacancy {vacancy} does not match its site"
            ),
            Self::VacancyExchange { vacancy, direction } => write!(
                f,
                "vacancy {vacancy} cannot jump in direction {direction} onto another vacancy"
            ),
        }
    }
}

impl Error for MultiVacancyError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn model() -> MultiVacancyDiffusion {
        MultiVacancyDiffusion::new(BccLattice::new([2, 2, 2]).unwrap(), 1.5)
    }

    fn pair_state(model: &MultiVacancyDiffusion) -> MultiVacancyState {
        let lattice = BccLattice::new([2, 2, 2]).unwrap();
        model.initial_state(&[0, lattice.neighbor(0, 3)]).unwrap()
    }

    #[test]
    fn opposite_direction_is_seven_minus_direction() {
        let lattice = BccLattice::new([2, 2, 2]).unwrap();
        for d in 0..8 {
            let negated = DISPLACEMENTS[d].map(|c| -c);
            assert_eq!(DISPLACEMENTS[7 - d], negated);
            for site in 0..lattice.site_count() {
                assert_eq!(lattice.neighbor(lattice.neighbor(site, d), 7 - d), site);
            }
        }
    }

    #[test]
    fn enumerate_zeroes_directions_onto_vacancies() {
        let model = model();
        let state = pair_state(&model);
        let mut events = Vec::new();
        model.enumerate_events(&state, &mut |e, r| events.push((e, r)));
        assert_eq!(events.len(), 16);
        for (slot, (event, rate)) in events.iter().enumerate() {
            assert_eq!(*event, model.slot_event(&state, slot));
            assert_eq!(*rate, model.slot_rate(&state, slot));
            let blocked = (event.vacancy == 0 && event.direction == 3)
                || (event.vacancy == 1 && event.direction == 4);
            assert_eq!(*rate, if blocked { 0.0 } else { 1.5 }, "slot {slot}");
        }
        assert_eq!(state.vv_neighbor_pairs(model.lattice()), 1);
    }

    #[test]
    fn apply_onto_vacancy_errors_and_keeps_state() {
        let model = model();
        let mut state = pair_state(&model);
        let before = state.clone();
        let event = MultiVacancyJump {
            vacancy: 0,
            direction: 3,
        };
        assert_eq!(
            model.apply(&mut state, &event),
            Err(MultiVacancyError::VacancyExchange {
                vacancy: 0,
                direction: 3
            })
        );
        assert_eq!(state, before);
    }

    #[test]
    fn invariants_detect_corruption() {
        let model = model();
        let valid = pair_state(&model);
        assert_eq!(model.check_invariants(&valid), Ok(()));

        let mut s = valid.clone();
        s.site_vacancy.pop();
        assert_eq!(
            model.check_invariants(&s),
            Err(MultiVacancyError::SiteCount {
                expected: 16,
                found: 15
            })
        );

        let mut s = valid.clone();
        s.origins.pop();
        assert_eq!(
            model.check_invariants(&s),
            Err(MultiVacancyError::LengthMismatch)
        );

        let mut s = valid.clone();
        s.sites[1] = 16;
        assert_eq!(
            model.check_invariants(&s),
            Err(MultiVacancyError::SiteOutOfRange {
                site: 16,
                sites: 16
            })
        );

        let mut s = valid.clone();
        s.origins[0] = 16;
        assert_eq!(
            model.check_invariants(&s),
            Err(MultiVacancyError::SiteOutOfRange {
                site: 16,
                sites: 16
            })
        );

        let mut s = valid.clone();
        s.sites.swap(0, 1);
        assert_eq!(
            model.check_invariants(&s),
            Err(MultiVacancyError::VacancyMismatch { vacancy: 0 })
        );

        let mut s = valid.clone();
        s.site_vacancy[5] = Some(0);
        assert_eq!(
            model.check_invariants(&s),
            Err(MultiVacancyError::VacancyCount {
                expected: 2,
                found: 3
            })
        );

        let mut s = valid.clone();
        s.displacements[1] = [1, 1, 1];
        assert_eq!(
            model.check_invariants(&s),
            Err(MultiVacancyError::UnwrappedMismatch { vacancy: 1 })
        );
    }

    #[test]
    fn initial_state_errors() {
        let model = model();
        assert_eq!(
            model.initial_state(&[]),
            Err(MultiVacancyError::NoVacancies)
        );
        assert_eq!(
            model.initial_state(&[16]),
            Err(MultiVacancyError::SiteOutOfRange {
                site: 16,
                sites: 16
            })
        );
        assert_eq!(
            model.initial_state(&[2, 2]),
            Err(MultiVacancyError::DuplicateSite(2))
        );
    }

    #[test]
    fn random_sites_are_distinct_in_range_and_deterministic() {
        let model = model();
        let sites = model.random_sites(10, 42).unwrap();
        assert_eq!(sites.len(), 10);
        assert!(sites.iter().all(|&s| s < 16));
        let mut sorted = sites.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), 10);
        assert_eq!(model.random_sites(10, 42).unwrap(), sites);
        assert_eq!(model.random_sites(16, 1).unwrap().len(), 16);
        assert_eq!(
            model.random_sites(0, 1),
            Err(MultiVacancyError::NoVacancies)
        );
        assert_eq!(
            model.random_sites(17, 1),
            Err(MultiVacancyError::TooManyVacancies {
                count: 17,
                sites: 16
            })
        );
    }

    #[test]
    fn checkpoint_state_round_trips_and_rejects_bad_bytes() {
        let model = model();
        let mut state = pair_state(&model);
        model
            .apply(
                &mut state,
                &MultiVacancyJump {
                    vacancy: 0,
                    direction: 0,
                },
            )
            .unwrap();
        let mut bytes = Vec::new();
        model.encode_state(&state, &mut bytes);
        assert_eq!(bytes.len(), 8 + 40 * 2);
        assert_eq!(model.decode_state(&bytes).unwrap(), state);

        assert!(model.decode_state(&bytes[..bytes.len() - 1]).is_err());
        assert!(model.decode_state(&[]).is_err());

        let mut duplicate = bytes.clone();
        let first_site: [u8; 8] = duplicate[8..16].try_into().unwrap();
        duplicate[48..56].copy_from_slice(&first_site);
        assert!(model.decode_state(&duplicate).is_err());

        let mut out_of_range = bytes.clone();
        out_of_range[8..16].copy_from_slice(&99u64.to_le_bytes());
        assert!(model.decode_state(&out_of_range).is_err());
    }
}
