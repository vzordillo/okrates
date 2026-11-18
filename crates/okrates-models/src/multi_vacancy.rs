//! Multi-vacancy site-exclusion diffusion on a periodic BCC lattice.

use std::error::Error;
use std::fmt;

use okrates_core::{
    Checkpointable, IncrementalModel, IndependentSnapshot, KmcRng, Model, ReversibleModel,
};

use crate::bcc::{BccLattice, DISPLACEMENTS};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MultiVacancyState {
    site_vacancy: Vec<Option<usize>>,
    sites: Vec<usize>,
    origins: Vec<usize>,
    displacements: Vec<[i64; 3]>,
}

impl IndependentSnapshot for MultiVacancyState {
    fn snapshot(&self) -> Self {
        self.clone()
    }
}

impl MultiVacancyState {
    pub fn vacancy_count(&self) -> usize {
        self.sites.len()
    }

    pub fn sites(&self) -> &[usize] {
        &self.sites
    }

    pub fn site_vacancy(&self) -> &[Option<usize>] {
        &self.site_vacancy
    }

    /// Per-vacancy unwrapped displacements in `a/2` units.
    pub fn displacements(&self) -> &[[i64; 3]] {
        &self.displacements
    }

    pub fn squared_displacement(&self, id: usize, lattice_parameter: f64) -> f64 {
        let sq_sum = self.displacements[id]
            .iter()
            .map(|&component| (component as f64) * (component as f64))
            .sum::<f64>();
        sq_sum * lattice_parameter * lattice_parameter / 4.0
    }

    /// Number of unordered 1NN vacancy–vacancy pairs.
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MultiVacancyJump {
    pub vacancy: usize,
    pub direction: usize,
}

/// Hard-core multi-vacancy BCC model (`0.0` rate onto occupied vacancy sites).
#[derive(Clone, Debug)]
pub struct MultiVacancyDiffusion {
    lattice: BccLattice,
    jump_rate: f64,
}

impl MultiVacancyDiffusion {
    pub fn new(lattice: BccLattice, jump_rate: f64) -> Self {
        Self { lattice, jump_rate }
    }

    pub fn lattice(&self) -> &BccLattice {
        &self.lattice
    }

    pub fn jump_rate(&self) -> f64 {
        self.jump_rate
    }

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

    pub fn random_sites(&self, count: usize, seed: u64) -> Result<Vec<usize>, MultiVacancyError> {
        let mut rng = KmcRng::from_seed(seed);
        self.random_sites_with_rng(count, &mut rng)
    }

    pub fn random_sites_with_rng(
        &self,
        count: usize,
        rng: &mut KmcRng,
    ) -> Result<Vec<usize>, MultiVacancyError> {
        let n = self.lattice.site_count();
        if count == 0 {
            return Err(MultiVacancyError::NoVacancies);
        }
        if count > n {
            return Err(MultiVacancyError::TooManyVacancies { count, sites: n });
        }
        let mut all: Vec<usize> = (0..n).collect();
        for i in 0..count {
            let j = i + rng.index(n - i);
            all.swap(i, j);
        }
        all.truncate(count);
        Ok(all)
    }

    pub fn check_invariants(&self, state: &MultiVacancyState) -> Result<(), MultiVacancyError> {
        let site_count = self.lattice.site_count();
        if state.site_vacancy.len() != site_count {
            return Err(MultiVacancyError::SiteCount {
                expected: site_count,
                found: state.site_vacancy.len(),
            });
        }
        let n = state.sites.len();
        if n == 0 {
            return Err(MultiVacancyError::NoVacancies);
        }
        if state.origins.len() != n || state.displacements.len() != n {
            return Err(MultiVacancyError::LengthMismatch);
        }
        let mut seen_origins = vec![false; site_count];
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
            if seen_origins[state.origins[v]] {
                return Err(MultiVacancyError::DuplicateSite(state.origins[v]));
            }
            seen_origins[state.origins[v]] = true;
        }
        let found = state.site_vacancy.iter().filter(|s| s.is_some()).count();
        if found != n {
            return Err(MultiVacancyError::VacancyCount { expected: n, found });
        }
        for v in 0..n {
            let origin = self.lattice.position(state.origins[v]);
            let wrapped = std::array::from_fn(|axis| {
                let period = 2 * self.lattice.cells()[axis] as i128;
                (origin[axis] as i128 + state.displacements[v][axis] as i128).rem_euclid(period)
                    as i64
            });
            if self.lattice.site_at(wrapped) != Some(state.sites[v]) {
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
        if vacancy >= state.sites.len() {
            return Err(MultiVacancyError::VacancyOutOfRange {
                vacancy,
                count: state.sites.len(),
            });
        }
        let Some(&shift) = DISPLACEMENTS.get(direction) else {
            return Err(MultiVacancyError::InvalidDirection(direction));
        };
        let old = state.sites[vacancy];
        let target = self.lattice.neighbor(old, direction);
        if state.site_vacancy[target].is_some() {
            return Err(MultiVacancyError::VacancyExchange { vacancy, direction });
        }
        let mut next_disp = [0_i64; 3];
        for axis in 0..3 {
            next_disp[axis] = state.displacements[vacancy][axis]
                .checked_add(shift[axis])
                .ok_or(MultiVacancyError::DisplacementOverflow { vacancy })?;
        }
        state.site_vacancy[old] = None;
        state.site_vacancy[target] = Some(vacancy);
        state.sites[vacancy] = target;
        state.displacements[vacancy] = next_disp;
        Ok(())
    }
}

impl ReversibleModel for MultiVacancyDiffusion {
    type Undo = MultiVacancyJump;

    fn apply_reversible(
        &self,
        state: &mut Self::State,
        event: &Self::Event,
    ) -> Result<Self::Undo, Self::Error> {
        self.apply(state, event)?;
        Ok(MultiVacancyJump {
            vacancy: event.vacancy,
            direction: 7 - event.direction,
        })
    }

    fn undo(&self, state: &mut Self::State, reverse: Self::Undo) {
        let _ = self.apply(state, &reverse);
    }
}

impl IncrementalModel for MultiVacancyDiffusion {
    fn slot_count(&self, state: &Self::State) -> usize {
        8 * state.sites.len()
    }

    fn slot_rate(&self, state: &Self::State, slot: usize) -> f64 {
        let vacancy = slot / 8;
        if vacancy >= state.sites.len() {
            return f64::NAN;
        }
        self.rate_of(state, vacancy, slot % 8)
    }

    fn slot_event(&self, _state: &Self::State, slot: usize) -> Self::Event {
        MultiVacancyJump {
            vacancy: slot / 8,
            direction: slot % 8,
        }
    }

    fn affected_slots(&self, state: &Self::State, event: &Self::Event, out: &mut Vec<usize>) {
        let Some(&new) = state.sites.get(event.vacancy) else {
            return;
        };
        if event.direction >= 8 {
            return;
        }
        out.extend((0..8).map(|d| event.vacancy * 8 + d));
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MultiVacancyError {
    SiteOutOfRange { site: usize, sites: usize },
    DuplicateSite(usize),
    NoVacancies,
    TooManyVacancies { count: usize, sites: usize },
    SiteCount { expected: usize, found: usize },
    LengthMismatch,
    VacancyMismatch { vacancy: usize },
    VacancyCount { expected: usize, found: usize },
    UnwrappedMismatch { vacancy: usize },
    DisplacementOverflow { vacancy: usize },
    VacancyExchange { vacancy: usize, direction: usize },
    VacancyOutOfRange { vacancy: usize, count: usize },
    InvalidDirection(usize),
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
            Self::DisplacementOverflow { vacancy } => write!(
                f,
                "unwrapped displacement of vacancy {vacancy} overflowed i64"
            ),
            Self::VacancyExchange { vacancy, direction } => write!(
                f,
                "vacancy {vacancy} cannot jump in direction {direction} onto another vacancy"
            ),
            Self::VacancyOutOfRange { vacancy, count } => {
                write!(f, "vacancy {vacancy} is out of range for {count} vacancies")
            }
            Self::InvalidDirection(dir) => {
                write!(f, "multi-vacancy jump direction {dir} is out of 0..8")
            }
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
    fn blocks_vacancy_vacancy_exchange_and_reversible_undo_works() {
        let m = model();
        let mut state = pair_state(&m);
        let unchanged = state.clone();
        assert_eq!(
            m.apply(
                &mut state,
                &MultiVacancyJump {
                    vacancy: 0,
                    direction: 3
                }
            ),
            Err(MultiVacancyError::VacancyExchange {
                vacancy: 0,
                direction: 3
            })
        );
        assert_eq!(state, unchanged);

        let before = state.clone();
        let undo = m
            .apply_reversible(
                &mut state,
                &MultiVacancyJump {
                    vacancy: 0,
                    direction: 0,
                },
            )
            .unwrap();
        assert_ne!(state, before);
        m.undo(&mut state, undo);
        assert_eq!(state, before);

        let near_max = i64::MAX - 3;
        let mut raw = Vec::new();
        m.encode_state(&m.initial_state(&[0]).unwrap(), &mut raw);
        raw[24..32].copy_from_slice(&near_max.to_le_bytes());
        let mut hi_state = m.decode_state(&raw).unwrap();
        assert!(hi_state.squared_displacement(0, 1.0).is_finite());
        for _ in 0..3 {
            m.apply(
                &mut hi_state,
                &MultiVacancyJump {
                    vacancy: 0,
                    direction: 7,
                },
            )
            .unwrap();
        }
        let snap = hi_state.clone();
        assert_eq!(
            m.apply(
                &mut hi_state,
                &MultiVacancyJump {
                    vacancy: 0,
                    direction: 7,
                }
            ),
            Err(MultiVacancyError::DisplacementOverflow { vacancy: 0 })
        );
        assert_eq!(hi_state, snap);
    }

    #[test]
    fn check_invariants_rejects_duplicate_origins_and_invalid_apply_indices() {
        let m = model();
        let mut state = pair_state(&m);
        let base = state.clone();
        assert_eq!(
            m.apply(
                &mut state,
                &MultiVacancyJump {
                    vacancy: 9,
                    direction: 0,
                }
            ),
            Err(MultiVacancyError::VacancyOutOfRange {
                vacancy: 9,
                count: 2,
            })
        );
        assert_eq!(state, base);
        assert_eq!(
            m.apply(
                &mut state,
                &MultiVacancyJump {
                    vacancy: 0,
                    direction: 8,
                }
            ),
            Err(MultiVacancyError::InvalidDirection(8))
        );
        assert_eq!(state, base);

        let mut dup = pair_state(&m);
        dup.origins[1] = 0;
        dup.displacements[1] = DISPLACEMENTS[3];
        assert_eq!(
            m.check_invariants(&dup),
            Err(MultiVacancyError::DuplicateSite(0))
        );
        let mut raw = Vec::new();
        m.encode_state(&dup, &mut raw);
        assert!(m.decode_state(&raw).is_err());
    }
}
