//! Single-vacancy BCC Fe–Cr alloy model with pluggable energetics and KRA barriers.
//!
//! [`PairEnergetics::illustrative`] provides round-number test parameters with no
//! DFT or literature source; use [`crate::fecr_martinez::Martinez2012`] or
//! [`crate::tabulated::TabulatedEnergetics`] for parameterized alloy studies.

use std::error::Error;
use std::fmt;

use okrates_core::{
    Checkpointable, IncrementalModel, IndependentSnapshot, KmcRng, Model, ReversibleModel,
};

use crate::bcc::{BccLattice, DISPLACEMENTS};
use crate::vacancy::BOLTZMANN_EV_PER_K;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Species {
    Fe,
    Cr,
    Vacancy,
}

impl Species {
    /// Index into symmetric 3×3 pair matrices (`Fe = 0`, `Cr = 1`, `Vacancy = 2`).
    pub fn index(self) -> usize {
        match self {
            Self::Fe => 0,
            Self::Cr => 1,
            Self::Vacancy => 2,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Fe => "Fe",
            Self::Cr => "Cr",
            Self::Vacancy => "Vacancy",
        }
    }

    pub(crate) fn to_byte(self) -> u8 {
        self.index() as u8
    }

    pub(crate) fn from_byte(byte: u8) -> Option<Self> {
        match byte {
            0 => Some(Self::Fe),
            1 => Some(Self::Cr),
            2 => Some(Self::Vacancy),
            _ => None,
        }
    }
}

/// Energetics contract (energies in eV, frequencies in s⁻¹) for single-vacancy Fe–Cr states.
///
/// Conservative Hamiltonians (`supports_equilibrium_swaps() == true`) define `total_energy`,
/// `swap_delta`, and path-independent `exchange_delta == E_after - E_before`; rate-only tables
/// (`false`) return `NaN` on those methods and supply local jump parameters directly.
pub trait Energetics {
    fn total_energy(&self, lattice: &BccLattice, occupancy: &[Species]) -> f64;

    /// Local `E_after - E_before` in eV for exchanging `vacancy_site` with neighbour `atom_site`.
    fn exchange_delta(
        &self,
        lattice: &BccLattice,
        occupancy: &[Species],
        vacancy_site: usize,
        atom_site: usize,
    ) -> f64;

    /// `E_after - E_before` in eV for swapping non-vacancy sites `i` and `j`.
    fn swap_delta(&self, lattice: &BccLattice, occupancy: &[Species], i: usize, j: usize) -> f64 {
        if occupancy.len() != lattice.site_count() || i >= occupancy.len() || j >= occupancy.len() {
            return f64::NAN;
        }
        let before = self.total_energy(lattice, occupancy);
        let mut after = occupancy.to_vec();
        after.swap(i, j);
        self.total_energy(lattice, &after) - before
    }

    /// Forward migration barrier in eV for `atom_site` jumping into `vacancy_site`.
    fn barrier(
        &self,
        lattice: &BccLattice,
        occupancy: &[Species],
        vacancy_site: usize,
        atom_site: usize,
    ) -> f64;

    /// Species-specific attempt frequency in s⁻¹ (`None` uses model default; `0.0` disables jumps).
    fn attempt_frequency(&self, atom: Species) -> Option<f64> {
        let _ = atom;
        None
    }

    /// Environment-specific attempt frequency in s⁻¹ (`0.0` disables jumps; negative/non-finite values surface `SimError::InvalidRate`).
    fn jump_attempt_frequency(
        &self,
        lattice: &BccLattice,
        occupancy: &[Species],
        vacancy_site: usize,
        atom_site: usize,
    ) -> Option<f64> {
        let _ = (lattice, vacancy_site);
        occupancy
            .get(atom_site)
            .copied()
            .and_then(|s| self.attempt_frequency(s))
    }

    /// Combined `(barrier_ev, attempt_frequency_hz)` lookup for single-pass table resolution.
    fn jump_parameters(
        &self,
        lattice: &BccLattice,
        occupancy: &[Species],
        vacancy_site: usize,
        atom_site: usize,
    ) -> (f64, Option<f64>) {
        (
            self.barrier(lattice, occupancy, vacancy_site, atom_site),
            self.jump_attempt_frequency(lattice, occupancy, vacancy_site, atom_site),
        )
    }

    /// Whether `total_energy` and `swap_delta` define a conservative Hamiltonian for Kawasaki swaps.
    fn supports_equilibrium_swaps(&self) -> bool {
        true
    }

    fn validate(&self, lattice: &BccLattice, temperature_k: f64) -> Result<(), FeCrError> {
        let _ = (lattice, temperature_k);
        Ok(())
    }
}

/// Nearest-neighbour pair model with kinetically resolved activation (KRA) barriers.
#[derive(Clone, Debug, PartialEq)]
pub struct PairEnergetics {
    pair: [[f64; 3]; 3],
    kra_fe: f64,
    kra_cr: f64,
}

impl PairEnergetics {
    pub fn kra(&self, atom: Species) -> f64 {
        match atom {
            Species::Fe => self.kra_fe,
            _ => self.kra_cr,
        }
    }

    /// Upper bound `B` with `|exchange_delta| <= B` for any jump of `atom`.
    pub fn delta_bound(&self, atom: Species) -> f64 {
        let values = [self.d(atom, Species::Fe), self.d(atom, Species::Cr)];
        let max = values[0].max(values[1]);
        let min = values[0].min(values[1]);
        7.0 * (max - min)
    }

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

    /// Synthetic round-number parameters (eV) for testing only, not DFT/literature values.
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

    pub fn pair(&self) -> &[[f64; 3]; 3] {
        &self.pair
    }

    fn d(&self, atom: Species, other: Species) -> f64 {
        self.pair[atom.index()][other.index()] - self.pair[Species::Vacancy.index()][other.index()]
    }
}

impl Energetics for PairEnergetics {
    fn total_energy(&self, lattice: &BccLattice, occupancy: &[Species]) -> f64 {
        if occupancy.len() != lattice.site_count() {
            return f64::NAN;
        }
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
        if occupancy.len() != lattice.site_count()
            || vacancy_site >= occupancy.len()
            || atom_site >= occupancy.len()
        {
            return f64::NAN;
        }
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
        if occupancy.len() != lattice.site_count() || i >= occupancy.len() || j >= occupancy.len() {
            return f64::NAN;
        }
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
        let Some(&jumper) = occupancy.get(atom_site) else {
            return f64::NAN;
        };
        self.kra(jumper) + self.exchange_delta(lattice, occupancy, vacancy_site, atom_site) / 2.0
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FeCrJump {
    pub direction: usize,
}

/// Lattice occupancy and per-entity unwrapped displacements (in `a/2` integer units).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FeCrState {
    occupancy: Vec<Species>,
    vacancy: usize,
    atom_at: Vec<usize>,
    origin: Vec<usize>,
    disp: Vec<[i64; 3]>,
}

impl IndependentSnapshot for FeCrState {
    fn snapshot(&self) -> Self {
        self.clone()
    }
}

impl FeCrState {
    pub fn occupancy(&self) -> &[Species] {
        &self.occupancy
    }

    pub fn vacancy(&self) -> usize {
        self.vacancy
    }

    /// Counts `[n_fe, n_cr, n_vacancy]`.
    pub fn counts(&self) -> [usize; 3] {
        let mut counts = [0; 3];
        for species in &self.occupancy {
            counts[species.index()] += 1;
        }
        counts
    }

    pub fn vacancy_displacement(&self) -> [i64; 3] {
        self.disp[self.atom_at[self.vacancy]]
    }

    pub fn displacement_of_site(&self, site: usize) -> [i64; 3] {
        self.disp[self.atom_at[site]]
    }

    pub fn atom_displacements(&self) -> impl Iterator<Item = (Species, [i64; 3])> + '_ {
        self.occupancy
            .iter()
            .enumerate()
            .filter(|&(_, &species)| species != Species::Vacancy)
            .map(|(site, &species)| (species, self.displacement_of_site(site)))
    }

    pub fn collective_displacement(&self, species: Species) -> [i128; 3] {
        let mut sum = [0_i128; 3];
        for (s, d) in self.atom_displacements() {
            if s == species {
                for axis in 0..3 {
                    sum[axis] += d[axis] as i128;
                }
            }
        }
        sum
    }

    pub fn tracer_squared_sum(&self, species: Species) -> f64 {
        self.atom_displacements()
            .filter(|&(s, _)| s == species)
            .map(|(_, d)| d.iter().map(|&c| (c as f64) * (c as f64)).sum::<f64>())
            .sum()
    }
}

#[derive(Clone, Debug)]
pub struct FeCrModel<E: Energetics> {
    lattice: BccLattice,
    energetics: E,
    attempt_frequency: f64,
    temperature_k: f64,
}

impl<E: Energetics> FeCrModel<E> {
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

    pub fn lattice(&self) -> &BccLattice {
        &self.lattice
    }

    pub fn energetics(&self) -> &E {
        &self.energetics
    }

    pub fn attempt_frequency(&self) -> f64 {
        self.attempt_frequency
    }

    pub fn temperature_k(&self) -> f64 {
        self.temperature_k
    }

    /// Thermal energy `k_B T` in eV.
    pub fn kt(&self) -> f64 {
        BOLTZMANN_EV_PER_K * self.temperature_k
    }

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

    pub fn random_state(&self, cr_count: usize, seed: u64) -> Result<FeCrState, FeCrError> {
        let mut rng = KmcRng::from_seed(seed);
        self.random_state_with_rng(cr_count, &mut rng)
    }

    pub fn random_state_with_rng(
        &self,
        cr_count: usize,
        rng: &mut KmcRng,
    ) -> Result<FeCrState, FeCrError> {
        let n = self.lattice.site_count();
        if cr_count >= n {
            return Err(FeCrError::TooManyAtoms);
        }
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

    pub fn barrier(&self, state: &FeCrState, direction: usize) -> f64 {
        if direction >= 8 {
            return f64::NAN;
        }
        let target = self.lattice.neighbor(state.vacancy, direction);
        self.energetics
            .barrier(&self.lattice, &state.occupancy, state.vacancy, target)
    }

    /// Jump rate in s⁻¹ along `direction` (`0.0` when attempt frequency is `0.0`, `NaN` on invalid prefactor or non-finite barrier).
    pub fn rate(&self, state: &FeCrState, direction: usize) -> f64 {
        if direction >= 8 {
            return f64::NAN;
        }
        let target = self.lattice.neighbor(state.vacancy, direction);
        let (b, nu_opt) =
            self.energetics
                .jump_parameters(&self.lattice, &state.occupancy, state.vacancy, target);
        let nu = nu_opt.unwrap_or(self.attempt_frequency);
        if !nu.is_finite() || nu < 0.0 || !b.is_finite() {
            return f64::NAN;
        }
        if nu == 0.0 {
            return 0.0;
        }
        nu * (-b / self.kt()).exp()
    }

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
        let mut total = [0_i128; 3];
        for d in &state.disp {
            for axis in 0..3 {
                total[axis] += d[axis] as i128;
            }
        }
        if total != [0; 3] {
            return Err(FeCrError::DisplacementSum);
        }
        let period: [i128; 3] = std::array::from_fn(|axis| 2 * self.lattice.cells()[axis] as i128);
        let mut seen_entities = vec![false; expected];
        let mut seen_origins = vec![false; expected];
        for site in 0..expected {
            let entity = state.atom_at[site];
            if entity >= expected {
                return Err(FeCrError::UnwrappedMismatch { site });
            }
            let orig = state.origin[entity];
            if orig >= expected || seen_entities[entity] || seen_origins[orig] {
                return Err(FeCrError::UnwrappedMismatch { site });
            }
            seen_entities[entity] = true;
            seen_origins[orig] = true;
            let here = self.lattice.position(site);
            let start = self.lattice.position(orig);
            let d = state.disp[entity];
            if (0..3).any(|axis| {
                (here[axis] as i128 - start[axis] as i128 - d[axis] as i128)
                    .rem_euclid(period[axis])
                    != 0
            }) {
                return Err(FeCrError::UnwrappedMismatch { site });
            }
        }
        Ok(())
    }
}

pub(crate) fn encode_fecr_common_params<E: Energetics>(model: &FeCrModel<E>, out: &mut Vec<u8>) {
    for c in model.lattice.cells() {
        out.extend_from_slice(&(c as u64).to_le_bytes());
    }
    out.extend_from_slice(&model.attempt_frequency.to_bits().to_le_bytes());
    out.extend_from_slice(&model.temperature_k.to_bits().to_le_bytes());
}

pub(crate) fn encode_fecr_state(state: &FeCrState, out: &mut Vec<u8>) {
    let n = state.occupancy.len();
    out.reserve(16 + 41 * n);
    out.extend_from_slice(&(n as u64).to_le_bytes());
    out.extend_from_slice(&(state.vacancy as u64).to_le_bytes());
    for &sp in &state.occupancy {
        out.push(sp.to_byte());
    }
    for &id in &state.atom_at {
        out.extend_from_slice(&(id as u64).to_le_bytes());
    }
    for &orig in &state.origin {
        out.extend_from_slice(&(orig as u64).to_le_bytes());
    }
    for d in &state.disp {
        for &c in d {
            out.extend_from_slice(&c.to_le_bytes());
        }
    }
}

pub(crate) fn decode_fecr_state<E: Energetics>(
    model: &FeCrModel<E>,
    bytes: &[u8],
) -> Result<FeCrState, Box<dyn Error + Send + Sync>> {
    let n = model.lattice.site_count();
    let expected_bytes = 16 + 41 * n;
    if bytes.len() != expected_bytes {
        return Err(format!(
            "FeCrState requires {expected_bytes} bytes, got {}",
            bytes.len()
        )
        .into());
    }
    let mut pos = 0_usize;
    let u64_at = |buf: &[u8], p: &mut usize| -> u64 {
        let w: [u8; 8] = buf[*p..*p + 8].try_into().unwrap();
        *p += 8;
        u64::from_le_bytes(w)
    };
    let stored_n = usize::try_from(u64_at(bytes, &mut pos))?;
    if stored_n != n {
        return Err(FeCrError::SiteCount {
            expected: n,
            found: stored_n,
        }
        .into());
    }
    let vacancy = usize::try_from(u64_at(bytes, &mut pos))?;
    let mut occupancy = Vec::with_capacity(n);
    for &b in &bytes[pos..pos + n] {
        let sp = Species::from_byte(b).ok_or_else(|| format!("invalid species tag {b}"))?;
        occupancy.push(sp);
    }
    pos += n;
    let mut atom_at = Vec::with_capacity(n);
    for _ in 0..n {
        atom_at.push(usize::try_from(u64_at(bytes, &mut pos))?);
    }
    let mut origin = Vec::with_capacity(n);
    for _ in 0..n {
        origin.push(usize::try_from(u64_at(bytes, &mut pos))?);
    }
    let mut disp = Vec::with_capacity(n);
    for _ in 0..n {
        let mut v = [0_i64; 3];
        for item in &mut v {
            *item = u64_at(bytes, &mut pos) as i64;
        }
        disp.push(v);
    }
    let state = FeCrState {
        occupancy,
        vacancy,
        atom_at,
        origin,
        disp,
    };
    model.check_invariants(&state)?;
    Ok(state)
}

impl<E: Energetics> Model for FeCrModel<E> {
    type State = FeCrState;
    type Event = FeCrJump;
    type Error = FeCrError;

    fn enumerate_events(&self, state: &Self::State, emit: &mut dyn FnMut(Self::Event, f64)) {
        for direction in 0..8 {
            emit(FeCrJump { direction }, self.rate(state, direction));
        }
    }

    fn apply(&self, state: &mut Self::State, event: &Self::Event) -> Result<(), Self::Error> {
        let Some(&shift) = DISPLACEMENTS.get(event.direction) else {
            return Err(FeCrError::InvalidDirection(event.direction));
        };
        let v = state.vacancy;
        let target = self.lattice.neighbor(v, event.direction);
        let vacancy_id = state.atom_at[v];
        let atom_id = state.atom_at[target];
        let mut next_vac_disp = [0_i64; 3];
        let mut next_atom_disp = [0_i64; 3];
        for axis in 0..3 {
            next_vac_disp[axis] = state.disp[vacancy_id][axis]
                .checked_add(shift[axis])
                .ok_or(FeCrError::DisplacementOverflow)?;
            next_atom_disp[axis] = state.disp[atom_id][axis]
                .checked_sub(shift[axis])
                .ok_or(FeCrError::DisplacementOverflow)?;
        }
        state.disp[vacancy_id] = next_vac_disp;
        state.disp[atom_id] = next_atom_disp;
        state.occupancy.swap(v, target);
        state.atom_at.swap(v, target);
        state.vacancy = target;
        Ok(())
    }
}

impl<E: Energetics> ReversibleModel for FeCrModel<E> {
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
            &FeCrJump {
                direction: opposite,
            },
        );
    }
}

impl<E: Energetics> IncrementalModel for FeCrModel<E> {
    fn slot_count(&self, _state: &Self::State) -> usize {
        8
    }

    fn slot_rate(&self, state: &Self::State, slot: usize) -> f64 {
        self.rate(state, slot)
    }

    fn slot_event(&self, _state: &Self::State, slot: usize) -> Self::Event {
        FeCrJump { direction: slot }
    }

    fn affected_slots(&self, _state: &Self::State, _event: &Self::Event, out: &mut Vec<usize>) {
        out.extend(0..8);
    }
}

impl Checkpointable for FeCrModel<PairEnergetics> {
    const MODEL_ID: &'static str = "okrates.fecr_pair";

    fn encode_params(&self, out: &mut Vec<u8>) {
        encode_fecr_common_params(self, out);
        for row in &self.energetics.pair {
            for &val in row {
                out.extend_from_slice(&val.to_bits().to_le_bytes());
            }
        }
        out.extend_from_slice(&self.energetics.kra_fe.to_bits().to_le_bytes());
        out.extend_from_slice(&self.energetics.kra_cr.to_bits().to_le_bytes());
    }

    fn encode_state(&self, state: &Self::State, out: &mut Vec<u8>) {
        encode_fecr_state(state, out);
    }

    fn decode_state(&self, bytes: &[u8]) -> Result<Self::State, Box<dyn Error + Send + Sync>> {
        decode_fecr_state(self, bytes)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum FeCrError {
    NonFinite,
    Asymmetric,
    NegativeKra(Species),
    AttemptFrequency(f64),
    Temperature(f64),
    NegativeBarrier { atom: Species, kra: f64, bound: f64 },
    SiteCount { expected: usize, found: usize },
    VacancyCount(usize),
    VacancyMismatch,
    TooManyAtoms,
    DisplacementSum,
    DisplacementOverflow,
    InvalidDirection(usize),
    UnwrappedMismatch { site: usize },
    LatticeTooSmall { min_cells: usize },
    TemperatureMismatch { model: f64, energetics: f64 },
    BarrierTable(String),
    UnsupportedEquilibration(&'static str),
    SweepAllocation(u64),
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
            Self::DisplacementOverflow => write!(f, "entity unwrapped displacement overflowed i64"),
            Self::InvalidDirection(dir) => {
                write!(f, "Fe-Cr jump direction {dir} is out of 0..8")
            }
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
            Self::BarrierTable(msg) => write!(f, "invalid barrier table: {msg}"),
            Self::UnsupportedEquilibration(msg) => write!(f, "equilibration unsupported: {msg}"),
            Self::SweepAllocation(sweeps) => {
                write!(f, "cannot allocate anneal trace for {sweeps} sweeps")
            }
        }
    }
}

impl Error for FeCrError {}

#[cfg(test)]
mod tests {
    use super::{Energetics, FeCrError, FeCrJump, FeCrModel, PairEnergetics, Species};
    use crate::bcc::BccLattice;
    use crate::fecr_martinez::Martinez2012;
    use okrates_core::{Model, ReversibleModel};

    #[derive(Clone, Debug)]
    struct ZeroAttemptEnergetics(PairEnergetics);

    impl Energetics for ZeroAttemptEnergetics {
        fn total_energy(&self, l: &BccLattice, occ: &[Species]) -> f64 {
            self.0.total_energy(l, occ)
        }
        fn exchange_delta(&self, l: &BccLattice, occ: &[Species], v: usize, a: usize) -> f64 {
            self.0.exchange_delta(l, occ, v, a)
        }
        fn barrier(&self, l: &BccLattice, occ: &[Species], v: usize, a: usize) -> f64 {
            self.0.barrier(l, occ, v, a)
        }
        fn attempt_frequency(&self, atom: Species) -> Option<f64> {
            match atom {
                Species::Cr => Some(0.0),
                _ => Some(1e13),
            }
        }
    }

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

    #[test]
    fn reversible_undo_restores_exact_state() {
        let model = FeCrModel::new(
            BccLattice::new([3, 3, 3]).unwrap(),
            PairEnergetics::illustrative(),
            1e13,
            800.0,
        )
        .unwrap();
        let mut state = model.random_state(11, 42).unwrap();
        let before = state.clone();
        let undo = model
            .apply_reversible(&mut state, &FeCrJump { direction: 3 })
            .unwrap();
        assert_ne!(state, before);
        model.undo(&mut state, undo);
        assert_eq!(state, before);
    }

    #[test]
    fn zero_attempt_frequency_disables_species_jumps_with_zero_rate() {
        let lattice = BccLattice::new([3, 3, 3]).unwrap();
        let frozen_cr = FeCrModel::new(
            lattice.clone(),
            ZeroAttemptEnergetics(PairEnergetics::illustrative()),
            1e13,
            800.0,
        )
        .unwrap();
        let mut occ = vec![Species::Fe; lattice.site_count()];
        occ[0] = Species::Vacancy;
        occ[lattice.neighbor(0, 0)] = Species::Cr;
        let st = frozen_cr.state(occ).unwrap();
        assert_eq!(frozen_cr.rate(&st, 0), 0.0);
        assert!(frozen_cr.rate(&st, 1) > 0.0);
        assert!(frozen_cr.rate(&st, 8).is_nan());
        assert!(frozen_cr.barrier(&st, 8).is_nan());
    }

    #[test]
    fn short_occupancy_slices_return_nan_across_energetics() {
        let lattice = BccLattice::new([3, 3, 3]).unwrap();
        let short = [Species::Vacancy, Species::Fe];
        let m2012 = Martinez2012::new(800.0).unwrap();
        let default_swap = ZeroAttemptEnergetics(PairEnergetics::illustrative());
        for e in [
            PairEnergetics::illustrative().total_energy(&lattice, &short),
            PairEnergetics::illustrative().exchange_delta(&lattice, &short, 0, 1),
            PairEnergetics::illustrative().swap_delta(&lattice, &short, 0, 1),
            PairEnergetics::illustrative().barrier(&lattice, &short, 0, 1),
            default_swap.swap_delta(&lattice, &short, 0, 1),
            m2012.total_energy(&lattice, &short),
            m2012.exchange_delta(&lattice, &short, 0, 1),
            m2012.swap_delta(&lattice, &short, 0, 1),
            m2012.barrier(&lattice, &short, 0, 1),
        ] {
            assert!(e.is_nan());
        }
    }

    #[test]
    fn check_invariants_rejects_duplicate_origins() {
        let lattice = BccLattice::new([3, 3, 3]).unwrap();
        let model = FeCrModel::new(lattice, PairEnergetics::illustrative(), 1e13, 800.0).unwrap();
        let mut sneaky = model.random_state(0, 9).unwrap();
        sneaky.origin[1] = 0;
        sneaky.origin[2] = 0;
        sneaky.disp[1] = [0, 0, 2];
        sneaky.disp[2] = [0, 0, -2];
        assert!(matches!(
            model.check_invariants(&sneaky),
            Err(FeCrError::UnwrappedMismatch { .. })
        ));
    }

    #[test]
    fn apply_rejects_invalid_direction_and_displacement_overflow_atomically() {
        let lattice = BccLattice::new([3, 3, 3]).unwrap();
        let model =
            FeCrModel::new(lattice.clone(), PairEnergetics::illustrative(), 1e13, 800.0).unwrap();
        let n = lattice.site_count();
        let mut occ = vec![Species::Fe; n];
        occ[0] = Species::Vacancy;
        let mut state = model.state(occ).unwrap();
        let base = state.clone();
        assert_eq!(
            model.apply(&mut state, &FeCrJump { direction: 8 }),
            Err(FeCrError::InvalidDirection(8))
        );
        assert_eq!(state, base);

        let mult6 = i64::MAX - (i64::MAX % 6);
        state.disp[0] = [mult6, 0, 0];
        state.disp[1] = [-mult6, 0, 0];
        assert_eq!(model.check_invariants(&state), Ok(()));
        assert!(state.tracer_squared_sum(Species::Fe).is_finite());
        while state.disp[0][0] < i64::MAX {
            model.apply(&mut state, &FeCrJump { direction: 7 }).unwrap();
        }
        let snap = state.clone();
        assert_eq!(
            model.apply(&mut state, &FeCrJump { direction: 7 }),
            Err(FeCrError::DisplacementOverflow)
        );
        assert_eq!(state, snap);
    }

    #[test]
    fn random_state_rejects_oversized_cr_count() {
        let lattice = BccLattice::new([3, 3, 3]).unwrap();
        let model = FeCrModel::new(lattice, PairEnergetics::illustrative(), 1e13, 800.0).unwrap();
        assert_eq!(
            model.random_state(usize::MAX, 1),
            Err(FeCrError::TooManyAtoms)
        );
    }
}
