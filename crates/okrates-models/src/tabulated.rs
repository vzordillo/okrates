//! Tabulated Fe–Cr barrier energetics backed by a validated [`okrates_schema::BarrierSet`].
//!
//! Lookup probes `(migrating_species, env_key)` in decreasing specificity:
//! 1. `"cr_saddle={cs},cr_atom={ca},cr_vac={cv}"` (or reverse `"cr_saddle={cs},cr_atom={cv},cr_vac={ca}"`)
//! 2. `"cr_atom={ca},cr_vac={cv}"` (or reverse `"cr_atom={cv},cr_vac={ca}"`)
//! 3. `"cr_saddle={cs}"`
//! 4. `"*"`

use std::collections::HashMap;
use std::error::Error;

use okrates_core::Checkpointable;
use okrates_schema::{BarrierRecord, BarrierSet};

use crate::bcc::BccLattice;
use crate::fecr::{
    Energetics, FeCrError, FeCrModel, Species, decode_fecr_state, encode_fecr_common_params,
    encode_fecr_state,
};

const MIN_CELLS_SADDLE: usize = 3;

#[derive(Clone, Copy, Debug)]
struct ResolvedJump {
    barrier_ev: f64,
    delta_ev: f64,
    attempt_hz: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum EnvKey {
    Wildcard,
    Saddle { cs: u8 },
    Pair { ca: u8, cv: u8 },
    Full { cs: u8, ca: u8, cv: u8 },
}

impl EnvKey {
    fn uses_saddle(self) -> bool {
        matches!(self, Self::Saddle { .. } | Self::Full { .. })
    }

    fn reverse(self) -> Self {
        match self {
            Self::Wildcard => Self::Wildcard,
            Self::Saddle { cs } => Self::Saddle { cs },
            Self::Pair { ca, cv } => Self::Pair { ca: cv, cv: ca },
            Self::Full { cs, ca, cv } => Self::Full { cs, ca: cv, cv: ca },
        }
    }

    fn is_symmetric(self) -> bool {
        self.reverse() == self
    }
}

fn parse_env_key(env_key: &str) -> Result<EnvKey, String> {
    if env_key == "*" {
        return Ok(EnvKey::Wildcard);
    }
    let mut cs: Option<u8> = None;
    let mut ca: Option<u8> = None;
    let mut cv: Option<u8> = None;
    for part in env_key.split(',') {
        let Some((k, raw_v)) = part.split_once('=') else {
            return Err("expected \"*\" or comma-separated key=value clauses".to_string());
        };
        let parse_bounded = |limit: u8, label: &str| -> Result<u8, String> {
            let val = raw_v
                .parse::<u8>()
                .map_err(|_| format!("{label} count {raw_v:?} must be a non-negative integer"))?;
            if val > limit {
                return Err(format!(
                    "{label} count {val} exceeds BCC shell maximum {limit}"
                ));
            }
            Ok(val)
        };
        match k {
            "cr_saddle" => {
                if cs.replace(parse_bounded(6, "cr_saddle")?).is_some() {
                    return Err("duplicate cr_saddle clause".to_string());
                }
            }
            "cr_atom" => {
                if ca.replace(parse_bounded(7, "cr_atom")?).is_some() {
                    return Err("duplicate cr_atom clause".to_string());
                }
            }
            "cr_vac" => {
                if cv.replace(parse_bounded(7, "cr_vac")?).is_some() {
                    return Err("duplicate cr_vac clause".to_string());
                }
            }
            other => {
                return Err(format!(
                    "unknown clause {other:?}; expected cr_saddle, cr_atom, or cr_vac"
                ));
            }
        }
    }
    match (cs, ca, cv) {
        (Some(cs), None, None) => Ok(EnvKey::Saddle { cs }),
        (None, Some(ca), Some(cv)) => Ok(EnvKey::Pair { ca, cv }),
        (Some(cs), Some(ca), Some(cv)) => Ok(EnvKey::Full { cs, ca, cv }),
        _ => Err("cr_atom and cr_vac must be specified together".to_string()),
    }
}

/// Rate-table Fe–Cr energetics backed by a validated [`BarrierSet`].
///
/// Defines local transition barriers, attempt frequencies, and [`Self::transition_delta_ev`] rather
/// than a global Hamiltonian (`total_energy`, `exchange_delta`, and `swap_delta` return `NaN`, and
/// [`Energetics::supports_equilibrium_swaps`] is `false`).
#[derive(Clone, Debug)]
pub struct TabulatedEnergetics {
    barrier_set: BarrierSet,
    sha256: String,
    index: HashMap<(Species, EnvKey), usize>,
    uses_saddle_shell: bool,
}

impl TabulatedEnergetics {
    pub fn new(barrier_set: BarrierSet) -> Result<Self, FeCrError> {
        barrier_set
            .validate()
            .map_err(|err| FeCrError::BarrierTable(err.to_string()))?;
        let sha256 = barrier_set
            .content_sha256()
            .map_err(|err| FeCrError::BarrierTable(err.to_string()))?;

        let mut index = HashMap::with_capacity(barrier_set.records.len());
        let mut parsed_keys = Vec::with_capacity(barrier_set.records.len());
        let mut uses_saddle_shell = false;
        for (idx, rec) in barrier_set.records.iter().enumerate() {
            let sp = match rec.migrating_species.as_str() {
                "Fe" => Species::Fe,
                "Cr" => Species::Cr,
                other => {
                    return Err(FeCrError::BarrierTable(format!(
                        "unsupported migrating_species {other:?} in record {:?}; expected \"Fe\" or \"Cr\"",
                        rec.id
                    )));
                }
            };
            let key = parse_env_key(&rec.env_key).map_err(|reason| {
                FeCrError::BarrierTable(format!(
                    "invalid env_key {:?} in record {:?}: {reason}",
                    rec.env_key, rec.id
                ))
            })?;
            if key.uses_saddle() {
                uses_saddle_shell = true;
            }
            if key.is_symmetric()
                && (rec.delta_energy_ev.abs() > barrier_set.tolerance_ev
                    || (rec.forward_barrier_ev - rec.reverse_barrier_ev).abs()
                        > barrier_set.tolerance_ev)
            {
                return Err(FeCrError::BarrierTable(format!(
                    "symmetric environment key {:?} in record {:?} requires |delta_energy_ev| <= {} and |forward - reverse| <= {}",
                    rec.env_key, rec.id, barrier_set.tolerance_ev, barrier_set.tolerance_ev
                )));
            }
            if let Some(rev_hz) = rec.reverse_attempt_frequency_hz
                && (rev_hz - rec.attempt_frequency_hz).abs()
                    > 1e-12 * rec.attempt_frequency_hz.min(rev_hz)
            {
                return Err(FeCrError::BarrierTable(format!(
                    "record {:?} specifies unequal forward/reverse attempt frequencies ({} vs {} Hz) incompatible with energy-only detailed balance",
                    rec.id, rec.attempt_frequency_hz, rev_hz
                )));
            }
            if index.insert((sp, key), idx).is_some() {
                return Err(FeCrError::BarrierTable(format!(
                    "duplicate canonical environment key ({}, {:?}) in record {:?}",
                    sp.name(),
                    rec.env_key,
                    rec.id
                )));
            }
            parsed_keys.push((sp, key));
        }

        for (idx, (&(sp, key), rec)) in parsed_keys.iter().zip(&barrier_set.records).enumerate() {
            let rev_key = key.reverse();
            if rev_key != key
                && let Some(&other_idx) = index.get(&(sp, rev_key))
                && other_idx > idx
            {
                let other = &barrier_set.records[other_idx];
                let tol = barrier_set.tolerance_ev;
                if (rec.forward_barrier_ev - other.reverse_barrier_ev).abs() > tol
                    || (rec.reverse_barrier_ev - other.forward_barrier_ev).abs() > tol
                    || (rec.delta_energy_ev + other.delta_energy_ev).abs() > tol
                    || ((rec.forward_barrier_ev - other.forward_barrier_ev) - rec.delta_energy_ev)
                        .abs()
                        > tol
                    || ((other.forward_barrier_ev - rec.forward_barrier_ev) - other.delta_energy_ev)
                        .abs()
                        > tol
                {
                    return Err(FeCrError::BarrierTable(format!(
                        "complementary records {:?} and {:?} disagree beyond tolerance {tol} eV",
                        rec.id, other.id
                    )));
                }
                let rec_rev_hz = rec
                    .reverse_attempt_frequency_hz
                    .unwrap_or(rec.attempt_frequency_hz);
                let other_rev_hz = other
                    .reverse_attempt_frequency_hz
                    .unwrap_or(other.attempt_frequency_hz);
                if (rec.attempt_frequency_hz - other_rev_hz).abs()
                    > 1e-12 * rec.attempt_frequency_hz.min(other_rev_hz)
                    || (rec_rev_hz - other.attempt_frequency_hz).abs()
                        > 1e-12 * rec_rev_hz.min(other.attempt_frequency_hz)
                {
                    return Err(FeCrError::BarrierTable(format!(
                        "complementary records {:?} and {:?} disagree in attempt frequency",
                        rec.id, other.id
                    )));
                }
            }
        }

        Ok(Self {
            barrier_set,
            sha256,
            index,
            uses_saddle_shell,
        })
    }

    pub fn barrier_set(&self) -> &BarrierSet {
        &self.barrier_set
    }

    pub fn barrier_set_sha256(&self) -> &str {
        &self.sha256
    }

    /// Local reaction-energy difference `E_after - E_before` in eV from the matched barrier record (`NaN` if unresolved).
    pub fn transition_delta_ev(
        &self,
        lattice: &BccLattice,
        occupancy: &[Species],
        vacancy_site: usize,
        atom_site: usize,
    ) -> f64 {
        self.resolve(lattice, occupancy, vacancy_site, atom_site)
            .map_or(f64::NAN, |j| j.delta_ev)
    }

    fn record_forward(&self, rec: &BarrierRecord) -> ResolvedJump {
        ResolvedJump {
            barrier_ev: rec.forward_barrier_ev,
            delta_ev: rec.delta_energy_ev,
            attempt_hz: rec.attempt_frequency_hz,
        }
    }

    fn record_reverse(&self, rec: &BarrierRecord) -> ResolvedJump {
        ResolvedJump {
            barrier_ev: rec.reverse_barrier_ev,
            delta_ev: -rec.delta_energy_ev,
            attempt_hz: rec
                .reverse_attempt_frequency_hz
                .unwrap_or(rec.attempt_frequency_hz),
        }
    }

    fn resolve(
        &self,
        lattice: &BccLattice,
        occupancy: &[Species],
        vacancy_site: usize,
        atom_site: usize,
    ) -> Option<ResolvedJump> {
        if occupancy.len() != lattice.site_count()
            || vacancy_site >= occupancy.len()
            || atom_site >= occupancy.len()
            || occupancy[vacancy_site] != Species::Vacancy
        {
            return None;
        }
        let sp = occupancy[atom_site];
        if sp == Species::Vacancy {
            return None;
        }

        let ca = lattice
            .neighbors(atom_site)
            .iter()
            .filter(|&&k| k != vacancy_site && occupancy[k] == Species::Cr)
            .count() as u8;
        let cv = lattice
            .neighbors(vacancy_site)
            .iter()
            .filter(|&&k| k != atom_site && occupancy[k] == Species::Cr)
            .count() as u8;
        let cs = if self.uses_saddle_shell {
            let mut count = 0_u8;
            for &k in lattice
                .neighbors(atom_site)
                .iter()
                .filter(|k| lattice.second_neighbors(vacancy_site).contains(k))
                .chain(
                    lattice
                        .second_neighbors(atom_site)
                        .iter()
                        .filter(|k| lattice.neighbors(vacancy_site).contains(k)),
                )
            {
                if k != atom_site && k != vacancy_site && occupancy[k] == Species::Cr {
                    count += 1;
                }
            }
            count
        } else {
            0
        };

        if self.uses_saddle_shell {
            if let Some(&idx) = self.index.get(&(sp, EnvKey::Full { cs, ca, cv })) {
                return Some(self.record_forward(&self.barrier_set.records[idx]));
            }
            if ca != cv
                && let Some(&idx) = self.index.get(&(sp, EnvKey::Full { cs, ca: cv, cv: ca }))
            {
                return Some(self.record_reverse(&self.barrier_set.records[idx]));
            }
        }

        if let Some(&idx) = self.index.get(&(sp, EnvKey::Pair { ca, cv })) {
            return Some(self.record_forward(&self.barrier_set.records[idx]));
        }
        if ca != cv
            && let Some(&idx) = self.index.get(&(sp, EnvKey::Pair { ca: cv, cv: ca }))
        {
            return Some(self.record_reverse(&self.barrier_set.records[idx]));
        }

        if self.uses_saddle_shell
            && let Some(&idx) = self.index.get(&(sp, EnvKey::Saddle { cs }))
        {
            return Some(self.record_forward(&self.barrier_set.records[idx]));
        }

        if let Some(&idx) = self.index.get(&(sp, EnvKey::Wildcard)) {
            return Some(self.record_forward(&self.barrier_set.records[idx]));
        }
        None
    }
}

impl Energetics for TabulatedEnergetics {
    fn total_energy(&self, _lattice: &BccLattice, _occupancy: &[Species]) -> f64 {
        f64::NAN
    }

    fn exchange_delta(
        &self,
        _lattice: &BccLattice,
        _occupancy: &[Species],
        _vacancy_site: usize,
        _atom_site: usize,
    ) -> f64 {
        f64::NAN
    }

    fn swap_delta(
        &self,
        _lattice: &BccLattice,
        _occupancy: &[Species],
        _i: usize,
        _j: usize,
    ) -> f64 {
        f64::NAN
    }

    fn barrier(
        &self,
        lattice: &BccLattice,
        occupancy: &[Species],
        vacancy_site: usize,
        atom_site: usize,
    ) -> f64 {
        self.resolve(lattice, occupancy, vacancy_site, atom_site)
            .map_or(f64::NAN, |j| j.barrier_ev)
    }

    fn jump_attempt_frequency(
        &self,
        lattice: &BccLattice,
        occupancy: &[Species],
        vacancy_site: usize,
        atom_site: usize,
    ) -> Option<f64> {
        self.resolve(lattice, occupancy, vacancy_site, atom_site)
            .map(|j| j.attempt_hz)
    }

    fn jump_parameters(
        &self,
        lattice: &BccLattice,
        occupancy: &[Species],
        vacancy_site: usize,
        atom_site: usize,
    ) -> (f64, Option<f64>) {
        self.resolve(lattice, occupancy, vacancy_site, atom_site)
            .map_or((f64::NAN, None), |j| (j.barrier_ev, Some(j.attempt_hz)))
    }

    fn supports_equilibrium_swaps(&self) -> bool {
        false
    }

    fn validate(&self, lattice: &BccLattice, temperature_k: f64) -> Result<(), FeCrError> {
        if self.uses_saddle_shell && lattice.cells().iter().any(|&c| c < MIN_CELLS_SADDLE) {
            return Err(FeCrError::LatticeTooSmall {
                min_cells: MIN_CELLS_SADDLE,
            });
        }
        self.barrier_set
            .validate_at_temperature(temperature_k)
            .map_err(|err| FeCrError::BarrierTable(err.to_string()))
    }
}

impl Checkpointable for FeCrModel<TabulatedEnergetics> {
    const MODEL_ID: &'static str = "okrates.fecr_tabulated";

    fn encode_params(&self, out: &mut Vec<u8>) {
        encode_fecr_common_params(self, out);
        out.extend_from_slice(self.energetics().barrier_set_sha256().as_bytes());
    }

    fn encode_state(&self, state: &Self::State, out: &mut Vec<u8>) {
        encode_fecr_state(state, out);
    }

    fn decode_state(&self, bytes: &[u8]) -> Result<Self::State, Box<dyn Error + Send + Sync>> {
        decode_fecr_state(self, bytes)
    }
}
