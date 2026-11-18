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

#[derive(Clone, Debug)]
pub struct TabulatedEnergetics {
    barrier_set: BarrierSet,
    sha256: String,
    index: HashMap<(Species, String), usize>,
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
            if rec.env_key.contains("cr_saddle=") {
                uses_saddle_shell = true;
            }
            if is_symmetric_key(&rec.env_key)
                && rec.delta_energy_ev.abs() > barrier_set.tolerance_ev
            {
                return Err(FeCrError::BarrierTable(format!(
                    "symmetric environment key {:?} in record {:?} requires |delta_energy_ev| <= {}, got {}",
                    rec.env_key, rec.id, barrier_set.tolerance_ev, rec.delta_energy_ev
                )));
            }
            index.insert((sp, rec.env_key.clone()), idx);
        }

        // Verify cross-record consistency if both forward and reverse asymmetric keys exist.
        for (idx, rec) in barrier_set.records.iter().enumerate() {
            if let Some(rev_key) = reverse_env_key(&rec.env_key)
                && rev_key != rec.env_key
            {
                let sp = if rec.migrating_species == "Fe" {
                    Species::Fe
                } else {
                    Species::Cr
                };
                if let Some(&other_idx) = index.get(&(sp, rev_key))
                    && other_idx > idx
                {
                    let other = &barrier_set.records[other_idx];
                    let tol = barrier_set.tolerance_ev;
                    if (rec.forward_barrier_ev - other.reverse_barrier_ev).abs() > tol
                        || (rec.reverse_barrier_ev - other.forward_barrier_ev).abs() > tol
                        || (rec.delta_energy_ev + other.delta_energy_ev).abs() > tol
                    {
                        return Err(FeCrError::BarrierTable(format!(
                            "complementary records {:?} and {:?} disagree beyond tolerance {tol} eV",
                            rec.id, other.id
                        )));
                    }
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
        if vacancy_site >= occupancy.len()
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
            .count();
        let cv = lattice
            .neighbors(vacancy_site)
            .iter()
            .filter(|&&k| k != atom_site && occupancy[k] == Species::Cr)
            .count();
        let cs = if self.uses_saddle_shell {
            let mut count = 0_usize;
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
            let fwd = format!("cr_saddle={cs},cr_atom={ca},cr_vac={cv}");
            if let Some(&idx) = self.index.get(&(sp, fwd)) {
                return Some(self.record_forward(&self.barrier_set.records[idx]));
            }
            let rev = format!("cr_saddle={cs},cr_atom={cv},cr_vac={ca}");
            if let Some(&idx) = self.index.get(&(sp, rev)) {
                return Some(self.record_reverse(&self.barrier_set.records[idx]));
            }
        }

        let pair_fwd = format!("cr_atom={ca},cr_vac={cv}");
        if let Some(&idx) = self.index.get(&(sp, pair_fwd)) {
            return Some(self.record_forward(&self.barrier_set.records[idx]));
        }
        let pair_rev = format!("cr_atom={cv},cr_vac={ca}");
        if let Some(&idx) = self.index.get(&(sp, pair_rev)) {
            return Some(self.record_reverse(&self.barrier_set.records[idx]));
        }

        if self.uses_saddle_shell {
            let saddle_key = format!("cr_saddle={cs}");
            if let Some(&idx) = self.index.get(&(sp, saddle_key)) {
                return Some(self.record_forward(&self.barrier_set.records[idx]));
            }
        }

        if let Some(&idx) = self.index.get(&(sp, "*".to_string())) {
            return Some(self.record_forward(&self.barrier_set.records[idx]));
        }
        None
    }
}

fn reverse_env_key(env_key: &str) -> Option<String> {
    let mut cs: Option<&str> = None;
    let mut ca: Option<&str> = None;
    let mut cv: Option<&str> = None;
    for part in env_key.split(',') {
        let (k, v) = part.split_once('=')?;
        match k {
            "cr_saddle" => cs = Some(v),
            "cr_atom" => ca = Some(v),
            "cr_vac" => cv = Some(v),
            _ => return None,
        }
    }
    match (cs, ca, cv) {
        (Some(s), Some(a), Some(v)) => Some(format!("cr_saddle={s},cr_atom={v},cr_vac={a}")),
        (None, Some(a), Some(v)) => Some(format!("cr_atom={v},cr_vac={a}")),
        _ => None,
    }
}

fn is_symmetric_key(env_key: &str) -> bool {
    if env_key == "*" || (env_key.starts_with("cr_saddle=") && !env_key.contains(',')) {
        return true;
    }
    reverse_env_key(env_key).is_some_and(|rev| rev == env_key)
}

impl Energetics for TabulatedEnergetics {
    fn total_energy(&self, _lattice: &BccLattice, _occupancy: &[Species]) -> f64 {
        0.0
    }

    fn exchange_delta(
        &self,
        lattice: &BccLattice,
        occupancy: &[Species],
        vacancy_site: usize,
        atom_site: usize,
    ) -> f64 {
        self.resolve(lattice, occupancy, vacancy_site, atom_site)
            .map_or(f64::NAN, |j| j.delta_ev)
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
