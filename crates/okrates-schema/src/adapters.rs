//! JSON data-contract adapters for `pymatgen`/`atomate2` (VASP NEB) and `AiiDA-QuantumESPRESSO`,
//! preserving optional `temp_range_k` validity windows and barrier uncertainties across import and export.

use std::error::Error;
use std::fmt;

use serde::{Deserialize, Serialize};

use crate::barrier::{BarrierProvenance, BarrierRecord, BarrierSet, BarrierValidationError};

#[derive(Debug)]
pub enum AdapterError {
    Json(serde_json::Error),
    TooFewNebImages { task_id: String, count: usize },
    NonFiniteImageEnergy { task_id: String },
    Validation(BarrierValidationError),
}

impl fmt::Display for AdapterError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Json(err) => write!(f, "JSON adapter error: {err}"),
            Self::TooFewNebImages { task_id, count } => write!(
                f,
                "NEB task {task_id:?} requires at least 3 image energies, found {count}"
            ),
            Self::NonFiniteImageEnergy { task_id } => {
                write!(f, "NEB task {task_id:?} contains non-finite image energy")
            }
            Self::Validation(err) => write!(f, "{err}"),
        }
    }
}

impl Error for AdapterError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Json(err) => Some(err),
            Self::Validation(err) => Some(err),
            Self::TooFewNebImages { .. } | Self::NonFiniteImageEnergy { .. } => None,
        }
    }
}

impl From<serde_json::Error> for AdapterError {
    fn from(err: serde_json::Error) -> Self {
        Self::Json(err)
    }
}

impl From<BarrierValidationError> for AdapterError {
    fn from(err: BarrierValidationError) -> Self {
        Self::Validation(err)
    }
}

/// `pymatgen` / `atomate2` VASP NEB task summary entry.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PymatgenNebTask {
    pub task_id: String,
    pub migrating_specie: String,
    pub env_key: String,
    /// Total energies (eV) along NEB images `[initial, ..., final]`.
    pub image_energies_ev: Vec<f64>,
    pub v0_hz: f64,
    pub functional: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub barrier_std_ev: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temp_range_k: Option<[f64; 2]>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PymatgenNebCollection {
    pub name: String,
    pub tolerance_ev: f64,
    pub tasks: Vec<PymatgenNebTask>,
}

pub fn import_pymatgen_vasp_json(json: &str) -> Result<BarrierSet, AdapterError> {
    let doc: PymatgenNebCollection = serde_json::from_str(json)?;
    let mut records = Vec::with_capacity(doc.tasks.len());
    for task in doc.tasks {
        if task.image_energies_ev.len() < 3 {
            return Err(AdapterError::TooFewNebImages {
                task_id: task.task_id,
                count: task.image_energies_ev.len(),
            });
        }
        if task.image_energies_ev.iter().any(|e| !e.is_finite()) {
            return Err(AdapterError::NonFiniteImageEnergy {
                task_id: task.task_id,
            });
        }
        let e_init = task.image_energies_ev[0];
        let e_final = *task.image_energies_ev.last().unwrap();
        // Interior maximum defines the transition state; if interior < endpoints, forward/reverse becomes negative and fails validation.
        let e_saddle = task.image_energies_ev[1..task.image_energies_ev.len() - 1]
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        records.push(BarrierRecord {
            id: task.task_id.clone(),
            migrating_species: task.migrating_specie,
            env_key: task.env_key,
            forward_barrier_ev: e_saddle - e_init,
            reverse_barrier_ev: e_saddle - e_final,
            delta_energy_ev: e_final - e_init,
            attempt_frequency_hz: task.v0_hz,
            reverse_attempt_frequency_hz: None,
            barrier_std_ev: task.barrier_std_ev,
            temp_range_k: task.temp_range_k,
            provenance: Some(BarrierProvenance {
                source_code: "VASP (pymatgen/atomate2)".to_string(),
                functional: task.functional,
                workflow_id: Some(task.task_id),
            }),
        });
    }
    Ok(BarrierSet::new(doc.name, doc.tolerance_ev, records)?)
}

pub fn export_pymatgen_vasp_json(set: &BarrierSet) -> Result<String, AdapterError> {
    set.validate()?;
    let tasks = set
        .records
        .iter()
        .map(|rec| PymatgenNebTask {
            task_id: rec.id.clone(),
            migrating_specie: rec.migrating_species.clone(),
            env_key: rec.env_key.clone(),
            image_energies_ev: vec![0.0, rec.forward_barrier_ev, rec.delta_energy_ev],
            v0_hz: rec.attempt_frequency_hz,
            functional: rec.provenance.as_ref().and_then(|p| p.functional.clone()),
            barrier_std_ev: rec.barrier_std_ev,
            temp_range_k: rec.temp_range_k,
        })
        .collect();
    Ok(serde_json::to_string_pretty(&PymatgenNebCollection {
        name: set.name.clone(),
        tolerance_ev: set.tolerance_ev,
        tasks,
    })?)
}

/// `AiiDA-QuantumESPRESSO` NEB WorkChain path entry (`attempt_frequency_thz` in THz = `1e12 Hz`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AiidaQePath {
    pub label: String,
    pub species: String,
    pub env_key: String,
    pub forward_barrier_ev: f64,
    pub reverse_barrier_ev: f64,
    pub reaction_energy_ev: f64,
    pub attempt_frequency_thz: f64,
    pub barrier_uncertainty_ev: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temp_range_k: Option<[f64; 2]>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AiidaQeNebExport {
    pub workchain_uuid: String,
    pub name: String,
    pub dft_functional: Option<String>,
    pub tolerance_ev: f64,
    pub paths: Vec<AiidaQePath>,
}

pub fn import_aiida_qe_json(json: &str) -> Result<BarrierSet, AdapterError> {
    let doc: AiidaQeNebExport = serde_json::from_str(json)?;
    let records = doc
        .paths
        .into_iter()
        .map(|p| BarrierRecord {
            id: p.label,
            migrating_species: p.species,
            env_key: p.env_key,
            forward_barrier_ev: p.forward_barrier_ev,
            reverse_barrier_ev: p.reverse_barrier_ev,
            delta_energy_ev: p.reaction_energy_ev,
            attempt_frequency_hz: p.attempt_frequency_thz * 1e12,
            reverse_attempt_frequency_hz: None,
            barrier_std_ev: p.barrier_uncertainty_ev,
            temp_range_k: p.temp_range_k,
            provenance: Some(BarrierProvenance {
                source_code: "QuantumESPRESSO (AiiDA)".to_string(),
                functional: doc.dft_functional.clone(),
                workflow_id: Some(doc.workchain_uuid.clone()),
            }),
        })
        .collect();
    Ok(BarrierSet::new(doc.name, doc.tolerance_ev, records)?)
}

pub fn export_aiida_qe_json(
    set: &BarrierSet,
    workchain_uuid: &str,
) -> Result<String, AdapterError> {
    set.validate()?;
    let functional = set
        .records
        .first()
        .and_then(|r| r.provenance.as_ref())
        .and_then(|p| p.functional.clone());
    let paths = set
        .records
        .iter()
        .map(|r| AiidaQePath {
            label: r.id.clone(),
            species: r.migrating_species.clone(),
            env_key: r.env_key.clone(),
            forward_barrier_ev: r.forward_barrier_ev,
            reverse_barrier_ev: r.reverse_barrier_ev,
            reaction_energy_ev: r.delta_energy_ev,
            attempt_frequency_thz: r.attempt_frequency_hz * 1e-12,
            barrier_uncertainty_ev: r.barrier_std_ev,
            temp_range_k: r.temp_range_k,
        })
        .collect();
    Ok(serde_json::to_string_pretty(&AiidaQeNebExport {
        workchain_uuid: workchain_uuid.to_string(),
        name: set.name.clone(),
        dft_functional: functional,
        tolerance_ev: set.tolerance_ev,
        paths,
    })?)
}
