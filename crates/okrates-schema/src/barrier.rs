//! BarrierRecord v1 schema and thermodynamic detailed-balance validator.

use std::collections::HashSet;
use std::error::Error;
use std::fmt;

use serde::{Deserialize, Serialize};

use crate::digest::sha256_hex;

pub const BARRIER_SCHEMA_VERSION: u32 = 1;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BarrierProvenance {
    pub source_code: String,
    pub functional: Option<String>,
    pub workflow_id: Option<String>,
}

/// Elementary transition barrier record with forward/reverse consistency and optional UQ (`barrier_std_ev`).
///
/// Sign convention: `delta_energy_ev = E_final - E_initial`, so energy-level detailed balance
/// checks `(forward_barrier_ev - reverse_barrier_ev) == delta_energy_ev` (equal forward/reverse prefactors).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BarrierRecord {
    pub id: String,
    pub migrating_species: String,
    pub env_key: String,
    pub forward_barrier_ev: f64,
    pub reverse_barrier_ev: f64,
    pub delta_energy_ev: f64,
    pub attempt_frequency_hz: f64,
    pub reverse_attempt_frequency_hz: Option<f64>,
    pub barrier_std_ev: Option<f64>,
    pub temp_range_k: Option<[f64; 2]>,
    pub provenance: Option<BarrierProvenance>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum BarrierValidationError {
    UnsupportedSchema(u32),
    InvalidTolerance(f64),
    InvalidTemperature(f64),
    EmptyBarrierSet,
    DuplicateEnvironment {
        species: String,
        env_key: String,
    },
    NonFiniteField {
        id: String,
        field: &'static str,
    },
    NegativeBarrier {
        id: String,
        direction: &'static str,
        value_ev: f64,
    },
    InvalidAttemptFrequency {
        id: String,
        value_hz: f64,
    },
    NegativeUncertainty {
        id: String,
        std_ev: f64,
    },
    InvalidTemperatureBounds {
        id: String,
        min_k: f64,
        max_k: f64,
    },
    TemperatureOutOfRange {
        id: String,
        temperature_k: f64,
        min_k: f64,
        max_k: f64,
    },
    DetailedBalanceViolation {
        id: String,
        forward_ev: f64,
        reverse_ev: f64,
        delta_ev: f64,
        residual_ev: f64,
        tolerance_ev: f64,
    },
}

impl fmt::Display for BarrierValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedSchema(v) => write!(
                f,
                "unsupported barrier schema version {v}, expected {BARRIER_SCHEMA_VERSION}"
            ),
            Self::InvalidTolerance(tol) => write!(f, "invalid tolerance {tol} eV"),
            Self::InvalidTemperature(t) => {
                write!(f, "temperature must be finite and positive, got {t} K")
            }
            Self::EmptyBarrierSet => write!(f, "barrier set must contain at least one record"),
            Self::DuplicateEnvironment { species, env_key } => {
                write!(
                    f,
                    "duplicate barrier key ({species}, {env_key:?}) in barrier set"
                )
            }
            Self::NonFiniteField { id, field } => {
                write!(f, "barrier record {id:?} has non-finite {field}")
            }
            Self::NegativeBarrier {
                id,
                direction,
                value_ev,
            } => write!(
                f,
                "barrier record {id:?} has negative {direction} barrier {value_ev} eV"
            ),
            Self::InvalidAttemptFrequency { id, value_hz } => write!(
                f,
                "barrier record {id:?} has invalid attempt frequency {value_hz} Hz"
            ),
            Self::NegativeUncertainty { id, std_ev } => write!(
                f,
                "barrier record {id:?} has invalid uncertainty {std_ev} eV"
            ),
            Self::InvalidTemperatureBounds { id, min_k, max_k } => write!(
                f,
                "barrier record {id:?} has invalid temperature bounds [{min_k}, {max_k}] K"
            ),
            Self::TemperatureOutOfRange {
                id,
                temperature_k,
                min_k,
                max_k,
            } => write!(
                f,
                "temperature {temperature_k} K is outside [{min_k}, {max_k}] K for barrier {id:?}"
            ),
            Self::DetailedBalanceViolation {
                id,
                forward_ev,
                reverse_ev,
                delta_ev,
                residual_ev,
                tolerance_ev,
            } => write!(
                f,
                "detailed balance violated for {id:?}: |(Ef={forward_ev} - Er={reverse_ev}) - dE={delta_ev}| = {residual_ev:.3e} > {tolerance_ev:.3e} eV"
            ),
        }
    }
}

impl Error for BarrierValidationError {}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BarrierSet {
    pub schema_version: u32,
    pub name: String,
    pub tolerance_ev: f64,
    pub records: Vec<BarrierRecord>,
}

impl BarrierSet {
    pub fn new(
        name: impl Into<String>,
        tolerance_ev: f64,
        records: Vec<BarrierRecord>,
    ) -> Result<Self, BarrierValidationError> {
        let set = Self {
            schema_version: BARRIER_SCHEMA_VERSION,
            name: name.into(),
            tolerance_ev,
            records,
        };
        set.validate()?;
        Ok(set)
    }

    pub fn validate(&self) -> Result<(), BarrierValidationError> {
        if self.schema_version != BARRIER_SCHEMA_VERSION {
            return Err(BarrierValidationError::UnsupportedSchema(
                self.schema_version,
            ));
        }
        if !self.tolerance_ev.is_finite() || self.tolerance_ev < 0.0 {
            return Err(BarrierValidationError::InvalidTolerance(self.tolerance_ev));
        }
        if self.records.is_empty() {
            return Err(BarrierValidationError::EmptyBarrierSet);
        }

        let mut seen = HashSet::with_capacity(self.records.len());
        for rec in &self.records {
            if !seen.insert((rec.migrating_species.as_str(), rec.env_key.as_str())) {
                return Err(BarrierValidationError::DuplicateEnvironment {
                    species: rec.migrating_species.clone(),
                    env_key: rec.env_key.clone(),
                });
            }
            for (field, val) in [
                ("forward_barrier_ev", rec.forward_barrier_ev),
                ("reverse_barrier_ev", rec.reverse_barrier_ev),
                ("delta_energy_ev", rec.delta_energy_ev),
                ("attempt_frequency_hz", rec.attempt_frequency_hz),
            ] {
                if !val.is_finite() {
                    return Err(BarrierValidationError::NonFiniteField {
                        id: rec.id.clone(),
                        field,
                    });
                }
            }
            if rec.forward_barrier_ev < 0.0 {
                return Err(BarrierValidationError::NegativeBarrier {
                    id: rec.id.clone(),
                    direction: "forward",
                    value_ev: rec.forward_barrier_ev,
                });
            }
            if rec.reverse_barrier_ev < 0.0 {
                return Err(BarrierValidationError::NegativeBarrier {
                    id: rec.id.clone(),
                    direction: "reverse",
                    value_ev: rec.reverse_barrier_ev,
                });
            }
            if rec.attempt_frequency_hz <= 0.0 {
                return Err(BarrierValidationError::InvalidAttemptFrequency {
                    id: rec.id.clone(),
                    value_hz: rec.attempt_frequency_hz,
                });
            }
            if let Some(rev_v0) = rec.reverse_attempt_frequency_hz
                && (!rev_v0.is_finite() || rev_v0 <= 0.0)
            {
                return Err(BarrierValidationError::InvalidAttemptFrequency {
                    id: rec.id.clone(),
                    value_hz: rev_v0,
                });
            }
            if let Some(std_ev) = rec.barrier_std_ev
                && (!std_ev.is_finite() || std_ev < 0.0)
            {
                return Err(BarrierValidationError::NegativeUncertainty {
                    id: rec.id.clone(),
                    std_ev,
                });
            }
            if let Some([min_k, max_k]) = rec.temp_range_k
                && (!min_k.is_finite() || !max_k.is_finite() || min_k <= 0.0 || min_k > max_k)
            {
                return Err(BarrierValidationError::InvalidTemperatureBounds {
                    id: rec.id.clone(),
                    min_k,
                    max_k,
                });
            }

            let residual =
                (rec.forward_barrier_ev - rec.reverse_barrier_ev - rec.delta_energy_ev).abs();
            if residual > self.tolerance_ev {
                return Err(BarrierValidationError::DetailedBalanceViolation {
                    id: rec.id.clone(),
                    forward_ev: rec.forward_barrier_ev,
                    reverse_ev: rec.reverse_barrier_ev,
                    delta_ev: rec.delta_energy_ev,
                    residual_ev: residual,
                    tolerance_ev: self.tolerance_ev,
                });
            }
        }
        Ok(())
    }

    pub fn validate_at_temperature(
        &self,
        temperature_k: f64,
    ) -> Result<(), BarrierValidationError> {
        if !temperature_k.is_finite() || temperature_k <= 0.0 {
            return Err(BarrierValidationError::InvalidTemperature(temperature_k));
        }
        self.validate()?;
        for rec in &self.records {
            if let Some([min_k, max_k]) = rec.temp_range_k
                && (temperature_k < min_k || temperature_k > max_k)
            {
                return Err(BarrierValidationError::TemperatureOutOfRange {
                    id: rec.id.clone(),
                    temperature_k,
                    min_k,
                    max_k,
                });
            }
        }
        Ok(())
    }

    pub fn content_sha256(&self) -> Result<String, serde_json::Error> {
        Ok(sha256_hex(&serde_json::to_vec(self)?))
    }
}
