//! RunManifest v1 provenance contract for reproducible KMC simulations.

use std::collections::BTreeMap;
use std::error::Error;
use std::fmt;

use serde::{Deserialize, Serialize};

use crate::digest::sha256_hex;

pub const MANIFEST_SCHEMA_VERSION: u32 = 1;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RngPolicy {
    pub algorithm: String,
    pub base_seed: u64,
    pub stream_policy: String,
}

impl RngPolicy {
    pub fn chacha12_substreams(base_seed: u64) -> Self {
        Self {
            algorithm: "ChaCha12Rng".to_string(),
            base_seed,
            stream_policy: "key[0..8]=seed,key[8..16]=replica,key[16..24]=role".to_string(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Conditions {
    pub cells: [usize; 3],
    pub lattice_parameter: f64,
    pub temperature_k: f64,
    pub composition: BTreeMap<String, usize>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EstimatorSpec {
    pub fit_method: String,
    pub t_equil: f64,
    pub t_end: f64,
    pub sample_count: usize,
    pub replicas: u64,
    pub anneal_sweeps: u64,
}

/// Self-describing provenance record tying outputs to code, parameters, RNG streams, and lineage.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RunManifest {
    pub schema_version: u32,
    pub crate_version: String,
    pub git_sha: String,
    pub reproducibility_tier: String,
    pub model_id: String,
    pub params_sha256: String,
    pub barrier_set_sha256: Option<String>,
    pub parent_checkpoint_sha256: Option<String>,
    pub catalog_backend: String,
    pub rng: RngPolicy,
    pub conditions: Conditions,
    pub estimator: Option<EstimatorSpec>,
}

#[derive(Debug)]
pub enum ManifestError {
    UnsupportedVersion(u32),
    Json(serde_json::Error),
}

impl fmt::Display for ManifestError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedVersion(v) => write!(
                f,
                "unsupported manifest schema version {v}, expected {MANIFEST_SCHEMA_VERSION}"
            ),
            Self::Json(err) => write!(f, "manifest JSON error: {err}"),
        }
    }
}

impl Error for ManifestError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::UnsupportedVersion(_) => None,
            Self::Json(err) => Some(err),
        }
    }
}

impl From<serde_json::Error> for ManifestError {
    fn from(err: serde_json::Error) -> Self {
        Self::Json(err)
    }
}

impl RunManifest {
    pub fn new(
        model_id: impl Into<String>,
        encoded_params: &[u8],
        catalog_backend: impl Into<String>,
        rng: RngPolicy,
        conditions: Conditions,
    ) -> Self {
        Self {
            schema_version: MANIFEST_SCHEMA_VERSION,
            crate_version: env!("CARGO_PKG_VERSION").to_string(),
            git_sha: env!("OKRATES_GIT_SHA").to_string(),
            reproducibility_tier: "bitwise_same_crate_same_fp".to_string(),
            model_id: model_id.into(),
            params_sha256: sha256_hex(encoded_params),
            barrier_set_sha256: None,
            parent_checkpoint_sha256: None,
            catalog_backend: catalog_backend.into(),
            rng,
            conditions,
            estimator: None,
        }
    }

    pub fn with_parent_checkpoint(mut self, checkpoint_bytes: &[u8]) -> Self {
        self.parent_checkpoint_sha256 = Some(sha256_hex(checkpoint_bytes));
        self
    }

    pub fn to_json_pretty(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    pub fn from_json(json: &str) -> Result<Self, ManifestError> {
        let parsed: Self = serde_json::from_str(json)?;
        if parsed.schema_version != MANIFEST_SCHEMA_VERSION {
            return Err(ManifestError::UnsupportedVersion(parsed.schema_version));
        }
        Ok(parsed)
    }

    pub fn manifest_sha256(&self) -> Result<String, serde_json::Error> {
        let canonical = serde_json::to_vec(self)?;
        Ok(sha256_hex(&canonical))
    }
}
