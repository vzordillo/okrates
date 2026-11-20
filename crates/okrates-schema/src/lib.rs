//! Provenance manifests, thermodynamic barrier schemas, and DFT workflow JSON adapters.

#![forbid(unsafe_code)]

pub mod adapters;
pub mod barrier;
pub mod digest;
pub mod manifest;

pub use adapters::{
    AdapterError, AiidaQeNebExport, AiidaQePath, PymatgenNebCollection, PymatgenNebTask,
    export_aiida_qe_json, export_pymatgen_vasp_json, import_aiida_qe_json,
    import_pymatgen_vasp_json,
};
pub use barrier::{
    BARRIER_SCHEMA_VERSION, BarrierJsonError, BarrierProvenance, BarrierRecord, BarrierSet,
    BarrierValidationError,
};
pub use digest::sha256_hex;
pub use manifest::{
    Conditions, EstimatorSpec, MANIFEST_SCHEMA_VERSION, ManifestError, RngPolicy, RunManifest,
};
