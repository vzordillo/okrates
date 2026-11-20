//! Tests for RunManifest v1, BarrierRecord detailed-balance validation, and DFT JSON adapters.

use std::collections::BTreeMap;

use okrates_schema::{
    AdapterError, AiidaQeNebExport, BarrierJsonError, BarrierRecord, BarrierSet,
    BarrierValidationError, Conditions, EstimatorSpec, ManifestError, RngPolicy, RunManifest,
    export_aiida_qe_json, export_pymatgen_vasp_json, import_aiida_qe_json,
    import_pymatgen_vasp_json, sha256_hex,
};

#[test]
fn run_manifest_round_trips_and_tracks_checkpoint_lineage() {
    let mut comp = BTreeMap::new();
    comp.insert("Fe".to_string(), 391);
    comp.insert("Cr".to_string(), 40);
    comp.insert("Vacancy".to_string(), 1);
    let parent_bytes = b"OKRCKPT\0...";
    let mut manifest = RunManifest::new(
        "okrates.fecr_martinez2012",
        b"encoded-params-bytes",
        "sum_tree",
        RngPolicy::chacha12_substreams(42),
        Conditions {
            cells: [6, 6, 6],
            lattice_parameter: 2.8553,
            temperature_k: 773.0,
            composition: comp,
        },
    )
    .with_parent_checkpoint(parent_bytes);
    manifest.estimator = Some(EstimatorSpec {
        fit_method: "windowed_ols".to_string(),
        t_equil: 1e-6,
        t_end: 5e-6,
        sample_count: 10,
        replicas: 16,
        anneal_sweeps: 20,
    });

    let json = manifest.to_json_pretty().unwrap();
    let decoded = RunManifest::from_json(&json).unwrap();
    assert_eq!(manifest, decoded);
    assert!(decoded.build_target().is_some_and(|t| !t.is_empty()));
    assert_eq!(manifest.reproducibility_tier, "bitwise_same_crate_same_fp");
    assert_eq!(manifest.params_sha256, sha256_hex(b"encoded-params-bytes"));
    assert_eq!(
        manifest.parent_checkpoint_sha256.as_deref(),
        Some(sha256_hex(parent_bytes).as_str())
    );
    assert_eq!(
        manifest.manifest_sha256().unwrap(),
        decoded.manifest_sha256().unwrap()
    );

    // Legacy manifests omitting optional `git_dirty` and `build_target` remain valid.
    let mut legacy_val: serde_json::Value = serde_json::from_str(&json).unwrap();
    legacy_val.as_object_mut().unwrap().remove("git_dirty");
    legacy_val.as_object_mut().unwrap().remove("build_target");
    let legacy = RunManifest::from_json(&legacy_val.to_string()).unwrap();
    assert_eq!(legacy.git_dirty(), None);
    assert_eq!(legacy.build_target(), None);

    let bad_ver = json.replace("\"schema_version\": 1", "\"schema_version\": 99");
    assert!(matches!(
        RunManifest::from_json(&bad_ver),
        Err(ManifestError::UnsupportedVersion(99))
    ));
}

#[test]
fn barrier_set_enforces_detailed_balance_positivity_temperature_and_float_roundtrip() {
    let sub_ulp = f64::from_bits(1.0_f64.to_bits() - 1);
    let valid_record = BarrierRecord {
        id: "fe_nn1_cr2".to_string(),
        migrating_species: "Fe".to_string(),
        env_key: "cr_saddle=2".to_string(),
        forward_barrier_ev: sub_ulp,
        reverse_barrier_ev: sub_ulp - 0.05,
        delta_energy_ev: 0.05,
        attempt_frequency_hz: 1e13,
        reverse_attempt_frequency_hz: None,
        barrier_std_ev: Some(0.015),
        temp_range_k: Some([500.0, 1100.0]),
        provenance: None,
    };
    let set = BarrierSet::new("fe-cr-test", 1e-12, vec![valid_record.clone()]).unwrap();
    let json = serde_json::to_string_pretty(&set).unwrap();
    let reloaded: BarrierSet = serde_json::from_str(&json).unwrap();
    assert_eq!(
        reloaded.records[0].forward_barrier_ev.to_bits(),
        sub_ulp.to_bits()
    );
    assert_eq!(
        set.content_sha256().unwrap(),
        reloaded.content_sha256().unwrap()
    );

    assert!(set.validate_at_temperature(773.0).is_ok());
    assert!(matches!(
        set.validate_at_temperature(300.0),
        Err(BarrierValidationError::TemperatureOutOfRange { .. })
    ));
    for bad_t in [f64::NAN, f64::INFINITY, 0.0, -50.0] {
        assert!(matches!(
            set.validate_at_temperature(bad_t),
            Err(BarrierValidationError::InvalidTemperature(_))
        ));
    }

    // Break Ef - Er == dE by 2 meV.
    let mut broken = valid_record.clone();
    broken.reverse_barrier_ev -= 0.002;
    assert!(matches!(
        BarrierSet::new("broken", 1e-4, vec![broken]),
        Err(BarrierValidationError::DetailedBalanceViolation { .. })
    ));

    // Negative barrier rejected.
    let mut neg = valid_record;
    neg.forward_barrier_ev = -0.02;
    neg.reverse_barrier_ev = 0.03;
    neg.delta_energy_ev = -0.05;
    assert!(matches!(
        BarrierSet::new("neg", 1e-4, vec![neg]),
        Err(BarrierValidationError::NegativeBarrier { .. })
    ));
}

#[test]
fn pymatgen_vasp_and_aiida_qe_adapters_round_trip_and_reject_unphysical_paths() {
    let vasp_json = r#"{
        "name": "vasp-neb-fecr",
        "tolerance_ev": 1e-9,
        "tasks": [
            {
                "task_id": "mp-neb-01",
                "migrating_specie": "Fe",
                "env_key": "cr=0",
                "image_energies_ev": [-354.120, -353.810, -353.470, -353.790, -354.100],
                "v0_hz": 1.0e13,
                "functional": "PBE",
                "temp_range_k": [500.0, 1100.0]
            },
            {
                "task_id": "mp-neb-02",
                "migrating_specie": "Cr",
                "env_key": "cr=0",
                "image_energies_ev": [-352.500, -352.100, -351.960, -352.200, -352.540],
                "v0_hz": 1.05e13,
                "functional": "PBE"
            }
        ]
    }"#;
    let mut set = import_pymatgen_vasp_json(vasp_json).unwrap();
    assert_eq!(set.records.len(), 2);
    assert!((set.records[0].forward_barrier_ev - 0.65).abs() < 1e-12);
    assert!((set.records[0].delta_energy_ev - 0.02).abs() < 1e-12);
    assert_eq!(set.records[0].temp_range_k, Some([500.0, 1100.0]));
    assert_eq!(set.records[1].temp_range_k, None);

    let reexported = export_pymatgen_vasp_json(&set).unwrap();
    assert!(!reexported.contains("\"temp_range_k\": null"));
    let reimported = import_pymatgen_vasp_json(&reexported).unwrap();
    assert_eq!(reimported.records.len(), 2);
    assert_eq!(reimported.records[0].temp_range_k, Some([500.0, 1100.0]));
    assert_eq!(reimported.records[1].temp_range_k, None);

    // Interior NEB image lower than endpoint produces negative barrier and is rejected.
    let bad_vasp = r#"{
        "name": "bad",
        "tolerance_ev": 1e-6,
        "tasks": [{
            "task_id": "bad-saddle",
            "migrating_specie": "Fe",
            "env_key": "cr=0",
            "image_energies_ev": [-10.0, -10.5, -10.0],
            "v0_hz": 1e13,
            "functional": "PBE"
        }]
    }"#;
    assert!(matches!(
        import_pymatgen_vasp_json(bad_vasp),
        Err(AdapterError::Validation(
            BarrierValidationError::NegativeBarrier { .. }
        ))
    ));

    let mut bad_vasp_temp: serde_json::Value = serde_json::from_str(&reexported).unwrap();
    bad_vasp_temp["tasks"][0]["temp_range_k"] = serde_json::json!([900.0, 400.0]);
    assert!(matches!(
        import_pymatgen_vasp_json(&bad_vasp_temp.to_string()),
        Err(AdapterError::Validation(
            BarrierValidationError::InvalidTemperatureBounds {
                id,
                min_k: 900.0,
                max_k: 400.0,
            }
        )) if id == "mp-neb-01"
    ));

    set.records[1].barrier_std_ev = Some(0.02);
    let vasp_with_std = export_pymatgen_vasp_json(&set).unwrap();
    assert!(!vasp_with_std.contains("\"barrier_std_ev\": null"));
    assert_eq!(
        import_pymatgen_vasp_json(&vasp_with_std).unwrap().records[1].barrier_std_ev,
        Some(0.02)
    );

    let aiida_json = export_aiida_qe_json(&set, "3e4a-uuid").unwrap();
    assert!(!aiida_json.contains("\"temp_range_k\": null"));
    let raw_aiida: AiidaQeNebExport = serde_json::from_str(&aiida_json).unwrap();
    assert!((raw_aiida.paths[1].attempt_frequency_thz - 10.5).abs() < 1e-12);
    assert_eq!(raw_aiida.paths[1].barrier_uncertainty_ev, Some(0.02));
    assert_eq!(raw_aiida.paths[0].temp_range_k, Some([500.0, 1100.0]));
    assert_eq!(raw_aiida.paths[1].temp_range_k, None);

    let aiida_set = import_aiida_qe_json(&aiida_json).unwrap();
    assert_eq!(aiida_set.records.len(), 2);
    assert!(
        (aiida_set.records[1].attempt_frequency_hz - 1.05e13).abs() < 1.0,
        "got {}",
        aiida_set.records[1].attempt_frequency_hz
    );
    assert_eq!(aiida_set.records[1].barrier_std_ev, Some(0.02));
    assert_eq!(aiida_set.records[0].temp_range_k, Some([500.0, 1100.0]));
    assert_eq!(aiida_set.records[1].temp_range_k, None);
    assert!(aiida_set.validate_at_temperature(773.0).is_ok());
    assert!(matches!(
        aiida_set.validate_at_temperature(300.0),
        Err(BarrierValidationError::TemperatureOutOfRange {
            id,
            temperature_k: 300.0,
            min_k: 500.0,
            max_k: 1100.0,
        }) if id == "mp-neb-01"
    ));

    let mut bad_aiida_temp: serde_json::Value = serde_json::from_str(&aiida_json).unwrap();
    bad_aiida_temp["paths"][0]["temp_range_k"] = serde_json::json!([0.0, 500.0]);
    assert!(matches!(
        import_aiida_qe_json(&bad_aiida_temp.to_string()),
        Err(AdapterError::Validation(
            BarrierValidationError::InvalidTemperatureBounds {
                id,
                min_k: 0.0,
                max_k: 500.0,
            }
        )) if id == "mp-neb-01"
    ));

    let bad_aiida = r#"{
        "workchain_uuid": "uuid-1",
        "name": "broken-qe",
        "dft_functional": "PBEsol",
        "tolerance_ev": 1e-4,
        "paths": [{
            "label": "p1",
            "species": "Fe",
            "env_key": "cr=1",
            "forward_barrier_ev": 0.70,
            "reverse_barrier_ev": 0.70,
            "reaction_energy_ev": 0.15,
            "attempt_frequency_thz": 10.0,
            "barrier_uncertainty_ev": 0.02
        }]
    }"#;
    assert!(matches!(
        import_aiida_qe_json(bad_aiida),
        Err(AdapterError::Validation(
            BarrierValidationError::DetailedBalanceViolation { .. }
        ))
    ));
}

#[test]
fn lossless_barrier_json_and_equilibrium_rate_ratio() {
    let mut rec = BarrierRecord {
        id: "db_check".to_string(),
        migrating_species: "Cr".to_string(),
        env_key: "pair=1".to_string(),
        forward_barrier_ev: 0.65,
        reverse_barrier_ev: 0.55,
        delta_energy_ev: 0.10,
        attempt_frequency_hz: 1.0e13,
        reverse_attempt_frequency_hz: Some(2.0e13),
        barrier_std_ev: Some(0.01),
        temp_range_k: Some([500.0, 1000.0]),
        provenance: None,
    };

    // Asymmetric reverse attempt frequency keeps rate ratio = nu_fwd / nu_rev when energies are reciprocal.
    let asym = rec.equilibrium_rate_ratio(750.0).unwrap();
    assert!((asym - 0.5).abs() < 1e-14, "got {asym}");

    rec.reverse_attempt_frequency_hz = None;
    let exact = rec.equilibrium_rate_ratio(750.0).unwrap();
    assert!((exact - 1.0).abs() < 1e-14, "got {exact}");

    // Combined log-space prefactor and barrier factor stays finite when individual factors would overflow.
    let stiff_rec = BarrierRecord {
        id: "log_space_overflow_guard".to_string(),
        migrating_species: "Cr".to_string(),
        env_key: "stiff=1".to_string(),
        forward_barrier_ev: 0.0,
        reverse_barrier_ev: 0.612,
        delta_energy_ev: 0.0,
        attempt_frequency_hz: 1.0e12,
        reverse_attempt_frequency_hz: Some(1.0e13),
        barrier_std_ev: None,
        temp_range_k: None,
        provenance: None,
    };
    let stiff_ratio = stiff_rec.equilibrium_rate_ratio(10.0).unwrap();
    assert!(
        stiff_ratio.is_finite() && stiff_ratio > 1.0e300,
        "got {stiff_ratio}"
    );

    // Export preserves reverse_attempt_frequency_hz losslessly; validation runs on import and export.
    rec.reverse_attempt_frequency_hz = Some(2.0e13);
    let set = BarrierSet::new("lossless-test", 1e-12, vec![rec.clone()]).unwrap();
    let json = set.to_json_pretty().unwrap();
    let roundtrip = BarrierSet::from_json_str(&json).unwrap();
    assert_eq!(roundtrip, set);
    assert_eq!(
        roundtrip.records[0].reverse_attempt_frequency_hz,
        Some(2.0e13)
    );

    let broken_json = json.replace(
        "\"reverse_barrier_ev\": 0.55",
        "\"reverse_barrier_ev\": -0.1",
    );
    assert!(matches!(
        BarrierSet::from_json_str(&broken_json),
        Err(BarrierJsonError::Validation(
            BarrierValidationError::NegativeBarrier { .. }
        ))
    ));
    assert!(matches!(
        rec.equilibrium_rate_ratio(0.0),
        Err(BarrierValidationError::InvalidTemperature(0.0))
    ));
}
