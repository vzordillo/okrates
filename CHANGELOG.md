# Changelog

All notable changes to `okrates` are documented in this file.

## [0.1.0] - 2026-11-18

### Added
- **`okrates-core` 3-seam architecture**: `TransitionTopology` (`apply` + `undo`), `EnumeratedTopology`, `SlottedTopology`, `RateModel<T>`, `CoupledModel<T, R>`, and unified transactional `Engine<M, C: CatalogBackend<M>>` (`DirectCatalog`, `SumTreeCatalog`).
- **Compensated time integration**: Error-free TwoSum `Clock` preserving sub-ULP residence-time increments in long trajectories and surfacing `ClockError::Stagnation` instead of silently stalling.
- **Lazy piecewise-constant trajectory sampling**: `SampleGrid` (`windowed` and `from_origin`) and `sample_trajectory` evaluating observables only when crossing sample times without mutating or panicking on error.
- **Domain-separated RNG streams**: `StreamRole` (`Dynamics`, `InitialState`, `Equilibration`, `Diagnostic`), `KmcRng::for_replica`, `ReplicaContext`, and `run_ensemble` supporting every `u64` base seed (`0..=u64::MAX`).
- **`okrates-schema` crate**: `RunManifest` v1 (SHA-256 parameter, barrier-table, and checkpoint lineage + build-time git SHA), `BarrierRecord` / `BarrierSet` forward-reverse detailed-balance validator, and JSON adapters for `pymatgen`/`atomate2` (VASP NEB) and `AiiDA-QuantumESPRESSO`.
- **`okrates-models` physics & estimators**:
  - `TabulatedEnergetics` for DFT/NEB Fe–Cr barrier sets with automatic reverse-environment detailed balance.
  - `Checkpointable` and `ReversibleModel` across single-vacancy, multi-vacancy, and all three `FeCrModel<E>` energetics (`PairEnergetics`, `Martinez2012`, `TabulatedEnergetics`), plus `IncrementalModel` (`SumTreeCatalog`) for `FeCrModel<PairEnergetics>`.
  - `BarrierFloorPolicy` (`SymmetricFloor`, `Unfloored`, `Strict`) on `Martinez2012`.
  - Stateful `KawasakiAnnealer`, Warren–Cowley 1NN short-range order `warren_cowley_alpha1`, and tail energy-drift diagnostics.
  - `TransportConfig` (`WindowedOls` burn-in window `[t_equil, t_end]` and `OriginSlope`), split-window drift diagnostic `window_drift_ratio`, and unbiased ratio-of-means + delete-1 jackknife standard error (`jackknife_ratio_mean_se`) for the correlation factor $f$.
- **Typed Python API (`okrates`)**: Frozen configuration and result dataclasses (`LatticeConfig`, `PairEnergeticsConfig`, `Martinez2012Config`, `BarrierTableConfig`, `AnnealConfig`, `SamplingConfig`, `DiffusionResult`, `AnnealResult`, `RunManifest`), `run_fecr_diffusion`, `run_anneal`, `validate_barrier_set`, `import_pymatgen_vasp_barriers`, `import_aiida_qe_barriers`, and PEP 561 `.pyi` stubs.
- **Repository standards**: `REPRODUCIBILITY.md` and `.coderabbit.yaml`.

### Changed (Breaking)
- **Checkpoint schema bumped to v2 (`SCHEMA_VERSION = 2`)**: Persists TwoSum clock compensation bits alongside `time`. v1 checkpoints fail cleanly with `CheckpointError::UnsupportedSchema(1)`.
- **Ensemble RNG seeding**: `run_transport` / `run_composition` now draw from `ChaCha12` substreams keyed by `(replica_id, StreamRole)` instead of `seed + 1_000_000_000` arithmetic offsets.
- **Correlation factor $f$ estimator**: `summarize` now computes $f = N_{\mathrm{atoms}} \overline{D^*_{\mathrm{all}}} / \overline{D_v}$ via `jackknife_ratio_mean_se` instead of averaging noisy per-replica ratios $\overline{D^*_r / D_{v,r}}$ (eliminating small-$t$ Jensen upward bias).
- **Fallible lazy `sample_run` & Python error mapping**: `sample_run` returns `Result<..., TransportError>` instead of panicking; Python bindings map invalid inputs to `ValueError` and runtime simulation failures to `RuntimeError` with zero `PanicException` paths.

## [0.0.1] - Initial Prototype
- Serial direct-catalog residence-time kMC engine, `SumTreeCatalog` incremental prototype, Checkpoint v1, BCC single/multi-vacancy diffusion, Fe–Cr pair and Martínez (2012) models, Kawasaki anneal function, and initial PyO3 bindings.
