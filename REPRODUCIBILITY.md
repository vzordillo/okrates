# Reproducibility Contract

`okrates` defines three reproducibility tiers for kinetic Monte Carlo (kMC) simulations and records the active tier in `RunManifest` (`okrates-schema`).

## 1. Tiers

| Tier | Identifier | Guarantee | Scope & Requirements |
| --- | --- | --- | --- |
| **Bitwise** | `bitwise_same_crate_same_fp` | Identical event sequence, residence times, and final floating-point bits (`u64::to_bits`) | Fixed crate version, matching target architecture, Rust toolchain, and platform `libm` (`f64::ln` / `f64::exp`), IEEE-754 binary64 arithmetic without FMA contraction (`rustc` default), identical `CatalogBackend` (`direct` or `sum_tree`), identical `seed` and `StreamRole`, and Checkpoint v2 resume |
| **Algorithmic** | `algorithmic_cross_backend` | Identical Markov jump chain in distribution; equivalent physical observables within Monte Carlo sampling error | Switching between `DirectCatalog` ($O(M)$) and `SumTreeCatalog` ($O(k \log M)$), cross-platform `libm` builds, or altering Rayon thread counts across independent `run_ensemble` replicas |
| **Statistical** | `statistical_ensemble` | Consistent convergence to analytical/thermodynamic reference values within $z$-score gates | Ensemble transport estimators (`WindowedOls`, ratio-of-means + delete-1 jackknife correlation factor $f$), Boltzmann occupancy distributions, and Kawasaki canonical equilibration |

## 2. Deterministic RNG Substreams

`KmcRng` wraps `rand_chacha::ChaCha12Rng` (`seed_from_u64(base_seed)`), partitioning its $2^{64}$ stream selector across `(replica_id, StreamRole)` via SplitMix64 mixing:

- `StreamRole::Dynamics` (`0`) — kMC event selection ($u_1$) and exponential waiting times ($-\ln u_2 / R_{\text{tot}}$), with $u \in (0, 1)$ strictly open.
- `StreamRole::InitialState` (`1`) — random alloy/defect placement (`random_state_with_rng`).
- `StreamRole::Equilibration` (`2`) — Kawasaki Metropolis swap proposals and acceptance draws (`KawasakiAnnealer`).
- `StreamRole::Diagnostic` (`3`) — auxiliary sampling or bootstrap/jackknife checks.

Because `replica_id` and `StreamRole` map to orthogonal `ChaCha12` streams rather than `seed + offset` arithmetic, any `base_seed` in `0..=u64::MAX` is valid without overflow or inter-role stream collision, and Rayon scheduling order never affects per-replica results.

## 3. Clock & Checkpoint v2

- **Compensated Clock**: `Clock` accumulates residence times using error-free Knuth TwoSum (`time + compensation`), preventing deep-run $\Delta t \ll t$ mantissa truncation, and returns `ClockError::Stagnation` if `time` fails to advance at finite `dt`.
- **Checkpoint Schema v2**: Binary checkpoints (`OKRATES\0`, `SCHEMA_VERSION = 2`) persist model `MODEL_ID`, encoded model parameters, `steps`, `time`, TwoSum `compensation` bits, serialized `ChaCha12` RNG word index/seed, and model state. On the same target/toolchain/`libm` environment, resuming from a v2 checkpoint reproduces the uninterrupted trajectory bit-for-bit.

## 4. Provenance Manifests (`okrates-schema`)

`RunManifest` (schema v1) records:

- `crate_version` and build-time `git_sha`
- `model_id` and `params_sha256` (SHA-256 over canonical `Checkpointable::encode_params` bytes)
- `barrier_set_sha256` (SHA-256 over canonical `BarrierSet` JSON for tabulated DFT/NEB runs)
- `parent_checkpoint_sha256` (lineage link when continuing from a saved checkpoint)
- `catalog_backend`, `rng` policy, thermodynamic `conditions`, and `estimator` window/fit specification
