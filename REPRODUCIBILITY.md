# Reproducibility Contract

`okrates` defines three reproducibility tiers for kinetic Monte Carlo (kMC) simulations and records the active tier in `RunManifest` (`okrates-schema`).

## 1. Tiers

| Tier | Identifier | Guarantee | Scope & Requirements |
| --- | --- | --- | --- |
| **Bitwise** | `bitwise_same_crate_same_fp` | Identical event sequence, residence times, and final floating-point bits (`u64::to_bits`) | Fixed crate version, matching target architecture, Rust toolchain, and platform `libm` (`f64::ln` / `f64::exp`), IEEE-754 binary64 arithmetic without FMA contraction (`rustc` default), identical `CatalogBackend` (`direct` or `sum_tree`), identical `seed` and `StreamRole`, and Checkpoint v2 resume |
| **Algorithmic** | `algorithmic_cross_backend` | Identical Markov jump chain in distribution; equivalent physical observables within Monte Carlo sampling error | Switching between `DirectCatalog` ($O(M)$) and `SumTreeCatalog` ($O(k \log M)$), cross-platform `libm` builds, or altering Rayon thread counts across independent `run_ensemble` replicas |
| **Statistical** | `statistical_ensemble` | Consistent convergence to analytical/thermodynamic reference values within $z$-score gates | Ensemble transport estimators (`WindowedOls`, ratio-of-means + delete-1 jackknife correlation factor $f$), Boltzmann occupancy distributions, and Kawasaki canonical equilibration |

## 2. Deterministic RNG Substreams

`KmcRng::for_replica(base_seed, replica_id, role)` constructs `rand_chacha::ChaCha12Rng` (stream `0`) from a 32-byte little-endian key (`key[0..8] = base_seed`, `key[8..16] = replica_id`, `key[16..24] = role.tag()`, `key[24..32] = 0`):

- `StreamRole::Dynamics` (`1`) — kMC event selection ($u_1$) and exponential waiting times ($-\ln u_2 / R_{\text{tot}}$), with $u \in (0, 1)$ strictly open.
- `StreamRole::InitialState` (`2`) — random alloy/defect placement (`random_state_with_rng`).
- `StreamRole::Equilibration` (`3`) — Kawasaki Metropolis swap proposals and acceptance draws (`KawasakiAnnealer`).
- `StreamRole::Custom(id)` (`0x1_0000 | u64::from(id)`) — caller-defined auxiliary domain-separated substreams.

Because `base_seed`, `replica_id`, and `StreamRole` occupy disjoint little-endian slices of the 256-bit `ChaCha12` key rather than `seed + offset` arithmetic, any `base_seed` in `0..=u64::MAX` is valid without overflow or inter-role key collision, and Rayon scheduling order never affects per-replica results.

## 3. Clock & Checkpoint v2

- **Compensated Clock**: `Clock` accumulates residence times using error-free Knuth TwoSum (`time + compensation`), preventing deep-run $\Delta t \ll t$ mantissa truncation, and returns `ClockError::Stagnation` if a finite `dt > 0` leaves both `time` and `compensation` unchanged.
- **Checkpoint Schema v2**: Binary checkpoints (`OKRCKPT\0`, `SCHEMA_VERSION = 2`) persist model `MODEL_ID`, encoded model parameters, `steps`, `time`, TwoSum `compensation` bits, serialized `ChaCha12` RNG state (32-byte `seed` key, `stream`, and `word_pos`), and model state. On the same target/toolchain/`libm` environment, resuming from a v2 checkpoint reproduces the uninterrupted trajectory bit-for-bit.

## 4. Provenance Manifests (`okrates-schema`)

`RunManifest` (schema v1) records:

- `crate_version` and build-time `git_sha`
- `model_id` and `params_sha256` (SHA-256 over canonical `Checkpointable::encode_params` bytes)
- `barrier_set_sha256` (SHA-256 over canonical `BarrierSet` JSON for tabulated DFT/NEB runs)
- `parent_checkpoint_sha256` (lineage link when continuing from a saved checkpoint)
- `catalog_backend`, `rng` policy, thermodynamic `conditions`, and `estimator` window/fit specification
