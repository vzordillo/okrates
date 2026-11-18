# okrates — Kinetic Monte Carlo Simulation in Rust and Python

[![status](https://img.shields.io/badge/status-v0.1.0-blue)](https://github.com/vzordillo/okrates)
[![rust](https://github.com/vzordillo/okrates/actions/workflows/rust.yml/badge.svg?branch=main)](https://github.com/vzordillo/okrates/actions/workflows/rust.yml)
[![python](https://github.com/vzordillo/okrates/actions/workflows/python.yml/badge.svg?branch=main)](https://github.com/vzordillo/okrates/actions/workflows/python.yml)
[![license](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

`okrates` is a modular residence-time kinetic Monte Carlo (kMC) framework in Rust with typed Python bindings (`abi3-py310`). It separates discrete state topology, rate/barrier models, and catalog backends from statistical trajectory estimators and provenance manifests.

<p align="center"><img src="assets/fecr-snapshot.gif" width="560" alt="Fe-20%Cr box: Cr atoms cluster during a Monte Carlo anneal, then a vacancy diffuses through the alloy"></p>

## Workspace Crates

- **`okrates-core`** — Geometry-agnostic 3-seam kMC engine (`TransitionTopology`, `RateModel`, `CatalogBackend`: $O(M)$ `DirectCatalog` and $O(k \log M)$ `SumTreeCatalog`), error-free TwoSum `Clock`, lazy piecewise-constant `sample_trajectory`, `ChaCha12` `StreamRole` substreams (`run_ensemble`), and Checkpoint v2.
- **`okrates-schema`** — `RunManifest` v1 (SHA-256 parameter, barrier, and checkpoint lineage), `BarrierSet` forward/reverse detailed-balance validator, and JSON adapters for `pymatgen`/`atomate2` (VASP NEB) and `AiiDA-QuantumESPRESSO`.
- **`okrates-models`** — BCC single- and multi-vacancy diffusion, Fe–Cr vacancy-mediated transport (`PairEnergetics`, Martínez et al. 2012 `Martinez2012` with `BarrierFloorPolicy`, and DFT `TabulatedEnergetics`), stateful `KawasakiAnnealer` with Warren–Cowley 1NN SRO $\alpha_1$, windowed OLS transport estimators, and delete-1 jackknife correlation factor $f$.
- **`okrates-python`** — PyO3 extension backing `okrates` (`LatticeConfig`, `Martinez2012Config`, `BarrierTableConfig`, `SamplingConfig`, `DiffusionResult`, `AnnealResult`, `RunManifest`).

## Getting started

```console
cargo test
cargo run --example random_walk -- 42 5
```

Python (in any virtualenv or conda environment; `numpy>=2.2` is installed as a dependency):

```console
pip install maturin==1.16.0
maturin develop
python examples/python/vacancy_diffusivity.py
```

### Python Fe–Cr Transport Example

```python
import okrates

result = okrates.run_fecr_diffusion(
    lattice=okrates.LatticeConfig(cells=(4, 4, 4), lattice_parameter_m=2.8665e-10),
    energetics=okrates.Martinez2012Config(barrier_policy="symmetric_floor"),
    cr_count=25,
    temperature_k=800.0,
    sampling=okrates.SamplingConfig(t_end=5e-8, sample_count=20, fit_method="windowed_ols"),
    anneal=okrates.AnnealConfig(sweeps=50),
    runs=16,
    seed=2026,
)
print(f"D*_Cr = {result.D_Cr:.3e} +/- {result.D_Cr_se:.1e} m^2/s, f = {result.f:.4f} +/- {result.f_se:.4f}")
print(f"params_sha256 = {result.manifest.params_sha256}")
```

## License

[MIT](LICENSE)
