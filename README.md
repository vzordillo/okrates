# okrates — Kinetic Monte Carlo Simulation in Rust and Python

[![status](https://img.shields.io/badge/status-experimental-orange)](https://github.com/vzordillo/okrates)
[![rust](https://github.com/vzordillo/okrates/actions/workflows/rust.yml/badge.svg?branch=main)](https://github.com/vzordillo/okrates/actions/workflows/rust.yml)
[![python](https://github.com/vzordillo/okrates/actions/workflows/python.yml/badge.svg?branch=main)](https://github.com/vzordillo/okrates/actions/workflows/python.yml)
[![license](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

`okrates` is a residence-time (rejection-free) kinetic Monte Carlo (kMC) framework written in Rust, with typed Python bindings. It is built for atomistic transport in alloys: vacancy- and interstitial-mediated diffusion in BCC Fe–Cr today, and radiation-induced segregation (RIS) as the scientific goal.

<p align="center"><img src="assets/fecr-snapshot.gif" width="560" alt="Fe-20%Cr box: Cr atoms cluster during a Monte Carlo anneal, then a vacancy diffuses through the alloy"></p>

## About this repository

`okrates` is an experiment in fully agentic scientific software development. The code, tests and reviews are produced by AI coding agents working under the maintainer's direction:
- [Google Antigravity](https://antigravity.google/use-cases/science/) with Google's frontier Gemini models, through Google's free access offer for scientists;
- Anthropic Claude models through [Claude Code](https://claude.com/claude-code).

Every change goes through a pull request with CI, an automated review, and a test that is checked to fail when the targeted bug is injected.

## What it does

- **Engine.** A model supplies its states, transitions and rates; the engine selects events, advances time and keeps the event catalog consistent. Two catalog backends give the same physics at different cost: a direct catalog (O(M) per step) and an incremental sum tree (O(log M) per update), about 150× faster at 32,768 events.
- **Reproducibility.** Seeded ChaCha12 random streams, separated by replica and purpose, and a compensated (TwoSum) clock. Checkpoints restore a run bit for bit on the same platform.
- **Models.** BCC vacancy diffusion, Fe–Cr transport with the Martínez et al. (2012) energetics or tabulated DFT barriers, and an Fe–Cr RIS model with ⟨110⟩ self-interstitial dumbbells, Frenkel-pair creation, recombination and a planar sink.
- **Analysis.** Mean-squared-displacement transport estimators, tracer and correlation factors with jackknife errors, Kawasaki annealing with Warren–Cowley short-range order, and extended-XYZ snapshots for visualisation (e.g. OVITO).
- **Provenance.** Run manifests with SHA-256 hashes of parameters, barrier tables and checkpoints, and validation of DFT barrier sets (detailed balance, temperature ranges).
- **Python.** Frozen configuration and result classes, numpy arrays out, and Python exceptions instead of crashes.

## Workspace

| Crate | Contents |
|---|---|
| `okrates-core` | Engine (`TransitionTopology`, `RateModel`, `CatalogBackend` with `DirectCatalog` and `SumTreeCatalog`), TwoSum `Clock`, `sample_trajectory`, `StreamRole` random substreams and `run_ensemble`, checkpoints (schema v2) |
| `okrates-schema` | `RunManifest` v1, `BarrierSet` validation, JSON adapters in an okrates-defined layout for barrier data exported from pymatgen/atomate2 (VASP NEB) and AiiDA-QuantumESPRESSO |
| `okrates-models` | BCC lattice, single- and multi-vacancy models, Fe–Cr (`PairEnergetics`, `Martinez2012`, `TabulatedEnergetics`), Fe–Cr RIS (`fecr_ris`), annealing and transport estimators |
| `okrates-python` | PyO3 extension behind the `okrates` Python package |

## Validation status

| Model | Check | Result |
|---|---|---|
| Single vacancy, BCC | D_v against the analytic r·a² | agrees within statistical error |
| Fe–Cr, Martínez et al. (2012) | Dilute D*_Cr/D*_Fe against the paper's Fig. 4 model lines | 1.86–2.39 against 2.03–2.31, within 2.1 standard errors |
| Fe–Cr, Martínez et al. (2012) | Self-diffusion correlation factor in Fe | 0.729–0.738 (finite-cell value 0.7314 ± 0.0025) |
| Fe–Cr RIS | Interstitial parameters against Senninger et al. (2016) binding energies and barriers | fitted; transport and RIS replication in progress |

The interstitial parameters are a reconstruction: the fitted values are not published, so they were refitted to the paper's tables and figures.

## Getting started

Rust 1.100.0 is pinned in `rust-toolchain.toml` (minimum supported 1.88).

```console
cargo test
cargo run --example random_walk -- 42 5
```

Python ≥ 3.10, in any virtualenv or conda environment (`numpy>=2.2` is installed as a dependency):

```console
pip install maturin==1.16.0
maturin develop
python examples/python/vacancy_diffusivity.py
```

### Python: Fe–Cr transport

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

### Rust examples

Run with `cargo run --release -p <crate> --example <name> -- <args>`.

| Example | Crate | Arguments | Purpose |
|---|---|---|---|
| `random_walk` | okrates-core | `[SEED] [STEPS]` | Minimal engine demo |
| `catalog_scaling` | okrates-core | — | Direct vs sum-tree step cost against catalog size |
| `vacancy_trajectory` | okrates-models | `SEED STEPS` | One vacancy trajectory |
| `vacancy_msd` | okrates-models | `[RUNS] [SEED_BASE]` | Vacancy diffusivity from the mean-squared displacement |
| `multi_vacancy_scaling` | okrates-models | — | Many-vacancy step cost |
| `fecr_diffusion` | okrates-models | env `RUNS`, `RUNS_F`, `ANNEAL_SWEEPS` | Fe–Cr tracer diffusion and correlation factor |
| `fecr_snapshot` | okrates-models | `cr_count seed anneal_sweeps sweeps_per_frame jumps jumps_per_frame temperature_k` | Extended-XYZ frames of an anneal and vacancy walk |
| `fecr_ris_pilot` | okrates-models | `--temperature K --k0 dpa_per_s --dpa-end x --capacity P --seed s [--cells nx ny nz]` | Fe–Cr RIS run with a sink plane |

## Testing

Tests are split into tiers. The fast tier runs in a few seconds; the full tier adds statistical, equivalence and end-to-end tests:

```text
cargo test --locked                          # fast tier
cargo test --locked -- --include-ignored     # full tier
pytest -q python/tests                       # Python, after maturin develop
```

Regression tests pin results bit for bit. Where platform maths libraries differ in the last bit, values are pinned per target (architecture and operating system).

## Roadmap

1. Interstitial transport in Fe–Cr against Senninger et al. (2016): diffusion coefficients and Onsager-coefficient ratios.
2. RIS at grain boundaries: Cr enrichment and depletion against the same paper, then a map over 573–773 K and 1e-7 to 1e-3 dpa/s.
3. An offline DFT pipeline (Quantum ESPRESSO first) that produces parameter sets with uncertainties.

## References

- E. Martínez, O. Senninger, C.-C. Fu, F. Soisson, "Decomposition kinetics of Fe–Cr solid solutions during thermal aging", Phys. Rev. B 86, 224109 (2012). [doi:10.1103/PhysRevB.86.224109](https://doi.org/10.1103/PhysRevB.86.224109)
- O. Senninger et al., "Modeling radiation induced segregation in iron–chromium alloys", Acta Mater. 103, 1–11 (2016). [doi:10.1016/j.actamat.2015.09.058](https://doi.org/10.1016/j.actamat.2015.09.058)

## License

[MIT](LICENSE)
