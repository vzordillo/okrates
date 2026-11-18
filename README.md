# okrates — A Rust Framework for Kinetic Monte Carlo Simulation

[![status](https://img.shields.io/badge/status-under%20active%20development-orange)](https://github.com/vzordillo/okrates)
[![rust](https://github.com/vzordillo/okrates/actions/workflows/rust.yml/badge.svg?branch=main)](https://github.com/vzordillo/okrates/actions/workflows/rust.yml)
[![python](https://github.com/vzordillo/okrates/actions/workflows/python.yml/badge.svg?branch=main)](https://github.com/vzordillo/okrates/actions/workflows/python.yml)
[![license](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

A kinetic Monte Carlo (kMC) framework in Rust, with Python bindings.

okrates aims to be a general, reusable kMC engine: a small, statistically tested core that samples events and advances time, with the physics supplied by interchangeable models. The goal is to make kMC models fast to build, easy to verify, and reproducible.

<p align="center"><img src="assets/fecr-snapshot.gif" width="560" alt="Fe-20%Cr box: Cr atoms cluster during a Monte Carlo anneal, then a vacancy diffuses through the alloy"></p>

## Planned

- A model-agnostic engine that is not tied to lattices, vacancies or a particular geometry.
- Efficient event selection with local catalog updates.
- Parallel replicas and parameter sweeps.
- Materials models, starting with vacancy-mediated diffusion in BCC Fe, then DFT-informed Fe–Cr alloys and radiation-induced segregation.
- A Python interface for building models, running simulations and analysing results.

## Status

Early development. The serial engine, a BCC vacancy diffusion model, checkpoint/restart and basic Python bindings are in place.

## Getting started

```console
cargo test
cargo run --example random_walk -- 42 5
```

Python (in any virtualenv or conda environment; numpy >= 2.2 is installed as a dependency):

```console
pip install maturin==1.16.0
maturin develop
python examples/python/vacancy_diffusivity.py
```

## License

[MIT](LICENSE)
