# okrates

A kinetic Monte Carlo (kMC) framework in Rust, with Python bindings.

okrates aims to be a general, reusable kMC engine: a small, statistically tested core that samples events and advances time, with the physics supplied by interchangeable models. The goal is to make kMC models fast to build, easy to verify, and reproducible.

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

Python (in any virtualenv or conda environment):

```console
pip install maturin==1.16.0
maturin develop
python examples/python/vacancy_diffusivity.py
```

## License

[MIT](LICENSE)
