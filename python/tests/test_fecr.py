"""Rust/Python parity, smoke and error tests for okrates.fecr_diffusion."""

import math
import subprocess
from pathlib import Path

import numpy as np
import pytest

import okrates

ROOT = Path(__file__).resolve().parents[2]
# Must match the constants in crates/okrates-models/examples/fecr_observables.rs.
CELLS, NU, T, K_COUNT, LATTICE_PARAMETER = (6, 6, 6), 1e13, 800.0, 10, 1.0


def rust_observables(cr_count, seed, runs, anneal_sweeps,
                     example="fecr_observables"):
    out = subprocess.run(
        ["cargo", "run", "--locked", "-q", "-p", "okrates-models",
         "--example", example, "--", str(cr_count), str(seed),
         str(runs), str(anneal_sweeps)],
        cwd=ROOT, check=True, capture_output=True, text=True,
    ).stdout.splitlines()
    rows = {}
    for line in out:
        key, *values = line.split()
        rows[key] = tuple(float(v) for v in values)
    return rows


def assert_matches_rust(rust, result):
    scalar_keys = {
        "vacancy": "D_v", "tracer_fe": "D_Fe", "tracer_cr": "D_Cr",
        "tracer_all": "D_all", "f": "f",
    }
    for rust_key, name in scalar_keys.items():
        mean, se = rust[rust_key]
        assert result[name] == mean, name
        assert result[name + "_se"] == se, name + "_se"
    lambdas = {"lambda_fefe": (0, 0), "lambda_fecr": (0, 1),
               "lambda_crcr": (1, 1)}
    for rust_key, (i, j) in lambdas.items():
        mean, se = rust[rust_key]
        assert result["Lambda"][i, j] == mean, rust_key
        assert result["Lambda_se"][i, j] == se, rust_key + " se"
    assert result["Lambda"][1, 0] == rust["lambda_fecr"][0]
    assert result["Lambda_se"][1, 0] == rust["lambda_fecr"][1]
    assert result["mean_jumps"] == rust["mean_jumps"][0]


PARITY_CASES = [(22, 7, 2, 20), (43, 11, 3, 0)]


@pytest.mark.parametrize("cr_count, seed, runs, anneal_sweeps", PARITY_CASES)
def test_parity_with_rust_example(cr_count, seed, runs, anneal_sweeps):
    rust = rust_observables(cr_count, seed, runs, anneal_sweeps)
    result = okrates.fecr_diffusion(
        cells=CELLS, cr_count=cr_count, temperature_k=T, attempt_frequency=NU,
        t_end=rust["t_end"][0], k_count=K_COUNT,
        lattice_parameter_m=LATTICE_PARAMETER, seed=seed, runs=runs,
        anneal_sweeps=anneal_sweeps,
    )
    assert_matches_rust(rust, result)


@pytest.mark.parametrize("cr_count, seed, runs, anneal_sweeps", PARITY_CASES)
def test_martinez_parity_with_rust_example(
    cr_count, seed, runs, anneal_sweeps
):
    rust = rust_observables(
        cr_count, seed, runs, anneal_sweeps, "fecr_observables_martinez"
    )
    result = okrates.fecr_diffusion(
        cells=CELLS, cr_count=cr_count, temperature_k=T,
        t_end=rust["t_end"][0], k_count=K_COUNT,
        lattice_parameter_m=LATTICE_PARAMETER, seed=seed, runs=runs,
        anneal_sweeps=anneal_sweeps, energetics="martinez2012",
    )
    assert_matches_rust(rust, result)


SMALL = dict(
    cells=(3, 3, 3), cr_count=5, temperature_k=800.0, attempt_frequency=1e13,
    t_end=1e-8, k_count=4, lattice_parameter_m=2.8665e-10, seed=3, runs=2,
    anneal_sweeps=5,
)
FLOAT_KEYS = ["D_v", "D_v_se", "D_Fe", "D_Fe_se", "D_Cr", "D_Cr_se", "D_all",
              "D_all_se", "f", "f_se", "mean_jumps"]


def test_explicit_illustrative_energetics_match_defaults():
    energetics = okrates.fecr_illustrative_energetics()
    pair = energetics["pair_energies_ev"]
    assert isinstance(pair, np.ndarray)
    assert pair.dtype == np.float64
    assert pair.shape == (3, 3)
    assert len(energetics["kra_ev"]) == 2
    default = okrates.fecr_diffusion(**SMALL)
    explicit = okrates.fecr_diffusion(
        **SMALL, pair_energies_ev=pair, kra_ev=energetics["kra_ev"]
    )
    assert set(default) == set(FLOAT_KEYS) | {
        "Lambda", "Lambda_se", "barrier_floor_activations"
    }
    for key in FLOAT_KEYS:
        assert type(default[key]) is float
    for key in ("Lambda", "Lambda_se"):
        assert default[key].dtype == np.float64
        assert default[key].shape == (2, 2)
        assert default[key][0, 1] == default[key][1, 0]
    for key in default:
        np.testing.assert_array_equal(default[key], explicit[key])


def test_zero_cr_gives_nan_cr_diffusivity():
    result = okrates.fecr_diffusion(**{**SMALL, "cr_count": 0})
    assert math.isnan(result["D_Cr"])


@pytest.mark.parametrize("kwargs", [
    {"runs": 1},
    {"cells": (0, 6, 6)},
    {"cells": (6, 6, 6), "cr_count": 432},
    {"pair_energies_ev": np.zeros((3, 3))},
    {"kra_ev": (0.6, 0.55)},
    {"pair_energies_ev": np.zeros((3, 3)), "kra_ev": (-1.0, 0.55)},
    {"temperature_k": 0},
])
def test_invalid_arguments_raise_value_error(kwargs):
    with pytest.raises(ValueError):
        okrates.fecr_diffusion(**{**SMALL, **kwargs})


MARTINEZ = {k: v for k, v in SMALL.items() if k != "attempt_frequency"}
MARTINEZ["energetics"] = "martinez2012"


@pytest.mark.parametrize("kwargs", [
    {"attempt_frequency": 1e13},
    {"pair_energies_ev": np.zeros((3, 3)), "kra_ev": (0.6, 0.55)},
    {"cells": (2, 2, 2)},
    {"energetics": "bogus"},
])
def test_invalid_martinez_arguments_raise_value_error(kwargs):
    with pytest.raises(ValueError):
        okrates.fecr_diffusion(**{**MARTINEZ, **kwargs})


def test_illustrative_without_attempt_frequency_raises_value_error():
    kwargs = {k: v for k, v in SMALL.items() if k != "attempt_frequency"}
    with pytest.raises(ValueError):
        okrates.fecr_diffusion(**kwargs)


def test_negative_kra_and_negative_barrier_raise_value_error():
    pair = okrates.fecr_illustrative_energetics()["pair_energies_ev"]
    for kra in [(-1.0, 0.55), (0.0, 0.0)]:
        with pytest.raises(ValueError):
            okrates.fecr_diffusion(
                **SMALL, pair_energies_ev=pair, kra_ev=kra
            )


def test_negative_seed_raises_overflow_error():
    with pytest.raises(OverflowError):
        okrates.fecr_diffusion(**{**SMALL, "seed": -1})


def test_barrier_floor_activations_is_an_int():
    martinez = okrates.fecr_diffusion(**MARTINEZ)
    assert isinstance(martinez["barrier_floor_activations"], int)
    assert martinez["barrier_floor_activations"] >= 0
    assert okrates.fecr_diffusion(**SMALL)["barrier_floor_activations"] == 0
