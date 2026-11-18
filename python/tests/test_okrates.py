"""Rust/Python parity and error tests for single-vacancy bindings."""

import subprocess
from pathlib import Path

import numpy as np
import pytest

import okrates

ROOT = Path(__file__).resolve().parents[2]
# Must match the constants in crates/okrates-models/examples/vacancy_trajectory.rs.
CELLS = (4, 4, 4)
NU, EM, T, A = 1e13, 0.65, 800.0, 2.8665e-10


def rust_trajectory(seed, steps):
    out = subprocess.run(
        ["cargo", "run", "--locked", "-q", "-p", "okrates-models",
         "--example", "vacancy_trajectory", "--", str(seed), str(steps)],
        cwd=ROOT, check=True, capture_output=True, text=True,
    ).stdout.splitlines()
    rate = float(out[0].split()[1])
    rows = [tuple(map(float, line.split())) for line in out[1:]]
    return rate, [r[0] for r in rows], [r[1] for r in rows]


@pytest.mark.parametrize("seed", [0, 7, 20261117])
def test_parity_with_rust_example(seed):
    steps = 500
    rate, times, sq = rust_trajectory(seed, steps)
    assert okrates.jump_rate(NU, EM, T) == rate
    py_times, py_sq = okrates.vacancy_trajectory(
        cells=CELLS, jump_rate=rate, lattice_parameter_m=A, seed=seed, steps=steps
    )
    assert len(py_times) == steps + 1
    for array in (py_times, py_sq):
        assert isinstance(array, np.ndarray)
        assert array.dtype == np.float64
        assert array.ndim == 1
    assert py_times.tolist() == times
    assert py_sq.tolist() == sq


@pytest.mark.parametrize("kwargs", [
    {"jump_rate": 0.0}, {"jump_rate": float("nan")},
    {"lattice_parameter_m": -1.0}, {"cells": (0, 4, 4)},
])
def test_invalid_arguments_raise_value_error(kwargs):
    args = dict(cells=CELLS, jump_rate=1.0, lattice_parameter_m=A, seed=1, steps=1)
    args.update(kwargs)
    with pytest.raises(ValueError):
        okrates.vacancy_trajectory(**args)


def test_invalid_rate_parameters_raise_value_error():
    with pytest.raises(ValueError):
        okrates.jump_rate(1e13, 0.65, -1.0)


def test_negative_seed_raises_overflow_error():
    with pytest.raises(OverflowError):
        okrates.vacancy_trajectory(cells=CELLS, jump_rate=1.0, lattice_parameter_m=A, seed=-1, steps=1)
