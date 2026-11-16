"""Python bindings for the okrates kinetic Monte Carlo framework."""

import numpy as np
import numpy.typing as npt

from okrates import _okrates
from okrates._okrates import jump_rate


def vacancy_trajectory(
    *,
    cells: tuple[int, int, int],
    jump_rate: float,
    lattice_parameter_m: float,
    seed: int,
    steps: int,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Run a BCC single-vacancy trajectory; return ``(times, squared_displacements)``.

    Both are new float64 numpy arrays (copies) of ``steps + 1`` points, in s
    and m^2, starting at t = 0. Errors are as in the native function:
    ``ValueError`` for invalid arguments, ``RuntimeError`` if stepping fails,
    ``OverflowError`` for a seed or step count outside u64.
    """
    times, squared = _okrates.vacancy_trajectory(
        cells=cells,
        jump_rate=jump_rate,
        lattice_parameter_m=lattice_parameter_m,
        seed=seed,
        steps=steps,
    )
    return np.array(times, dtype=np.float64), np.array(squared, dtype=np.float64)


__all__ = ["jump_rate", "vacancy_trajectory"]
