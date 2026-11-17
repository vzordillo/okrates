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


def fecr_illustrative_energetics() -> dict[str, object]:
    """Return the ILLUSTRATIVE Fe-Cr energetics used by default.

    ``{"pair_energies_ev": 3x3 float64 array (eV, indexed Fe, Cr, vacancy),
    "kra_ev": (kra_fe, kra_cr)}``. The values are round numbers with no source;
    they are not DFT or literature values.
    """
    pair, kra_fe, kra_cr = _okrates.fecr_illustrative_energetics()
    return {
        "pair_energies_ev": np.array(pair, dtype=np.float64),
        "kra_ev": (kra_fe, kra_cr),
    }


def fecr_diffusion(
    *,
    cells: tuple[int, int, int],
    cr_count: int,
    temperature_k: float,
    attempt_frequency: float,
    t_end: float,
    k_count: int,
    lattice_parameter_m: float,
    seed: int,
    runs: int,
    anneal_sweeps: int = 0,
    pair_energies_ev: npt.ArrayLike | None = None,
    kra_ev: tuple[float, float] | None = None,
) -> dict[str, float | npt.NDArray[np.float64]]:
    """Estimate Fe-Cr single-vacancy diffusion observables for one composition.

    ``pair_energies_ev`` (symmetric 3x3, eV, indexed Fe, Cr, vacancy) and
    ``kra_ev`` (Fe, Cr; eV) must be given together or both omitted. When
    omitted, the default energetics are ILLUSTRATIVE: round numbers with no
    source, not DFT or literature values.

    Units: ``temperature_k`` in K, ``t_end`` in s, energies in eV. Lengths enter
    squared, so D and Lambda are in m^2/s when ``lattice_parameter_m`` is in m.

    Seeds: replica i runs dynamics with seed ``seed + i``; its random initial
    state uses ``seed + 1e9`` and the anneal ``seed + 2e9``.
    ``f = N_atoms * D*_all / D_v`` is computed per replica, then averaged. Each
    ``*_se`` is the sample standard deviation (n - 1) over replicas divided by
    ``sqrt(runs)``. ``D_Cr`` is NaN when ``cr_count`` is 0.

    Returns a dict with float keys ``D_v, D_v_se, D_Fe, D_Fe_se, D_Cr, D_Cr_se,
    D_all, D_all_se, f, f_se, mean_jumps`` and float64 2x2 arrays ``Lambda`` and
    ``Lambda_se`` laid out ``[[FeFe, FeCr], [FeCr, CrCr]]``.

    Raises ``ValueError`` for invalid arguments and ``OverflowError`` for
    negative or oversized integers. A failure during stepping surfaces as a
    pyo3 ``PanicException`` (the sampler panics; there is no ``RuntimeError``
    path).
    """
    pair = None
    if pair_energies_ev is not None:
        pair = np.asarray(pair_energies_ev, dtype=np.float64).tolist()
    scalars, lam, lam_se, mean_jumps = _okrates.fecr_diffusion(
        cells=cells,
        cr_count=cr_count,
        temperature_k=temperature_k,
        attempt_frequency=attempt_frequency,
        t_end=t_end,
        k_count=k_count,
        lattice_parameter_m=lattice_parameter_m,
        seed=seed,
        runs=runs,
        anneal_sweeps=anneal_sweeps,
        pair_energies_ev=pair,
        kra_ev=kra_ev,
    )
    keys = ["D_v", "D_Fe", "D_Cr", "D_all", "f"]
    result: dict[str, float | npt.NDArray[np.float64]] = {}
    for i, key in enumerate(keys):
        result[key] = scalars[2 * i]
        result[key + "_se"] = scalars[2 * i + 1]
    result["mean_jumps"] = mean_jumps

    def matrix(v: list[float]) -> npt.NDArray[np.float64]:
        return np.array([[v[0], v[1]], [v[1], v[2]]], dtype=np.float64)

    result["Lambda"] = matrix(lam)
    result["Lambda_se"] = matrix(lam_se)
    return result


__all__ = [
    "fecr_diffusion",
    "fecr_illustrative_energetics",
    "jump_rate",
    "vacancy_trajectory",
]
