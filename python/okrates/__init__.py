"""Python interface to the okrates kinetic Monte Carlo framework."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Literal, TypeAlias

import numpy as np
import numpy.typing as npt

from okrates import _okrates

FitMethodName: TypeAlias = Literal["windowed_ols", "origin_slope"]
BarrierPolicyName: TypeAlias = Literal["symmetric_floor", "unfloored", "strict"]


@dataclass(frozen=True, slots=True)
class LatticeConfig:
    """Periodic BCC supercell dimensions and lattice constant in metres."""

    cells: tuple[int, int, int]
    lattice_parameter_m: float


@dataclass(frozen=True, slots=True)
class PairEnergeticsConfig:
    """1NN broken-bond + KRA Fe–Cr energetics (defaults to round illustrative values, not DFT)."""

    attempt_frequency_hz: float
    pair_energies_ev: npt.ArrayLike | None = None
    kra_ev: tuple[float, float] | None = None


@dataclass(frozen=True, slots=True)
class Martinez2012Config:
    """Martínez et al. PRB 86, 224109 (2012) Fe–Cr energetics (requires cells >= 3 per axis)."""

    barrier_policy: BarrierPolicyName = "symmetric_floor"
    attempt_frequencies_hz: tuple[float, float] | None = None
    saddle_shift_ev: float = 0.0


@dataclass(frozen=True, slots=True)
class BarrierTableConfig:
    """Tabulated Fe–Cr NEB barrier set validated for forward/reverse detailed balance."""

    barrier_set_json: str
    default_attempt_frequency_hz: float = 1e13


EnergeticsConfig: TypeAlias = (
    PairEnergeticsConfig | Martinez2012Config | BarrierTableConfig
)


@dataclass(frozen=True, slots=True)
class AnnealConfig:
    """Pre-dynamics Kawasaki Metropolis equilibration (`N_Fe + N_Cr` swap proposals per sweep)."""

    sweeps: int = 0


@dataclass(frozen=True, slots=True)
class SamplingConfig:
    """Trajectory observation window `[t_equil, t_end]` in seconds (`t_equil` defaults to `0.2 * t_end` for `windowed_ols`)."""

    t_end: float
    sample_count: int = 20
    t_equil: float | None = None
    fit_method: FitMethodName = "windowed_ols"
    max_steps: int | None = None

    def resolved_t_equil(self) -> float:
        if self.t_equil is not None:
            return self.t_equil
        return 0.2 * self.t_end if self.fit_method == "windowed_ols" else 0.0


@dataclass(frozen=True, slots=True)
class RunManifest:
    """Provenance record capturing model parameter SHA-256, optional barrier SHA-256, RNG streams, and estimator settings."""

    schema_version: int
    crate_version: str
    git_sha: str
    reproducibility_tier: str
    model_id: str
    params_sha256: str
    barrier_set_sha256: str | None
    parent_checkpoint_sha256: str | None
    catalog_backend: str
    rng: dict[str, Any]
    conditions: dict[str, Any]
    estimator: dict[str, Any] | None
    raw_json: str

    @classmethod
    def from_json(cls, raw_json: str) -> RunManifest:
        data = json.loads(raw_json)
        return cls(
            schema_version=int(data["schema_version"]),
            crate_version=str(data["crate_version"]),
            git_sha=str(data["git_sha"]),
            reproducibility_tier=str(data["reproducibility_tier"]),
            model_id=str(data["model_id"]),
            params_sha256=str(data["params_sha256"]),
            barrier_set_sha256=data.get("barrier_set_sha256"),
            parent_checkpoint_sha256=data.get("parent_checkpoint_sha256"),
            catalog_backend=str(data["catalog_backend"]),
            rng=dict(data["rng"]),
            conditions=dict(data["conditions"]),
            estimator=data.get("estimator"),
            raw_json=raw_json,
        )

    def to_json(self) -> str:
        return self.raw_json


@dataclass(frozen=True, slots=True)
class DiffusionResult:
    """Fe–Cr transport coefficients in `m^2/s`, jackknife correlation factor `f`, and provenance manifest."""

    D_v: float
    D_v_se: float
    D_Fe: float
    D_Fe_se: float
    D_Cr: float
    D_Cr_se: float
    D_all: float
    D_all_se: float
    f: float
    f_se: float
    Lambda: npt.NDArray[np.float64]
    Lambda_se: npt.NDArray[np.float64]
    mean_jumps: float
    window_drift_ratio: float
    barrier_floor_activations: int
    mean_final_sro_alpha1: float | None
    manifest: RunManifest

    def to_dict(self) -> dict[str, float | int | None | npt.NDArray[np.float64]]:
        return {
            "D_v": self.D_v,
            "D_v_se": self.D_v_se,
            "D_Fe": self.D_Fe,
            "D_Fe_se": self.D_Fe_se,
            "D_Cr": self.D_Cr,
            "D_Cr_se": self.D_Cr_se,
            "D_all": self.D_all,
            "D_all_se": self.D_all_se,
            "f": self.f,
            "f_se": self.f_se,
            "Lambda": self.Lambda.copy(),
            "Lambda_se": self.Lambda_se.copy(),
            "mean_jumps": self.mean_jumps,
            "window_drift_ratio": self.window_drift_ratio,
            "barrier_floor_activations": self.barrier_floor_activations,
            "mean_final_sro_alpha1": self.mean_final_sro_alpha1,
        }


@dataclass(frozen=True, slots=True)
class AnnealResult:
    """Kawasaki equilibration energy trajectory (eV), Warren–Cowley 1NN `alpha_1` trajectory, and diagnostics."""

    energy: npt.NDArray[np.float64]
    sro_alpha1: npt.NDArray[np.float64]
    attempts: int
    accepted: int
    acceptance_rate: float
    final_sro_alpha1: float
    tail_energy_drift_ev: float


def _unpack_energetics(config: EnergeticsConfig) -> dict[str, Any]:
    if isinstance(config, PairEnergeticsConfig):
        pair = (
            np.asarray(config.pair_energies_ev, dtype=np.float64).tolist()
            if config.pair_energies_ev is not None
            else None
        )
        return {
            "energetics": "pair",
            "attempt_frequency": config.attempt_frequency_hz,
            "pair_energies_ev": pair,
            "kra_ev": config.kra_ev,
        }
    if isinstance(config, Martinez2012Config):
        return {
            "energetics": "martinez2012",
            "barrier_policy": config.barrier_policy,
            "attempt_frequencies_hz": config.attempt_frequencies_hz,
            "saddle_shift_ev": config.saddle_shift_ev,
        }
    if isinstance(config, BarrierTableConfig):
        return {
            "energetics": "tabulated",
            "attempt_frequency": config.default_attempt_frequency_hz,
            "barrier_set_json": config.barrier_set_json,
        }
    raise TypeError(f"unsupported energetics config: {type(config)!r}")


def _symmetric_2x2(v: list[float]) -> npt.NDArray[np.float64]:
    return np.array([[v[0], v[1]], [v[1], v[2]]], dtype=np.float64)


def jump_rate(
    attempt_frequency: float,
    migration_energy_ev: float,
    temperature_k: float,
) -> float:
    """Compute Arrhenius rate `nu * exp(-E_m / (k_B T))` in `s^-1` (`ValueError` on invalid inputs)."""
    return _okrates.jump_rate(attempt_frequency, migration_energy_ev, temperature_k)


def vacancy_trajectory(
    *,
    cells: tuple[int, int, int],
    jump_rate: float,
    lattice_parameter_m: float,
    seed: int,
    steps: int,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Run a BCC single-vacancy trajectory and return `(times_s, squared_displacements_m2)`."""
    times, squared = _okrates.vacancy_trajectory(
        cells=cells,
        jump_rate=jump_rate,
        lattice_parameter_m=lattice_parameter_m,
        seed=seed,
        steps=steps,
    )
    return np.array(times, dtype=np.float64), np.array(squared, dtype=np.float64)


def fecr_illustrative_energetics() -> dict[str, object]:
    """Return round illustrative Fe–Cr pair energies (3x3 eV array) and `(kra_fe, kra_cr)` in eV."""
    pair, kra_fe, kra_cr = _okrates.fecr_illustrative_energetics()
    return {
        "pair_energies_ev": np.array(pair, dtype=np.float64),
        "kra_ev": (kra_fe, kra_cr),
    }


def run_fecr_diffusion(
    *,
    lattice: LatticeConfig,
    energetics: EnergeticsConfig,
    cr_count: int,
    temperature_k: float,
    sampling: SamplingConfig,
    runs: int,
    seed: int,
    anneal: AnnealConfig = AnnealConfig(),
) -> DiffusionResult:
    """Run an ensemble Fe–Cr diffusion calculation (`ValueError` on invalid config, `RuntimeError` on dynamics failure)."""
    kwargs = _unpack_energetics(energetics)
    (
        scalars,
        lam,
        lam_se,
        mean_jumps,
        window_drift_ratio,
        floor_activations,
        mean_final_sro,
        manifest_json,
    ) = _okrates.run_fecr_transport(
        cells=lattice.cells,
        cr_count=cr_count,
        temperature_k=temperature_k,
        t_equil=sampling.resolved_t_equil(),
        t_end=sampling.t_end,
        sample_count=sampling.sample_count,
        fit_method=sampling.fit_method,
        max_steps=sampling.max_steps,
        lattice_parameter_m=lattice.lattice_parameter_m,
        seed=seed,
        runs=runs,
        anneal_sweeps=anneal.sweeps,
        **kwargs,
    )
    return DiffusionResult(
        D_v=scalars[0],
        D_v_se=scalars[1],
        D_Fe=scalars[2],
        D_Fe_se=scalars[3],
        D_Cr=scalars[4],
        D_Cr_se=scalars[5],
        D_all=scalars[6],
        D_all_se=scalars[7],
        f=scalars[8],
        f_se=scalars[9],
        Lambda=_symmetric_2x2(lam),
        Lambda_se=_symmetric_2x2(lam_se),
        mean_jumps=mean_jumps,
        window_drift_ratio=window_drift_ratio,
        barrier_floor_activations=floor_activations,
        mean_final_sro_alpha1=mean_final_sro,
        manifest=RunManifest.from_json(manifest_json),
    )


def run_anneal(
    *,
    cells: tuple[int, int, int],
    energetics: EnergeticsConfig,
    cr_count: int,
    temperature_k: float,
    sweeps: int,
    seed: int,
) -> AnnealResult:
    """Run Kawasaki Fe↔Cr Metropolis equilibration and return energy, 1NN SRO `alpha_1`, and diagnostics."""
    kwargs = _unpack_energetics(energetics)
    (
        energy,
        sro_alpha1,
        attempts,
        accepted,
        acceptance_rate,
        final_sro_alpha1,
        tail_energy_drift_ev,
    ) = _okrates.run_kawasaki_anneal(
        cells=cells,
        cr_count=cr_count,
        temperature_k=temperature_k,
        sweeps=sweeps,
        seed=seed,
        **kwargs,
    )
    return AnnealResult(
        energy=np.array(energy, dtype=np.float64),
        sro_alpha1=np.array(sro_alpha1, dtype=np.float64),
        attempts=attempts,
        accepted=accepted,
        acceptance_rate=acceptance_rate,
        final_sro_alpha1=final_sro_alpha1,
        tail_energy_drift_ev=tail_energy_drift_ev,
    )


def validate_barrier_set(
    barrier_set_json: str,
    *,
    temperature_k: float | None = None,
) -> str:
    """Validate a `BarrierSet` JSON document and return its canonical SHA-256 hex digest."""
    return _okrates.validate_barrier_set_json(
        barrier_set_json,
        temperature_k=temperature_k,
    )


def import_pymatgen_vasp_barriers(
    payload_json: str,
    *,
    default_attempt_frequency_hz: float = 1e13,
) -> BarrierTableConfig:
    """Convert pymatgen/atomate2 VASP NEB JSON into a validated `BarrierTableConfig`."""
    barrier_json = _okrates.import_pymatgen_vasp_barriers_json(payload_json)
    return BarrierTableConfig(
        barrier_set_json=barrier_json,
        default_attempt_frequency_hz=default_attempt_frequency_hz,
    )


def import_aiida_qe_barriers(
    payload_json: str,
    *,
    default_attempt_frequency_hz: float = 1e13,
) -> BarrierTableConfig:
    """Convert AiiDA-QuantumESPRESSO NEB JSON into a validated `BarrierTableConfig`."""
    barrier_json = _okrates.import_aiida_qe_barriers_json(payload_json)
    return BarrierTableConfig(
        barrier_set_json=barrier_json,
        default_attempt_frequency_hz=default_attempt_frequency_hz,
    )


def fecr_diffusion(
    *,
    cells: tuple[int, int, int],
    cr_count: int,
    temperature_k: float,
    attempt_frequency: float | None = None,
    t_end: float,
    k_count: int,
    lattice_parameter_m: float,
    seed: int,
    runs: int,
    anneal_sweeps: int = 0,
    pair_energies_ev: npt.ArrayLike | None = None,
    kra_ev: tuple[float, float] | None = None,
    energetics: str = "illustrative",
    t_equil: float = 0.0,
    fit_method: FitMethodName = "origin_slope",
) -> dict[str, float | int | None | npt.NDArray[np.float64]]:
    """Keyword convenience wrapper returning Fe–Cr transport observables as a dictionary."""
    pair = (
        np.asarray(pair_energies_ev, dtype=np.float64).tolist()
        if pair_energies_ev is not None
        else None
    )
    (
        scalars,
        lam,
        lam_se,
        mean_jumps,
        window_drift_ratio,
        floor_activations,
        mean_final_sro,
        manifest_json,
    ) = _okrates.run_fecr_transport(
        cells=cells,
        cr_count=cr_count,
        temperature_k=temperature_k,
        t_equil=t_equil,
        t_end=t_end,
        sample_count=k_count,
        fit_method=fit_method,
        max_steps=None,
        lattice_parameter_m=lattice_parameter_m,
        seed=seed,
        runs=runs,
        anneal_sweeps=anneal_sweeps,
        energetics=energetics,
        attempt_frequency=attempt_frequency,
        pair_energies_ev=pair,
        kra_ev=kra_ev,
    )
    res = DiffusionResult(
        D_v=scalars[0],
        D_v_se=scalars[1],
        D_Fe=scalars[2],
        D_Fe_se=scalars[3],
        D_Cr=scalars[4],
        D_Cr_se=scalars[5],
        D_all=scalars[6],
        D_all_se=scalars[7],
        f=scalars[8],
        f_se=scalars[9],
        Lambda=_symmetric_2x2(lam),
        Lambda_se=_symmetric_2x2(lam_se),
        mean_jumps=mean_jumps,
        window_drift_ratio=window_drift_ratio,
        barrier_floor_activations=floor_activations,
        mean_final_sro_alpha1=mean_final_sro,
        manifest=RunManifest.from_json(manifest_json),
    )
    return res.to_dict()


__all__ = [
    "AnnealConfig",
    "AnnealResult",
    "BarrierPolicyName",
    "BarrierTableConfig",
    "DiffusionResult",
    "EnergeticsConfig",
    "FitMethodName",
    "LatticeConfig",
    "Martinez2012Config",
    "PairEnergeticsConfig",
    "RunManifest",
    "SamplingConfig",
    "fecr_diffusion",
    "fecr_illustrative_energetics",
    "import_aiida_qe_barriers",
    "import_pymatgen_vasp_barriers",
    "jump_rate",
    "run_anneal",
    "run_fecr_diffusion",
    "vacancy_trajectory",
    "validate_barrier_set",
]
