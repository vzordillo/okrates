"""Python interface to the okrates kinetic Monte Carlo framework."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
from types import MappingProxyType
from typing import Any, Literal, TypeAlias, TypedDict

import numpy as np
import numpy.typing as npt

from okrates import _okrates

FitMethodName: TypeAlias = Literal["windowed_ols", "origin_slope"]
BarrierPolicyName: TypeAlias = Literal["symmetric_floor", "unfloored", "strict"]


class IllustrativeEnergeticsDict(TypedDict):
    """Symmetric 3x3 pair energies ordered `(Fe, Cr, Vacancy)` on both axes and `(Fe, Cr)` KRA barriers in eV."""

    pair_energies_ev: npt.NDArray[np.float64]
    kra_ev: tuple[float, float]


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
    """Martínez et al., PRB 86, 224109 (2012), doi:10.1103/PhysRevB.86.224109 Fe–Cr energetics
    with Levesque et al., PRB 84, 184205 (2011), doi:10.1103/PhysRevB.84.184205 mixing (cells >= 3)."""

    barrier_policy: BarrierPolicyName = "symmetric_floor"
    attempt_frequencies_hz: tuple[float, float] | None = None
    saddle_shift_ev: float = 0.0


@dataclass(frozen=True, slots=True)
class BarrierTableConfig:
    """Tabulated Fe–Cr NEB barrier set validated for forward/reverse detailed balance (CI-NEB: Henkelman et al., J. Chem. Phys. 113, 9901 (2000), doi:10.1063/1.1329672)."""

    barrier_set_json: str


EnergeticsConfig: TypeAlias = (
    PairEnergeticsConfig | Martinez2012Config | BarrierTableConfig
)


@dataclass(frozen=True, slots=True)
class AnnealConfig:
    """Pre-dynamics Kawasaki (Phys. Rev. 145, 224 (1966), doi:10.1103/PhysRev.145.224) Metropolis (J. Chem. Phys. 21, 1087 (1953), doi:10.1063/1.1699114) equilibration (`N_Fe + N_Cr` swap proposals per sweep)."""

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


def _req_obj(obj: dict[str, Any], key: str, ctx: str) -> dict[str, Any]:
    val = obj.get(key)
    if not isinstance(val, dict):
        raise ValueError(f"{ctx}.{key} must be a JSON object")
    return val


def _req_str(obj: dict[str, Any], key: str, ctx: str) -> str:
    val = obj.get(key)
    if not isinstance(val, str):
        raise ValueError(f"{ctx}.{key} must be a string")
    return val


def _opt_str(obj: dict[str, Any], key: str, ctx: str) -> str | None:
    if key not in obj or obj[key] is None:
        return None
    val = obj[key]
    if not isinstance(val, str):
        raise ValueError(f"{ctx}.{key} must be a string or null")
    return val


def _req_nonneg_int(obj: dict[str, Any], key: str, ctx: str) -> int:
    val = obj.get(key)
    if isinstance(val, bool) or not isinstance(val, int) or val < 0:
        raise ValueError(f"{ctx}.{key} must be a non-negative integer")
    return val


def _req_float(obj: dict[str, Any], key: str, ctx: str) -> float:
    val = obj.get(key)
    if isinstance(val, bool) or not isinstance(val, (int, float)):
        raise ValueError(f"{ctx}.{key} must be a number")
    return float(val)


@dataclass(frozen=True, slots=True)
class RngPolicy:
    """Deterministic RNG algorithm, base seed, and 256-bit key domain-separation policy."""

    algorithm: str
    base_seed: int
    stream_policy: str


@dataclass(frozen=True, slots=True)
class ManifestConditions:
    """Supercell dimensions, lattice constant in metres, temperature in kelvin, and site counts by species."""

    cells: tuple[int, int, int]
    lattice_parameter: float
    temperature_k: float
    composition: Mapping[str, int]

    def __post_init__(self) -> None:
        object.__setattr__(self, "composition", MappingProxyType(dict(self.composition)))


@dataclass(frozen=True, slots=True)
class EstimatorSpec:
    """Trajectory window `[t_equil, t_end]` in seconds, sample/replica counts, and pre-run Kawasaki sweeps."""

    fit_method: str
    t_equil: float
    t_end: float
    sample_count: int
    replicas: int
    anneal_sweeps: int


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
    rng: RngPolicy
    conditions: ManifestConditions
    estimator: EstimatorSpec | None
    raw_json: str

    @classmethod
    def from_json(cls, raw_json: str) -> RunManifest:
        if not isinstance(raw_json, str):
            raise ValueError("manifest input must be a JSON string")
        data = json.loads(raw_json)
        if not isinstance(data, dict):
            raise ValueError("manifest JSON root must be an object")
        version = data.get("schema_version")
        if isinstance(version, bool) or not isinstance(version, int):
            raise ValueError("manifest.schema_version must be an integer")
        if version != 1:
            raise ValueError(f"unsupported manifest schema version {version}, expected 1")

        rng_obj = _req_obj(data, "rng", "manifest")
        rng = RngPolicy(
            algorithm=_req_str(rng_obj, "algorithm", "rng"),
            base_seed=_req_nonneg_int(rng_obj, "base_seed", "rng"),
            stream_policy=_req_str(rng_obj, "stream_policy", "rng"),
        )

        cond_obj = _req_obj(data, "conditions", "manifest")
        raw_cells = cond_obj.get("cells")
        if (
            not isinstance(raw_cells, list)
            or len(raw_cells) != 3
            or any(isinstance(c, bool) or not isinstance(c, int) or c < 0 for c in raw_cells)
        ):
            raise ValueError("conditions.cells must be a 3-element non-negative integer list")
        raw_comp = _req_obj(cond_obj, "composition", "conditions")
        comp: dict[str, int] = {}
        for sp, count in raw_comp.items():
            if (
                not isinstance(sp, str)
                or isinstance(count, bool)
                or not isinstance(count, int)
                or count < 0
            ):
                raise ValueError(
                    "conditions.composition must map species names to non-negative integers"
                )
            comp[sp] = count
        conditions = ManifestConditions(
            cells=(raw_cells[0], raw_cells[1], raw_cells[2]),
            lattice_parameter=_req_float(cond_obj, "lattice_parameter", "conditions"),
            temperature_k=_req_float(cond_obj, "temperature_k", "conditions"),
            composition=MappingProxyType(comp),
        )

        est_raw = data.get("estimator")
        if est_raw is None:
            estimator = None
        elif isinstance(est_raw, dict):
            estimator = EstimatorSpec(
                fit_method=_req_str(est_raw, "fit_method", "estimator"),
                t_equil=_req_float(est_raw, "t_equil", "estimator"),
                t_end=_req_float(est_raw, "t_end", "estimator"),
                sample_count=_req_nonneg_int(est_raw, "sample_count", "estimator"),
                replicas=_req_nonneg_int(est_raw, "replicas", "estimator"),
                anneal_sweeps=_req_nonneg_int(est_raw, "anneal_sweeps", "estimator"),
            )
        else:
            raise ValueError("manifest.estimator must be a JSON object or null")

        return cls(
            schema_version=version,
            crate_version=_req_str(data, "crate_version", "manifest"),
            git_sha=_req_str(data, "git_sha", "manifest"),
            reproducibility_tier=_req_str(data, "reproducibility_tier", "manifest"),
            model_id=_req_str(data, "model_id", "manifest"),
            params_sha256=_req_str(data, "params_sha256", "manifest"),
            barrier_set_sha256=_opt_str(data, "barrier_set_sha256", "manifest"),
            parent_checkpoint_sha256=_opt_str(data, "parent_checkpoint_sha256", "manifest"),
            catalog_backend=_req_str(data, "catalog_backend", "manifest"),
            rng=rng,
            conditions=conditions,
            estimator=estimator,
            raw_json=raw_json,
        )

    def to_json(self) -> str:
        return self.raw_json


@dataclass(frozen=True, slots=True)
class DiffusionResult:
    """Fe-Cr transport coefficients in `m^2/s`, ratio-of-means correlation factor `f` with delete-1 jackknife standard error `f_se` (Quenouille, J. R. Stat. Soc. B 11, 68 (1949), doi:10.1111/j.2517-6161.1949.tb00023.x; Biometrika 43, 353 (1956), doi:10.1093/biomet/43.3-4.353; Tukey, Ann. Math. Stat. 29, 614 (1958), doi:10.1214/aoms/1177706647), and provenance manifest."""

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
    """Kawasaki equilibration energy trajectory (eV), Warren–Cowley 1NN `alpha_1` trajectory (Cowley, Phys. Rev. 77, 669 (1950), doi:10.1103/PhysRev.77.669; Warren, 1969), and diagnostics."""

    energy: npt.NDArray[np.float64]
    sro_alpha1: npt.NDArray[np.float64]
    attempts: int
    accepted: int
    acceptance_rate: float
    final_sro_alpha1: float
    tail_energy_drift_ev: float


def _coerce_pair_matrix(pair_energies_ev: npt.ArrayLike | None) -> list[list[float]] | None:
    if pair_energies_ev is None:
        return None
    arr = np.asarray(pair_energies_ev, dtype=np.float64)
    if arr.shape != (3, 3):
        raise ValueError(f"pair_energies_ev must have shape (3, 3), got {arr.shape}")
    return arr.tolist()


def _unpack_energetics(config: EnergeticsConfig) -> dict[str, Any]:
    if isinstance(config, PairEnergeticsConfig):
        return {
            "energetics": "pair",
            "attempt_frequency": config.attempt_frequency_hz,
            "pair_energies_ev": _coerce_pair_matrix(config.pair_energies_ev),
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
            "barrier_set_json": config.barrier_set_json,
        }
    raise TypeError(f"unsupported energetics config: {type(config)!r}")


def _symmetric_2x2(v: list[float]) -> npt.NDArray[np.float64]:
    return np.array([[v[0], v[1]], [v[1], v[2]]], dtype=np.float64)


def _decode_diffusion_result(
    raw: tuple[
        list[float],
        list[float],
        list[float],
        float,
        float,
        int,
        float | None,
        str,
    ],
) -> DiffusionResult:
    (
        scalars,
        lam,
        lam_se,
        mean_jumps,
        window_drift_ratio,
        floor_activations,
        mean_final_sro,
        manifest_json,
    ) = raw
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


def jump_rate(
    attempt_frequency: float,
    migration_energy_ev: float,
    temperature_k: float,
) -> float:
    """Compute Arrhenius rate `nu * exp(-E_m / (k_B T))` in `s^-1`."""
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


def fecr_illustrative_energetics() -> IllustrativeEnergeticsDict:
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
    """Run an ensemble Fe–Cr diffusion calculation."""
    kwargs = _unpack_energetics(energetics)
    raw = _okrates.run_fecr_transport(
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
    return _decode_diffusion_result(raw)


def run_anneal(
    *,
    cells: tuple[int, int, int],
    energetics: EnergeticsConfig,
    cr_count: int,
    temperature_k: float,
    sweeps: int,
    seed: int,
) -> AnnealResult:
    """Run Kawasaki Fe↔Cr Metropolis equilibration (Kawasaki, Phys. Rev. 145, 224 (1966), doi:10.1103/PhysRev.145.224; Metropolis et al., J. Chem. Phys. 21, 1087 (1953), doi:10.1063/1.1699114) and return energy, 1NN SRO `alpha_1`, and diagnostics."""
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


def import_pymatgen_vasp_barriers(payload_json: str) -> BarrierTableConfig:
    """Convert pymatgen/atomate2 VASP NEB JSON into a validated `BarrierTableConfig`."""
    return BarrierTableConfig(
        barrier_set_json=_okrates.import_pymatgen_vasp_barriers_json(payload_json)
    )


def import_aiida_qe_barriers(payload_json: str) -> BarrierTableConfig:
    """Convert AiiDA-QuantumESPRESSO NEB JSON into a validated `BarrierTableConfig`."""
    return BarrierTableConfig(
        barrier_set_json=_okrates.import_aiida_qe_barriers_json(payload_json)
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
    pair = _coerce_pair_matrix(pair_energies_ev)
    raw = _okrates.run_fecr_transport(
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
    return _decode_diffusion_result(raw).to_dict()


__all__ = [
    "AnnealConfig",
    "AnnealResult",
    "BarrierPolicyName",
    "BarrierTableConfig",
    "DiffusionResult",
    "EnergeticsConfig",
    "EstimatorSpec",
    "FitMethodName",
    "IllustrativeEnergeticsDict",
    "LatticeConfig",
    "ManifestConditions",
    "Martinez2012Config",
    "PairEnergeticsConfig",
    "RngPolicy",
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
