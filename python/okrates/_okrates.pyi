from typing import Sequence

def jump_rate(
    attempt_frequency: float,
    migration_energy_ev: float,
    temperature_k: float,
) -> float: ...
def vacancy_trajectory(
    *,
    cells: Sequence[int],
    jump_rate: float,
    lattice_parameter_m: float,
    seed: int,
    steps: int,
) -> tuple[list[float], list[float]]: ...
def fecr_illustrative_energetics() -> tuple[list[list[float]], float, float]: ...
def run_fecr_transport(
    *,
    cells: Sequence[int],
    cr_count: int,
    temperature_k: float,
    t_equil: float,
    t_end: float,
    sample_count: int,
    fit_method: str = "windowed_ols",
    max_steps: int | None = None,
    lattice_parameter_m: float,
    seed: int,
    runs: int,
    anneal_sweeps: int = 0,
    energetics: str = "illustrative",
    attempt_frequency: float | None = None,
    pair_energies_ev: Sequence[Sequence[float]] | None = None,
    kra_ev: tuple[float, float] | None = None,
    barrier_policy: str = "symmetric_floor",
    attempt_frequencies_hz: tuple[float, float] | None = None,
    saddle_shift_ev: float = 0.0,
    barrier_set_json: str | None = None,
) -> tuple[
    list[float],
    list[float],
    list[float],
    float,
    float,
    int,
    float | None,
    str,
]: ...
def run_kawasaki_anneal(
    *,
    cells: Sequence[int],
    cr_count: int,
    temperature_k: float,
    sweeps: int,
    seed: int,
    energetics: str = "illustrative",
    attempt_frequency: float | None = None,
    pair_energies_ev: Sequence[Sequence[float]] | None = None,
    kra_ev: tuple[float, float] | None = None,
    barrier_policy: str = "symmetric_floor",
    attempt_frequencies_hz: tuple[float, float] | None = None,
    saddle_shift_ev: float = 0.0,
    barrier_set_json: str | None = None,
) -> tuple[list[float], list[float], int, int, float, float, float]: ...
def validate_barrier_set_json(
    barrier_set_json: str,
    *,
    temperature_k: float | None = None,
) -> str: ...
def import_pymatgen_vasp_barriers_json(payload_json: str) -> str: ...
def import_aiida_qe_barriers_json(payload_json: str) -> str: ...
