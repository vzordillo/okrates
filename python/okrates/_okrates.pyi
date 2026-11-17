def jump_rate(
    attempt_frequency: float, migration_energy_ev: float, temperature_k: float
) -> float: ...
def vacancy_trajectory(
    *,
    cells: tuple[int, int, int],
    jump_rate: float,
    lattice_parameter_m: float,
    seed: int,
    steps: int,
) -> tuple[list[float], list[float]]: ...
def fecr_illustrative_energetics() -> tuple[list[list[float]], float, float]: ...
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
    pair_energies_ev: list[list[float]] | None = None,
    kra_ev: tuple[float, float] | None = None,
) -> tuple[list[float], list[float], list[float], float]: ...
