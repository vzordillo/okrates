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
