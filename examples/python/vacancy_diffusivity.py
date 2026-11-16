"""Estimate the vacancy diffusivity in BCC Fe from okrates trajectories.

Illustrative parameters, not a validated Fe parameterization. Usage:
    python examples/python/vacancy_diffusivity.py [RUNS] [STEPS]
"""

import sys

import okrates

runs = int(sys.argv[1]) if len(sys.argv) > 1 else 200
steps = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
a = 2.8665e-10  # m
rate = okrates.jump_rate(1e13, 0.65, 800.0)  # s^-1

sum_sq = sum_t = 0.0
for seed in range(runs):
    times, sq = okrates.vacancy_trajectory(
        cells=(4, 4, 4), jump_rate=rate, lattice_parameter_m=a, seed=seed, steps=steps
    )
    sum_sq += sq[-1]
    sum_t += times[-1]

# E[R^2] / (6 E[t]) after a fixed number of jumps equals D = r a^2.
d_hat = sum_sq / (6.0 * sum_t)
d_theory = rate * a * a
print(f"jump rate r = {rate:.6e} s^-1")
print(f"D_hat    = {d_hat:.6e} m^2/s ({runs} runs x {steps} jumps)")
print(f"D_theory = {d_theory:.6e} m^2/s")
print(f"D_hat / D_theory = {d_hat / d_theory:.4f}")
