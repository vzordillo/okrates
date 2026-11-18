"""Fe-Cr tracer diffusion with Martinez 2012 energetics, compared with their Fig. 4.

Computes, for 2 Cr atoms in a 432-site bcc box with one vacancy, the tracer
diffusivities of Fe and Cr and their ratio D_Cr/D_Fe at 1000/T = 1.0, 1.1, 1.2
and 1.3 (T in K), after a Kawasaki anneal of the Cr arrangement. It also
computes the Fe self-diffusion correlation factor f. Larger Cr contents
(x = 0.10 and 0.20, at 1000/T = 1.1 and 1.3) follow the four dilute points.

Reference: E. Martinez, O. Senninger, C.-C. Fu and F. Soisson, Phys. Rev. B 86,
224109 (2012), doi:10.1103/PhysRevB.86.224109. Read from their Fig. 4, the
ratio D*_Cr/D*_Fe is 2.03, 2.12, 2.21 and 2.31 at 1000/T = 1.0, 1.1, 1.2, 1.3.

Usage:
    python examples/python/fecr_martinez_comparison.py JUMPS_PER_REPLICA RUNS [SUBSET]

JUMPS_PER_REPLICA is the approximate number of vacancy jumps in each replica
(for example 1e6), RUNS the number of replicas (for example 64), and SUBSET the
number of leading points to compute (4 gives the dilute points only; the
default is all eight). Run it against a release build (maturin develop
--release); the full calculation is long.

Lengths are in units of the lattice parameter (lattice_parameter_m = 1), so
diffusivities are in a^2/s at the kMC vacancy fraction 1/N. The same
calculation can be written with okrates.run_fecr_diffusion; this script uses
the keyword wrapper okrates.fecr_diffusion, whose seeds and arguments give the
recorded numbers.
"""

import math
import sys
import time

import okrates

K_B = 1.380649e-23 / 1.602176634e-19  # Boltzmann constant, eV/K
CELLS = (6, 6, 6)
N = 432  # sites in the box
# Martinez et al. Table I and p. 6: attempt frequency (1/s), migration entropy
# (k_B), migration enthalpy and vacancy formation enthalpy (eV), formation
# entropy (k_B).
NU, DS_MIG, DH_MIG, DH_FORM, DS_FORM = 1e13, 2.1, 0.69, 2.18, 4.1
ANNEAL_SWEEPS = 200
K_COUNT = 20  # sample points on the mean-squared-displacement curve

# (Cr atoms, T): 2 Cr in 431 atoms at 1000/T = 1.0-1.3 (Fig. 4 range), then
# x = 0.10 (43 Cr) and x = 0.20 (86 Cr).
POINTS = [(2, 1000 / v) for v in (1.0, 1.1, 1.2, 1.3)] + [
    (c, 1000 / v) for c in (43, 86) for v in (1.1, 1.3)
]


def main() -> None:
    jumps = float(sys.argv[1])
    runs = int(sys.argv[2])
    subset = sys.argv[3] if len(sys.argv) > 3 else "all"
    points = POINTS if subset == "all" else POINTS[: int(subset)]

    print(f"# N={N} anneal_sweeps={ANNEAL_SWEEPS} jumps/replica~{jumps:g} runs={runs} k_count={K_COUNT}")
    print(
        "# cr T 1000/T  Gamma0  D_Fe/a2 se  D_Cr/a2 se  D_v se  f se  ratioCr/Fe se  "
        "f_kmc=N*D_Fe/Gamma0  D_Fe_eq/a2  D_Cr_eq/a2  mean_jumps floor_activations wall_s"
    )
    for i, (cr, temp) in enumerate(points):
        kt = K_B * temp
        gamma0 = NU * math.exp(DS_MIG) * math.exp(-DH_MIG / kt)  # vacancy jump rate in pure Fe
        c_vac = math.exp(DS_FORM) * math.exp(-DH_FORM / kt)  # equilibrium vacancy fraction
        t0 = time.time()
        r = okrates.fecr_diffusion(
            cells=CELLS,
            cr_count=cr,
            temperature_k=temp,
            t_end=jumps / (8 * gamma0),
            k_count=K_COUNT,
            lattice_parameter_m=1.0,
            seed=8_000_000 + 1000 * i,
            runs=runs,
            anneal_sweeps=ANNEAL_SWEEPS,
            energetics="martinez2012",
        )
        ratio = r["D_Cr"] / r["D_Fe"]
        ratio_se = ratio * math.hypot(r["D_Cr_se"] / r["D_Cr"], r["D_Fe_se"] / r["D_Fe"])
        print(
            f"{cr} {temp:.2f} {1000 / temp:.2f} {gamma0:.6e} "
            f"{r['D_Fe']:.6e} {r['D_Fe_se']:.2e} {r['D_Cr']:.6e} {r['D_Cr_se']:.2e} "
            f"{r['D_v']:.6e} {r['D_v_se']:.2e} {r['f']:.5f} {r['f_se']:.5f} "
            f"{ratio:.4f} {ratio_se:.4f} {N * r['D_Fe'] / gamma0:.5f} "
            f"{N * c_vac * r['D_Fe']:.6e} {N * c_vac * r['D_Cr']:.6e} "
            f"{r['mean_jumps']:.4g} {r['barrier_floor_activations']} {time.time() - t0:.1f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
