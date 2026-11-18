"""End-to-end test of the Python example script.

The expected output was produced by running the script; never edit it by hand.
"""

import subprocess
import sys
from pathlib import Path

EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "python" / "vacancy_diffusivity.py"


def test_vacancy_diffusivity_example_small():
    out = subprocess.run(
        [sys.executable, str(EXAMPLE), "3", "10"],
        check=True, capture_output=True, text=True,
    )
    assert out.stdout == (
        "jump rate r = 8.038596e+08 s^-1\n"
        "D_hat    = 5.505709e-11 m^2/s (3 runs x 10 jumps)\n"
        "D_theory = 6.605172e-11 m^2/s\n"
        "D_hat / D_theory = 0.8335\n"
    )


FECR_EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "python" / "fecr_martinez_comparison.py"


def test_fecr_martinez_comparison_example_small():
    out = subprocess.run(
        [sys.executable, str(FECR_EXAMPLE), "1e3", "2", "1"],
        check=True, capture_output=True, text=True,
    )
    lines = out.stdout.splitlines()
    assert len(lines) == 3
    assert lines[0] == "# N=432 anneal_sweeps=200 jumps/replica~1000 runs=2 k_count=20"
    assert lines[1].startswith("# cr T 1000/T  Gamma0  D_Fe/a2 se")
    assert lines[1].endswith("mean_jumps floor_activations wall_s")
    row = lines[2].split()
    assert len(row) == 20
    assert row[:3] == ["2", "1000.00", "1.00"]
    assert float(row[4]) > 0 and float(row[6]) > 0  # D_Fe, D_Cr
