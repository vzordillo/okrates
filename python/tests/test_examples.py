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
