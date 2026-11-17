"""Print the commands of one README.md console block, for CI to run verbatim.

Usage: python .github/scripts/readme_commands.py rust|python

The README must contain exactly two ```console blocks: the Rust commands,
then the Python commands. Any other layout fails, so the README job cannot
silently run nothing.
"""

import re
import sys
from pathlib import Path

README = Path(__file__).resolve().parents[2] / "README.md"
BLOCKS = {"rust": 0, "python": 1}

blocks = re.findall(r"^```console\n(.*?)^```$", README.read_text(), re.M | re.S)
if len(blocks) != len(BLOCKS):
    sys.exit(f"expected {len(BLOCKS)} console blocks in README.md, found {len(blocks)}")
commands = [line for line in blocks[BLOCKS[sys.argv[1]]].splitlines() if line.strip()]
if not commands:
    sys.exit(f"the {sys.argv[1]} console block in README.md is empty")
print("\n".join(commands))
