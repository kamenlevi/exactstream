"""BREAK98c PREFIX EYES (FAR), arm kvx10: as levers/kvx3.py with the block at 1e-4 (400 steps is short for a block that
starts at zero: this arm guards against reading "too slow" as "no information"). Mechanism: sb/kvx.py; results/LEVERS_A.md
section B98C."""
import os
from sb.kvx import make, SPLIT_CLIP

setup, standin, PREFIX, MODULE_LR = make(float(os.environ.get("KVX10_LR", "1e-4")))
