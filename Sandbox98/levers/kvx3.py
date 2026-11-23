"""BREAK98c PREFIX EYES (FAR), arm kvx3: the head's layer gains a cross-attention block into the frozen 2B trunk's cached
K / V of the verified prefix at full-attention layers 15 and 19 (o_x zero-initialised; q from the trunk's own q_proj).
The block trains at 3e-5 (the harness's rate for a new attention block, deep2 / deep3). Mechanism: sb/kvx.py; the cache
from sb/kvxgen.py; results/LEVERS_A.md section B98C."""
import os
from sb.kvx import make, SPLIT_CLIP

setup, standin, PREFIX, MODULE_LR = make(float(os.environ.get("KVX3_LR", "3e-5")))
