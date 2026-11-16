"""MTP Head lever 'tapmlpk' (WEEK98 S15 SPARSETAP; NewIdeas 13:3x): Sandbox98's tapmlp (levers/tapmlp.py, loaded
read-only and unchanged: same modules, PREFIX, module lr, TAPMLP_TAP) with a SPARSE tap refresh. The true tap is read
only at refresh depths 2, 2 + K, 2 + 2K, ... (K = TAPMLP_K, default 8). In between, the head carries the last tap:
depth j uses the tap of the row at the last refresh depth jr <= j, through the same MLP. At runtime that is one
batched partial trunk pass per refresh (K drafted rows at once) instead of one per drafted token.
At K = 1 it is tapmlp exactly."""
import os
import importlib.util

_spec = importlib.util.spec_from_file_location("tapmlp_base_for_k", os.path.expanduser("~/Sandbox98/levers/tapmlp.py"))
B = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(B)
PREFIX, MODULE_LR, TAP = B.PREFIX, B.MODULE_LR, B.TAP
K = int(os.environ.get("TAPMLP_K", "8"))
setup = B.setup


def standin(j, h, anc):
    jr = 2 + ((j - 2) // K) * K if j >= 2 else j       # the last refresh depth <= j
    return B.standin(jr, h, anc)                        # tapmlp reads the tap of row anc + jr - 1
