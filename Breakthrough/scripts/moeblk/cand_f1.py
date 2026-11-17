"""cand_f1.py: bench arm 'f1' = msh.block_f1 (router + glue exact; routed tgl16eu + the shared expert in the same two
kernels; combine_phase + one compiled gated add)"""
import msh


def block(lay, x):
    return msh.block_f1(lay, x)
