"""Lever 'tie2' (T2's pointer, the zero-parameter version): the native layer applied TWICE (weight-tied) at every depth and
over the true pairs. No new parameters; step 0 differs from the base (the second pass is not an identity)."""


def setup(D, args):
    object.__setattr__(D, "_stack", [D, D])
