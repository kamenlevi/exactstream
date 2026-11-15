"""MTP Head lever 'carrya' (WEEK98 screen 2, THE CARRY; REPORT 'WEEK98 S2 CARRY400'). Loaded by Sandbox98's
sb/train_multi.py / sb/evalx.py through an absolute lever path (their scripts run unmodified).

Carry-specific capacity. The head's depth-j output h is both the guess (lm_head(norm(h))) and the carry (depth j+1's
stand-in for the true state). A residual adapter splits the two roles: depth j+1 reads A(h) = h + U(gelu(V(rms(h))))
(d -> 1024 -> d, 4.2M parameters) while the guess still reads h. A trains through the chain's KL only, on its own Adam
(MODULE_LR 1e-4; train_multi keeps it constant after the warm-up). U starts small and random (std 1e-3), not at zero, so
it has a gradient from step 0 and can grow (deep2's zero start barely moved)."""
import os
import mlx.core as mx
import mlx.nn as nn

PREFIX, MODULE_LR = "mhca", float(os.environ.get("CARRYA_LR", "1e-4"))
HID = 1024
_D = [None]


class Adapter(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.g = mx.ones((d,))
        self.v = nn.Linear(d, HID, bias=False)
        self.u = nn.Linear(HID, d, bias=False)
        self.u.weight = mx.random.normal(self.u.weight.shape, key=mx.random.key(1115)) * 1e-3

    def __call__(self, h):
        return h + self.u(nn.gelu(self.v(mx.fast.rms_norm(h.astype(mx.float32), self.g, 1e-6))))


def setup(D, args):
    D.mhca = [Adapter(D.d)]
    _D[0] = D


def standin(j, h, anc):
    return _D[0].mhca[0](h)
