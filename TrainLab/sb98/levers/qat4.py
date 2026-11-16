"""MTP Head lever 'qat4' (WEEK98 small drafter; REPORT 'WEEK98 S13 QATSCREEN'): fake quantization of the sandbox
head's 8 Linears (fc, q / k / v / o, gate / up / down), the way the engine stores them and M3's Q4 cut reads them:
bf16 -> affine group-64 quantize -> dequantize, at QAT_BITS (default 4). Straight-through: the forward uses the
quantized weight, the gradient reaches the latent fp32 weight unchanged. No new parameters and no stand-in.
Training with --lever qat4 is QAT; reading any run as run@qat4 scores it through the same fake quant (post-hoc Q4 is
s30@qat4). The target's own lm_head and embedding are not the head's and are already 4-bit."""
import os
import mlx.core as mx
import mlx.nn as nn

BITS, GROUP = int(os.environ.get("QAT_BITS", "4")), int(os.environ.get("QAT_GROUP", "64"))
NAMES = ("fc", "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def fq(w):
    q = mx.dequantize(*mx.quantize(w.astype(mx.bfloat16), group_size=GROUP, bits=BITS), group_size=GROUP, bits=BITS)
    return w + mx.stop_gradient(q.astype(w.dtype) - w)


class FQLinear(nn.Linear):
    def __call__(self, x):
        y = x @ fq(self["weight"]).T
        return y + self["bias"] if "bias" in self else y


def setup(D, args):
    for n in NAMES:
        old = getattr(D, n)
        if isinstance(old, FQLinear):
            continue
        m = FQLinear(old.weight.shape[1], old.weight.shape[0], bias="bias" in old)
        m.weight = old.weight
        if "bias" in old:
            m.bias = old.bias
        setattr(D, n, m)
