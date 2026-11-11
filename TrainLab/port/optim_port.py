"""PORT optimizer (MTP Head, Wed 11-04 20:2x): mlx.optimizers.AdamW and clip_grad_norm as MLX 0.32.2 implements them
(read from ~/BigMoE/.venv/.../mlx/optimizers/optimizers.py), so the PyTorch trainer reproduces the MLX trainer's steps:
- Adam WITHOUT bias correction (MLX's default): m = b1 m + (1-b1) g; v = b2 v + (1-b2) g^2; p -= lr * m / (sqrt(v) + eps);
  AdamW first scales p by (1 - lr * wd). torch.optim.AdamW always bias-corrects (its first step is ~3.2x smaller).
- clip: normalizer = min(max_norm / (||g|| + 1e-6), 1), every gradient multiplied by it.
A torch.optim.Optimizer subclass, so ZeroRedundancyOptimizer can shard it."""
import math
import torch


class MLXAdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        for grp in self.param_groups:
            lr, (b1, b2), eps, wd = grp["lr"], grp["betas"], grp["eps"], grp["weight_decay"]
            for p in grp["params"]:
                if p.grad is None:
                    continue
                st = self.state[p]
                if not st:
                    st["m"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    st["v"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                g = p.grad
                if wd:
                    p.mul_(1 - lr * wd)
                m, v = st["m"], st["v"]
                m.mul_(b1).add_(g, alpha=1 - b1)
                v.mul_(b2).addcmul_(g, g, value=1 - b2)
                p.sub_(lr * m / (v.sqrt() + eps))
        return None


@torch.no_grad()
def clip_grad_norm_mlx(params, max_norm=1.0):
    """MLX's clip_grad_norm on the .grad of params (in order); returns the total norm (before clipping)"""
    gs = [p.grad for p in params if p.grad is not None]
    sq = torch.zeros((), dtype=torch.float32, device=gs[0].device)
    for g in gs:
        sq = sq + g.float().square().sum()
    total = torch.sqrt(sq)
    normalizer = torch.clamp(max_norm / (total + 1e-6), max=1.0)
    for g in gs:
        g.mul_(normalizer.to(g.dtype))
    return float(total)


def lr_schedule(lr, steps):
    """the MLX trainer's lr_at: linear warm-up over min(20, steps // 10 + 1) steps from 1 %, then cosine to 10 %"""
    warm = min(20, steps // 10 + 1)

    def lr_at(s):
        if s < warm:
            return lr * (0.01 + 0.99 * s / warm)
        return lr * (0.1 + 0.45 * (1 + math.cos(math.pi * min(1.0, (s - warm) / max(1, steps - warm)))))
    return lr_at
