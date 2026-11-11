"""TrainLab (10-22): the per-pass expert budget as a router hook (batch-aware routing, published: XShare ICML'26, OEA
2511.02237). NON-EXACT, opt-in: a row's routing then depends on the other rows of its pass (in decoding: the drafts).
install(model, spec) wraps the router of every target layer (model.language_model.model.layers; the MTP head is left
alone). In calls of 2..MAXROWS rows (MTP verify passes) the router logits outside the pass set become -inf, so the
block's own softmax / top-k / renormalisation run unchanged and every row takes its top-k within the set. Calls of one
row (AR decode, drafts) and of more rows (prompt prefill) pass through untouched.
spec: "bU" = the U experts with the largest summed w^2 over the rows' top-k (w = top-k gate weights normalised over the
top-k; the energy-optimal set when a dropped expert costs w^2); "oK" = OEA's union of every row's top-K. None / "" = off.
Switch for drivers: env TL_BUDGET=b18 (TL_BUDGET_STATS=1 also counts distinct experts per call: natural vs used).
install_jit(model, spec): the budget-aware JIT prefetch. BigMoE's look-ahead (stream.py: layer L+d's router on layer L's
MoE input, top-B ranked experts prefetched, bm["next"]) holds the ORIGINAL routers captured at load, so under the budget
it predicts natural routing (turn14: 50-64 of ~240 prefetches per pass used, 104 natural). Rewiring those references to
budget-masked copies makes the prefetch rank experts inside the predicted pass set. Switch: TL_BUDGET_JIT=1."""
import mlx.core as mx
import mlx.nn as nn

MAXROWS = 8
STATS = {"calls": 0, "nat": [], "used": []}


def parse(spec):
    if not spec:
        return None, None
    if spec[0] == "b":
        return int(spec[1:]), None
    if spec[0] == "o":
        return None, int(spec[1:])
    raise ValueError(f"budget spec {spec!r}: bU or oK")


class BudgetGate(nn.Module):
    def __init__(self, g, k, U=None, k0=None, stats=False):
        super().__init__()
        self.g, self.k, self.U, self.k0, self.stats = g, k, U, k0, stats

    def __call__(self, x):
        lg = self.g(x)
        rows = lg.size // lg.shape[-1]
        if (self.U is None and self.k0 is None) or not 2 <= rows <= MAXROWS:
            return lg
        E = lg.shape[-1]
        lf = lg.astype(mx.float32).reshape(-1, E)
        top = mx.argpartition(-lf, kth=self.k - 1, axis=-1)[:, :self.k]
        if self.k0 is not None:
            keep = mx.zeros((E,), mx.float32).at[mx.argpartition(-lf, kth=self.k0 - 1, axis=-1)[:, :self.k0]
                                                 .reshape(-1)].add(1.0) > 0
            keep = mx.where(keep.sum() >= self.k, keep, mx.ones_like(keep))   # (near-identical rows: unrestricted)
        else:
            w = mx.softmax(mx.take_along_axis(lf, top, axis=-1), axis=-1)
            c = mx.zeros((E,), mx.float32).at[top.reshape(-1)].add((w * w).reshape(-1))
            keep = mx.zeros((E,), mx.float32).at[mx.argpartition(-c, kth=self.U - 1)[:self.U]].add(1.0) > 0
        out = mx.where(keep, lg, mx.array(-mx.inf, lg.dtype))
        if self.stats:
            STATS["calls"] += 1
            STATS["nat"].append((mx.zeros((E,), mx.float32).at[top.reshape(-1)].add(1.0) > 0).sum())
            sel = mx.argpartition(-out.astype(mx.float32).reshape(-1, E), kth=self.k - 1, axis=-1)[:, :self.k]
            STATS["used"].append((mx.zeros((E,), mx.float32).at[sel.reshape(-1)].add(1.0) > 0).sum())
        return out


def target_layers(model):                                  # as the Eval suite's model._text
    tm = model.language_model if hasattr(model, "language_model") else model
    return tm.model.layers


def install(model, spec, stats=False):
    """wrap every target layer's router (idempotent: re-install replaces the spec); returns the number wrapped"""
    U, k0 = parse(spec)
    n = 0
    for layer in target_layers(model):
        mlp = getattr(layer, "mlp", None)
        if mlp is None or not hasattr(mlp, "gate"):
            continue
        g = mlp.gate.g if isinstance(mlp.gate, BudgetGate) else mlp.gate
        mlp.gate = BudgetGate(g, mlp.top_k, U, k0, stats)
        n += 1
    return n


def install_jit(model, spec):
    """rewire every layer's look-ahead routers (switch_mlp._bm["next"]) to budget-masked copies (no stats); returns the
    number rewired"""
    U, k0 = parse(spec)
    n = 0
    for layer in target_layers(model):
        bm = getattr(getattr(getattr(layer, "mlp", None), "switch_mlp", None), "_bm", None)
        if not bm or not bm.get("next"):
            continue
        new = []
        for router, ratio, B in bm["next"]:
            g = router.g if isinstance(router, BudgetGate) else router
            new.append((BudgetGate(g, router.k if isinstance(router, BudgetGate) else layer.mlp.top_k, U, k0, False),
                        ratio, B))
            n += 1
        bm["next"] = new
    return n


def stats_summary():
    """mean distinct experts per wrapped call (one layer of one pass), natural vs used; clears the counters"""
    if not STATS["nat"]:
        return None
    mx.eval(STATS["nat"], STATS["used"])
    nat = sum(float(a) for a in STATS["nat"]) / len(STATS["nat"])
    used = sum(float(a) for a in STATS["used"]) / len(STATS["used"])
    STATS.update(calls=0, nat=[], used=[])
    return dict(natural=nat, used=used)
