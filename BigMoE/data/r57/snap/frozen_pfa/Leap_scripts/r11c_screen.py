"""R-11c SB (NewIdeas Tue ~20:31: "a small rapid screen"; Leap, pre-registered Tue 2026-10-27 20:34 EET in
~/Leap/TABLE_350.md 'R-11c'): the cheap exact cycle variants from R-11, proved bitwise and timed at 1K and 8K.
  S   the engine's cycle (R-11's S): draft, read its tokens, PLE + eager check build, acceptance read, lazy commit
  SB  S with the check built async every 4 layers (CA's M / b5's B) and the commit replay submitted on its own right
      after the acceptance read (A7)
Two lanes per context, each with its own identically prefilled 12-layer cache (CA's slice) and drafter cache, fed
the same text tokens (keep fixed at 10). The lanes run in alternating blocks of 8 cycles, so one lane's async work
never lands in the other's timing. Bitwise: every cycle's draft tokens and argmax tokens, and the final cache states.
Lines: PASS: SB bitwise = S at both contexts AND S - SB >= 1.0 ms per 12-layer cycle at both. KILL: not bitwise, or
< 0.3 ms at both.
Usage: [BigMoE venv] r11c_screen.py -> ~/Leap/data/r11c_screen.txt"""
import os, sys, time
sys.argv = [sys.argv[0], "r11c"]                         # ca_anat's MODE: load only, no run
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_pfa/Leap_scripts"))
import ca_anat as CA
import numpy as np
import mlx.core as mx
import mlx.nn as nn

Q, CS, model, D = CA.Q, CA.CS, CA.model, CA.D
PAUSE = os.path.expanduser("~/.local/share/quiet/PAUSE")
lines = []
def say(s):
    print(s, flush=True); lines.append(s)
med = lambda v: float(np.median(v))

sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_pfa/Leap_third_party"))
import dflash_model_mlx as DF
class Target(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.QuantizedEmbedding(248320, 2560, group_size=64, bits=4)
        self.lm_head = nn.QuantizedLinear(2560, 248320, bias=False, group_size=64, bits=4)
w = mx.load(os.path.expanduser("~/BigMoE/models/bare/model-00016-of-00017.safetensors"))
tgt = Target()
tgt.load_weights([(f"embed_tokens.{s}", w[f"language_model.model.embed_tokens.{s}"]) for s in ("weight", "scales",
                  "biases")] + [(f"lm_head.{s}", w[f"language_model.lm_head.{s}"]) for s in ("weight", "scales",
                  "biases")])
del w
draft = DF.load_draft("incoai/Qwen3.8-Flash-Next-DFlash2")
draft.bind(tgt)
draft.load_weights(os.path.join(D, "l120_ft.safetensors"), strict=False)
mx.eval(draft.parameters(), tgt.parameters())
MASK = int(draft.config.mask_token_id)
F = int(draft.fc.weight.shape[-1])

def select(block, hidden):
    logits = draft.compute_logits(hidden)
    sel = draft.candidate_selector
    cand = mx.argpartition(logits, -sel.top_k, axis=-1)[..., -sel.top_k:]
    unary = mx.take_along_axis(logits, cand, axis=-1)
    hp = sel.hidden_projection(hidden)
    pred, path = block[:, 0], []
    for p in range(hp.shape[1]):
        sc = unary[:, p] + mx.sum(sel.predecessor_codebook(pred)[:, None] * hp[:, p, None]
                                  * sel.successor_codebook(cand[:, p]), axis=-1)
        pred = mx.take_along_axis(cand[:, p], mx.argmax(sc, axis=-1)[:, None], axis=-1)[:, 0]
        path.append(pred)
    return mx.stack(path, axis=1)

R, BS = 16, 16
K = CA.KEEP[R]
MASKS = mx.array([[MASK] * (BS - 1)])

class Lane:
    def __init__(self, P):
        self.cache = CA.prefill(P)
        self.dcache = draft.make_cache()
        self.pos, self.dpos = P, 0
        self.tok = self.ctx = None
        self.rec = []                                    # per cycle: (draft tokens, argmax tokens)
    def states(self):
        return CA.leaves(CA.states(self.cache))

def dbuild(L, tok, ctx):
    blk = mx.concatenate([mx.array([[tok]]), MASKS], axis=1)
    d = select(blk, draft.hidden_states(blk, ctx, L.dcache, 1))
    L.dpos += ctx.shape[1]
    if (trim := L.dcache[0].offset - L.dpos) > 0:
        DF._trim_recent_cache(L.dcache, trim)
    return d

def first(L):
    x = mx.array([CA.ids[L.pos:L.pos + R]])
    snap = CS.snapshot_untrimmable_cache_lazy(L.cache)
    lg = CA.forward("A", x, L.cache)
    g = mx.argmax(lg[0], axis=-1).tolist()
    model.commit_verified_window(L.cache, snap.states, keep_tokens=K, verified_tokens=R)
    L.pos += K
    L.tok, L.ctx = g[K - 1], lg[:, :K, :F].astype(mx.bfloat16)
    mx.eval(L.ctx, *L.states())

def cycle(L, cheap):
    d = dbuild(L, L.tok, L.ctx)
    dl = d.tolist()                                      # sync 1: the draft tokens
    x = mx.array([CA.ids[L.pos:L.pos + R]])
    snap = CS.snapshot_untrimmable_cache_lazy(L.cache)
    lg = CA.forward("M" if cheap else "A", x, L.cache)
    g = mx.argmax(lg[0], axis=-1).tolist()               # sync 2: the acceptance read
    model.commit_verified_window(L.cache, snap.states, keep_tokens=K, verified_tokens=R)
    L.pos += K
    if cheap:
        mx.async_eval(*L.states())                       # the commit replay goes out on its own (A7)
    L.tok, L.ctx = g[K - 1], lg[:, :K, :F].astype(mx.bfloat16)
    L.rec.append((dl, g))

def block(L, cheap, n):
    while os.path.exists(PAUSE):
        time.sleep(5)
    paused = os.path.exists(PAUSE)
    mx.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        cycle(L, cheap)
    mx.eval(L.ctx, *L.states())
    mx.synchronize()
    return 1e3 * (time.perf_counter() - t0) / n, paused or os.path.exists(PAUSE)

RES = {}
for P in (1024, 8192):
    LS, LB = Lane(P), Lane(P)
    first(LS); first(LB)
    block(LS, False, 3); block(LB, True, 3)              # warm-up (also kept in the bitwise record)
    WS, WB = [], []
    for r in range(5):
        order = ((LS, False, WS), (LB, True, WB)) if r % 2 == 0 else ((LB, True, WB), (LS, False, WS))
        got = [(lst, block(L, cheap, 8)) for L, cheap, lst in order]
        if any(p for _, (_, p) in got):
            say(f"  ctx {P}: round {r} had a PAUSE, dropped")
            continue
        for lst, (wall, _) in got:
            lst.append(wall)
    same_tok = LS.rec == LB.rec
    same_state = all(bool(mx.array_equal(a, b).item()) for a, b in zip(LS.states(), LB.states()))
    diff = med([a - b for a, b in zip(WS, WB)])
    RES[P] = (same_tok and same_state, diff)
    say(f"  ctx {P}: S {med(WS):.2f} ms, SB {med(WB):.2f} ms per 12-layer cycle; S - SB {diff:.2f} ms (paired median of "
        f"{len(WS)} rounds x 8 cycles); bitwise: drafts + tokens of {len(LS.rec)} cycles {same_tok}, final states "
        f"{same_state}")
    del LS, LB
    mx.clear_cache()
ok = all(b for b, _ in RES.values())
gain = min(d for _, d in RES.values())
v = "KILL (not bitwise)" if not ok else "PASS" if gain >= 1.0 else "KILL" if max(d for _, d in RES.values()) < 0.3 \
    else "BETWEEN"
say(f"R-11c verdict (PASS: bitwise AND S - SB >= 1.0 ms per 12-layer cycle at 1K and 8K; KILL: not bitwise or < 0.3 ms "
    f"at both): {v}; per-check saving at 48 layers ~{4 * gain:.1f} ms if the check-build part scales with layers")
open(os.path.join(D, "r11c_screen.txt"), "w").write("\n".join(lines) + "\n")
