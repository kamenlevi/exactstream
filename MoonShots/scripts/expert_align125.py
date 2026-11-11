"""Are the 125B's routed experts the same up to a permutation of their 640 intermediate neurons? (Follow-up to
expert_similarity.py, which found raw-coordinate correlation 0; Git Re-Basin-style weight matching.) A SwiGLU expert is
invariant to permuting its intermediate neurons (rows of gate/up, columns of down) and to flipping the sign of one neuron's
(up row, down column) pair. Per neuron j the features are g_j, u_j, d_j (each length 2560, standardized).
  joint match : score(j, k) = corr(g_aj, g_bk) + |corr(u_aj, u_bk) + corr(d_aj, d_bk)|, greedy max-weight matching (no
                scipy here), then the aligned correlation of gate / up / down (sign-corrected per neuron).
  held-out    : match on gate rows only, then score up/down under that permutation. A true permutation aligns them too;
                a spurious one leaves them at the ~0.02 noise level.
Null: the same matching between experts of different layers (no shared origin); greedy matching of 640 neurons in 2560
dims still finds spurious maxima, so the same-layer number must be read against it. Delta coding an expert against an
aligned partner saves ~0.5*log2(1/(1-rho^2)) bits/weight (Gaussian rate-distortion), so rho must reach ~0.87 to save 1 bit.
CPU only, < 1 GB.   usage: expert_align125.py [experts_per_layer]"""
import json, os, struct, sys, time
import numpy as np
NE = int(sys.argv[1]) if len(sys.argv) > 1 else 6
MD = os.path.expanduser("~/BigMoE/models/bare")
idx = json.load(open(f"{MD}/model.safetensors.index.json"))["weight_map"]
rng = np.random.default_rng(3)


def raw(name, sel, dt):
    f = f"{MD}/{idx[name]}"
    with open(f, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]; m = json.loads(fh.read(n))[name]
    a, b = m["data_offsets"]
    mm = np.memmap(f, dt, "r", offset=8 + n + a, shape=((b - a) // np.dtype(dt).itemsize,)).reshape(m["shape"])
    return np.array(mm[sel])


def std(x):                                     # standardize each row
    x = x - x.mean(-1, keepdims=True)
    return x / x.std(-1, keepdims=True)


def experts(L, sel):
    """[{g, u, d}] per expert, each [640, 2560] with standardized rows (d = down columns)."""
    f32 = lambda u: (u.astype(np.uint32) << 16).view(np.float32)
    m = {}
    for proj in ("gate_proj", "up_proj", "down_proj"):
        pre = f"language_model.model.layers.{L}.mlp.switch_mlp.{proj}."
        w = raw(pre + "weight", sel, np.uint32); s = f32(raw(pre + "scales", sel, np.uint16)); b = f32(raw(pre + "biases", sel, np.uint16))
        q = np.stack([(w >> (4 * j)) & 15 for j in range(8)], -1).reshape(len(sel), w.shape[1], -1).astype(np.float32)
        m[proj] = (q.reshape(len(sel), q.shape[1], -1, 64) * s[..., None] + b[..., None]).reshape(len(sel), q.shape[1], -1)
    return [{"g": std(m["gate_proj"][i]), "u": std(m["up_proj"][i]), "d": std(m["down_proj"][i].T)} for i in range(len(sel))]


def greedy_match(S):
    n = S.shape[0]
    used_r, used_c, perm, k = np.zeros(n, bool), np.zeros(n, bool), np.full(n, -1), 0
    for flat in np.argsort(-S, axis=None):
        i, j = divmod(int(flat), n)
        if used_r[i] or used_c[j]:
            continue
        perm[i] = j; used_r[i] = used_c[j] = True; k += 1
        if k == n:
            break
    return perm


def compare(A, B):
    D = A["g"].shape[1]
    Cg, Cu, Cd = (A[x] @ B[x].T / D for x in "gud")
    j = np.arange(Cg.shape[0])
    su = Cu + Cd
    p = greedy_match(Cg + np.abs(su))                      # joint match
    sg = np.sign(su[j, p])
    joint = (Cg[j, p].mean(), (sg * Cu[j, p]).mean(), (sg * Cd[j, p]).mean())
    pg = greedy_match(Cg)                                  # gate-only match, up/down held out
    held = (Cg[j, pg].mean(), np.abs(Cu[j, pg]).mean(), np.abs(Cd[j, pg]).mean())
    ident = (np.diag(Cg).mean(), np.diag(Cu).mean(), np.diag(Cd).mean())
    return np.array(joint + held + ident)


def bits(rho):
    return 0.5 * np.log2(1 / max(1e-9, 1 - rho ** 2))


t0 = time.time()
LAYERS = (0, 24, 47)
E = {L: experts(L, np.sort(rng.choice(512, NE, replace=False))) for L in LAYERS}
print(f"125B, {NE} experts per layer, layers {LAYERS}; loaded in {time.time() - t0:.0f}s", flush=True)
hdr = "joint match g/u/d | gate-only match: g, held-out |u|/|d| | identity g/u/d"
for L in LAYERS:
    same = np.mean([compare(E[L][a], E[L][b]) for a in range(NE) for b in range(a + 1, NE)], 0)
    L2 = LAYERS[(LAYERS.index(L) + 1) % len(LAYERS)]
    null = np.mean([compare(E[L][a], E[L2][a]) for a in range(NE)], 0)
    for tag, v in (("same layer", same), (f"null L{L}-L{L2}", null)):
        print(f"L{L:2d} {tag:14s}: joint {v[0]:.3f}/{v[1]:.3f}/{v[2]:.3f} | gate-only {v[3]:.3f}, held-out {v[4]:.3f}/{v[5]:.3f} "
              f"| identity {v[6]:+.3f}/{v[7]:+.3f}/{v[8]:+.3f}", flush=True)
    rho = same[:3].mean()
    print(f"L{L:2d} delta coding vs the aligned partner would save ~{bits(rho):.3f} bits/weight (rho {rho:.3f}; null rho "
          f"{null[:3].mean():.3f})", flush=True)
print(f"({hdr}; {time.time() - t0:.0f}s)")
