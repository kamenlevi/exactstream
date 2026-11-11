"""PORT walk (MTP Head, Wed 11-04 21:0x): sticky/d02b_walk.py's teacher-forced walk in PyTorch, so a Kaggle session can
score its own checkpoints (kept per 16-row check, per-token rate, per-depth rate, P(first 15 right)), as the Mac does.
Per sample (the 125B's own continuation: f = its 5 taps in fp16, ids, off, gen): checks from a = gen; the drafter's 15
picks (full logits -> top-16, the set R-06's exact two-stage head returns -> the selector chain from the anchor token)
vs ids[a+1..a+15]; k = the leading matches; next anchor a + k + 1, while a < off + n - 1.
- context = the rows the official rotating cache holds at anchor a: positions max(start, a - 2047) .. a - 1, with
  start = max(off, gen - 2047); h = hidden_norm(fc(bf16(f))) computed once per sample with MLX's bf16 arithmetic;
- the drafter runs the trainer's forward (DFlash2.fwd: fp32, = the official path, MLX --check) on all active samples
  at once (S = samples, A = 1, BS = 16); the MLX walk runs it in bf16, so a few argmax ties can differ.
Validation target (L1-20 weights): data/d02b_walk.txt, code kept 9.57 / rate 0.9239, math 7.00 / 0.8707.
Usage: python walk_port.py [--draft W.safetensors] [--sets code=DIR,math=DIR] [--batch 30] [--threads 4] [--bf16-store]"""
import argparse, glob, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
import dflash2_torch as M

ap = argparse.ArgumentParser()
ap.add_argument("--draft", default="~/Leap/data/l120_ft.safetensors")
ap.add_argument("--sets", default="code=~/Leap/data/dfl2d_code,math=~/Leap/data/dfl2d_math")
ap.add_argument("--batch", type=int, default=30); ap.add_argument("--threads", type=int, default=4)
ap.add_argument("--bf16-store", action="store_true"); ap.add_argument("--layers", type=int, default=None)
ap.add_argument("--expand", default=None); ap.add_argument("--out", default="")
A = ap.parse_args()
torch.set_num_threads(A.threads)
BS, SLIDE = 16, M.SLIDE
m = M.build(init=os.path.expanduser(A.draft) if A.draft != "hf" else "hf", expand=A.expand,
            param_dtype=torch.bfloat16 if A.bf16_store else torch.float32)
m.head.q4.unpack_u8(); m.head.q4.packed = torch.empty(0, dtype=torch.int32)
if A.bf16_store:
    M.bf16_storage(m)
w = M.st_load(os.path.join(M.DRAFT_DIR, "model.safetensors"), {"fc.weight", "hidden_norm.weight"})
FC = M.to_torch(*w["fc.weight"])                     # (2560, 12800), bf16 values
HN = M.to_torch(*w["hidden_norm.weight"])
EPS = float(m.c["rms_norm_eps"])
lines = []


def say(s):
    print(s, flush=True); lines.append(s)


@torch.no_grad()
def ctx_features(f16):
    """h = hidden_norm(fc(x)) as MLX computes it on bf16: x = bf16(f), fc output bf16 (fp32 accumulation), RMSNorm
    in fp32 with a bf16 result"""
    x = M.round_bf16(torch.from_numpy(f16.reshape(f16.shape[0], -1).astype(np.float32)))
    out = []
    for r0 in range(0, x.shape[0], 512):
        y = M.round_bf16(x[r0:r0 + 512] @ FC.T)
        out.append(M.round_bf16(y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + EPS) * HN).to(torch.bfloat16))
    return torch.cat(out)


@torch.no_grad()
def picks(hid, anchor_tok, rows=256):
    """the selector chain on the full logits' top-16 (R-06's exact set): 15 picks per sample"""
    S = hid.shape[0]
    h2 = hid.reshape(-1, hid.shape[-1])
    cand, unary = [], []
    for r0 in range(0, h2.shape[0], rows):
        lg = m.head(h2[r0:r0 + rows]) * m.OUTM
        c = torch.topk(lg, m.top_k, dim=-1).indices
        cand.append(c); unary.append(torch.gather(lg, -1, c))
    cand, unary = torch.cat(cand).reshape(S, BS - 1, -1), torch.cat(unary).reshape(S, BS - 1, -1)
    sel = m.candidate_selector
    hp = sel.hidden_projection(hid)
    pred, path = anchor_tok, []
    for p in range(BS - 1):
        sc = unary[:, p] + (sel.rows("predecessor_codebook", pred)[:, None] * hp[:, p, None]
                            * sel.rows("successor_codebook", cand[:, p])).sum(-1)
        pred = torch.gather(cand[:, p], -1, sc.argmax(-1, keepdim=True))[:, 0]
        path.append(pred)
    return torch.stack(path, 1)


@torch.no_grad()
def walk(samples):
    """samples: list of (h bf16 (n, D), ids, off, gen); returns per sample the (k, checkable, anchor) of every check"""
    st = []
    for h, ids, off, gen in samples:
        st.append(dict(h=h, ids=ids, off=off, n=len(ids), a=gen, start=max(off, gen - (SLIDE - 1)), out=[]))
    active = [s for s in st if s["a"] < s["off"] + s["n"] - 1]
    while active:
        U = max(s["a"] - max(s["start"], s["a"] - (SLIDE - 1)) for s in active)
        U = -(-max(1, U) // 128) * 128
        S, D = len(active), active[0]["h"].shape[1]
        hc = torch.zeros((S, U, D)); upos = torch.full((S, U), -1, dtype=torch.long)
        anc = torch.zeros((S, 1), dtype=torch.long); tok = torch.full((S, 1, BS), m.MASK, dtype=torch.long)
        for i, s in enumerate(active):
            a, off = s["a"], s["off"]
            lo = max(s["start"], a - (SLIDE - 1))
            hc[i, :a - lo] = s["h"][lo - off:a - off].to(torch.float32)
            upos[i, :a - lo] = torch.arange(lo, a)
            anc[i, 0] = a; tok[i, 0, 0] = int(s["ids"][a - off])
        hid = m.fwd(hc, upos, anc, tok)                                   # (S, 15, D)
        dr = picks(hid, tok[:, 0, 0]).numpy()
        nxt = []
        for i, s in enumerate(active):
            a, off = s["a"], s["off"]
            true = s["ids"][a - off + 1:a - off + BS]
            k = 0
            while k < len(true) and dr[i, k] == true[k]:
                k += 1
            s["out"].append((k, len(true), a))
            s["a"] = a + k + 1
            if s["a"] < off + s["n"] - 1:
                nxt.append(s)
        active = nxt
    return [s["out"] for s in st]


def stats(name, per_s):
    recs = [(k, m_) for r in per_s for k, m_, _ in r]
    full = [k for k, m_ in recs if m_ == 15]
    kept = np.mean([k + 1 for k, m_ in recs])
    attempted = sum(min(k + 1, m_) for k, m_ in recs)
    rate = sum(k for k, m_ in recs) / max(attempted, 1)
    reach, right = np.zeros(16), np.zeros(16)
    for k, m_ in recs:
        for dd in range(1, min(k + 1, m_) + 1):
            reach[dd] += 1; right[dd] += dd <= k
    per = [right[x] / reach[x] if reach[x] else float("nan") for x in range(1, 16)]
    p15 = float(np.mean([k == 15 for k in full])) if full else float("nan")
    say(f"  {name}: {len(per_s)} samples, {len(recs)} checks; kept per check {kept:.2f}; per-token rate {rate:.4f}; "
        f"P(first 15 right) {p15:.3f} (of {len(full)} full checks)")
    say("    rate per depth 1-15: " + " ".join(f"{x:.3f}" for x in per))


say(f"PORT walk: {A.draft} ({len(m.layers)} layers, {'bf16 storage' if A.bf16_store else 'fp32'}), block 16, top-16 + "
    f"the selector chain, trainer forward (fp32)")
for spec in A.sets.split(","):
    name, folder = spec.split("=")
    files = sorted(glob.glob(os.path.join(os.path.expanduser(folder), "*.npz")))
    t0, per_s = time.time(), []
    for b0 in range(0, len(files), A.batch):                  # samples in batches (memory), each walked to its end
        smp = []
        for p in files[b0:b0 + A.batch]:
            z = np.load(p)
            smp.append((ctx_features(z["f"]), z["ids"], int(z["off"]), int(z["gen"])))
        per_s += walk(smp)
    stats(f"{name} ({time.time() - t0:.0f}s)", per_s)
if A.out:
    open(os.path.expanduser(A.out), "w").write("\n".join(lines) + "\n")
