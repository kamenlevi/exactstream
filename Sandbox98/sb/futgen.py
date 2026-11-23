"""BREAK98 Screen 1 data (sb/futstate.py): the frozen 2B's state at K = 8 PLACEHOLDER rows hung as a chain under each
anchor s (positions s+1 .. s+K; plain causal attention over the true prefix ids[0..s] + the chain).

Method: B capture files in lockstep (one shared position, so one KV offset; right-padded), the target's own caches
(mlx_lm make_cache: KVCache for the 6 full-attention layers, ArraysCache for the 18 Gated-DeltaNet layers). At each
anchor s the caches hold ids[0..s]: snapshot (the ArraysCache slots are rebound, never written in place, so keeping the
list is a snapshot; the KV caches keep their offset), run the K placeholder rows as one chunk, keep the post-norm
states (what the head reads), restore (trim the KV caches by K, put the slots back), then advance by the real token.
Gaps with no needed anchor in the whole batch advance as one chunk.

Variants: A <|fim_pad|> (248063); B the mean of the input-embedding table (an embedding override); C <|vision_pad|>
(248055); T the TRUE tokens ids[s+1 .. s+K] (the check: states must match the stored true W at those positions, which
tests the positions, the snapshot / restore and the chunked-vs-recurrent numerics in one go).
Sets: train = the anchors of the first STEPS slices of core.pass_slices(pool2, ROWS, SEED) (train_multi's defaults:
400, 96, 1115); mbpp / he = the first NM / NH files, anchors gen-2 .. S-2 (core.chain_table's).
Output: data/fut_V/<set dir>/<key>.safetensors: anc [n] int32, W [n, K, d] bf16, top1 [n, K] int32 (the target's own
argmax at each placeholder row: it predicts ids[s + p + 1], the head's depth-p target; eval sets only, else -1).
usage: futgen.py --variant A --sets train,mbpp,he [--batch 16] [--nm 50 --nh 30] [--files N] [--steps-cap N]"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import mlx.core as mx
from sb import core as C
from sb.futstate import K, fut_path

TOK = {"A": 248063, "C": 248055}
PAD = 248044                                  # <|endoftext|>: right padding past a file's end (outputs unused)


def jobs(sets, nm, nh, steps, rows, seed, nfiles=0):
    """-> list of (set, file, sorted needed anchors, ids)"""
    out = []
    for st in sets:
        if st == "train":
            from sb.train import split
            sl = C.pass_slices(split("pool2")[0], rows, seed)[:steps]
            need = {}
            for f, b, r in sl:
                need.setdefault(f, set()).update(range(b, b + r))
            for f in sorted(need):
                out.append((st, f, np.array(sorted(need[f]), np.int64), np.asarray(np.load(f)["ids"], np.int32)))
        else:
            for f in C.files(st)[:{"mbpp": nm, "he": nh}[st]]:
                z = np.load(f)
                ids = np.asarray(z["ids"], np.int32); a0 = max(0, int(z["gen"]) - 2)
                out.append((st, f, np.arange(a0, len(ids) - 1, dtype=np.int64), ids))
    return out[:nfiles] if nfiles else out


def placeholder(variant, t):
    if variant in TOK:
        return t.embed_tokens(mx.array([TOK[variant]]))[0]
    if variant == "B":
        E = t.embed_tokens
        ref = t.embed_tokens(mx.array([TOK["A"]]))[0]
        s, n = mx.zeros((ref.shape[-1],), mx.float32), 0
        V = E.weight.shape[0]
        for c0 in range(0, V, 16384):
            w = mx.dequantize(E.weight[c0:c0 + 16384], E.scales[c0:c0 + 16384], E.biases[c0:c0 + 16384],
                              group_size=E.group_size, bits=E.bits).astype(mx.float32)
            s = s + w.sum(0); n += w.shape[0]
            mx.eval(s)
        return (s / n).astype(ref.dtype)
    raise ValueError(variant)


def cache_arrays(caches):
    out = []
    for c in caches:
        out += [x for x in c.cache if x is not None] if hasattr(c, "cache") else [c.keys, c.values]
    return out


def run_batch(t, lmh, batch, variant, E1, want_top1, log):
    """batch: list of (set, f, anchors, ids) -> {f: (anc, W [n,K,d] bf16 mx, top1 [n,K] np)}"""
    B = len(batch)
    s_lo = min(int(a[0]) for _, _, a, _ in batch); s_hi = max(int(a[-1]) for _, _, a, _ in batch)
    L = s_hi + K + 2
    ids = np.full((B, L), PAD, np.int32)
    for b, (_, _, _, x) in enumerate(batch):
        ids[b, :min(len(x), L)] = x[:L]
    needed = np.zeros((B, L), bool)
    for b, (_, _, a, _) in enumerate(batch):
        needed[b, a] = True
    any_need = needed.any(0)
    caches = lmh.make_cache()
    rows = {b: [] for b in range(B)}; tops = {b: [] for b in range(B)}
    # prefill ids[0 .. s_lo] in chunks
    p = 0
    while p <= s_lo:
        q = min(s_lo + 1, p + 256)
        h = t(mx.array(ids[:, p:q]), caches)
        mx.eval(h, *cache_arrays(caches))
        p = q
    s = s_lo
    nchain = 0
    while True:
        if any_need[s]:
            snap = [(list(c.cache) if hasattr(c, "cache") else c.offset) for c in caches]
            if variant == "T":
                emb = t.embed_tokens(mx.array(ids[:, s + 1:s + 1 + K]))
            else:
                emb = mx.broadcast_to(E1[None, None], (B, K, E1.shape[-1]))
            out = t(None, caches, input_embeddings=emb)                    # [B, K, d] post-norm
            tt = mx.argmax(t.embed_tokens.as_linear(out), -1) if want_top1 else None
            mx.eval(out, *([tt] if tt is not None else []))
            for c, sv in zip(caches, snap):                               # restore
                if hasattr(c, "cache"):
                    c.cache = sv
                else:
                    c.trim(c.offset - sv)
            nb = np.flatnonzero(needed[:, s])
            sel = out.astype(mx.bfloat16)[mx.array(nb)]                      # only the rows kept (frees `out`)
            mx.eval(sel)
            tt_np = np.array(tt) if tt is not None else None
            for i, b in enumerate(nb):
                rows[b].append(sel[i]); tops[b].append(tt_np[b] if tt_np is not None else np.full(K, -1, np.int64))
            nchain += 1
        if s >= s_hi:
            break
        nxt = s + 1 + int(np.argmax(any_need[s + 1:])) if any_need[s + 1:].any() else s_hi
        h = t(mx.array(ids[:, s + 1:nxt + 1]), caches)                   # advance to the next needed anchor
        mx.eval(h, *cache_arrays(caches))
        s = nxt
    res = {}
    for b, (_, f, a, _) in enumerate(batch):
        assert len(rows[b]) == len(a), (f, len(rows[b]), len(a))
        res[f] = (a, mx.stack(rows[b]), np.stack(tops[b]).astype(np.int32))
    log["chains"] += nchain; log["positions"] += s_hi - s_lo + 1
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=["A", "B", "C", "T"])
    ap.add_argument("--sets", default="train,mbpp,he"); ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--nm", type=int, default=50); ap.add_argument("--nh", type=int, default=30)
    ap.add_argument("--steps", type=int, default=400); ap.add_argument("--rows", type=int, default=96)
    ap.add_argument("--seed", type=int, default=1115); ap.add_argument("--files", type=int, default=0)
    ap.add_argument("--mem", type=float, default=4.5); ap.add_argument("--out-root", default="")
    ap.add_argument("--check-out", default="", help="variant T: write the per-offset agreement with the stored W here")
    a = ap.parse_args()
    C.limit_memory(a.mem)
    t0 = time.time()
    m, _ = C.load_target()
    lmh = m.language_model
    t = lmh.model
    E1 = None if a.variant == "T" else placeholder(a.variant, t)
    J = jobs(a.sets.split(","), a.nm, a.nh, a.steps, a.rows, a.seed, a.files)
    J.sort(key=lambda x: (x[0] == "train", int(x[2][0]), int(x[2][-1])))   # lockstep batches of similar ranges
    nanc = sum(len(x[2]) for x in J)
    print(f"[fg] variant {a.variant}: {len(J)} files, {nanc} anchors, batch {a.batch}; ready in {time.time() - t0:.0f}s", flush=True)
    log = {"chains": 0, "positions": 0}
    chk = {"cos": [[] for _ in range(K)], "top1": [[0, 0] for _ in range(K)]}
    done = 0; t1 = time.time()
    for i in range(0, len(J), a.batch):
        C.wait_pause()
        batch = J[i:i + a.batch]
        want = a.variant != "T" and any(st != "train" for st, _, _, _ in batch)
        res = run_batch(t, lmh, batch, a.variant, E1, want, log)
        for st, f, anc, ids in batch:
            an, W, tp = res[f]
            if a.variant == "T":                                           # compare with the stored true states
                z = np.load(f)
                Wt = C.final_norm(mx.array(z["res"])).astype(mx.float32)
                S = len(ids)
                for p in range(1, K + 1):
                    ok = an + p <= S - 2                                    # stored res rows: 0 .. S-2
                    if not ok.any():
                        continue
                    A_ = W[mx.array(np.flatnonzero(ok)), p - 1].astype(mx.float32)
                    T_ = Wt[mx.array(an[ok] + p)]
                    cs = (A_ * T_).sum(-1) / (mx.linalg.norm(A_, axis=-1) * mx.linalg.norm(T_, axis=-1) + 1e-9)
                    ta = mx.argmax(C.lm_logits(A_.astype(mx.bfloat16)), -1); tb = mx.argmax(C.lm_logits(T_.astype(mx.bfloat16)), -1)
                    chk["cos"][p - 1] += np.array(cs).tolist()
                    chk["top1"][p - 1][0] += int((ta == tb).sum()); chk["top1"][p - 1][1] += int(ok.sum())
                continue
            p_out = fut_path(a.variant, f) if not a.out_root else os.path.join(a.out_root, os.path.basename(fut_path(a.variant, f)))
            os.makedirs(os.path.dirname(p_out), exist_ok=True)
            mx.save_safetensors(p_out + ".tmp.safetensors", {"anc": mx.array(an.astype(np.int32)), "W": W,
                                                             "top1": mx.array(tp)})
            os.replace(p_out + ".tmp.safetensors", p_out)
        done += sum(len(x[2]) for x in batch)
        el = time.time() - t1
        print(f"[fg] {min(i + a.batch, len(J))}/{len(J)} files, {done}/{nanc} anchors, {el:.0f}s "
              f"({1000 * el / max(done, 1):.2f} ms/anchor), peak {mx.get_peak_memory() / 1e9:.2f} GB", flush=True)
        mx.clear_cache()
    if a.variant == "T":
        rep = {"cos_mean": [float(np.mean(c)) if c else None for c in chk["cos"]],
               "cos_min": [float(np.min(c)) if c else None for c in chk["cos"]],
               "top1_agree": [x[0] / x[1] if x[1] else None for x in chk["top1"]], "n": [x[1] for x in chk["top1"]]}
        print("[fg] CHECK (true-token chains vs the stored true W, by offset 1..8): cos mean "
              + " ".join(f"{x:.5f}" for x in rep["cos_mean"]) + " | cos min " + " ".join(f"{x:.4f}" for x in rep["cos_min"])
              + " | top-1 agree " + " ".join(f"{100 * x:.2f}" for x in rep["top1_agree"]), flush=True)
        if a.check_out:
            json.dump(rep, open(a.check_out, "w"), indent=1)
    print(f"[fg] DONE variant {a.variant}: {nanc} anchors, {log['chains']} chain steps over {log['positions']} positions in "
          f"{time.time() - t0:.0f}s; peak {mx.get_peak_memory() / 1e9:.2f} GB", flush=True)


if __name__ == "__main__":
    main()
