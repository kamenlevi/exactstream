"""T2 context-truncation oracle (results/T2_TRUNCATION.md, lines written first). For sampled MBPP anchors s, the 2B runs
on only the K tokens ending at s plus the drafted rows s+1..s+30; its post-norm final hidden at s+j-1 replaces s30's
stand-in at depth j. Scored on the sampled anchors beside the chain and the oracle. No training.
usage: trunc.py [--ks 0,16,64,256] [--n 50] [--every 3] [--sink 0] -> results/t2_trunc[_sinkN].json"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import mlx.core as mx
from sb import core as C


def fwd_states(seqs):
    """equal-length token lists -> [B, L, d] post-norm final hidden (fp32), one batched forward without a cache"""
    from mlx_lm.models.base import create_attention_mask, create_ssm_mask
    t = C.tm()
    x = t.embed_tokens(mx.array(np.asarray(seqs, np.int32)))
    fa, ssm = create_attention_mask(x, None), create_ssm_mask(x, None)
    for layer in t.layers:
        x = layer(x, mask=ssm if layer.is_linear else fa, cache=None)
    return t.norm(x).astype(mx.float32)


def trunc_table(ids, a0, samp, K, sink, S, d):
    """R [n_all, 30, d] (zeros off the sample) + mask [n_all]: the truncated-context state at s+1..s+30 per sampled s"""
    n_all = S - 1 - a0
    R = np.zeros((n_all, 30, d), np.float32)
    mask = np.zeros(n_all, bool)
    groups = {}
    for s in samp:
        lo = max(0, s - K + 1) if K > 0 else s + 1
        hi = min(S, s + 31)
        seq = list(ids[lo:hi])
        if sink and lo > sink:
            seq = list(ids[:sink]) + seq
        nd = hi - (s + 1)
        groups.setdefault(len(seq), []).append((s, seq, nd))
    for L, items in groups.items():
        for b0 in range(0, len(items), 48):
            part = items[b0:b0 + 48]
            H = np.array(fwd_states([p[1] for p in part]))
            for (s, _, nd), h in zip(part, H):
                R[s - a0, :nd] = h[L - nd:]
                mask[s - a0] = True
    return mx.array(R), mx.array(mask)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ks", default="0,16,64,256"); ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--every", type=int, default=3); ap.add_argument("--sink", type=int, default=0)
    ap.add_argument("--keyframes", default="", help="LEVERS_A lever 2: refresh only at depths j %% k == 0, e.g. 1,4,8,16")
    ap.add_argument("--kfK", type=int, default=64, help="the composer's context K for the keyframes")
    a = ap.parse_args()
    C.limit_memory(6)
    C.load_target()                                      # the trunk stays: this test runs the 2B
    D = C.load_drafter("s30")
    Ks = [int(k) for k in a.ks.split(",") if k]
    KF = [int(k) for k in a.keyframes.split(",") if k]
    fs = C.files("mbpp")[:a.n]
    modes = ["chain", "true"] + [f"K{k}" for k in Ks] + [f"kf{k}_c{a.kfK}" for k in KF] + [f"kf{k}_true" for k in KF]
    cnt = {m: {f"{lo}-{hi}": [0, 0] for lo, hi in C.BANDS} for m in modes}
    t0, ntrunc, nsamp, check = time.time(), {k: 0 for k in Ks}, 0, None
    for wi, f in enumerate(fs):
        C.wait_pause()
        z = np.load(f)
        ids = np.asarray(z["ids"], np.int32); gen = int(z["gen"]); S = len(ids); a0 = max(0, gen - 2)
        W = C.load_W(z).astype(mx.float32)
        samp = [s for s in range(a0, S - 1) if (s - a0) % a.every == 0]
        nsamp += len(samp)
        if wi == 0:                                      # sanity: the batched forward over the FULL prefix = the captured state
            s = samp[len(samp) // 2]
            h = np.array(fwd_states([list(ids[:min(S, s + 31)])]))[0]
            nd = min(S, s + 31) - (s + 1)
            ref = np.array(W[0, s + 1:s + 1 + nd])
            hn = h[-nd:] / np.linalg.norm(h[-nd:], axis=-1, keepdims=True); rn = ref / np.linalg.norm(ref, axis=-1, keepdims=True)
            check = float(np.sqrt(((hn - rn) ** 2).sum(-1).mean()))
            print(f"[t2] sanity: full-prefix forward vs captured state, direction error {check:.4f}", flush=True)
        tabs = {"chain": C.chain_table(D, f)[0], "true": C.chain_table(D, f, R=W)[0]}
        for k in Ks:
            ntrunc[k] += sum(1 for s in samp if k == 0 or s - k + 1 > 0)
            R, mk = trunc_table(ids, a0, samp, k, a.sink, S, W.shape[-1])

            def hook(j, h, anc, R=R, mk=mk):
                i = anc - a0
                return mx.where(mk[i][:, None], R[i, j - 2], h[0].astype(mx.float32))[None]
            tabs[f"K{k}"] = C.chain_table(D, f, fn=hook)[0]
        if KF:
            Rc, mkc = trunc_table(ids, a0, samp, a.kfK, a.sink, S, W.shape[-1])
            Rt = np.zeros((S - 1 - a0, 30, W.shape[-1]), np.float32); Wn = np.array(W[0])
            for s in samp:
                nd = min(30, S - 1 - s); Rt[s - a0, :nd] = Wn[s + 1:s + 1 + nd]
            Rt = mx.array(Rt)
            for kf in KF:
                for src, R_ in (("c%d" % a.kfK, Rc), ("true", Rt)):
                    def hookk(j, h, anc, R=R_, mk=mkc, kf=kf):
                        if j % kf:
                            return h
                        i = anc - a0
                        return mx.where(mk[i][:, None], R[i, j - 2], h[0].astype(mx.float32))[None]
                    tabs[f"kf{kf}_{src}"] = C.chain_table(D, f, fn=hookk)[0]
        rows = np.array([s - a0 for s in samp])
        for m, rt in tabs.items():
            r = rt[rows]
            for lo, hi in C.BANDS:
                c = r[:, lo:hi + 1]
                cnt[m][f"{lo}-{hi}"][0] += int((c == 1).sum()); cnt[m][f"{lo}-{hi}"][1] += int((c >= 0).sum())
        mx.clear_cache()
        if (wi + 1) % 10 == 0 or wi == 0:
            print(f"[t2] {wi + 1}/{len(fs)} windows, {nsamp} anchors, {time.time() - t0:.0f}s: 23-30 " + " ".join(
                f"{m} {100 * cnt[m]['23-30'][0] / max(cnt[m]['23-30'][1], 1):.2f}" for m in modes), flush=True)
    out = {"args": vars(a), "anchors": nsamp, "truncated_share": {k: ntrunc[k] / max(nsamp, 1) for k in Ks}, "sanity_full_dir_err": check,
           "bands": {m: {b: (v[0] / v[1] if v[1] else None) for b, v in cnt[m].items()} for m in modes},
           "counts": cnt, "wall_s": time.time() - t0}
    if KF:                                               # the composer's block pass: 64 + k tokens, 24 layers, no lm_head
        z = np.load(fs[0]); ids0 = list(np.asarray(z["ids"], np.int32))
        out["composer_ms"] = {}
        for kf in KF:
            seq = [ids0[:a.kfK + kf]]
            for _ in range(3):
                mx.eval(fwd_states(seq))
            ts = []
            for _ in range(10):
                t = time.perf_counter(); mx.eval(fwd_states(seq)); ts.append(time.perf_counter() - t)
            out["composer_ms"][kf] = 1e3 * float(np.median(ts))
            print(f"[t2] composer block pass ({a.kfK} + {kf} tokens): {out['composer_ms'][kf]:.2f} ms = {out['composer_ms'][kf] / kf:.2f} ms per drafted token at k = {kf}", flush=True)
    name = "t2_trunc" + (f"_sink{a.sink}" if a.sink else "") + ("_keyframes" if KF else "") + ".json"
    json.dump(out, open(os.path.join(C.RES, name), "w"), indent=1)
    print("[t2] DONE " + json.dumps({m: {b: round(100 * v, 2) for b, v in out["bands"][m].items() if v is not None and b in ("0-0", "1-2", "7-14", "23-30")} for m in modes}), flush=True)


if __name__ == "__main__":
    main()
