"""CHAIN SIZE follow-up (eval-only, s30 unchanged): does a PROMPT-AWARE lm_head list close the static list's loss?
For each window and anchor s, the drafter's argmax is taken over:
  static K      the K most frequent answer tokens (data/shortlist_counts.npy: Magicoder python + captures)
  +prompt       static K plus every token of this window's prompt (known before drafting)
  +context      static K plus the prompt plus the answer tokens already verified at anchor s (ids[gen .. s + 1])
against the full vocabulary. Teacher-forced chain on MBPP-50 / HE-30 (s30), bands 0-0 / 23-30, kept/32, paired vs full.
Rows read per step = K + the window's extra tokens (logged): the list's size for the cache-resident chain.
-> results/listctx.json   usage: listctx.py [--ks 4096,8192,16384] [--nm 50] [--nh 30]"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import mlx.core as mx
from sb import core as C
from sb.compare import paired

V = 248320


def window_tables(D, f, statics):
    """per rule: rt [n_anchors, 31] (1 right / 0 wrong / -1 none); extra-token counts"""
    z = np.load(f)
    ids = np.asarray(z["ids"], np.int32); gen = int(z["gen"]); W = C.load_W(z)
    a0 = max(0, gen - 2); S = len(ids); n = S - 1 - a0; Dp = 31
    C.REPL.update(R=None, K=1, fn=None)
    C.CUR["f"], C.CUR["train"], C.CUR["x"] = f, False, None
    hs = []

    def on_depth(j, h, anc):
        m = min(h.shape[1], S - 1 - j - a0)
        if m > 0:
            hs.append((j, m, D.norm(h[0, :m]).astype(mx.bfloat16)))
    C.chain(D, W, mx.array(ids), Dp, on_depth, a0)
    # cell order = concatenation over depths j of anchors a0 .. a0 + m - 1
    anc = np.concatenate([a0 + np.arange(m) for _, m, _ in hs])
    tgt = np.concatenate([ids[a0 + j + 1:a0 + j + 1 + m] for j, m, _ in hs])
    Hs = mx.concatenate([x for _, _, x in hs], 0)
    prompt = np.unique(ids[:gen])
    ans_tok = ids[gen:]
    first = {}
    for k, tkn in enumerate(ans_tok):                     # first position of each answer token (absolute index)
        first.setdefault(int(tkn), gen + k)
    cand = np.array(sorted(first), np.int32)              # answer tokens of this window
    cpos = np.array([first[int(c)] for c in cand], np.int64)
    masks = {"full": None}
    for K, sm in statics.items():
        masks[f"S{K // 1024}K"] = sm
        mp = np.array(sm); mp[prompt] = 0.0
        masks[f"S{K // 1024}K+prompt"] = mx.array(mp)
    am = {k: [] for k in masks}
    amc = {K: [] for K in statics}                        # +context: (static + prompt) vs answer-prefix candidates
    CH = 512
    for c0 in range(0, Hs.shape[0], CH):
        lg = C.lm_logits(Hs[c0:c0 + CH])
        for k, mk in masks.items():
            am[k].append(np.asarray(mx.argmax(lg if mk is None else lg + mk, -1)))
        a_c = anc[c0:c0 + CH]
        if len(cand):
            lc = np.asarray(lg[:, mx.array(cand)])                       # [rows, n_cand]
            ok = cpos[None, :] <= (a_c[:, None] + 1)                     # verified by anchor s: ids[.. s + 1]
            lc = np.where(ok, lc, -np.inf)
            best_c = lc.max(1); arg_c = cand[np.argmax(lc, 1)]
        for K in statics:
            k = f"S{K // 1024}K+prompt"
            a_sp = am[k][-1]
            l_sp = np.asarray(mx.take_along_axis(lg, mx.array(a_sp[:, None].astype(np.int32)), -1))[:, 0]
            if len(cand):
                amc[K].append(np.where(best_c > l_sp, arg_c, a_sp))
            else:
                amc[K].append(a_sp)
        mx.eval(lg)
    rules = {k: np.concatenate(v) for k, v in am.items()}
    for K in statics:
        rules[f"S{K // 1024}K+context"] = np.concatenate(amc[K])
    out = {}
    for k, a in rules.items():
        rt = np.full((n, Dp), -1, np.int8); o = 0
        for j, m, _ in hs:
            rt[:m, j - 1] = (a[o:o + m] == ids[a0 + j + 1:a0 + j + 1 + m]).astype(np.int8); o += m
        out[k] = rt
    extra = {"prompt": int(len(prompt)), "answer": int(len(cand))}
    return out, gen, a0, extra


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ks", default="4096,8192,16384"); ap.add_argument("--nm", type=int, default=50)
    ap.add_argument("--nh", type=int, default=30); ap.add_argument("--run", default="s30")
    a = ap.parse_args()
    C.limit_memory(5); C.load_target(); C.drop_trunk()
    t0 = time.time()
    D = C.load_drafter(a.run)
    cnt = np.load(os.path.join(C.DATA, "shortlist_counts.npy"))
    order = np.argsort(-cnt, kind="stable")
    statics = {}
    for K in [int(k) for k in a.ks.split(",") if k]:
        m = np.full(V, -1e9, np.float32); m[order[:K]] = 0.0
        statics[K] = mx.array(m)
    sets = {"mbpp": C.files("mbpp")[:a.nm], "he": C.files("he")[:a.nh]}
    res, per_all, extras = {}, {}, {}
    for st, fs in sets.items():
        tabs = {}
        ex = []
        for f in fs:
            C.wait_pause()
            rts, gen, a0, extra = window_tables(D, f, statics)
            for k, rt in rts.items():
                tabs.setdefault(k, []).append((os.path.basename(f)[:-4], rt, gen, a0))
            ex.append(extra); mx.clear_cache()
        extras[st] = {"prompt_tokens_mean": float(np.mean([e["prompt"] for e in ex])),
                      "answer_tokens_mean": float(np.mean([e["answer"] for e in ex]))}
        summ = {k: C.summarize(v) for k, v in tabs.items()}
        rng = np.random.default_rng(1114)
        res[st] = {}
        for k, (s_, per) in summ.items():
            row = {"0-0": round(100 * s_["0-0"], 2), "23-30": round(100 * s_["23-30"], 2), "kept32": round(s_["kept_per_32"], 3)}
            if k != "full":
                for b in ("0-0", "23-30", "kept"):
                    d = paired(summ["full"][1], per, b, rng)
                    sc = 1 if b == "kept" else 100
                    row["d" + b] = round(sc * d["diff"], 3); row["ci" + b] = [round(sc * x, 3) for x in d["ci95"]]
            res[st][k] = row
            print(f"[lc] {st} {k:16s} 0-0 {row['0-0']:6.2f} 23-30 {row['23-30']:6.2f} kept {row['kept32']:.3f}"
                  + (f"  (vs full: 0-0 {row['d0-0']:+.2f}, kept {row['dkept']:+.3f})" if k != "full" else ""), flush=True)
    out = {"args": vars(a), "res": res, "extra_tokens": extras, "wall_s": round(time.time() - t0, 1)}
    json.dump(out, open(os.path.join(C.RES, "listctx.json"), "w"), indent=1)
    print(f"[lc] DONE in {out['wall_s']:.0f}s; extra list rows per window: {json.dumps(extras)}", flush=True)


if __name__ == "__main__":
    main()
