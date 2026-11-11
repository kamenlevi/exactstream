"""D-22 real-runtime read-out (MTP Head, Fri 10-30; REPORT 'D-22'): the trained MTP head inside MTPLX's own runtime,
the same decode as D-17 (sticky/d17_mtpchain.py: spec125.generate, max_draft=0 so one token per pass = plain greedy,
k_mtp=15 so the head drafts a 15-token chain at every position from the true prefix), with the head's trained parts
installed in bf16 (mtp125.install: fc_hidden, fc_embedding, attention q/k/v/o, router, shared expert; the MoE through
MTPLX's plain path). Prompts: D-05's 8 code / gsm / ifeval / multi (data/d05_prompts.jsonl; D-17 decoded the same with
the stock head into data/d17). Drafts never change the output: every answer must equal D-17's token for token.
  decode RUN FRAC N_TOKENS [DOMAINS]  -> data/d22r_RUN/<key>.npz (resumable)
  read RUN [BASE]                     -> data/d22r_RUN_vs_BASE.txt: tokens per pass K3 / K7 / K15 vs the stock head (data/d17),
                                         paired per answer, 90 % bootstrap CI; the conditional rate per depth
Usage: [BigMoE venv] python sticky/d22_real.py decode base 0.25 300 code,gsm"""
import glob, json, os, sys, time
os.environ["TL_INFER"] = "1"                                 # mtp125: keep MTPLX's fused HC read (inference only)
os.environ.setdefault("MTPLX_NGRAM_PREWARM", "off")
for _k, _v in (("MTPLX_FAMILY_CAPTURE_COMMIT", "1"), ("MTPLX_DROP_EVENTS", "1"), ("MTPLX_LAZY_MTP_HISTORY_APPEND", "1"),
               ("MTPLX_DEFER_REPAIR_EVAL", "1")):
    os.environ.setdefault(_k, _v)
for k in ("MTPLX_COMPILED_VERIFY", "MTPLX_QWEN4EXP_COMPILE", "MTPLX_COMPILE_AR_FORWARD", "MTPLX_COMPILED_TARGET_PREFIX"):
    os.environ.setdefault(k, "0")
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_p1a/TrainLab_root"))
sys.path.insert(1, os.path.expanduser("~/SpecDecode"))
import numpy as np

H = os.path.expanduser
TD = H("~/TrainLab/data")
CK = H("~/TrainLab/ckpt")
MD = H("~/BigMoE/models/bare")


def decode(run, frac, N, doms):
    import mlx.core as mx
    import mtp125 as M
    from bigmoe.loader import load_streamed
    sys.path.insert(1, H("/Users/kamen/BigMoE/data/r57/snap/frozen_p1a/Leap_scripts"))
    import pausewait                                         # passes wait at their boundary while PAUSE exists
    import spec125
    out_dir = os.path.join(TD, f"d22r_{run}"); os.makedirs(out_dir, exist_ok=True)
    tc = json.load(open(os.path.join(MD, "config.json"))); tc = tc.get("text_config", tc)
    P = [json.loads(l) for l in open(os.path.join(TD, "d05_prompts.jsonl"))]
    P = [r for r in P if r["domain"] in doms]
    t0 = time.time()
    C = max(16, round(frac * tc["num_experts"]))
    rt, st = load_streamed(MD, C, mtp=True)
    head = rt.model.language_model.mtp
    if run != "stock":                                       # stockinst: the untrained copies through the same install
        T = M.Trainable(head)
        if run != "stockinst":
            T.load_weights(os.path.join(CK, f"d22_{run}.safetensors"))
        M.install(head, T, dtype=mx.bfloat16)
    print(f"[d22r] loaded {time.time() - t0:.0f}s (C={C}), head '{run}'; {len(P)} prompts", flush=True)
    spec125.generate(rt.model, rt.tokenizer.apply_chat_template([{"role": "user", "content": "Hi"}], add_generation_prompt=True,
                     enable_thinking=False), 8, k_mtp=3, max_draft=0, lookup=0)   # warm-up
    for r in P:
        key = r["key"].replace("/", "_")
        path = os.path.join(out_dir, key + ".npz")
        if os.path.exists(path):
            continue
        ids = [int(t) for t in r["ids"]]
        s = {"trace": []}
        t1 = time.time()
        out, s = spec125.generate(rt.model, ids, N, k_mtp=15, max_draft=0, lookup=0, stats=s)
        st.gate.check()
        T_ = len(out)
        mtp = np.full((T_, 15), -1, np.int64)
        pos = 0
        for x in s["trace"]:
            m = x.get("mtp") or []
            if pos < T_ and m:
                mtp[pos, :min(15, len(m))] = m[:15]
            pos += x["n"] + 1
        np.savez_compressed(path, out=np.array(out), mtp=mtp, prompt=np.array(ids), domain=r["domain"])
        print(f"[d22r] {r['domain']} {key}: {T_} tokens in {time.time() - t1:.0f}s; depth-1 hit "
              f"{np.mean(mtp[:-1, 0] == np.array(out)[:-1]) if T_ > 1 else 0:.3f}", flush=True)
    print(f"[d22r] done in {time.time() - t0:.0f}s, peak {mx.get_peak_memory() / 1e9:.1f} GB", flush=True)


def read(run, base="stock"):
    from tokenizers import Tokenizer
    tk = Tokenizer.from_file(os.path.join(MD, "tokenizer.json"))
    END = {tk.token_to_id(t) for t in ("<|im_end|>", "<|endoftext|>")} - {None}

    def per_answer(out, mtp, K):
        L = next((i for i, t in enumerate(out) if t in END), len(out) - 1) + 1
        p, tok, n = 1, 0, 0
        while p < L:
            k = 0
            while k < K and p + k < L and mtp[p][k] >= 0 and mtp[p][k] == out[p + k]:
                k += 1
            tok += k + 1; n += 1; p += k + 1
        return tok, n

    def rates(items):
        reach, right = np.zeros(16), np.zeros(16)
        for out, mtp in items:
            L = next((i for i, t in enumerate(out) if t in END), len(out) - 1) + 1
            for p in range(1, L):
                for j in range(15):
                    if p + j >= L or mtp[p][j] < 0:
                        break
                    reach[j + 1] += 1
                    if mtp[p][j] != out[p + j]:
                        break
                    right[j + 1] += 1
        return [right[j] / reach[j] if reach[j] else float("nan") for j in range(1, 16)]

    bdir = os.path.join(TD, "d17") if base == "stock" else os.path.join(TD, f"d22r_{base}")
    lines = [f"D-22 real runtime: head '{run}' vs head '{base}' inside MTPLX (spec125, k_mtp 15, the same greedy answers)",
             "tokens per pass (accepted + 1, answers cut at their end token), paired per answer, 90 % bootstrap CI of the ratio"]
    rng = np.random.default_rng(0)
    groups = {}
    for f in sorted(glob.glob(os.path.join(TD, f"d22r_{run}", "*.npz"))):
        key = os.path.basename(f)
        if not os.path.exists(os.path.join(bdir, key)):
            continue
        z, s = np.load(f), np.load(os.path.join(bdir, key))
        o1, o0 = [int(t) for t in z["out"]], [int(t) for t in s["out"]]
        assert o1 == o0, f"{key}: the answers differ (drafts must not change the output)"
        groups.setdefault(str(z["domain"]), []).append((o1, z["mtp"], s["mtp"]))
    for dom, items in groups.items():
        row = []
        for K in (3, 7, 15):
            A = np.array([per_answer(o, m1, K) + per_answer(o, m0, K) for o, m1, m0 in items], float)
            r1, r0 = A[:, 0].sum() / A[:, 1].sum(), A[:, 2].sum() / A[:, 3].sum()
            idx = rng.integers(0, len(A), (4000, len(A)))
            b = (A[idx, 0].sum(1) / A[idx, 1].sum(1)) / (A[idx, 2].sum(1) / A[idx, 3].sum(1))
            row.append(f"K{K} {r0:.2f} -> {r1:.2f} = {r1 / r0:.3f}x (CI {np.quantile(b, .05):.3f}-{np.quantile(b, .95):.3f})")
        lines.append(f"  {dom} ({len(items)} answers, identical): " + "; ".join(row))
        r1, r0 = rates([(o, m1) for o, m1, _ in items]), rates([(o, m0) for o, _, m0 in items])
        lines.append("    rate per depth 1-15 stock -> trained: " + " ".join(f"{a:.2f}>{b:.2f}" for a, b in zip(r0, r1)))
    print("\n".join(lines))
    open(os.path.join(TD, f"d22r_{run}_vs_{base}.txt"), "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    if sys.argv[1] == "decode":
        decode(sys.argv[2], float(sys.argv[3]), int(sys.argv[4]), set((sys.argv[5] if len(sys.argv) > 5 else "code,gsm,ifeval,multi").split(",")))
    else:
        read(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "stock")
