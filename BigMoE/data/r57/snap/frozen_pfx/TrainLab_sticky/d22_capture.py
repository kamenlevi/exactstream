"""D-22 chat data (MTP Head, Fri 10-30): training windows for the MTP head on chat-type text, in L1-20's window format
plus the head's input, from ONE prefill per sequence (BigMoE's streamed loader, compiled paths off, as d16_extract.py):
- data/d22c/<key>.npz: ids, asst (1 on the answer's tokens), tk / lp (the 125B's top-32 next-token ids / log-probs per
  row, fp16), split ('train');
- data/d16w/<key>.npz: wd (the final pre-mixer widened stream, fp16), off 0, ids (sticky/d22_ttt.py's window()).
Sources (the exact ref's own answers, ~/EvalSuite/results/ref/decode.jsonl, prompts from the suite's items): IFEval
prompts outside D-20's 3 (97; tf_gen's 30 eval items are a different set) and MGSM items NOT among data/mgsm_items.jsonl's 30 (50); multi has none
left (all 24 are tf_gen's eval items). Only sequences below 2,051 tokens (QSA = dense attention there).
Resumable; stops under 110 GB free disk; passes wait while PAUSE exists.
Usage: [BigMoE venv] python sticky/d22_capture.py [C]"""
import glob, json, os, shutil, sys, time
for _k, _v in (("MTPLX_NGRAM_PREWARM", "off"), ("MTPLX_COMPILED_VERIFY", "0"), ("MTPLX_QWEN4EXP_COMPILE", "0"),
               ("MTPLX_COMPILED_GDN", "0"), ("MTPLX_COMPILE_AR_FORWARD", "0"), ("MTPLX_COMPILED_TARGET_PREFIX", "0")):
    os.environ[_k] = _v
os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, os.path.expanduser("~/BigMoE"))
sys.path.insert(1, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_pfx/Leap_scripts"))
sys.path.insert(2, os.path.expanduser("~/EvalSuite"))
import numpy as np

H = os.path.expanduser
TD = H("~/TrainLab/data")
TOPK, MAXLEN = 32, 2051


def sources():
    from evalsuite import data as ED                       # the suite's items (prompt ids); read-only
    ref = {}
    for l in open(H("~/EvalSuite/results/ref/decode.jsonl")):
        r = json.loads(l)
        ref[r["key"]] = r["completion_ids"]
    held = {json.loads(l)["key"] for l in open(H("~/EvalSuite/cache/ref/tf_gen.jsonl"))}            # tf_* eval items
    held |= {"mgsm/" + json.loads(l)["key"][len("mgsm_"):] for l in open(os.path.join(TD, "mgsm_items.jsonl"))}
    held |= {json.loads(l)["key"] for l in open(os.path.join(TD, "d05_prompts.jsonl"))}             # D-17's prompts
    held |= {"ifeval/1220", "ifeval/3280", "ifeval/1561", "multi/bg0", "multi/de3", "agent/bash17", "agent/edit00"}  # D-20's
    held_tf = {k.replace("tf_", "", 1) for k in held}
    out = []
    for s in ("ifeval", "mgsm", "multi"):
        for it in sorted(ED.load("dec_" + s), key=lambda x: x["key"]):
            k = it["key"]
            if k in held or k in held_tf or ("tf_" + k) in held or k not in ref:
                continue
            ids = list(it["prompt_ids"]) + list(ref[k])
            if len(ids) >= MAXLEN:
                continue
            out.append(("d22c_" + k.replace("/", "_"), ids, len(it["prompt_ids"])))
    return out


def main():
    items = sources()
    if len(sys.argv) > 2 and sys.argv[2] == "list":
        import collections
        print(len(items), collections.Counter(k.split("_")[1] for k, _, _ in items), sum(len(i) for _, i, _ in items), "tokens")
        return
    import mlx.core as mx
    from bigmoe.loader import load_streamed
    import pausewait                                      # passes wait at their boundary while PAUSE exists
    from mtplx.models import qwen4_exp as Q
    from mtplx import generation as G
    os.makedirs(os.path.join(TD, "d22c"), exist_ok=True); os.makedirs(os.path.join(TD, "d16w"), exist_ok=True)
    REC = {"on": False, "tk": [], "lp": [], "wd": []}
    _text_call = Q.Qwen4ExpTextModel.__call__

    def _text(self, inputs, cache=None, input_embeddings=None):
        out = _text_call(self, inputs, cache, input_embeddings)
        if REC["on"]:
            REC["wd"].append(self._last_widened[0].astype(mx.float16))
            h = out[0]
            for i in range(0, h.shape[0], 256):
                lg = HEAD(h[i:i + 256]).astype(mx.float32)
                top = mx.argpartition(-lg, TOPK - 1, axis=-1)[:, :TOPK]
                v = mx.take_along_axis(lg, top, axis=-1)
                o = mx.argsort(-v, axis=-1)
                top, v = mx.take_along_axis(top, o, axis=-1), mx.take_along_axis(v, o, axis=-1)
                lp = (v - mx.logsumexp(lg, axis=-1, keepdims=True)).astype(mx.float16)
                mx.async_eval(top, lp)
                REC["tk"].append(top.astype(mx.int32)); REC["lp"].append(lp)
        return out
    Q.Qwen4ExpTextModel.__call__ = _text
    t0 = time.time()
    rt, st = load_streamed(H("~/BigMoE/models/bare"), int(sys.argv[1]) if len(sys.argv) > 1 else 51, mtp=False)
    model = rt.model if hasattr(rt, "model") else rt
    HEAD = model.language_model.lm_head
    print(f"[d22c] loaded {time.time() - t0:.0f}s; {len(items)} sequences, {sum(len(x[1]) for x in items)} tokens", flush=True)
    for k, (key, ids, gen) in enumerate(items):
        p1, p2 = os.path.join(TD, "d22c", key + ".npz"), os.path.join(TD, "d16w", key + ".npz")
        if os.path.exists(p1) and os.path.exists(p2):
            continue
        if shutil.disk_usage(H("~")).free / 1e9 < 110:
            print(f"[d22c] free disk under 110 GB: stopping at {k}/{len(items)} (resumable)", flush=True)
            sys.exit(2)
        t1 = time.time()
        REC.update(tk=[], lp=[], wd=[]); REC["on"] = True
        G.restore_or_prefill_prompt_state(rt, ids, base_hidden_variant=None, mtp_history_policy="cycle", capture_hidden=False)
        REC["on"] = False
        st.gate.check()
        tk, lp, wd = mx.concatenate(REC["tk"]), mx.concatenate(REC["lp"]), mx.concatenate(REC["wd"])
        n = int(tk.shape[0])                              # rows prefilled (the last token is not)
        assert int(wd.shape[0]) == n and n in (len(ids), len(ids) - 1), (key, n, int(wd.shape[0]), len(ids))
        N = len(ids)
        asst = np.zeros(N, np.uint8); asst[gen:] = 1
        np.savez(p1, ids=np.array(ids, np.int32), asst=asst, tk=np.array(tk[:N - 1]), lp=np.array(lp[:N - 1]), split="train")
        w = np.zeros((N, wd.shape[1]), np.float16); w[:n] = np.array(wd)
        np.savez(p2, wd=w, off=0, ids=np.array(ids, np.int32))
        mx.clear_cache()
        print(f"[d22c] {k + 1}/{len(items)} {key}: {N} tokens ({N - gen} answer) in {time.time() - t1:.1f}s", flush=True)
    print(f"[d22c] done in {time.time() - t0:.0f}s, peak {mx.get_peak_memory() / 1e9:.1f} GB", flush=True)


if __name__ == "__main__":
    main()
