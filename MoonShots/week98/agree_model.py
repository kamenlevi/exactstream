"""WEEK98 size scaling (MoonShots, Sun 15 Nov 2026, orchestrator 01:3x): lever7_agree.py's teacher-forced agreement for any
same-tokenizer model, here Qwen3.5-4B (mlx-community 4-bit, on disk; no download). Per answer row of the bank's HE+ /
MBPP files: the model's greedy top-1 against the 125B's runtime top-1 (tk[:, 0]), all rows and by the 125B's margin. It
also saves per-row picks (rows<tag>_<set>.npz, same fields as rows2b_*) for the joins.
Question: does agreement rise with auditor size (2B: HE+ 94.42, MBPP 85.67) and close the oracle gap?
Read-only: the bank, the model from the HF cache. Writes ~/MoonShots/week98/ only.
usage: ~/BigMoE/.venv/bin/python -I agree_model.py --model mlx-community--Qwen3.5-4B-MLX-4bit --tag 4b [--sets heplus,mbpp]"""
import argparse, glob, json, os, time
import numpy as np

H = os.path.expanduser
BANK = H("~/datasets/deepdraft/bank")
SPLIT = H("~/BigMoE/data/deepdraft/bank/split.json")
OUT = H("~/MoonShots/week98")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--tag", required=True)
    ap.add_argument("--sets", default="heplus,mbpp")
    a = ap.parse_args()
    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.models.base import create_attention_mask, create_ssm_mask
    mx.set_memory_limit(int(9 * 2 ** 30)); mx.set_cache_limit(int(1 * 2 ** 30))
    path = sorted(glob.glob(H(f"~/.cache/huggingface/hub/models--{a.model}/snapshots/*")))[0]
    model, _ = load(path); model.eval(); model.freeze()
    lm = model.language_model if hasattr(model, "language_model") else model
    t = lm.model
    head = lm.lm_head if hasattr(lm, "lm_head") else t.embed_tokens.as_linear
    print(f"loaded {a.model}: {len(t.layers)} layers, lm_head {'own' if hasattr(lm, 'lm_head') else 'tied'}", flush=True)
    split = json.load(open(SPLIT))
    t0 = time.time()
    res = {"model": a.model}
    for st in a.sets.split(","):
        dump = {"fname": [], "row": [], "top1": [], "margin2b": [], "tk0": [], "margin125": []}
        bins = {"all": [0, 0], "m<0.25": [0, 0], "0.25-1": [0, 0], "1-4": [0, 0], "m>=4": [0, 0]}
        for k, nm in enumerate(split[st]):
            z = np.load(os.path.join(BANK, nm))
            ids, lp, tk = np.asarray(z["ids"]), np.asarray(z["lp"], np.float32), np.asarray(z["tk"])
            n, gen = z["raw"].shape[0], int(z["gen"])
            rr = np.array([r for r in range(gen - 1, n) if r + 1 < len(ids)])
            x = t.embed_tokens(mx.array(ids.astype(np.int32))[None])
            fa, ssm = create_attention_mask(x, None), create_ssm_mask(x, None)
            for layer in t.layers:
                x = layer(x, mask=ssm if layer.is_linear else fa, cache=None)
            post = t.norm(x)[0]
            top, mg = [], []
            for i in range(0, len(rr), 256):
                lg = head(post[mx.array(rr[i:i + 256])]).astype(mx.float32)
                ls = lg - mx.logsumexp(lg, axis=-1, keepdims=True)
                t2 = mx.topk(ls, 2, axis=-1)
                top.append(np.asarray(mx.argmax(lg, axis=-1))); mg.append(np.abs(np.asarray(t2[:, 0] - t2[:, 1])))
            top, mg = np.concatenate(top), np.concatenate(mg)
            hit = top == tk[rr, 0]; m = lp[rr, 0] - lp[rr, 1]
            for key, sel in (("all", np.ones_like(hit)), ("m<0.25", m < 0.25), ("0.25-1", (m >= 0.25) & (m < 1)),
                             ("1-4", (m >= 1) & (m < 4)), ("m>=4", m >= 4)):
                bins[key][0] += int((hit & sel).sum()); bins[key][1] += int(sel.sum())
            dump["fname"] += [nm] * len(rr); dump["row"].append(rr); dump["top1"].append(top); dump["margin2b"].append(mg)
            dump["tk0"].append(tk[rr, 0]); dump["margin125"].append(m)
            mx.clear_cache()
        res[st] = {k: dict(agree=round(100 * v[0] / max(v[1], 1), 2), rows=v[1]) for k, v in bins.items()}
        print(f"[{a.tag}] {st}: " + "  ".join(f"{k} {v['agree']:.2f} % ({v['rows']})" for k, v in res[st].items())
              + f"  {time.time() - t0:.0f}s", flush=True)
        np.savez_compressed(os.path.join(OUT, f"rows{a.tag}_{st}.npz"), fname=np.array(dump["fname"]),
                            **{k: np.concatenate(v) for k, v in dump.items() if k != "fname"})
    res["peak_gb"] = round(mx.get_peak_memory() / 2 ** 30, 2)
    json.dump(res, open(os.path.join(OUT, f"agree_{a.tag}.json"), "w"), indent=1)
    print("peak GB", res["peak_gb"])


if __name__ == "__main__":
    main()
