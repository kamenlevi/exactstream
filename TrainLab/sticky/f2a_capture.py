"""F2a step 2 (MTP Head, Sun 11-08; GPU, the night's phase 1): the 125B's features for the chat head, written to a disk ring.
Per sequence of data/f2a/seqs.npz (f2a_data.py): one streamed prefill (mtp125.trunk_features, chunks of 2048) gives the
widened stream W [S, 10240] (what the engine feeds the MTP head) and the 125B's own top-32 ids + logprobs at every
position. Saved per window, data/f2a/ring/w_<i>.npz: ids int32 [S], wd uint16 [S, 10240] (W's raw bf16 bits, exact),
tk int32 [S, 32], lp float16 [S, 32], asst uint8 [S]. The ring is the night's training set (20.5 KB per token).
First, --eval: the 125B's own greedy IFEval answers as on-policy eval windows (data/f2a/eval/), from EvalSuite's
reference decodes (results/ref/decode.jsonl, the IFEval keys NOT used by D-22's general_items training: none were, it
fed D-22 b2 only; checked in f2a_eval) -> same format.
Stops at: the ring's disk budget (F2A_RING_GB, default 28), free disk under F2A_DISK_MIN GB (default 28), the wall deadline (F2A_UNTIL epoch s), SIGTERM, or the
corpus end; resumes from data/f2a/ring/cursor.json. Waits while ~/.local/share/quiet/PAUSE exists.
Usage: [BigMoE venv] python3 -I sticky/f2a_capture.py [--frac 0.5] [--eval N] [--max N]"""
import os, sys, json, time, signal, argparse, glob, shutil
sys.path.insert(0, os.path.expanduser("~/TrainLab"))
import mtp125 as M
import numpy as np
import mlx.core as mx

TD = os.path.expanduser("~/TrainLab/data/f2a")
RING, EVAL = os.path.join(TD, "ring"), os.path.join(TD, "eval")
PAUSE = os.path.expanduser("~/.local/share/quiet/PAUSE")
ap = argparse.ArgumentParser()
ap.add_argument("--frac", type=float, default=0.5)              # expert pool share for the prefill (0.5 = C 256)
ap.add_argument("--eval", type=int, default=24)                  # on-policy IFEval eval windows to capture first
ap.add_argument("--max", type=int, default=10 ** 9)              # corpus windows (smoke tests)
a = ap.parse_args()
os.makedirs(RING, exist_ok=True); os.makedirs(EVAL, exist_ok=True)
budget = float(os.environ.get("F2A_RING_GB", "28")) * 2 ** 30
disk_min = float(os.environ.get("F2A_DISK_MIN", "28")) * 2 ** 30   # free-disk floor (amendment 2): the runner TERMs any phase < 25 GB
until = float(os.environ.get("F2A_UNTIL", "1e12"))
stop = {"now": False}
signal.signal(signal.SIGTERM, lambda *_: stop.update(now=True))
M.TOPK = 32
M.limit_memory()
t0 = time.time()
rt, lm, head = M.load(a.frac)
print(f"[f2a-cap] model loaded in {time.time() - t0:.0f} s (frac {a.frac})", flush=True)


def save(path, ids, asst):
    W, top, tlp = M.trunk_features(lm, [int(x) for x in ids])
    wd = (np.array(W[0].astype(mx.float32)).view(np.uint32) >> 16).astype(np.uint16)   # bf16 -> fp32 is exact: top 16 bits
    np.savez(path + ".tmp.npz", ids=np.asarray(ids, np.int32), wd=wd, tk=np.array(top[0]).astype(np.int32),
             lp=np.array(tlp[0]).astype(np.float16), asst=np.asarray(asst, np.uint8))
    os.replace(path + ".tmp.npz", path + ".npz")
    del W, top, tlp
    return len(ids)


def ring_bytes():
    return sum(os.path.getsize(f) for f in glob.glob(os.path.join(RING, "w_*.npz")))


# ---- on-policy eval windows (amendment 1, Sun 01:2x): EvalSuite's reference IFEval decodes are in b5's training (92 of 95;
# all 24 planned), so the 125B answers IFEval prompts that neither b5 (data/d22c) nor dec_ifeval holds: greedy, <= 192 tokens,
# stop at <|im_end|>, the template as EvalSuite's prompts; then the same capture as training windows.
if a.eval and not glob.glob(os.path.join(EVAL, "e_*.npz")):
    from tokenizers import Tokenizer
    tkz = Tokenizer.from_file(os.path.expanduser("~/BigMoE/models/bare/tokenizer.json"))
    IMS, IME = tkz.token_to_id("<|im_start|>"), tkz.token_to_id("<|im_end|>")
    E = os.path.expanduser("~/EvalSuite")
    seen = {os.path.basename(f)[:-4].split("_")[-1] for f in glob.glob(os.path.expanduser("~/TrainLab/data/d22c/*.npz"))}
    seen |= {str(json.loads(l)["key"]).split("/")[-1] for l in open(f"{E}/data/items/dec_ifeval.jsonl") if l.strip()}
    raw = [json.loads(l) for l in open(f"{E}/data/raw/ifeval_input_data.jsonl") if l.strip()]
    raw = [r for r in raw if str(r["key"]) not in seen][:a.eval]
    keys = []
    for j, r in enumerate(raw):
        p = [IMS] + tkz.encode("user\n" + r["prompt"], add_special_tokens=False).ids + [IME] + \
            tkz.encode("\n", add_special_tokens=False).ids + [IMS] + \
            tkz.encode("assistant\n<think>\n\n</think>\n\n", add_special_tokens=False).ids
        cache = lm.make_cache()
        lg = None
        for c0 in range(0, len(p), 2048):
            lg = lm(mx.array([p[c0:c0 + 2048]]), cache=cache)
        tok = int(mx.argmax(lg[0, -1]).item()); out = []
        while len(out) < int(os.environ.get("F2A_GEN_MAX", "192")):
            out.append(tok)
            if tok == IME:
                break
            lg = lm(mx.array([[tok]]), cache=cache)
            tok = int(mx.argmax(lg[0, -1]).item())
        del cache
        save(os.path.join(EVAL, f"e_{j:03d}"), p + out, [0] * len(p) + [1] * len(out))
        keys.append(r["key"])
    json.dump(keys, open(os.path.join(EVAL, "keys.json"), "w"))
    print(f"[f2a-cap] eval windows: {len(keys)} fresh greedy IFEval answers (not in b5's training) in {time.time() - t0:.0f} s",
          flush=True)

z = np.load(os.path.join(TD, "seqs.npz"))
ids_all, asst_all, off = z["ids"], z["asst"], z["off"]
cur_f = os.path.join(RING, "cursor.json")
cur = json.load(open(cur_f)) if os.path.exists(cur_f) else {"next": 0, "tokens": 0, "windows": 0, "seconds": 0.0}
n0, s0, t1 = cur["tokens"], cur["seconds"], time.time()
done = 0
while cur["next"] < len(off) - 1 and done < a.max:
    while os.path.exists(PAUSE) and not stop["now"]:
        time.sleep(2)
    if stop["now"] or time.time() >= until:
        why = "SIGTERM" if stop["now"] else "deadline"; break
    if ring_bytes() >= budget:
        why = "ring budget"; break
    if shutil.disk_usage(RING).free < disk_min:
        why = "disk floor"; break
    i = cur["next"]
    n = save(os.path.join(RING, f"w_{i:06d}"), ids_all[off[i]:off[i + 1]], asst_all[off[i]:off[i + 1]])
    cur["next"] += 1; cur["tokens"] += n; cur["windows"] += 1; done += 1
    cur["seconds"] = s0 + time.time() - t1
    json.dump(cur, open(cur_f + ".tmp", "w")); os.replace(cur_f + ".tmp", cur_f)
    if cur["windows"] % 50 == 0 or done <= 2:
        el = time.time() - t1
        print(f"[f2a-cap] {cur['windows']} windows, {cur['tokens']:,} tokens, {(cur['tokens'] - n0) / max(el, 1e-6):.0f} tok/s "
              f"this run, ring {ring_bytes() / 2 ** 30:.1f} GB, peak {mx.get_peak_memory() / 2 ** 30:.1f} GB", flush=True)
else:
    why = "corpus end" if cur["next"] >= len(off) - 1 else "--max"
print(f"[f2a-cap] STOP ({why}): {cur['windows']} windows, {cur['tokens']:,} tokens in the ring "
      f"({ring_bytes() / 2 ** 30:.1f} GB); {time.time() - t0:.0f} s", flush=True)
