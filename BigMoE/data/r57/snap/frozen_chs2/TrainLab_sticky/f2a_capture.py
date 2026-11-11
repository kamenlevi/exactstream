"""F2a step 2 (MTP Head, Sun 11-08; GPU, the night's phase 1): the 125B's features for the chat head, written to a disk ring.
Per sequence of data/f2a/seqs.npz (f2a_data.py): one streamed prefill (mtp125.trunk_features, chunks of 2048) gives the
widened stream W [S, 10240] (what the engine feeds the MTP head) and the 125B's own top-32 ids + logprobs at every
position. Saved per window, data/f2a/ring/w_<i>.npz: ids int32 [S], wd uint16 [S, 10240] (W's raw bf16 bits, exact),
tk int32 [S, 32], lp float16 [S, 32], asst uint8 [S]. The ring is the night's training set (20.5 KB per token).
First, --eval: the 125B's own greedy IFEval answers as on-policy eval windows (data/f2a/eval/), from EvalSuite's
reference decodes (results/ref/decode.jsonl, the IFEval keys NOT used by D-22's general_items training: none were, it
fed D-22 b2 only; checked in f2a_eval) -> same format.
Stops at: the ring's disk budget (F2A_RING_GB, default 28), the wall deadline (F2A_UNTIL epoch s), SIGTERM, or the
corpus end; resumes from data/f2a/ring/cursor.json. Waits while ~/.local/share/quiet/PAUSE exists.
Usage: [BigMoE venv] python3 -I sticky/f2a_capture.py [--frac 0.5] [--eval N] [--max N]"""
import os, sys, json, time, signal, argparse, glob
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_chs2/TrainLab_root"))
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


# ---- on-policy eval windows: the 125B's greedy IFEval answers (EvalSuite reference decodes), prompt + answer
if a.eval and not glob.glob(os.path.join(EVAL, "e_*.npz")):
    E = os.path.expanduser("~/EvalSuite")
    comp = {}
    for l in open(f"{E}/results/ref/decode.jsonl"):
        r = json.loads(l); comp[r["key"]] = r["completion_ids"]
    items = [json.loads(l) for l in open(f"{E}/data/items/dec_ifeval.jsonl")]
    items = [it for it in items if it["key"] in comp and len(comp[it["key"]]) >= 32][-a.eval:]   # the last N (D-22 b2 took the first)
    for j, it in enumerate(items):
        p, c = list(it["prompt_ids"]), list(comp[it["key"]])
        save(os.path.join(EVAL, f"e_{j:03d}"), p + c, [0] * len(p) + [1] * len(c))
    json.dump([it["key"] for it in items], open(os.path.join(EVAL, "keys.json"), "w"))
    print(f"[f2a-cap] eval windows: {len(items)} on-policy IFEval answers in {time.time() - t0:.0f} s", flush=True)

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
