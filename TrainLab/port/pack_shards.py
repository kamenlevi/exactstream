"""PORT data packer (MTP Head, Wed 11-04 20:3x): the extractor's npz windows -> flat shards a Kaggle dataset can hold
and the trainer memory-maps (port/data_port.py load_windows_shards), so a 30 GB-RAM Kaggle VM trains on ~195 GB of
features without loading them. Per shard: shardNNNN.h (uint16 bf16 bits, rows x D), .tk (int32 rows x 32), .lp (fp16
rows x 32), .ids (int32, n+1 per window), .asst (uint8, n+1 per window); index.json lists the windows in input order.
Bytes per token at D = 2560: 5120 (h) + 128 (tk) + 64 (lp) + 5 = 5,317 -> ~188K tokens per GB; a 195 GB dataset holds
~36.7M tokens (Kaggle caps a dataset at 200 GB). --max-gb stops before the budget; --budget only prints the arithmetic.
Usage: python pack_shards.py OUT_DIR NPZ_DIR [NPZ_DIR ...] [--shard-gb 2] [--max-gb 195] [--budget TOKENS]"""
import argparse, glob, json, os, sys
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("out"); ap.add_argument("src", nargs="*")
ap.add_argument("--shard-gb", type=float, default=2.0); ap.add_argument("--max-gb", type=float, default=195.0)
ap.add_argument("--budget", type=float, default=0)
a = ap.parse_args()


def per_token(D=2560):
    return D * 2 + 32 * 4 + 32 * 2 + 4 + 1


if a.budget:
    bt = per_token()
    print(f"{bt} bytes per token at D 2560: {a.budget / 1e6:.1f}M tokens = {a.budget * bt / 1e9:.1f} GB; "
          f"{a.max_gb:.0f} GB holds {a.max_gb * 1e9 / bt / 1e6:.1f}M tokens")
    sys.exit(0)
out = os.path.expanduser(a.out)
os.makedirs(out, exist_ok=True)
files = [p for d in a.src for p in sorted(glob.glob(os.path.join(os.path.expanduser(d), "*.npz")))]
idx, shard, fh, rows, toks, total, D = [], -1, None, 0, 0, 0, None


def open_shard(s):
    return {k: open(os.path.join(out, f"shard{s:04d}.{k}"), "wb") for k in ("h", "tk", "lp", "ids", "asst")}


for p in files:
    z = np.load(p)
    h, tk, lp, ids, asst = z["h"], z["tk"], z["lp"], z["ids"], z["asst"]
    n = tk.shape[0]
    D = D or h.shape[1]
    assert h.shape == (n, D) and lp.shape == (n, 32) and ids.shape == (n + 1,) and asst.shape == (n + 1,), p
    nb = n * per_token(D) + 5
    if total + nb > a.max_gb * 1e9:
        print(f"budget {a.max_gb} GB reached: stopping before {os.path.basename(p)}")
        break
    if fh is None or rows * per_token(D) + nb > a.shard_gb * 1e9:
        if fh:
            [f.close() for f in fh.values()]
        shard += 1; fh = open_shard(shard); rows = toks = 0
    idx.append(dict(key=os.path.basename(p)[:-4], split=str(z["split"]), shard=shard, n=int(n), row0=rows, tok0=toks))
    fh["h"].write(np.ascontiguousarray(h, np.uint16).tobytes()); fh["tk"].write(np.ascontiguousarray(tk, np.int32).tobytes())
    fh["lp"].write(np.ascontiguousarray(lp, np.float16).tobytes()); fh["ids"].write(np.ascontiguousarray(ids, np.int32).tobytes())
    fh["asst"].write(np.ascontiguousarray(asst, np.uint8).tobytes())
    rows += n; toks += n + 1; total += nb
if fh:
    [f.close() for f in fh.values()]
ntok = sum(e["n"] for e in idx)
json.dump(dict(hidden=D, windows=idx, tokens=ntok, bytes=total, shards=shard + 1), open(os.path.join(out, "index.json"), "w"))
print(f"packed {len(idx)} windows, {ntok} tokens, {total / 1e9:.2f} GB in {shard + 1} shards -> {out}")
