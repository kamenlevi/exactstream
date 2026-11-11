"""Sequence-clustered CIs for the dh_w4 gating run (Breakthrough's data/tfb12/{v3ctl,dhw4}_c128_e8.npz, read-only): the
gating table bootstrapped by position (optimistic: positions within a sequence are correlated). Positions are mapped back
to sequences from the Eval suite's tf_gen.jsonl (items sorted by key, the 4 own-output sets, every 8th of the combined
list, len(completion_ids) positions each), then the paired difference and dh_w4's mean are bootstrapped by sequence.
usage: gate_cluster.py   (CPU)"""
import json, os
import numpy as np
rows = sorted((json.loads(l) for l in open(os.path.expanduser("~/EvalSuite/cache/ref/tf_gen.jsonl"))), key=lambda r: r["key"])
SETS = ["tf_gsm", "tf_code", "tf_ifeval", "tf_multi"]
sel = [r for r in rows if r["set"] in SETS][::8]
B = os.path.expanduser("~/Breakthrough/data/tfb12")
a, b = np.load(f"{B}/v3ctl_c128_e8.npz"), np.load(f"{B}/dhw4_c128_e8.npz")
TARGET = {"tf_code": 0.00253, "tf_gsm": 0.00449, "tf_ifeval": 0.0204, "tf_multi": 0.0206}
rng = np.random.default_rng(0)
for s in SETS:
    ka, kb = a[f"kld_{s}"], b[f"kld_{s}"]
    lens = [len(r["completion_ids"]) for r in sel if r["set"] == s]
    assert sum(lens) == len(ka) == len(kb), (s, sum(lens), len(ka))
    cuts = np.cumsum([0] + lens)
    seqs = [np.arange(cuts[i], cuts[i + 1]) for i in range(len(lens))]
    d = kb - ka
    bp = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(4000)]
    bs, bm = [], []
    for _ in range(4000):
        pick = np.concatenate([seqs[j] for j in rng.integers(0, len(seqs), len(seqs))])
        bs.append(d[pick].mean()); bm.append(kb[pick].mean())
    per_seq = [f"{d[ix].mean():+.4f}" for ix in seqs]
    t = TARGET[s]
    print(f"{s}: {len(seqs)} seqs ({', '.join(map(str, lens))} positions) | control {ka.mean():.5f} dh_w4 {kb.mean():.5f} "
          f"({100 * kb.mean() / t:.0f} % of {t}) | diff {d.mean():+.5f}: by position [{np.percentile(bp, 2.5):+.5f}, "
          f"{np.percentile(bp, 97.5):+.5f}], by sequence [{np.percentile(bs, 2.5):+.5f}, {np.percentile(bs, 97.5):+.5f}] | "
          f"dh_w4 mean by-sequence 95 % upper {np.percentile(bm, 97.5):.5f} | per-seq diffs {' '.join(per_seq)}")
