"""D-05 read-out fix (MTP Head, Wed 10-28 17:2x): mtp_depth.py decodes N = 300 tokens without stopping at the end of the
turn, so short answers continue into post-turn text (new 'assistant' / 'user' turns, runs of newlines) where the MTP
head is out of distribution. Here each prompt's passes are cut at the first pass that commits an end token
(<|im_end|> or <|endoftext|>) among its accepted drafts or its saved bonus row; per-depth rates are recomputed on the
passes up to it (depth 1 over all kept passes, depth j over kept passes with depths < j accepted).
Usage: python sticky/d05_truncate.py -> data/d05_mtp_depth_trunc.txt"""
import json, os
from tokenizers import Tokenizer

tk = Tokenizer.from_file(os.path.expanduser("~/BigMoE/models/bare/tokenizer.json"))
END = {tk.token_to_id(t) for t in ("<|im_end|>", "<|endoftext|>")} - {None}
d = json.load(open(os.path.expanduser("~/TrainLab/data/d05_mtp_depth.json")))
K = d["k"]
lines = [f"D-05 native MTP chain (k = {K}), shipped 24 GB v3 engine, 300-token decodes cut at the first end token "
         f"{sorted(END)}:"]
agg, cnt = {}, {}
for P in d["prompts"]:
    kept = []
    for n, m, g in P["passes"]:
        kept.append((n, m, g))
        if END & set(g[:min(n + 1, len(g))]):            # accepted drafts + the bonus row when it was saved
            break
    a = agg.setdefault(P["domain"], [[0, 0] for _ in range(K)])
    c = cnt.setdefault(P["domain"], [0, 0, 0])
    c[0] += 1
    c[1] += len(kept)
    c[2] += len(P["passes"])
    for n, m, g in kept:
        for j in range(K):
            if j == 0 or n >= j:
                a[j][0] += int(m[j] == g[j])
                a[j][1] += 1
for dom, a in agg.items():
    r = [h / max(t, 1) for h, t in a]
    c = cnt[dom]
    lines.append(f"  {dom:6s}: depth 1 / 2 / 3 = " + " / ".join(f"{x:.3f}" for x in r)
                 + f"  ({c[0]} prompts, {c[1]} of {c[2]} passes kept; depth trials "
                 + "/".join(str(t) for _, t in a) + ")")
print("\n".join(lines))
open(os.path.expanduser("~/TrainLab/data/d05_mtp_depth_trunc.txt"), "w").write("\n".join(lines) + "\n")
