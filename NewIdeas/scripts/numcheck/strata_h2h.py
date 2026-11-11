"""NUMBER CHECK (NewIdeas, Sun 8 Nov): Strata head-to-head numbers re-derived from the per-request rows.
Per arm and set: pooled whole decode (sum generated / sum decode s), whole request (sum generated / sum wall_total s),
first token (median wall_ttft_s), requests. Default cache = passes 1 + 2 pooled; full cache (arena 33) = pass 1;
low-memory runs (results_low24 / results_low12) = every pass found.
usage: /usr/bin/python3 -I strata_h2h.py"""
import glob, json, os, statistics as st

H = os.path.expanduser("~/StrataEval/h2h")
SETS = ("humaneval", "mbpp", "ifeval", "math")


def rows(d, arm):
    out = []
    for p in sorted(glob.glob(f"{H}/{d}/{arm}[0-9].requests.jsonl")):
        out += [json.loads(l) for l in open(p)]
    return [r for r in out if r.get("role") == "measure"]


for label, d in (("default", "results"), ("full33", "results_arena33"), ("low24", "results_low24"),
                 ("low12", "results_low12")):
    for arm in "ABCD":
        R = rows(d, arm)
        if not R:
            continue
        cells = []
        for s in SETS + ("all",):
            x = [r for r in R if s == "all" or r["set"] == s]
            if not x:
                continue
            g = sum(r["generated"] for r in x)
            dec = g / (sum(r["decode_ms"] for r in x) / 1e3)
            wr = g / sum(r["wall_total_s"] for r in x)
            tt = st.median(r["wall_ttft_s"] for r in x)
            cells.append(f"{s} {dec:5.1f}/{wr:5.1f}/{tt:4.2f}")
        print(f"{label:7s} {arm} (n {len(R):3d}) decode/request/ttft: " + " | ".join(cells))
