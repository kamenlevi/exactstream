import json, glob, statistics as st
H = "/Users/kamen/StrataEval/h2h"
for d in ("results_low24", "results_low12", "results_fit12"):
    for arm in ("B1", "D1"):
        try:
            s = json.load(open(f"{H}/{d}/{arm}.summary.json"))
        except FileNotFoundError:
            continue
        pk = {k: v for k, v in s.items() if any(w in k for w in ("peak", "footprint", "rss", "time_l"))}
        R = [json.loads(l) for l in open(f"{H}/{d}/{arm}.requests.jsonl")]
        R = [r for r in R if r.get("role") == "measure"]
        cells = []
        for lbl, sets in (("code", ("humaneval", "mbpp")), ("chat", ("ifeval",)), ("math", ("math",))):
            x = [r for r in R if r["set"] in sets and "generated" in r and r.get("wall_total_s")]
            if not x:
                continue
            g = sum(r["generated"] for r in x)
            dec = g / (sum(r["decode_ms"] for r in x) / 1e3); wr = g / sum(r["wall_total_s"] for r in x)
            # the per-task aggregate candidates for the reports' END TO END column
            mean_rate = st.mean(r["generated"] / r["wall_total_s"] for r in x)
            geo = st.geometric_mean([r["generated"] / r["wall_total_s"] for r in x])
            cells.append(f"{lbl} n{len(x)} {dec:.2f} / {wr:.2f} / {st.median(r['wall_ttft_s'] for r in x):.2f} s "
                         f"[mean of rates {mean_rate:.2f}, geo {geo:.2f}]")
        print(d, arm, f"arena {s.get('arena_gb')}", "abort" if s.get("abort") else "", pk, "\n   " + "\n   ".join(cells))
