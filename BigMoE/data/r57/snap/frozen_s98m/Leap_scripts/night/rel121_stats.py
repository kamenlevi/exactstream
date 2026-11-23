"""rel121_stats.py (Leap 11-13): small summaries of the decode loop's runs.json for rel121_check.sh.
  p2smoke <runs.json>: first token (pfs) median / range over the runs, how many took P2, the lifetime peak
  one <runs.json>:     the first run's prompt length, P2 chunks, first token and lifetime peak"""
import json, statistics, sys

mode, path = sys.argv[1], sys.argv[2]
try:
    rr = [r for r in json.load(open(path)) if r.get("pfs") is not None]
except Exception as e:
    print(f"no runs ({type(e).__name__}: {e})"); sys.exit(0)
peak = lambda rows: max(r["mem"][6] for r in rows if r.get("mem") and len(r["mem"]) > 6)
if mode == "p2smoke":
    f = [r["pfs"] for r in rr]
    print(f"first token median {statistics.median(f):.2f} s ({min(f):.2f}-{max(f):.2f}) over {len(rr)} prompts, "
          f"P2 on {sum(1 for r in rr if r.get('p2'))}, lifetime peak {peak(rr):.2f} GB")
elif mode == "one":
    r = rr[0]
    print(f"prompt {r.get('P')} tokens, P2 chunks {r.get('p2')}, first token {r['pfs']:.2f} s, "
          f"{r.get('n')} tokens at {r.get('tps') or 0:.1f} tok/s, lifetime peak {peak([r]):.2f} GB")
