"""SSD expert reads per generated token, by set, from the rows' decode-phase counters (decode._gen_mtp: bt_b12's
ext.stats()['misses'] = blocking expert loads, defer_stats() = waited / late reads; disk = this process's bytes read).
Usage: .venv/bin/python tools/ssd_per_token.py b12c192 [more variants]"""
import json, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for v in sys.argv[1:]:
    rows = [r for r in map(json.loads, open(os.path.join(ROOT, "results", v, "decode.jsonl"))) if "ssd" in r]
    by = {}
    for r in rows:
        by.setdefault(r["set"], []).append(r["ssd"])
    out = []
    for s, xs in sorted(by.items()) + [("all", [x for xs in by.values() for x in xs])]:
        n = sum(x["n"] for x in xs) or 1
        out.append(f"{s} ({len(xs)} rows): reads {sum(x['reads'] for x in xs) / n:.1f}, waited {sum(x['waited'] for x in xs) / n:.1f},"
                   f" late {sum(x['late'] for x in xs) / n:.1f}, disk {sum(x['disk_mb'] for x in xs) / n:.0f} MB")
    print(f"{v}: SSD per generated token (decode only): " + "; ".join(out))
