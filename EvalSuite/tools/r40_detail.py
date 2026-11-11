"""R-40 detail (docs/RESULT_ref2_2026-10-28.md): discordant items per set (ref2 vs BASE, with the floor and the 10-20
ref beside them), loop rows in every arm, the exact REF2 counter check (48 experts / 12 attention / 96 hc_read /
1 hc_eager16 calls per AR step; steps = tokens - 1 at EOS, tokens at the cap), footprint of both arms, and the
baseline's chunk provenance. Usage: .venv/bin/python tools/r40_detail.py [BASE]   (default ref.r2)"""
import json, os, statistics, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from evalsuite import decode_report as dr

BASE = sys.argv[1] if len(sys.argv) > 1 else "ref.r2"


def main():
    its = dr.items()
    V = {n: dr.load(n) for n in ("ref2", BASE, "ref", "floor")}
    keys = sorted(set(V["ref2"]) & set(V[BASE]))
    print("rows:", {n: len(d) for n, d in V.items()})
    sc = {}
    for s, fn in dr.SCORERS.items():
        ks = [k for k in keys if its.get(k, {}).get("set") == s]
        if not ks:
            continue
        for n, d in V.items():
            sc.setdefault(n, {}).update(fn({k: d[k] for k in ks if k in d}, its))
        print(f"\n== {s}: {len(ks)} rows; discordant ref2 vs {BASE} (1 = correct)")
        for k in ks:
            a, b = sc["ref2"][k][1], sc[BASE][k][1]
            if a != b:
                r2, rb = V["ref2"][k], V[BASE][k]
                print(f"  {k:48s} ref2 {a:.0f} ({r2['finish']}, {len(r2['completion_ids'])} tok)  {BASE} {b:.0f} "
                      f"({rb['finish']}, {len(rb['completion_ids'])} tok)  floor {sc['floor'][k][1]:.0f}  "
                      f"ref(10-20) {sc['ref'][k][1]:.0f}")

    print("\n== loop rows (20-token n-gram x3 in the last 400 tokens): loop flag, score, finish per arm")
    for k in keys:
        L = {n: dr.looped(d[k]["completion_ids"]) for n, d in V.items() if k in d}
        if any(L.values()):
            print(f"  {k:40s} " + "  ".join(f"{n}: {'LOOP' if L[n] else '-':4s} {sc[n][k][1]:.0f} {V[n][k]['finish']}"
                                             for n in V))

    print("\n== REF2 counters vs AR steps (steps = n - 1 at EOS, n at the cap; n includes the prefill's first token)")
    bad = []
    for k in keys:
        r = V["ref2"][k]
        n, c = r["ref2"]["n"], r["ref2"]["calls"]
        steps = n if r["finish"] == "length" else n - 1
        exp = {"experts": 48 * steps, "attention": 12 * steps, "hc_read": 96 * steps, "hc_eager16": steps}
        if any(c.get(p) != v for p, v in exp.items()):
            bad.append((k, r["finish"], n, {p: c.get(p) for p in exp}))
    print(f"  rows where the calls are not 48 / 12 / 96 / 1 per AR step: {len(bad)}" + (f" -> {bad[:5]}" if bad else ""))

    print("\n== footprint (GB, row peak = max(sampled max, lifetime))")
    for n in ("ref2", BASE):
        pk = [max(V[n][k]["mem"]["max"], V[n][k]["mem"]["life"]) for k in keys if "mem" in V[n][k]]
        print(f"  {n:7s} rows {len(pk)}  max {max(pk):.2f}  median {statistics.median(pk):.2f}")

    print(f"\n== chunk provenance ({BASE}; ref2's is in tools/ref2_summary.py)")
    ch = [json.loads(l) for l in open(os.path.join(ROOT, "results", BASE, "chunks.jsonl"))]
    pv = sorted({json.dumps({k: x for k, x in c["provenance"].items()
                             if k.startswith(("bigmoe", "mlx", "mtplx", "evalsuite", "sha/"))}, sort_keys=True) for c in ch})
    print(f"  {len(ch)} chunks, {len(pv)} distinct:" + "".join("\n   " + p for p in pv))


if __name__ == "__main__":   # evalplus checks code in spawned processes, which re-import __main__ on macOS
    main()
