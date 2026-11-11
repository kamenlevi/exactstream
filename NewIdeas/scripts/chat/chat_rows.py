"""Chat (12 IFEval, BigMoE N45: nie = exact EB, nil = lossy L) and math (nme / nml): rows verified per check vs kept,
check and draft time by row count. Question: are most verified rows wasted, and what does a check cost at fewer rows?"""
import json, statistics as st, collections
D = "/Users/kamen/BigMoE/data/r57/"
for arm in ("nie", "nil", "nme", "nml"):
    rows, kept, ck, drf = [], [], [], []
    by = collections.defaultdict(list)
    for t in (arm + "1_256", arm + "2_256"):
        for c in json.load(open(D + t + "_checks.json")):
            n = len(c["ck"])
            for i in range(1, n):                      # skip check 0
                r, k = c["rows"][i], c["kept"][i]
                rows.append(r); kept.append(k); ck.append(c["ck"][i]); drf.append(c["drf"][i])
                by[r].append((c["ck"][i], c["drf"][i], k))
    hist = collections.Counter(rows)
    top = sorted(hist.items(), key=lambda x: -x[1])[:6]
    print(f"{arm}: checks {len(rows)}; rows mean {st.mean(rows):.1f} median {st.median(rows)}; kept mean {st.mean(kept):.2f} "
          f"median {st.median(kept)}; kept/rows {sum(kept)/sum(rows):.2f}; ck median {st.median(ck):.1f} ms, drf median {st.median(drf):.1f}")
    print("   rows histogram (top):", ", ".join(f"{r}:{n}" for r, n in top))
    line = []
    for r in sorted(by):
        if len(by[r]) >= 15:
            v = by[r]
            line.append(f"{r}r ck {st.median(x[0] for x in v):.1f} drf {st.median(x[1] for x in v):.1f} kept {st.mean(x[2] for x in v):.2f} (n {len(v)})")
    print("   by rows:", " | ".join(line))

print("\nmisses / SSD per check (checks >= 1):")
for arm in ("nie", "nil", "nme"):
    rec = collections.defaultdict(list)
    for t in (arm + "1_256", arm + "2_256"):
        for c in json.load(open(D + t + "_checks.json")):
            n = len(c["ck"])
            for i in range(1, n):
                r = c["rows"][i]
                b = "1-2" if r <= 2 else ("3-4" if r <= 4 else "5+")
                rec[b].append((c["ck"][i], c["miss"][i], c["io"][i], c["svc"][i] if c.get("svc") else 0.0, c.get("mb", [0]*n)[i]))
    for b in ("1-2", "3-4", "5+"):
        v = rec[b]
        if not v:
            continue
        print(f"  {arm} rows {b}: n {len(v)}; ck {st.median(x[0] for x in v):.1f} ms; misses {st.median(x[1] for x in v):.0f}; "
              f"demand SSD busy {st.median(x[2] for x in v):.1f} ms; gate service {st.median(x[3] for x in v):.1f} ms; MB read {st.median(x[4] for x in v):.0f}")
    lo = [x for b in rec for x in rec[b] if x[1] <= 10]
    if lo:
        print(f"  {arm}: checks with <= 10 misses: n {len(lo)}, ck median {st.median(x[0] for x in lo):.1f} ms (the no-SSD floor)")
