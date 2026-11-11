"""Fit12 (Thu 5 Nov 20:2x; 12 GB plan lever 1): offline replay of T3's two-pass walk under any certificate rule, on the
calibration records of arm g1k1u (/Users/kamen/Fit12/scripts/t3cal.sh -> /Users/kamen/Fit12/data/scr/t3cal/*_g1k1u_cal.json:
per check the draft d, its per-row cf, pass 1's per-row max softmax p1 / top-2 margin m1 / argmax g1, and the EXACT tokens
new, since g1k1u runs the exact check after pass 1 on every check).
Exact reference per check: new = d[:m] + [e_m], m = the first row whose draft the exact model rejects (or len(d): all kept).
Rule = (score, threshold, cfmin): score p1 or m1; a row is certified when score >= threshold; checks whose first draft cf <
cfmin go straight to the exact check (mode 0). The walk is t3_twopass.generate's: n = the first row that is uncertified or
where pass 1 disagrees with the draft; certified n -> commit d[:n] + [g1[n]] (mode 1); else pass 2 = the exact check of rows
n..j, j = pass 1's first predicted miss at or after n (mode 2). Divergence = a committed token that differs from the exact
model's (only at a certified row: inside the walk at row m, or the walk's last token); counted once per check.
Time model [K, T3R2 t80ck1u phases, /Users/kamen/Fit12/scripts/t3_exact_split.py]: draft 11 ms, pass 1 131, commit 13 per
check that runs pass 1; one exact call (<= 4 rows) 147 ms; calls = ceil(rows verified / 4), stopping at the first miss.
Usage: python -I -B t3cal_replay.py [CAL_DIR] | --selftest"""
import glob, json, math, os, sys

DRAFT, COMMIT, XCALL = 11.0, 13.0, 147.0
P1 = {"g1k1u": 131.0, "g1k2": 166.5}   # [K] T3R2 p47: pass 1 per phase, t80ck1u (P1-K1) / t80ck2 (top-2)
P1C = [131.0]


def exact_ref(c):
    d, new = c["d"], c["new"]
    m = len(new) - 1                                   # new = d[:m] + [e_m]
    assert new[:m] == d[:m], "new must extend the draft's accepted prefix"
    return m


def walk(c, score, thr, cfmin):
    """-> (tokens, ms, diverged, mode)"""
    d, new = c["d"], c["new"]
    m = exact_ref(c)
    if cfmin is not None and c["cf"] and c["cf"][0] < cfmin:
        k = len(new)                                   # the exact check: rows 0..m, <= 4-row calls up to the first miss
        return k, DRAFT + XCALL * math.ceil((m + 1) / 4), False, 0
    s, g1 = c[score], c["g1"]
    n = 0
    while n < len(d) and s[n] >= thr and d[n] == g1[n]:
        n += 1
    if s[n] >= thr:                                    # pass 1 decides: commit d[:n] + [g1[n]]
        div = n > m or (n == m and g1[n] != new[m]) or (n < m)   # n < m here means pass 1 rejected an exact-kept draft
        return n + 1, DRAFT + P1C[0] + COMMIT, div, 1
    if n > m:                                          # the walk certified a draft the exact model rejects (row m)
        return n, DRAFT + P1C[0] + COMMIT, True, 1
    j = n
    while j < len(d) and g1[j] == d[j]:
        j += 1
    k = min(m, j) + 1                                  # pass 2 = the exact check of rows n..j: exact up to its first miss
    return k, DRAFT + P1C[0] + COMMIT + XCALL * math.ceil((min(m, j) - n + 1) / 4), False, 2


def replay(cal, score, thr, cfmin):
    tok = ms = div = 0
    modes = [0, 0, 0]
    for c in cal:
        if score not in c:                             # pass 1 did not run on this check (cannot replay it)
            continue
        k, t, dv, md = walk(c, score, thr, cfmin)
        tok += k; ms += t; div += dv; modes[md] += 1
    return dict(tok=tok, s=ms / 1e3, tps=tok / max(ms / 1e3, 1e-9), div=div, div_per_k=1e3 * div / max(tok, 1), modes=modes)


def selftest():
    # draft 5 rows; exact model rejects row 2 (e_2 = 9)
    c = dict(d=[1, 2, 3, 4, 5], new=[1, 2, 9], cf=[0.95] * 5, g1=[1, 2, 3, 4, 5, 6], p1=[0.99, 0.99, 0.99, 0.99, 0.99, 0.99],
             m1=[0.9] * 6)
    assert walk(c, "p1", 0.8, 0.9) == (6, DRAFT + P1C[0] + COMMIT, True, 1)          # certifies the wrong row 2: divergence
    c2 = dict(c, p1=[0.99, 0.99, 0.5, 0.99, 0.99, 0.99])
    k, t, dv, md = walk(c2, "p1", 0.8, 0.9)                                     # row 2 uncertified -> pass 2 from row 2
    assert (k, dv, md) == (3, False, 2) and t == DRAFT + P1C[0] + COMMIT + XCALL, (k, t, dv, md)
    c3 = dict(c, g1=[1, 2, 9, 4, 5, 6])                                         # pass 1 rejects row 2 with the exact token
    assert walk(c3, "p1", 0.8, 0.9)[:3] == (3, DRAFT + P1C[0] + COMMIT, False)
    c4 = dict(c, cf=[0.5] * 5)                                                  # cf gate -> exact check
    assert walk(c4, "p1", 0.8, 0.9) == (3, DRAFT + XCALL, False, 0)
    c5 = dict(c, g1=[1, 7, 3, 4, 5, 6])                                         # pass 1 rejects an exact-kept row 1
    assert walk(c5, "p1", 0.8, 0.9)[2] is True
    print("selftest OK")


def main():
    if "--selftest" in sys.argv:
        return selftest()
    D = sys.argv[1] if len(sys.argv) > 1 else "/Users/kamen/Fit12/data/scr/t3cal"
    out = {}
    for form in ("g1k1u", "g1k2", "g1k3", "g1k4", "g1k5"):
        cal = [c for f in sorted(glob.glob(os.path.join(D, f"*_{form}_cal.json"))) for c in json.load(open(f))]
        if not cal:
            continue
        # pass 1's cost: T3R2's real-arm value where measured, else this calibration run's own p1 phases (G1 mode reads
        # 2-9 % high: g1k1u 141 vs 130, g1k2 170 vs 166 ms; so a conservative price for a new form)
        ph = [q for f in glob.glob(os.path.join(D, f"*_{form}_ph.json")) for q in json.load(open(f)) if q[0] == "p1"]
        P1C[0] = P1.get(form, 1e3 * sum(q[6] - q[2] for q in ph) / max(1, len(ph)))
        print(f"== {form} (pass 1 {P1C[0]:.0f} ms): {len(cal)} checks; divergences = committed tokens != the exact model's "
              f"(a LOWER bound on damage: pass 1's KV in certified rows also shifts later rows; quality = the map's)")
        for thr in (0.8, 0.9, 0.95, 0.98):
            for cfmin in (None, 0.9):
                r = replay(cal, "p1", thr, cfmin)
                print(f"  p1 >= {thr:<4} cfmin {str(cfmin):4s}: {r['tps']:5.1f} tok/s modelled | modes exact/certified/pass2 "
                      f"{r['modes']} | {r['div']} divergences ({r['div_per_k']:.2f} per 1k)")
                out[f"{form} p1 {thr} cf {cfmin}"] = r
    json.dump(out, open(os.path.join(D, "replay.json"), "w"), indent=1)


def _old_main(cal, D):
    base = replay(cal, "p1", 0.8, 0.9)
    print(f"today's rule (p1 >= 0.8, cf >= 0.9): {base['tps']:.1f} tok/s modelled, {base['div']} divergences "
          f"({base['div_per_k']:.2f} per 1k tokens), modes {base['modes']}")
    rows = []
    for score, thrs in (("p1", (0.5, 0.6, 0.7, 0.8, 0.9, 0.95)), ("m1", (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9))):
        for thr in thrs:
            for cfmin in (None, 0.5, 0.7, 0.8, 0.9, 0.95):
                r = replay(cal, score, thr, cfmin)
                rows.append((score, thr, cfmin, r))
    ok = [x for x in rows if x[3]["div"] <= base["div"]]
    ok.sort(key=lambda x: -x[3]["tps"])
    print("best rules at <= today's divergences:")
    for score, thr, cfmin, r in ok[:8]:
        print(f"  {score} >= {thr}, cfmin {cfmin}: {r['tps']:.1f} tok/s (x{r['tps'] / base['tps']:.3f}), {r['div']} div, modes {r['modes']}")
    json.dump(dict(base=base, rows=[dict(score=a, thr=b, cfmin=c, **r) for a, b, c, r in rows]),
              open(os.path.join(D, "replay.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
