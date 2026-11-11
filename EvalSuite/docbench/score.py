#!/usr/bin/env python3
"""docbench scoring and the pre-registered comparisons (PREREG.md).
  python score.py CONFIG [CONFIG ...]     per-config table (field accuracy, all-fields-correct documents, JSON
                                          validity, per type, per language) + per-field accuracy with --fields
  python score.py --verdict               the pre-registered claims on whatever configs have outputs
  python score.py --pair A B              one paired comparison (McNemar exact, document-clustered bootstrap CI)"""
from __future__ import annotations
import argparse, json, math, os, random, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import schema  # noqa: E402

OUT = os.path.join(os.path.dirname(HERE), "results", "docbench")
TYPES = ["invoice", "receipt", "form", "statement", "contract"]
BOOT, BOOT_SEED = 10000, 1


def docs():
    return [json.loads(l) for l in open(os.path.join(HERE, "set", "docs.jsonl"))]


def scored(config, file="outputs.jsonl"):
    p = os.path.join(OUT, config, file)
    if not os.path.exists(p):
        return None
    outs = {r["id"]: r for r in map(json.loads, open(p))}
    res = []
    for d in docs():
        if d["id"] in outs:
            s = schema.score_doc(d, outs[d["id"]]["text"])
            s["length_stop"] = outs[d["id"]]["finish"] == "length"
            res.append(s)
    return res


def summary(res):
    nf = sum(len(r["fields"]) for r in res)
    ok = sum(sum(r["fields"].values()) for r in res)
    out = dict(docs=len(res), fields=nf, field_acc=100 * ok / nf, doc_acc=100 * sum(r["all_correct"] for r in res) / len(res),
               json_strict=100 * sum(r["json_strict"] for r in res) / len(res),
               json_any=100 * sum(r["json_any"] for r in res) / len(res),
               length_stops=sum(r["length_stop"] for r in res))
    for t in TYPES:
        rs = [r for r in res if r["type"] == t]
        if rs:
            n = sum(len(r["fields"]) for r in rs)
            out[t] = 100 * sum(sum(r["fields"].values()) for r in rs) / n
    for lg in ("en", "de", "fr", "bg"):
        rs = [r for r in res if r["lang"] == lg]
        if rs:
            n = sum(len(r["fields"]) for r in rs)
            out["lang_" + lg] = 100 * sum(sum(r["fields"].values()) for r in rs) / n
    return out


def mcnemar_exact(b, c):
    """two-sided exact McNemar p (binomial on the discordant pairs)"""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def paired(ra, rb):
    """A vs B on the same fields: delta = acc(A) - acc(B) in points, McNemar exact p, 95 % CI by a document-clustered
    bootstrap (10,000 resamples of the documents, seed 1)"""
    da = {r["id"]: r for r in ra}
    db = {r["id"]: r for r in rb}
    ids = [i for i in da if i in db]
    per = []                                       # per document: (n fields, A correct, B correct)
    b = c = 0
    for i in ids:
        fa, fb = da[i]["fields"], db[i]["fields"]
        per.append((len(fa), sum(fa.values()), sum(fb.values())))
        for k in fa:
            b += fa[k] and not fb[k]
            c += fb[k] and not fa[k]
    n = sum(x[0] for x in per)
    delta = 100 * (sum(x[1] for x in per) - sum(x[2] for x in per)) / n
    rng = random.Random(BOOT_SEED)
    boots = []
    for _ in range(BOOT):
        s = [per[rng.randrange(len(per))] for _ in per]
        boots.append(100 * (sum(x[1] for x in s) - sum(x[2] for x in s)) / sum(x[0] for x in s))
    boots.sort()
    return dict(n_docs=len(ids), n_fields=n, delta=delta, a_only=b, b_only=c, p=mcnemar_exact(b, c),
                ci=(boots[int(0.025 * BOOT)], boots[int(0.975 * BOOT) - 1]))


def claim(name, better, worse, res, hi=3.0, lo=1.5):
    """pre-registered rule: TRUE if acc(better) - acc(worse) >= 3 points with p < 0.05; FALSE if the gap is < 1.5
    points (including a negative gap) or not significant; otherwise a small difference (the number)"""
    if res.get(better) is None or res.get(worse) is None:
        return f"{name}: not run ({better} vs {worse})"
    pr = paired(res[better], res[worse])
    d, p = pr["delta"], pr["p"]
    if d >= hi and p < 0.05:
        v = "TRUE"
    elif d < lo or p >= 0.05:
        v = "FALSE"
    else:
        v = f"SMALL DIFFERENCE ({d:+.1f} points)"
    robust = "" if (pr["ci"][0] > 0) == (p < 0.05) or d < lo else " [clustered CI includes 0: field-level only]"
    return (f"{name}: {v}{robust} - {better} minus {worse} = {d:+.2f} points (95 % CI {pr['ci'][0]:+.2f} to "
            f"{pr['ci'][1]:+.2f}), McNemar p = {pr['p']:.3g} ({pr['a_only']} fields right only in {better}, "
            f"{pr['b_only']} only in {worse}, {pr['n_fields']} fields)")


def table(configs, res):
    cols = ["field_acc", "doc_acc", "json_strict"] + TYPES + ["lang_en", "lang_de", "lang_fr", "lang_bg"]
    head = f"{'config':<16}" + "".join(f"{c.replace('lang_', ''):>10}" for c in cols)
    lines = [head]
    for c in configs:
        if res.get(c) is None:
            continue
        s = summary(res[c])
        lines.append(f"{c:<16}" + "".join(f"{s.get(k, float('nan')):>10.1f}" for k in cols) +
                     (f"   ({s['length_stops']} cut at max tokens)" if s["length_stops"] else ""))
    return "\n".join(lines)


def speed_table(configs):
    lines = [f"{'config':<16}{'decode tok/s':>14}{'prefill tok/s':>15}{'peak GB':>10}{'weights GB':>12}"
             f"{'docs/min':>10}"]
    for c in configs:
        p = os.path.join(OUT, c, "speed.json")
        m = os.path.join(OUT, c, "meta.json")
        if not os.path.exists(p):
            continue
        s = json.load(open(p))
        dpm = (60 * json.load(open(m))["docs"] / json.load(open(m))["wall_s"]) if os.path.exists(m) else float("nan")
        lines.append(f"{c:<16}{s['decode_tps']:>14.1f}{s['prefill_tps']:>15.0f}{s['peak_gb']:>10.2f}"
                     f"{s['weights_gb']:>12.2f}{dpm:>10.1f}")
    return "\n".join(lines)


ALL = ["9b-bf16", "9b-q8", "9b-q4", "9b-q4mixed", "27b-q4", "38-27b-q4", "38-27b-q6", "36-35b-a3b-q4"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("configs", nargs="*")
    ap.add_argument("--verdict", action="store_true")
    ap.add_argument("--pair", nargs=2)
    ap.add_argument("--fields", action="store_true")
    ap.add_argument("--file", default="outputs.jsonl")
    a = ap.parse_args()
    if a.pair:
        ra, rb = scored(a.pair[0], a.file), scored(a.pair[1], a.file)
        print(json.dumps(paired(ra, rb), indent=1))
        return
    if a.verdict:
        res = {c: scored(c) for c in ALL}
        print(claim("Claim 1 (9B Q4 much worse than 9B Q8)", "9b-q8", "9b-q4", res))
        print(claim("  same rule, mixed 4-bit (mixed_4_6)", "9b-q8", "9b-q4mixed", res))
        print(claim("Claim 2, headline (9B Q8 much better than Qwen3.8-27B Q4)", "9b-q8", "38-27b-q4", res))
        print(claim("  same rule vs Qwen3.8-27B 6-bit", "9b-q8", "38-27b-q6", res))
        if res.get("27b-q4"):
            print(claim("  same rule vs Qwen3.5-27B Q4 (same-family control)", "9b-q8", "27b-q4", res))
        if res.get("9b-bf16") and res.get("9b-q8"):
            pr = paired(res["9b-bf16"], res["9b-q8"])
            ok = abs(pr["delta"]) < 1.5 or pr["p"] >= 0.05
            print(f"Check (expected 9B Q8 = 9B BF16 within noise): {'YES' if ok else 'NO'} - BF16 minus Q8 = "
                  f"{pr['delta']:+.2f} points (95 % CI {pr['ci'][0]:+.2f} to {pr['ci'][1]:+.2f}), p = {pr['p']:.3g}")
        for x in ("9b-q4mixed", "36-35b-a3b-q4"):
            ref = "9b-q4" if x == "9b-q4mixed" else "9b-q8"
            if res.get(x) and res.get(ref):
                pr = paired(res[x], res[ref])
                print(f"Reported ({x} vs {ref}): {x} minus {ref} = {pr['delta']:+.2f} points (95 % CI "
                      f"{pr['ci'][0]:+.2f} to {pr['ci'][1]:+.2f}), p = {pr['p']:.3g}")
        print()
        print(table(ALL, res))
        print()
        print(speed_table(ALL))
        return
    res = {c: scored(c, a.file) for c in a.configs}
    print(table(a.configs, res))
    if a.fields:
        for c in a.configs:
            if res[c] is None:
                continue
            acc = {}
            for r in res[c]:
                for k, v in r["fields"].items():
                    acc.setdefault((r["type"], k), []).append(v)
            print(f"\n{c}: per field")
            for (t, k), v in acc.items():
                print(f"  {t:<10}{k:<20}{100 * sum(v) / len(v):6.1f}  ({len(v) - sum(v)} wrong)")


if __name__ == "__main__":
    main()
