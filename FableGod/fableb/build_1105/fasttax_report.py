"""fasttax_report.py (Fable B, build_1105): the FAST-TAX A-B-A read-out from r57_screen's per-check records.
usage: python -I scripts/fasttax_report.py <dir with {tag}_{C}_checks.json> <C> fta1 ftb [ftb2] fta2
Arms whose tag starts with 'ftb' are B (the all-in + R57_FASTTAX); the others are A (the all-in). Metric and line: the
header of fasttax_aba.sh (pre-registered before any run). CPU only; reads JSON, writes stdout."""
import json, math, os, statistics as S, sys

D, C, TAGS = sys.argv[1], sys.argv[2], sys.argv[3:]
g = lambda xs: math.exp(sum(map(math.log, xs)) / len(xs)) if xs else float("nan")
med = lambda xs: S.median(xs) if xs else float("nan")


def load(tag, kind):
    try:
        return json.load(open(os.path.join(D, f"{tag}_{C}_{kind}.json")))
    except OSError:
        return None


def fit(xs, ys):                      # ys = a + b * xs (least squares); the intercept is the fixed cost per check
    n = len(xs)
    if n < 3 or len(set(xs)) < 2:
        return float("nan"), float("nan")
    mx_, my_ = sum(xs) / n, sum(ys) / n
    b = sum((x - mx_) * (y - my_) for x, y in zip(xs, ys)) / sum((x - mx_) ** 2 for x in xs)
    return my_ - b * mx_, b


arms, L = {}, []
for t in TAGS:
    chk, runs, tok = load(t, "checks"), load(t, "runs"), load(t, "tokens")
    if not chk:
        L.append(f"{t:5s} MISSING ({t}_{C}_checks.json)")
        continue
    live = [e for e in chk if not e.get("paused")]
    ck = [x for e in live for x in e["ck"][1:]]
    rows = [x for e in live for x in e["rows"][1:]]
    warm = [c for e in live for c, m in zip(e["ck"][1:], e["miss"][1:]) if m == 0]
    gpu = [c - i for e in live for c, i in zip(e["ck"][1:], e["io"][1:])]
    a, b = fit(rows, ck)
    per = {e["key"]: med(e["ck"][1:]) for e in live if len(e["ck"]) > 1}
    kept = sum(sum(e["kept"]) for e in live) / max(1, sum(len(e["kept"]) for e in live))
    arms[t] = dict(med=med(ck), n=len(ck), warm=med(warm), nwarm=len(warm), gpu=med(gpu), a=a, b=b, per=per, kept=kept,
                   rows=sum(rows) / max(1, len(rows)), drf=med([x for e in live for x in e["drf"][1:]]),
                   tps={r["key"]: r["tpsw"] for r in (runs or []) if "PAUSED" not in r.get("bad", "")},
                   tok={k: ((next(iter(v.values()), None) or [[]])[0] if isinstance(v, dict) else []) for k, v in (tok or {}).items()}, paused=len(chk) - len(live))
    x = arms[t]
    L.append(f"{t:5s} median check {x['med']:6.2f} ms over {x['n']} checks ({len(live)} prompts, {x['paused']} paused); rows "
             f"{x['rows']:.2f}, kept {x['kept']:.2f} per check; check - SSD busy {x['gpu']:.2f}; miss-free checks {x['warm']:.2f} "
             f"(n {x['nwarm']}); fit {x['a']:.1f} + {x['b']:.2f} per row; draft {x['drf']:.2f}; geo steady {g(list(x['tps'].values())):.1f} tok/s")
A = [t for t in TAGS if not t.startswith("ftb") and t in arms]
B = [t for t in TAGS if t.startswith("ftb") and t in arms]
head = (f"FAST-TAX A-B-A at C {C} ({D}): A = the QOM all-in FAST arm, B = A + R57_FASTTAX (ref2 att_single; gdn_inproj only "
        f"without R57_DNR). Median check ms over all checks but each prompt's first. Line: fasttax_aba.sh's header.")
if len(A) < 2 or not B:
    print("\n".join([head] + L + [f"VERDICT: INCOMPLETE (A arms {A}, B arms {B}): no verdict"]))
    raise SystemExit
mA = sum(arms[t]["med"] for t in A) / len(A)
mB = sum(arms[t]["med"] for t in B) / len(B)
drift = max(arms[t]["med"] for t in A) - min(arms[t]["med"] for t in A)
save = mA - mB
keys = set.intersection(*[set(arms[t]["per"]) for t in A + B])
pp = sorted(sum(arms[t]["per"][k] for t in B) / len(B) - sum(arms[t]["per"][k] for t in A) / len(A) for k in keys)
kA = sum(arms[t]["kept"] for t in A) / len(A)
kB = sum(arms[t]["kept"] for t in B) / len(B)
tr = [g([arms[b]["tps"][k] for b in B if k in arms[b]["tps"]]) / g([arms[a]["tps"][k] for a in A if k in arms[a]["tps"]])
      for k in keys if all(k in arms[t]["tps"] for t in A + B)]
same = lambda p, q: sum(1 for k in keys if arms[p]["tok"].get(k) and arms[p]["tok"].get(k) == arms[q]["tok"].get(k))
L.append(f"A mean of medians {mA:.2f} ms, B {mB:.2f} ms: saving {save:+.2f} ms ({100 * save / mA:+.1f} %); the A medians differ by "
         f"{drift:.2f} ms; per prompt B - A: median {med(pp):+.2f} ms, B faster on {sum(1 for d in pp if d < 0)} of {len(pp)}")
L.append(f"miss-free checks: A {sum(arms[t]['warm'] for t in A) / len(A):.2f}, B {sum(arms[t]['warm'] for t in B) / len(B):.2f} ms; "
         f"check - SSD busy: A {sum(arms[t]['gpu'] for t in A) / len(A):.2f}, B {sum(arms[t]['gpu'] for t in B) / len(B):.2f} ms; "
         f"fixed cost (the fit's intercept): A {sum(arms[t]['a'] for t in A) / len(A):.1f}, B {sum(arms[t]['a'] for t in B) / len(B):.1f} ms")
L.append(f"kept per check: A {kA:.3f}, B {kB:.3f} ({100 * (kB / kA - 1):+.2f} %); geo steady tok/s per prompt, B / A: "
         f"{100 * (g(tr) - 1):+.1f} % over {len(tr)}; prompts with identical tokens: {A[0]} vs {A[-1]} {same(A[0], A[-1])} of {len(keys)} "
         f"(the control), {A[0]} vs {B[0]} {same(A[0], B[0])} of {len(keys)}")
eng = {}
for t in TAGS:                       # ref2's engagement counts, from the arm's screen.txt ("FAST-TAX engagement: att_single N, ...")
    try:
        s = open(os.path.join(D, f"{t}_{C}_screen.txt")).read()
        i = s.find("FAST-TAX engagement:")
        eng[t] = int(s[i:].split("att_single ")[1].split(",")[0]) if i >= 0 else 0
    except (OSError, IndexError, ValueError):
        eng[t] = None
L.append("ref2 att_single calls (graph build): " + ", ".join(f"{t} {eng[t]}" for t in TAGS))
engaged = all(eng.get(t) for t in B) and not any(eng.get(t) for t in A)
if drift > 2.0:
    v = f"VOID (the A medians differ by {drift:.2f} ms > 2.0: drift; rerun as the palindrome, FTA_PAL=1)"
elif not engaged:
    v = "VOID (ref2 did not count att_single calls in B only: the switch did not engage as planned)"
elif save >= 2.0 and abs(kB / kA - 1) <= 0.01:
    v = f"PASS (saving {save:.2f} ms >= 2.0, kept within 1 %)"
elif save < 1.0:
    v = f"KILL (saving {save:.2f} ms < 1.0)"
else:
    v = f"BETWEEN (saving {save:.2f} ms; kept {100 * (kB / kA - 1):+.2f} %)"
print("\n".join([head] + L + ["VERDICT: " + v]))
