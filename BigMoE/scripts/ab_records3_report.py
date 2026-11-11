"""THE records3 CONVERSION A/B night's report (scripts/ab_records3.sh): per arm (abA / abA2 row-major around abB / abB2
native+p11) every token vs data/ab/ref_tokens.json, steady tok/s (checks >= 2) and SSD MB per kept token per domain, the median
steady tok/s over prompts, lifetime peak; then the two pre-registered lines on abB2 vs abA (mine: docs/PLAN_records3_convert.md
section 4; M-27b: ~/NewIdeas/QUEUE.md) against A = the mean of abA / abA2, B vs A (the smaller parts alone) and abA2
vs abA (drift across the night).
usage: .venv/bin/python scripts/ab_records3_report.py [C=224] -> data/ab/ab_report.txt"""
import json, os, sys
import numpy as np

R = os.path.expanduser("~/BigMoE/data/r57")
CC = sys.argv[1] if len(sys.argv) > 1 else "224"
REF = json.load(open(os.path.expanduser("~/BigMoE/data/ab/ref_tokens.json")))
DOM = lambda k: "code" if k.startswith("humaneval") else "math" if k.startswith(("math500", "gsm8k")) else \
    "agent" if k.startswith("agent") else "chat"
L = []


def say(s=""):
    print(s); L.append(s)


def J(tag, kind):
    try:
        return json.load(open(os.path.join(R, f"{tag}_{CC}_{kind}.json")))
    except Exception:
        return None


A = {}
for t in ("abA", "abB", "abB2", "abB2z", "abA2"):
    rs = [r for r in (J(t, "runs") or []) if not r.get("bad")]
    if not rs:
        continue
    tk = {k: v["gate+t"] for k, v in (J(t, "tokens") or {}).items() if "gate+t" in v}
    pairs = [(k, o) for k, v in tk.items() if k in REF for o in v]
    ref = lambda k: REF[k]["gate+t"][0]
    bad = sorted({k for k, o in pairs if o[:min(len(o), len(ref(k)))] != ref(k)[:min(len(o), len(ref(k)))]})
    d = {}
    for dom in ("code", "math", "chat", "agent"):
        x = [r for r in rs if DOM(r["key"]) == dom]
        if x:
            d[dom] = dict(tpsw=float(np.mean([r["tpsw"] for r in x])), mb=float(np.mean([r["mb"] for r in x])))
    A[t] = dict(d=d, n=len(pairs), bad=bad, med=float(np.median([r["tpsw"] for r in rs])),
                tps=float(np.mean([d_["tpsw"] for d_ in d.values()])), peak=max(r["mem"][6] for r in rs), runs=len(rs))
say(f"THE records3 CONVERSION A/B at C {CC} (quiet): per domain steady tok/s / SSD MB per kept token")
say("| arm | store, sizes | tokens == ref | code | math | chat | agent | median steady | peak GB |")
say("|---|---|---|---|---|---|---|---|---|")
WHAT = {"abA": "row-major, M-17", "abB": "native+p11+pad, M-17", "abB2": "native+p11+pad, packed45", "abB2z": "+ zero-copy reads", "abA2": "row-major, M-17 (again)"}
for t, a in A.items():
    cells = [f"{a['d'][d_]['tpsw']:.1f} / {a['d'][d_]['mb']:.1f}" if d_ in a["d"] else "-" for d_ in ("code", "math", "chat", "agent")]
    say(f"| {t} | {WHAT[t]} | {a['n'] - len(a['bad'])} / {a['n']}{' BAD ' + ','.join(a['bad']) if a['bad'] else ''} | "
        + " | ".join(cells) + f" | {a['med']:.1f} | {a['peak']:.1f} |")


def rel(x, y):
    return 100.0 * (x / y - 1.0)


say()
def merged(tags):
    """A = the mean of the row-major arms that ran (abA, abA2): A brackets B, so the night's drift cancels"""
    xs = [A[t] for t in tags if t in A]
    if not xs:
        return None
    d = {dom: {k: float(np.mean([x["d"][dom][k] for x in xs if dom in x["d"]])) for k in ("tpsw", "mb")}
         for dom in ("code", "math", "chat", "agent") if all(dom in x["d"] for x in xs)}
    return dict(d=d, tps=float(np.mean([x["tps"] for x in xs])), med=float(np.mean([x["med"] for x in xs])))


AA = merged(("abA", "abA2"))
if "abB2" in A and AA and not ("code" in A["abB2"]["d"] and "code" in AA["d"]):
    say("no code prompts in abB2 / A: the lines need them")
elif "abB2" in A and AA:
    b2, a = A["abB2"], AA
    tok_ok = all(not v["bad"] and v["n"] > 0 for v in A.values())
    mb = rel(b2["d"]["code"]["mb"], a["d"]["code"]["mb"])
    st, md = rel(b2["tps"], a["tps"]), rel(b2["med"], a["med"])
    mine = "KILL" if (not tok_ok or mb > -3) else "PASS" if (mb <= -10 and st >= 0) else "BETWEEN"
    m27 = "KILL" if (not tok_ok or mb > -5 or md < 1) else "PASS" if (mb <= -10 and md >= 3) else "BETWEEN"
    say(f"abB2 vs A (mean of abA / abA2): every token equal in every arm {tok_ok}; code SSD MB per kept token {mb:+.1f} %; steady tok/s (mean of "
        f"domains) {st:+.1f} %; median steady {md:+.1f} %")
    say(f"LINE (BigMoE, PLAN 4): {mine}   |   LINE (M-27b): {m27}   (MoonShots' held-out prediction: code MB -15.0 % at C 224)")
    if not tok_ok:
        say("-> a token differs: --inverse + md5 (the store goes back to row-major), then find why")
if "abB" in A and AA and "code" in A["abB"]["d"] and "code" in AA["d"]:
    b, a = A["abB"], AA
    say(f"abB vs A (the smaller parts alone, same slot counts): code MB {rel(b['d']['code']['mb'], a['d']['code']['mb']):+.1f} % "
        f"(expected about -4.5 % MSB / -6.4 % sidecar bytes per miss), steady {rel(b['tps'], a['tps']):+.1f} %")
if "abA2" in A and "abA" in A and "code" in A["abA2"]["d"] and "code" in A["abA"]["d"]:
    say(f"abA2 vs abA (drift across the night): steady {rel(A['abA2']['tps'], A['abA']['tps']):+.1f} %, code MB "
        f"{rel(A['abA2']['d']['code']['mb'], A['abA']['d']['code']['mb']):+.1f} %")
open(os.path.expanduser("~/BigMoE/data/ab/ab_report.txt"), "w").write("\n".join(L) + "\n")
if "abB2z" in A and "abB2" in A and "code" in A["abB2z"]["d"] and "code" in A["abB2"]["d"]:
    say(f"abB2z vs abB2 (zero-copy reads, M-29): steady {rel(A['abB2z']['tps'], A['abB2']['tps']):+.1f} %, median "
        f"{rel(A['abB2z']['med'], A['abB2']['med']):+.1f} % (MB per kept token unchanged by design: "
        f"{rel(A['abB2z']['d']['code']['mb'], A['abB2']['d']['code']['mb']):+.1f} %)")
open(os.path.expanduser("~/BigMoE/data/ab/ab_report.txt"), "w").write("\n".join(L) + "\n")
