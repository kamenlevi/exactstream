"""Mission 20 report (Fable's PREFILL_WARM, ~/FableGod/CACHE.md 5): reads r57_screen's side files in data/pw (or data/r57)
for the arms pwc_C / pww_C / pwc_Cr / pww_Cr (C 160, 256) and prints per C: misses in the first three checks per prompt
(the mean of the two passes per arm), warm vs ctl in %, by domain and in total; the warm-up's own cost (ms, misses, MB) and
the end-to-end view (pre-loop + the first three cycles); every token vs chat23's REF2-T reference, HumanEval/16 (no stored
reference) across all its runs. Line (C 256): <= -8 % with every token equal = PASS, > -3 % = KILL, else BETWEEN.
usage: .venv/bin/python scripts/pw_report.py [DIR=data/pw] [OUT=DIR/pw_report.txt]"""
import json, os, sys
import numpy as np

D = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/BigMoE/data/pw")
OUTF = sys.argv[2] if len(sys.argv) > 2 else os.path.join(D, "pw_report.txt")
REF = json.load(open(os.path.expanduser("~/BigMoE/data/r57/chat23_ref_tokens.json")))
L = []
def say(s=""):
    print(s); L.append(s)
def J(tag, kind):
    for d in (D, os.path.expanduser("~/BigMoE/data/r57")):
        try:
            return json.load(open(os.path.join(d, f"{tag}_{kind}.json")))
        except Exception:
            pass
    return None
def per_prompt(tag):
    """key -> (first-3 misses, first-3 cycles ms, pre-loop ms, warm misses, warm MB) of each run of the arm (paused: left out)"""
    C, R = J(tag, "checks") or [], J(tag, "runs") or []
    out = {}
    for i, d in enumerate(C):
        if d.get("paused") or len(d["miss"]) < 3:
            continue
        pre = R[i]["a6"] if i < len(R) else {}
        out.setdefault(d["key"], []).append((sum(d["miss"][:3]), sum(d["drf"][:3]) + sum(d["ck"][:3]),
                                             sum(v for k, v in pre.items() if k.endswith("_ms")),
                                             pre.get("warm_miss", 0.0), pre.get("warm_mb", 0.0)))
    return out
def tokens(tags):
    """(runs equal to chat23's reference, runs compared, keys that differ); keys without a reference: all runs identical"""
    eq, nn, bad, own = 0, 0, set(), {}
    for t in tags:
        for k, d in (J(t, "tokens") or {}).items():
            for outs in d.values():
                for o in outs:
                    want = REF.get(k, {}).get("gate+t", [None])[0]
                    if want is None:
                        own.setdefault(k, []).append(tuple(o))
                        continue
                    nn += 1
                    m = min(len(o), len(want))
                    if o[:m] == want[:m]:
                        eq += 1
                    else:
                        bad.add(k)
    for k, v in own.items():
        nn += len(v)
        if len(set(v)) == 1:
            eq += len(v)
        else:
            bad.add(k)
    return eq, nn, sorted(bad), sorted(own)
say(f"== mission 20 PREFILL_WARM (Fable, CACHE.md 5): misses in the first three checks, warm vs ctl (mean of two passes)")
CS = [int(c) for c in os.environ.get("PW_CS", "160 256").split()]   # the line is on the largest C (11-03: 224 when 256
LINE_C = max(CS)                                       # did not fit the memory)
for C in CS:
    c_, w_ = [per_prompt(f"pwc_{C}"), per_prompt(f"pwc_{C}r")], [per_prompt(f"pww_{C}"), per_prompt(f"pww_{C}r")]
    keys = [k for k in (c_[0] or c_[1]) if any(k in x for x in w_)]
    if not keys:
        say(f"\nC {C}: no runs")
        continue
    mean = lambda arms, k, j: float(np.mean([r[j] for a in arms for r in a.get(k, [])]))
    say(f"\nC {C}:")
    tot = {"code": [0.0, 0.0], "agent": [0.0, 0.0]}
    for k in keys:
        mc, mw = mean(c_, k, 0), mean(w_, k, 0)
        dom = "agent" if k.startswith("agent") else "code"
        tot[dom][0] += mc; tot[dom][1] += mw
        say(f"  {k:24s} ctl {mc:6.0f} (passes {[r[0] for a in c_ for r in a.get(k, [])]}), warm {mw:6.0f} "
            f"({[r[0] for a in w_ for r in a.get(k, [])]}): {100 * (mw / mc - 1):+.1f} %; warm-up {mean(w_, k, 2):.0f} ms, "
            f"{mean(w_, k, 3):.0f} misses, {mean(w_, k, 4):.0f} MB; end to end (pre-loop + 3 cycles) ctl {mean(c_, k, 1):.0f} "
            f"vs warm {mean(w_, k, 1) + mean(w_, k, 2):.0f} ms")
    for dom, (a, b) in tot.items():
        if a:
            say(f"  {dom}: {a:.0f} -> {b:.0f} ({100 * (b / a - 1):+.1f} %)")
    a, b = tot["code"][0] + tot["agent"][0], tot["code"][1] + tot["agent"][1]
    eq, nn, bad, own = tokens([f"pw{x}_{C}{r}" for x in "cw" for r in ("", "r")])
    d = b / a - 1
    v = ("FAIL (a token differs)" if bad or not nn else "PASS" if d <= -0.08 else "KILL" if d > -0.03 else "BETWEEN")
    say(f"  total: {a:.0f} -> {b:.0f} ({100 * d:+.1f} %); tokens: {eq} / {nn} runs equal (REF2-T chat23"
        f"{'; ' + ', '.join(own) + ' across its runs' if own else ''}){'; differ: ' + ', '.join(bad) if bad else ''} -> {v}"
        f"{f' (the line is on C {LINE_C})' if C != LINE_C else ''}")
open(OUTF, "w").write("\n".join(L) + "\n")
