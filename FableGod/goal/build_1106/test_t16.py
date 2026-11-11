"""test_t16.py (GOAL, Fri 2026-11-06 21:2x): the guarded INERT arms T16 / TC16 / U16 of build_1106/inert_span.py.
CPU only, python -I. Run test_inert.py beside it first: that is the proof that the Thursday arms are unchanged.
  1  the arms: the Thursday arms' tables equal build_1105's; the guarded arms exist
  2  which rows a guarded arm opens: exactly the rows CD16 opens (same tagger, same guards), with a rule in place of
     'whole'; so the Thursday safety check (every open token inside Python's own comments / string statements) carries over
  3  what a guarded row keeps: the draft only when the 125B ranks it in its top 3 with p >= 0.1 p1; the first refused row
     ends the block; drafts equal to the argmax behave as the exact rule
  4  the cut: a guarded row's confidence is floored at 0.8 before the check (a block is cut later inside a span); rule
     C's rows as shipped are not
  5  the engine's failed CD16 run (BigMoE data/fday, read-only): what T16 would have refused of the tokens CD16 kept"""
import collections
import importlib.util
import json
import math
import os
import re
import sys

import numpy as np

H = os.path.expanduser("~")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import inert_span as INM                                             # noqa: E402


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


RXC = load("relax_core_bigmoe", f"{H}/BigMoE/scripts/relax_core.py")
OLD = load("inert_span_1105", f"{HERE}/../build_1105/inert_span.py")
fails = []


def check(cond, what):
    print(("  ok   " if cond else "  FAIL ") + what, flush=True)
    if not cond:
        fails.append(what)


def mk(text):
    s = INM.InertScan(RXC)
    s.feed(text)
    return s


def modes(s, d, a, arm):
    return [c[1] for c in INM.classify(RXC, s, d, a, arm)]


T16, TC16, U16, CD16, D16 = (INM.parse_arm(a) for a in ("T16", "TC16", "U16", "CD16", "D16"))

print("1  the arms")
check(all(INM.ARMS[k] == v for k, v in OLD.ARMS.items()) and set(INM.ARMS) - set(OLD.ARMS) == {"T16", "TC16", "U16"},
      "the Thursday arms' tables are unchanged; the new arms are T16, TC16, U16")
check(INM.RULE_T3[:3] == INM.RULE_C and INM.RULE_T3[3] == 0.8 and INM.RULE_T8[:3] == ("R2", 8, 0.01),
      "T16's rule is rule C's (top-3 at 0.1 p1); U16's is top-8 at 0.01 p1; both carry a 0.8 confidence floor for the cut")
try:
    INM.parse_arm("T32"); check(False, "an unknown arm is refused")
except ValueError:
    check(True, "an unknown arm is refused (there is no 32-row guarded arm)")

print("2  which rows are open")
sc = mk("Here it is:\n```python\ndef f(x):\n    y = 1\n    # compute the")
check(modes(sc, [" sum"], [" total"], T16) == [INM.RULE_T3] and modes(sc, [" sum"], [" total"], U16) == [INM.RULE_T8]
      and modes(sc, [" sum"], [" total"], CD16) == ["whole"], "a comment row: T16 -> its rule, U16 -> its rule, CD16 -> whole")
check(modes(sc, [" sum\n"], [" total"], T16) == [None] and modes(sc, ["\n"], [" total"], T16) == [None],
      "a draft that holds a newline is exact (the span's end is never relaxed)")
check(modes(sc, [" sum"], ["\n"], T16) == [None] and modes(sc, [" sum"], [" total\n"], T16) == [None],
      "when the 125B's own token ends the comment line, the row is exact (it can always end the comment)")
st = mk("```python\ndef f(x):\n    # type:")
check(modes(st, [" ignore"], [" int"], T16) == [None], "a comment that is or could become a tool comment is exact")
s2 = mk("```python\n# first line")
check(modes(s2, [" a"], [" b"], T16) == [None], "the fence's first two lines are exact (the tagger's rule)")
sc_ = mk("```c\nint f(int x) {\n    int y = 1;\n    # include the")
check(modes(sc_, [" a"], [" b"], T16) == [None], "a fence that is not python is never open")
sd = mk('```python\ndef f(x):\n    """Return the')
check(modes(sd, [" sum"], [" total"], T16) == [INM.RULE_T3] and modes(sd, [" sum"], [" total"], TC16) == [None],
      "a docstring row: T16 -> its rule; TC16 (comments only) -> exact")
check(modes(sd, [' sum"'], [" total"], T16) == [None] and modes(sd, [" sum"], ['"""'], T16) == [None],
      "the docstring's quote character in the draft or in the 125B's token: exact (its end is never relaxed)")
check(modes(mk('```python\ndef f(x):\n    """Add one.\n\n    >>> f(1'), [")"], [","], T16) == [None],
      "a doctest line is exact")
check(modes(mk('```python\ndef f(x):\n    msg = """Hello'), [" a"], [" b"], T16) == [None]
      and modes(mk('```python\ndef f(x):\n    f"""Hello'), [" a"], [" b"], T16) == [None],
      "a triple-quoted string that is data (assigned, or an f-string) is exact")
# the same rows as CD16 on the walks' answers
PI = load("precheck_inert", f"{HERE}/precheck_inert.py")
nopen = ndiff = 0
for dom, top in (("code", None), ("mbpp", None)):
    for x in PI.samples(dom, top):
        ids, gen = x["ids"], x["gen"]
        s0 = INM.InertScan(RXC); s0.feed(PI.dec(ids[gen]))
        txt = [PI.dec(t) for t in ids[gen + 1:]]
        a_ = INM.classify(RXC, s0, txt, txt, CD16); b_ = INM.classify(RXC, s0, txt, txt, T16)
        nopen += sum(1 for c in a_ if c[1] is not None)
        ndiff += sum(1 for c, d in zip(a_, b_) if (c[1] is None) != (d[1] is None) or (c[1] == "whole") != (d[1] == INM.RULE_T3))
check(ndiff == 0 and nopen > 8000, f"on the walks' 124 answers T16 opens exactly the rows CD16 opens ({nopen} rows, {ndiff} differ): "
      f"the Thursday safety check (all of them inside Python's own comments / string statements) carries over")

print("3  what a guarded row keeps")
cls = [("comment", INM.RULE_T3, "")] * 6
ln = math.log
k, rel = INM.inert_k([1, 2, 3, 4, 5, 6], [1, 9, 9, 9, 9, 9], [0.0, -1.0, ln(0.1) + 1e-6, -0.5, 0.0, 0.0], [0, 1, 2, 3, 0, 0], cls)
check(k == 3 and [r[0] for r in rel] == [1, 2] and all(r[2] == "rule" for r in rel),
      "rank 1 at -1.0 kept, rank 2 at 0.1 p1 kept, rank 3 refused (not in the top 3): the block ends there")
k, rel = INM.inert_k([1, 2, 3], [1, 9, 3], [0.0, ln(0.1) - 0.01, 0.0], [0, 1, 0], cls[:3])
check(k == 1 and rel == [], "rank 1 but below 0.1 p1: refused")
k, rel = INM.inert_k([1, 2, 3], [1, 2, 3], [0.0, 0.0, 0.0], [0, 0, 0], cls[:3])
check(k == 3 and rel == [], "drafts equal to the argmax: kept, nothing relaxed (the exact rule)")
k, rel = INM.inert_k([1, 2, 3], [1, 9, 3], [0.0, -30.0, 0.0], [0, 6000, 0], [("comment", "whole", "")] * 3)
check(k == 3 and len(rel) == 1, "the same far-off row under CD16's 'whole' is kept (what failed on MBPP); under T16 it is refused:")
k, rel = INM.inert_k([1, 2, 3], [1, 9, 3], [0.0, -30.0, 0.0], [0, 6000, 0], cls[:3])
check(k == 1 and rel == [], "   T16 refuses it")
k, rel = INM.inert_k([1, 2, 3], [1, 9, 3], [0.0, ln(0.02), 0.0], [0, 6, 0], [("docstring", INM.RULE_T8, "")] * 3)
check(k == 3 and len(rel) == 1, "U16 keeps rank 6 at 0.02 p1 (looser than T16)")

print("4  the cut before the check")
sc = mk("Here it is:\n```python\ndef f(x):\n    y = 1\n    # compute the")
dtx = [" sum", " of", " all", "\n", "    ", "return"]
n, cf2, whole = INM.pre_block(RXC, sc, [1, 2, 3, 4, 5, 6], [0.5, 0.5, 0.5, 0.9, 0.9, 0.9], dtx, T16)
check(cf2[:3] == [0.8, 0.8, 0.8] and cf2[3:] == [0.9, 0.9, 0.9] and whole == [False] * 6 and n == 6,
      "T16: the three comment rows are floored at 0.8 in the cut, the newline and the code rows keep the drafter's confidence")
n, cf2, whole = INM.pre_block(RXC, sc, [1, 2, 3, 4, 5, 6], [0.95, 0.5, 0.99, 0.9, 0.9, 0.9], dtx, T16)
check(cf2[:3] == [0.95, 0.8, 0.99], "   a guarded row the drafter is surer of than the floor keeps its own confidence")
n, cf2, whole = INM.pre_block(RXC, sc, [1, 2, 3, 4, 5, 6], [0.5, 0.5, 0.5, 0.9, 0.9, 0.9], dtx, D16)
check(cf2[:3] == [0.5, 0.5, 0.5], "D16 (rule C as shipped on comments): its rows keep the drafter's confidence, as before")
n, cf2, whole = INM.pre_block(RXC, sc, [1, 2, 3, 4, 5, 6], [0.5, 0.5, 0.5, 0.9, 0.9, 0.9], dtx, CD16)
check(cf2[:3] == [1.0, 1.0, 1.0] and whole[:3] == [True, True, True], "CD16: unchanged")

print("5  the engine's failed CD16 run (BigMoE data/fday), read through T16's rule")
logs = [(a, f"{H}/BigMoE/data/fday/{f}") for a, f in (("dmc", "mc_inert.jsonl"), ("dmcb", "mcb_inert.jsonl"))]
if all(os.path.exists(p) for _, p in logs) and os.path.exists(f"{H}/BigMoE/data/fday/mbpp12_verdicts.json"):
    ver = json.load(open(f"{H}/BigMoE/data/fday/mbpp12_verdicts.json"))
    ctl = {"dmc": "dmq", "dmcb": "dmqb"}
    for a, p in logs:
        byk = collections.defaultdict(list)
        for l in open(p):
            e = json.loads(l)
            byk["mbpp/" + e["prompt"].split("_")[1]].append(e)
        flipped = [k_ for k_ in byk if ver[ctl[a]][k_] and not ver[a][k_]]
        firsts = {}
        for k_ in flipped:
            firsts[k_] = next((j for j, e in enumerate(byk[k_]) if not INM.rule_ok(INM.RULE_T3, e["dlm"], e["rank"])), None)
        tot = sum(len(v) for v in byk.values())
        kept = sum(1 for v in byk.values() for e in v if INM.rule_ok(INM.RULE_T3, e["dlm"], e["rank"]))
        check(all(f is not None and f <= 3 for f in firsts.values()) and len(flipped) >= 3,
              f"{a}: in each of the {len(flipped)} runs that flipped to fail ({', '.join(sorted(flipped, key=lambda s: int(s[5:])))}) T16 refuses "
              f"the drafter's text within its first 4 deviating tokens (at #{sorted(firsts.values())}); of all {tot} tokens CD16 kept "
              f"against the 125B, T16 would keep {kept} ({100 * kept / tot:.0f} %)")
else:
    print("  (BigMoE's data/fday logs not found: part 5 skipped)")

print("\n" + ("ALL OK" if not fails else f"FAILED: {len(fails)}") + f" ({len(fails)} fail)")
sys.exit(1 if fails else 0)
