"""guard_pi.py (GOAL, Fri 2026-11-06 21:2x): how often would a guarded inert row be kept? On the 125B's OWN text (the walk),
the drafter's pick on the right prefix against the 125B's top-8: P(equal), P(passes T3 | differs), by span kind and depth."""
import importlib.util, math, os, re, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("kwr", f"{HERE}/kwall_replay.py"); K = importlib.util.module_from_spec(spec); spec.loader.exec_module(K)
G = K.load_mod("inert_span_1106", f"{HERE}/build_1106/inert_span.py")
sets = K.load_sets()
keys = set(open(f"{HERE}/mbpp12_1106/mbpp12_keys.txt").read().strip().split(","))
mb = sets[0][1]
for label, S in (("HumanEval+ fresh 12", sets[1][1]), ("MBPP 90 (answers that finish)", [x for x in mb if len(x["ids"]) - x["gen"] < 500])):
    print(f"\n=== {label}: rows CD16 / T16 would open, the drafter's pick on the right prefix")
    for kind in ("comment", "docstring"):
        eq = np.zeros(15); t3 = np.zeros(15); t8 = np.zeros(15); n = np.zeros(15)
        for x in S:
            ids, gen = x["ids"], x["gen"]
            for i, (k, m, a) in enumerate(x["ch"]):
                m, a = int(m), int(a)
                if a < gen:
                    continue
                for j in range(m):
                    p = a + 1 + j
                    kt = x["kt"].get(p)
                    if not kt or kt[0] != kind or not kt[1]:
                        continue
                    d = int(x["tt"][i][j]); n[j] += 1
                    if d == ids[p]:
                        eq[j] += 1
                    else:
                        dl, rk = K.PI.top8(x, p, d)
                        t3[j] += G.rule_ok(G.RULE_T3, dl, rk); t8[j] += G.rule_ok(G.RULE_T8, dl, rk)
        nd = np.maximum(n - eq, 1)
        f = lambda v: " ".join(f"{z:.2f}" for z in v)
        print(f"  {kind:9s} rows at depth 1 {int(n[0])}")
        print(f"     P(equal)             {f(eq / np.maximum(n, 1))}")
        print(f"     P(T3 | differs)      {f(t3 / nd)}")
        print(f"     kept under T16       {f((eq + t3) / np.maximum(n, 1))}   mean {np.sum(eq + t3) / np.sum(n):.3f}")
        print(f"     kept under U16       {f((eq + t8) / np.maximum(n, 1))}   mean {np.sum(eq + t8) / np.sum(n):.3f}")
