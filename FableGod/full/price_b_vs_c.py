"""price_b_vs_c.py (Fable FULL, LOSSY350, for the orchestrator's 16:4x ask): the comment-free mode (B) beside rule C on
the SAME rows and in rule C's own convention.
Machinery: MoonShots' span_relax_c.py (build / acc / prof and relax_curves' kept_from_profile: accepted drafts per 16-row
check from the per-depth acceptance, no bonus, the HPD-0 walk without a live cut), read-only import.
Top-8 sources: decision_map/out/baseline.json (the 12 map windows), Breakthrough/data/rulec/rulec_he12.json (the fresh 12),
rulec_mbpp.json (MBPP, 100 windows): the files M-41 used.
B in this convention: the rows whose target token is a comment token or the newline that ends a comment are gone; the
per-depth acceptance is read on the remaining rows with exact acceptance. Rule C + B = B (no in_comment row is left).
The walk npz is the project's own (the helpers load it). CPU, seconds.
Run: nice -n 15 ~/BigMoE/.venv/bin/python -I price_b_vs_c.py > price_b_vs_c.out"""
import importlib.util, os, sys
H = os.path.expanduser("~")
_argv = sys.argv; sys.argv = [_argv[0]]
spec = importlib.util.spec_from_file_location("span_relax_c_ro", f"{H}/MoonShots/scripts/span_relax_c.py")
S = importlib.util.module_from_spec(spec); spec.loader.exec_module(S)
RC = S.RCORE
SRC = {"map 12 (baseline.json)": ("code", [f"{H}/FableGod/decision_map/out/baseline.json"]),
       "fresh 12 (rulec_he12.json)": ("code", [f"{H}/Breakthrough/data/rulec/rulec_he12.json"]),
       "pooled 24": ("code", [f"{H}/FableGod/decision_map/out/baseline.json", f"{H}/Breakthrough/data/rulec/rulec_he12.json"]),
       "MBPP 100 (rulec_mbpp.json)": ("mbpp", [f"{H}/Breakthrough/data/rulec/rulec_mbpp.json"])}
_flags = {}
def flags(q):
    key = (q["win"], len(q["ids"]))
    if key not in _flags:
        sc = RC.CodeScan(); fl = []
        text = [S.dec(t) for t in q["ids"]]
        if "```" not in "".join(text): sc.code = True
        for s in text:
            before = bool(sc.code and sc.com); anyc = False
            for ch in s:
                sc._ch(ch); anyc = anyc or bool(sc.code and sc.com)
            fl.append(1 if anyc else (2 if before and s.startswith("\n") else (1 if before else 0)))
        _flags[key] = fl
    return _flags[key]
for label, (walk, srcs) in SRC.items():
    POS = []; nwin = 0
    for s in srcs:
        P, n = S.build(walk, s); POS += P; nwin += n
    com = [flags(q)[q["p"] + 1] != 0 if q["p"] + 1 < len(q["ids"]) else False for q in POS]
    a0 = [S.acc(q, "exact") for q in POS]; aC = [S.acc(q, "C") for q in POS]
    e16 = S.kfp(S.prof(POS, a0), 16); c16 = S.kfp(S.prof(POS, aC), 16)
    PB = [q for q, c in zip(POS, com) if not c]; aB = [x for x, c in zip(a0, com) if not c]
    b16 = S.kfp(S.prof(PB, aB), 16)
    aCB = [x for x, c in zip(aC, com) if not c]; cb16 = S.kfp(S.prof(PB, aCB), 16)
    miss = sum(1 for x in a0 if not x); cmiss = sum(1 for x, c in zip(a0, com) if c and not x)
    forg = sum(1 for x0, x, c in zip(a0, aC, com) if c and x and not x0); forg_out = sum(1 for x0, x, c in zip(a0, aC, com) if (not c) and x and not x0)
    print(f"== {label}: {nwin} windows, {len(POS)} tested rows; comment rows {sum(com)} ({100 * sum(com) / len(POS):.1f} %); first misses {miss}, on comment rows {cmiss} ({100 * cmiss / max(1, miss):.1f} %)")
    print(f"  exact kept(16) {e16:.2f} | rule C {c16:.2f} = x{c16 / e16:.3f} | comment-free {b16:.2f} = x{b16 / e16:.3f} | comment-free + rule C {cb16:.2f} = x{cb16 / e16:.3f}")
    print(f"  rule C forgives {forg} of the {cmiss} comment-row misses ({100 * forg / max(1, cmiss):.1f} %; outside comment rows {forg_out}); comment-free over rule C x{b16 / c16:.3f}")
