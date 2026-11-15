"""The lossy-check gate's verdicts (~/FableGod/deepdraft/LOSSY_GATE.md; docs/PREREG_lossygate_*): one arm run against a
reference (one run, or a comma list merged in order) on the panel of tools/lossy_panel.json. Scoring reuses
tools/cert_report.py: its items, row loaders and per-set scorers (HumanEval+ and IFEval via evalsuite.decode_report, MBPP
via tools/mbpp_score.py).
  .venv/bin/python tools/lossy_d4.py tier1 <arm run> <ref run[,ref run...]> [all|code] [ident]
  .venv/bin/python tools/lossy_d4.py interim <arm run> <ref runs> [all|code]
  .venv/bin/python tools/lossy_d4.py tier0 <arm run> <ref runs> [all|code]
Both orders pooled (PREREG_lossygate amendment 1c): join the orders' runs with '+', e.g. arm `x_fwd+x_rev` and
  reference `base_fwd+base_rev` (one reference group is reused for every order: exact rows are order-independent).
  Each order is paired on its own (arm order o vs reference order o), then the pairs pool, so a key counts once per
  order; merging the two orders by key would collapse them.
Scope all (default): HumanEval+ 164, MBPP 100 and IFEval 100 pooled (364 pairs).
Scope code (CODE-ONLY mode; the orchestrator 01:2x): HumanEval+ 164 + MBPP 100 pooled (264 pairs). IFEval 100 is the
  switch-off check: with `ident` (the reference is a cached base run, same order, IFEval in its own processes) its tokens
  must equal the reference on all 100; without it, decision 4 vs the reference ("no harm").
tier1: each set's guard is decision 4 (delta > -5 pt AND exact McNemar p >= 0.05; a gain with p < 0.05 is DIFFERS
(HIGHER)); the pooled line needs delta > -3 pt AND p >= 0.05 (DIFFERS (HIGHER) passes). PASS = all lines. The pooled
line is a SCREENING line: adopting a lossy check stays Kamen's call after a full verdict (tier 2, the card gate), and a
code-scoped PASS licenses a code-only mode only.
interim: z = (b - c) / sqrt(b + c) over the scope's pooled sets, b = arm-only correct, c = reference-only correct;
STOP-FAIL if z <= -2.8.
tier0: KILL if the reference-only minus arm-only count is >= 4 on the scope's 40 keys, or any arm row is empty; else GO.
RUNAWAY guard (amendment 3a, from MoonShots' J(CC) on the 2B: delta 2 failed through runaway answers, cap hits 26 -> 50):
  runaway = the answer hit the token cap (finish "length") or loops (evalsuite.decode_report.looped: a 20-token n-gram
  >= 3 times in the last 400 tokens). Over the scope's pooled sets, paired: arm-only runaway b vs reference-only c;
  FAIL if b > c with exact McNemar p < 0.05. Cap hits and loops are printed per set either way.
Last line: `LOSSY <mode>[ (CODE-ONLY mode)]: ...`; exit 0 PASS / GO / CONTINUE, 1 FAIL / KILL / STOP-FAIL, 2 rows missing."""
import json, math, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
import cert_report as cr
import d4x
from evalsuite.report import mcnemar_p
from evalsuite.decode_report import looped

PANEL = json.load(open(os.path.join(HERE, "lossy_panel.json")))
NAME = {"humaneval": "HumanEval+", "mbpp": "MBPP", "ifeval": "IFEval"}
POOL_D, Z_STOP, T0_LOSS = -3.0, -2.8, 4


def pairs(arm, refs, keys, sets):
    """per set: (keys present in both, arm correctness, reference correctness), scored by cert_report's scorers"""
    its = cr.items()
    A, R = cr.load(arm), cr.load(*refs)
    out, missing = {}, [k for k in keys if k not in A or k not in R]
    for s in sets:
        ks = [k for k in keys if k not in missing and its.get(k, {}).get("set") == s]
        if ks:
            out[s] = (ks, cr.score(A, its, s, ks), cr.score(R, its, s, ks))
    return out, missing, A, R


def pairs_orders(arm, ref, keys, sets):
    """pairs() per order ('+' joins orders; a single reference group serves every order), pooled with keys 'o:key'"""
    arms, refs = arm.split("+"), ref.split("+")
    if len(arms) == 1:
        return pairs(arm, ref.split(","), keys, sets)
    if len(refs) == 1:
        refs = refs * len(arms)
    if len(refs) != len(arms):
        raise SystemExit(f"{len(arms)} arm orders but {len(refs)} reference groups")
    out, missing, A, R = {}, [], {}, {}
    for o, (a, r) in enumerate(zip(arms, refs)):
        P, mis, Ao, Ro = pairs(a, r.split(","), keys, sets)
        missing += [f"{o}:{k}" for k in mis]
        for s_, (ks, sa, sr) in P.items():
            K0, SA, SR = out.setdefault(s_, ([], {}, {}))
            K0 += [f"{o}:{k}" for k in ks]
            SA.update({f"{o}:{k}": v for k, v in sa.items()}); SR.update({f"{o}:{k}": v for k, v in sr.items()})
        A.update({f"{o}:{k}": v for k, v in Ao.items()}); R.update({f"{o}:{k}": v for k, v in Ro.items()})
    return out, missing, A, R


def counts(ks, sa, sr):
    b = sum(1 for k in ks if sa[k][1] == 1 and sr[k][1] == 0)
    c = sum(1 for k in ks if sa[k][1] == 0 and sr[k][1] == 1)
    return b, c


def pooled_rule(b, c, n):
    d, p = 100.0 * (b - c) / n, mcnemar_p(b, c)
    if p < 0.05 and b > c:
        return "DIFFERS (HIGHER)", d, p
    return ("PASS" if d > POOL_D and p >= 0.05 else "FAIL"), d, p


def setline(s, ks, sa, sr):
    b, c = counts(ks, sa, sr)
    v, d, p = d4x.rule(b, c, len(ks))
    ca, cr_ = (100 * sum(x[k][1] for k in ks) / len(ks) for x in (sa, sr))
    print(f"{NAME[s]} (n {len(ks)}): {ca:.1f} % vs {cr_:.1f} %, delta {d:+.2f} pt, McNemar p {p:.4f} "
          f"(arm only {b} / reference only {c}): {v}")
    return v, b, c


def main(mode, arm, ref, scope="all", ident=""):
    if scope not in ("all", "code"):
        print("scope: all | code")
        return 2
    code = scope == "code"
    n_ord = len(arm.split("+"))
    lab = (" (CODE-ONLY mode)" if code else "") + (f" [{n_ord} orders pooled]" if n_ord > 1 else "")
    pool = ("humaneval", "mbpp") if code else ("humaneval", "mbpp", "ifeval")
    if mode == "tier1":
        keys = PANEL["code"] + PANEL["ifeval"] if code else PANEL["tier1"]
        P, missing, A, R = pairs_orders(arm, ref, keys, ("humaneval", "mbpp", "ifeval"))
        if missing:
            print(f"{len(missing)} of {len(keys)} panel keys missing in {arm} or the reference: {missing[:5]}")
            print(f"LOSSY tier1{lab}: INCOMPLETE")
            return 2
        bad, B, C, N = [], 0, 0, 0
        for s in pool:
            v, b, c = setline(s, *P[s])
            bad += [] if v in ("PASS", "DIFFERS (HIGHER)") else [NAME[s]]
            B, C, N = B + b, C + c, N + len(P[s][0])
        # the runaway guard: cap hits or loops, paired over the pooled sets
        run_ = lambda r: (r.get("finish") == "length") or looped(r.get("completion_ids") or [])
        rb = rc = 0
        for s_ in pool:
            ks = P[s_][0]
            ca, cr_ = (sum(1 for k in ks if X[k].get("finish") == "length") for X in (A, R))
            la, lr = (sum(1 for k in ks if looped(X[k].get("completion_ids") or [])) for X in (A, R))
            b_ = sum(1 for k in ks if run_(A[k]) and not run_(R[k])); c_ = sum(1 for k in ks if run_(R[k]) and not run_(A[k]))
            rb, rc = rb + b_, rc + c_
            print(f"  {NAME[s_]} runaway: cap hits {ca} vs {cr_}, loops {la} vs {lr} (arm only {b_} / reference only {c_})")
        rp = mcnemar_p(rb, rc)
        rv = "FAIL" if rb > rc and rp < 0.05 else "PASS"
        print(f"runaway guard (pooled{' code' if code else ''}): arm only {rb} / reference only {rc}, McNemar p {rp:.4f}: {rv}")
        bad += [] if rv == "PASS" else ["runaway"]
        v, d, p = pooled_rule(B, C, N)
        print(f"pooled{' code' if code else ''} (n {N}): delta {d:+.2f} pt, McNemar p {p:.4f} (arm only {B} / reference only "
              f"{C}; line > {POOL_D:+.0f} pt, a screening line): {v}")
        bad += [] if v in ("PASS", "DIFFERS (HIGHER)") else ["pooled"]
        if code:                                         # IFEval: the switch-off check
            ks = P["ifeval"][0]
            if ident == "ident":
                same = sum(1 for k in ks if A[k]["completion_ids"] == R[k]["completion_ids"])
                v = "PASS" if same == len(ks) else "FAIL"
                print(f"IFEval switch-off (n {len(ks)}): tokens identical to the reference on {same} of {len(ks)} "
                      f"(the lossy check is off for chat): {v}")
            else:
                v, _, _ = setline("ifeval", *P["ifeval"])
                print(f"IFEval switch-off: decision 4 vs the reference (no harm): {v}")
            bad += [] if v in ("PASS", "DIFFERS (HIGHER)") else ["IFEval switch-off"]
        print(f"LOSSY tier1{lab}: {'PASS' if not bad else 'FAIL (' + ', '.join(bad) + ')'}")
        return 1 if bad else 0
    if mode == "interim":
        keys = PANEL["code"] if code else PANEL["tier1"]
        P, _, _, _ = pairs_orders(arm, ref, keys, pool)
        B = C = N = 0
        for s in pool:
            if s in P:
                b, c = counts(*P[s])
                B, C, N = B + b, C + c, N + len(P[s][0])
        z = (B - C) / math.sqrt(B + C) if B + C else 0.0
        v = "STOP-FAIL" if z <= Z_STOP else "CONTINUE"
        print(f"interim{lab} (n {N} of {len(keys)}): arm only {B} / reference only {C}, z {z:+.2f} (stop at <= {Z_STOP}): {v}")
        print(f"LOSSY interim{lab}: {v}")
        return 1 if v == "STOP-FAIL" else 0
    if mode == "tier0":
        keys = PANEL["tier0_code"] if code else PANEL["tier0"]
        P, missing, A, _ = pairs_orders(arm, ref, keys, pool)
        if missing:
            print(f"{len(missing)} of {len(keys)} tier-0 keys missing: {missing[:5]}")
            print(f"LOSSY tier0{lab}: INCOMPLETE")
            return 2
        B = C = 0
        for s in pool:
            if s in P:
                b, c = counts(*P[s])
                B, C = B + b, C + c
        ks0 = set(keys)                                   # pooled keys are "o:key"
        empty = [k for k in A if k.split(":")[-1] in ks0 and not (A[k].get("completion_ids") or [])]
        v = "KILL" if C - B >= T0_LOSS or empty else "GO"
        print(f"tier 0{lab} (n {len(keys)}): arm only {B} / reference only {C}, net loss {C - B} (kill at >= {T0_LOSS}); "
              f"empty rows {empty}: {v}")
        print(f"LOSSY tier0{lab}: {v}")
        return 1 if v == "KILL" else 0
    print("mode: tier1 | interim | tier0")
    return 2


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:6]))
