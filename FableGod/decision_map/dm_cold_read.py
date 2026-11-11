"""Read-out of Fable C's COLD-CUT screen (~/FableGod/fablec/UNION_STACK.md section 5; pre-registered in SCOREBOARD before
the run; masks dm_cold_masks.py, live checks). hyb_cold4_t3 on code, all-row flips vs the baseline argmax: PASS <= Q1 at
0.99's level (gate_top2_cf099, 1.34 %), BETWEEN <= 2.0 %, KILL above; the mask is wrong (no verdict) unless its union bytes
on code are <= x0.55 of EXACT's (out/COLD_bytes.json). Its IFEval / multi columns are information (chat takes OEA). Only if
it PASSes: the chat form cold6_all on IFEval / multi, PASS <= Q1 at 0.99's level (5.42 / 3.69 %), BETWEEN <= 1.5x, KILL above
(oea_k8's 16-row 3.83 / 3.07 % quoted). Beside: gated code flips (cf >= 0.99), and McNemar pairs against q1u_0.9_live (the
live union arm that passed on code), gate_top2_cf099 and the CONTROL hyb_0.9_u8_live_t3 (the arm COLD-CUT would replace,
at the same live checks: COLD's own cost = only-COLD vs only-control flips, |z| < 2 = no measurable cost).
Usage (BigMoE venv): python dm_cold_read.py -> out/COLD_read.txt"""
import json, os
import numpy as np
import dm_gate
OUT = os.path.expanduser(os.environ.get("DM_OUT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")))
base = json.load(open(os.path.join(OUT, "baseline.json"))); G = dm_gate.load()
if os.environ.get("COLD_HELDOUT"):                        # Tue 22:3x (12b3d63): FLOOR's 78 minus the map's 36 = 42 held out
    _m36 = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "out", "baseline.json")))
    base = {k: v for k, v in base.items() if k not in _m36}
DOMS = ("code", "ifeval", "multi")
lines = []
def say(s=""): print(s, flush=True); lines.append(s)
def fv(name):
    """per domain: the flip vector over the answer rows (window order), and the gated (cf >= 0.99) code rows' flips"""
    p = os.path.join(OUT, name + ".json")
    if not os.path.exists(p): return None
    R = json.load(open(p)); out = {d: [] for d in DOMS}; gt = [0, 0]
    for nm, b in base.items():
        m = min(len(b["tk"]), len(R[nm]["tk"])); f = np.array(R[nm]["tk"][:m])[:, 0] != np.array(b["tk"][:m])[:, 0]
        out[b["dom"]].append(f)
        if b["dom"] == "code":
            g = np.nan_to_num(G[nm], nan=-1.0)[b["a"]:b["a"] + m] >= 0.99; gt[0] += int(f[g].sum()); gt[1] += int(g.sum())
    out = {d: np.concatenate(v) for d, v in out.items()}; out["g"] = gt
    return out
def pct(X, d): return 100 * X[d].mean()
def mcn(A, B, d):
    a, b = int((A[d] & ~B[d]).sum()), int((B[d] & ~A[d]).sum())
    return f"{a} / {b}, z {(a - b) / max(np.sqrt(a + b), 1):+.2f}"
ref = fv("gate_top2_cf099")                               # FLOOR's 78 windows have no Q1-at-0.99 run (Tue 22:3x): then the
lvl = {d: pct(ref, d) for d in DOMS} if ref is not None else {"code": 1.34, "ifeval": 5.42, "multi": 3.69}   # 36's levels
BY = json.load(open(os.path.join(OUT, "COLD_bytes.json"))) if os.path.exists(os.path.join(OUT, "COLD_bytes.json")) else {}
say(f"COLD-CUT screen (Fable C, UNION_STACK section 5): all-row flips vs Q1 at 0.99's level (code {lvl['code']:.2f} / IFEval "
    f"{lvl['ifeval']:.2f} / multi {lvl['multi']:.2f} %); live checks")
say("| arm | union bytes vs EXACT (code / IFEval / multi; groups on code) | code | IFEval | multi | gated code flips (cf >= 0.99) |")
say("|---|---|---|---|---|---|")
X = {}
for arm in ("hyb_cold4_t3", "hyb_0.9_u8_live_t3", "hyb_cold4_lowo_t3", "hyb_cold4_pmtc_t3", "hyb_cold4_pmt_t3", "hyb_cold4_len_t3", "cold6_all", "q1u_0.9_live", "reach_0.9_0.5_live", "hyb_0.9_u8_t3", "oea_k8", "gate_top2_cf099"):
    X[arm] = fv(arm)
    ba = "u8_t3" if arm == "hyb_0.9_u8_live_t3" else arm
    by = (" / ".join(f"x{BY[ba][d]['bytes']:.3f}" for d in DOMS) + f"; x{BY[ba]['code']['groups']:.3f}") if ba in BY else "-"
    if X[arm] is None: say(f"| {arm} | {by} | not run | | | |"); continue
    say(f"| {arm}{' (16-row checks)' if arm in ('hyb_0.9_u8_t3', 'oea_k8') else ''} | {by} | "
        + " | ".join(f"{pct(X[arm], d):.2f} % ({int(X[arm][d].sum())})" for d in DOMS)
        + f" | {100 * X[arm]['g'][0] / max(X[arm]['g'][1], 1):.3f} % ({X[arm]['g'][0]}/{X[arm]['g'][1]}) |")
say()
H, C = X["hyb_cold4_t3"], X["cold6_all"]; ok = BY.get("hyb_cold4_t3", {}).get("code", {}).get("bytes", 0) <= 0.55
if H is None: say("VERDICT hyb_cold4_t3: not run")
elif not ok: say(f"VERDICT hyb_cold4_t3: none, the mask is wrong (code union bytes x{BY['hyb_cold4_t3']['code']['bytes']:.3f} > x0.55)")
else:
    c = pct(H, "code"); v = "PASS" if c <= lvl["code"] else "BETWEEN" if c <= 2.0 else "KILL"
    say(f"VERDICT hyb_cold4_t3 (code): {v}: {c:.2f} % all-row flips (PASS <= {lvl['code']:.2f}, BETWEEN <= 2.0, KILL above); "
        f"union bytes x{BY['hyb_cold4_t3']['code']['bytes']:.3f} of EXACT's vs the u8_t3 rule's x{BY['u8_t3']['code']['bytes']:.3f}")
    for o in ("hyb_0.9_u8_live_t3", "q1u_0.9_live", "gate_top2_cf099"):
        if X[o] is not None: say(f"  McNemar vs {o} (only-COLD / only-{o}): " + "; ".join(f"{d} {mcn(H, X[o], d)}" for d in DOMS))
    K = X["hyb_0.9_u8_live_t3"]
    if K is not None:
        a, b = int((H["code"] & ~K["code"]).sum()), int((K["code"] & ~H["code"]).sum()); z = (a - b) / max(np.sqrt(a + b), 1)
        say(f"COST over the control (code, the arm COLD would replace at the same live checks): z {z:+.2f} -> "
            + ("no measurable cost (|z| < 2)" if abs(z) < 2 else "COLD costs flips" if z > 0 else "COLD has fewer flips"))
    if C is not None:
        vc = ["PASS" if pct(C, d) <= lvl[d] else "BETWEEN" if pct(C, d) <= 1.5 * lvl[d] else "KILL" for d in ("ifeval", "multi")]
        tag = "" if v == "PASS" else " (information only: the pre-registered condition, PASS on code, failed)"   # (PASS: 0a565a6)
        say(f"VERDICT cold6_all (chat form){tag}: IFEval {pct(C, 'ifeval'):.2f} % {vc[0]} / multi {pct(C, 'multi'):.2f} % {vc[1]} "
            f"at union bytes x{BY['cold6_all']['ifeval']['bytes']:.3f} / x{BY['cold6_all']['multi']['bytes']:.3f} (oea_k8, 16-row: 3.83 / 3.07 %)")
        if X["gate_top2_cf099"] is not None: say(f"  McNemar vs gate_top2_cf099: " + "; ".join(f"{d} {mcn(C, X['gate_top2_cf099'], d)}" for d in ("ifeval", "multi")))
L, K = X["hyb_cold4_lowo_t3"], X["hyb_0.9_u8_live_t3"]   # the follow-up (SCOREBOARD 0a565a6): the prior's domain or cold-dropping?
if L is not None and K is not None:
    zs = {}
    for d in DOMS:
        a, b = int((L[d] & ~K[d]).sum()), int((K[d] & ~L[d]).sum()); zs[d] = ((a - b) / max(np.sqrt(a + b), 1), a, b)
    cost = [d for d in ("ifeval", "multi") if zs[d][0] >= 2]
    say(f"FOLLOW-UP hyb_cold4_lowo_t3 (in-domain prior) vs the control on chat: "
        + "; ".join(f"{d} {pct(L, d):.2f} % vs {pct(K, d):.2f} % (only-lowo {zs[d][1]} / only-control {zs[d][2]}, z {zs[d][0]:+.2f})" for d in ("ifeval", "multi"))
        + (f" -> COST on {', '.join(cost)}: cold experts matter on chat even in-domain (COLD stays code-only)" if cost
           else " -> no measurable cost: the prior's domain was the cause; COLD carries to chat with a domain-covering prior (GO)"))
    say(f"  code (information, in-domain prior): {pct(L, 'code'):.2f} % (line {lvl['code']:.2f}); vs the control z {zs['code'][0]:+.2f}; "
        f"gated {L['g'][0]}/{L['g'][1]}")
_lj = os.path.join(OUT, "COLD_lru_heldout.json" if os.environ.get("COLD_HELDOUT") else "COLD_lru.json")
LJ = json.load(open(_lj)) if os.path.exists(_lj) else {}
gen = []                                                  # the prompt-table arms (SCOREBOARD 4236037)
for arm in ("hyb_cold4_pmtc_t3", "hyb_cold4_pmt_t3", "hyb_cold4_len_t3"):
    A_ = X.get(arm)
    if A_ is None or K is None: continue
    zs = {}
    for d in DOMS:
        a, b = int((A_[d] & ~K[d]).sum()), int((K[d] & ~A_[d]).sum()); zs[d] = ((a - b) / max(np.sqrt(a + b), 1), a, b)
    cheaper = all(LJ.get(arm, {}).get(d, 9) < LJ.get("u8_t3", {}).get(d, 0) for d in DOMS)
    nocost = [d for d in DOMS if zs[d][0] < 2]
    v = ("GENERAL" if len(nocost) == 3 and cheaper else "CODE+" if "code" in nocost and "ifeval" in nocost and "multi" not in nocost
         else "code-only (COLD keeps the cap45 table)")
    if v == "GENERAL": gen.append(arm)
    say(f"PROMPT {arm} vs the control: " + "; ".join(f"{d} {pct(A_, d):.2f} % vs {pct(K, d):.2f} % (z {zs[d][0]:+.2f})" for d in DOMS)
        + f"; cached bytes " + " / ".join(f"x{LJ.get(arm, {}).get(d, float('nan')):.3f}" for d in DOMS)
        + f" vs the control's " + " / ".join(f"x{LJ.get('u8_t3', {}).get(d, float('nan')):.3f}" for d in DOMS) + f" -> {v}")
if gen: say(f"PROMPT verdict: GENERAL method = {'hyb_cold4_pmt_t3 (fewer bytes)' if 'hyb_cold4_pmt_t3' in gen else gen[0]}")
if os.path.exists(os.path.join(OUT, "COLD_lru.txt")):
    say("Cached bytes (dm_cold_lru.py, 45 GB pools, plain LRU):"); [say("  " + x.rstrip()) for x in open(os.path.join(OUT, "COLD_lru.txt"))]
open(os.path.join(OUT, "COLD_read_heldout.txt" if os.environ.get("COLD_HELDOUT") else "COLD_read.txt"), "w").write("\n".join(lines) + "\n")
