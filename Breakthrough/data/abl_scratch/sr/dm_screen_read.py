"""Read-out of the Wed night map screens (Breakthrough, Wed 2026-11-04 04:1x; the orchestrator 04:0x / 04:08; lines copied into
SCOREBOARD before the run). All-row flips vs the baseline argmax, per domain, 36 windows, live checks; Q1 at 0.99's level
(gate_top2_cf099) = 1.34 / 5.42 / 3.69 % as pre-registered.
- Screen 1(a), Fable C (~/FableGod/fablec/BYTES10.md:213-226): qom_0_live_bonus = qom_0_live's masks with every drafted
  bonus row gated to ranks 0-1 + the block's union members. PASS = <= qom_0_live's own + 0.3 points AND <= the level;
  BETWEEN = otherwise <= 1.5x the level; KILL = above. Paired McNemar vs qom_0_live.
  `python dm_screen_read.py qom_0_live_bonus qom_0_live 0.3`
- Screen 2, NewIdeas (~/NewIdeas/QUEUE.md:1181-1185): hyb_0.9_k1_u10_live_t3 = sure rows (cf >= 0.9) keep rank 0 at 3 bits
  + the block's union members at 4 bits. PASS = <= the level; BETWEEN = <= 1.5x; KILL = above. Paired McNemar vs its
  top-2 twin q1u_0.9_live_t3.   `python dm_screen_read.py hyb_0.9_k1_u10_live_t3 q1u_0.9_live_t3`
Usage (BigMoE venv): python dm_screen_read.py <arm> <paired arm> [<points over the paired arm, also required for PASS>]
       [--also <arm> ...] (more McNemar partners, information; Wed 04:5x for qom_0_live_bq vs qom_0_live_bonus)
-> out/SCREEN_read_<arm>.txt
NON-TIE READ (Fable C, Thu 2026-11-05, WALL_1105.md section 1): the 125B's tie rows (floor_rows.json's cls, p1 - p2 <= 0.1) flip
20-45 % under every arm, so every table row gets a non-tie twin and the verdict a second line: NON-TIE PASS = non-tie flips <=
gate_top2_cf099's own non-tie level (0.87 / 2.22 / 1.63 %), and VERDICT2 = the all-row verdict AND the non-tie one. The
all-row VERDICT line is unchanged. FLOOR_ROWS = another class file; SCREEN_OUT = another directory for the read-out file."""
import json, os, sys
import numpy as np
D = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(D, "out")
argv = sys.argv[1:]; ALSO = []
NULL = "--null" in argv   # Wed 15:4x: a null check (Fable C's screens 1(b) / 1(c), BYTES10.md:255-260): PASS = within +<points> of the paired arm, whatever the level; else COST
if NULL: argv.remove("--null")
while "--also" in argv: i = argv.index("--also"); ALSO.append(argv[i + 1]); del argv[i:i + 2]
ARM, REF = argv[0], argv[1]; PLUS = float(argv[2]) if len(argv) > 2 else None
DOMS = ("code", "ifeval", "multi"); LVL = {"code": 1.34, "ifeval": 5.42, "multi": 3.69}   # Q1 at 0.99, as pre-registered
NTL = {"code": 0.87, "ifeval": 2.22, "multi": 1.63}   # the same arm's non-tie flips (gate_top2_cf099; recomputed below as information)
_fr = os.path.expanduser(os.environ.get("FLOOR_ROWS", "~/FableGod/floor_rows.json"))
CLS = {(r["nm"], int(r["i"])): r["cls"] for r in json.load(open(_fr))} if os.path.exists(_fr) else None   # (window, absolute row) -> sure / mid / tie
base = {k: v for k, v in json.load(open(os.path.join(OUT, "baseline.json"))).items() if isinstance(v, dict) and "tk" in v}
lines = []
def say(s=""): print(s, flush=True); lines.append(s)
def fv(name):
    """per domain: the flip vector over the answer rows (window order); None + why if the arm is missing or short"""
    p = os.path.join(OUT, name + ".json")
    if not os.path.exists(p): return None, f"{name}: not run"
    R = json.load(open(p)); out = {d: [] for d in DOMS}; warn = 0
    for nm, b in base.items():
        if nm not in R or "tk" not in R[nm]: return None, f"{name}: window {nm} missing"
        warn += "warn" in R[nm]
        m = min(len(b["tk"]), len(R[nm]["tk"]))
        out[b["dom"]].append(np.array(R[nm]["tk"][:m])[:, 0] != np.array(b["tk"][:m])[:, 0])
    return {d: np.concatenate(v) for d, v in out.items()}, f"{len(base)} windows, {warn} with row-count warnings"
def pct(X, d): return 100 * X[d].mean()
def ntmask(name):
    """per domain: True on the non-tie answer rows, in fv()'s row order (rows without a class count as non-tie)"""
    R = json.load(open(os.path.join(OUT, name + ".json"))); out = {d: [] for d in DOMS}
    for nm, b in base.items():
        m = min(len(b["tk"]), len(R[nm]["tk"]))
        out[b["dom"]].append(np.array([CLS.get((nm, int(b["a"]) + j)) != "tie" for j in range(m)]))
    return {d: np.concatenate(v) for d, v in out.items()}
def npct(X, M, d): return 100 * X[d][M[d]].mean()
def mcn(X, Y, d):
    a, b = int((X[d] & ~Y[d]).sum()), int((Y[d] & ~X[d]).sum()); return f"{d} {a} / {b}, z {(a - b) / max(np.sqrt(a + b), 1):+.2f}"
A, ia = fv(ARM); B, ib = fv(REF)
say(f"Map screen: {ARM} vs {REF} (paired); all-row flips vs the baseline argmax, live checks")
say(f"  {ARM}: {ia}; {REF}: {ib}")
if A is None or B is None: say(f"VERDICT {ARM}: INCOMPLETE ({ia if A is None else ib})")
else:
    ref, _ = fv("gate_top2_cf099")
    say("| arm | " + " | ".join(DOMS) + " |"); say("|---|---|---|---|")
    NT = ntmask(ARM) if CLS is not None else None
    for nm, X in ((REF, B), (ARM, A)):
        say(f"| {nm} | " + " | ".join(f"{pct(X, d):.2f} % ({int(X[d].sum())}/{len(X[d])})" for d in DOMS) + " |")
        if NT is not None and all(len(X[d]) == len(NT[d]) for d in DOMS):
            say(f"| {nm}, non-tie rows | " + " | ".join(f"{npct(X, NT, d):.2f} % ({int(X[d][NT[d]].sum())}/{int(NT[d].sum())})" for d in DOMS) + " |")
            say(f"| {nm}, tie rows | " + " | ".join(f"{100 * X[d][~NT[d]].mean():.1f} % ({int(X[d][~NT[d]].sum())}/{int((~NT[d]).sum())})" for d in DOMS) + " |")
    say("| Q1 at 0.99 (pre-registered level) | " + " | ".join(f"{LVL[d]:.2f} %" for d in DOMS) + " |")
    if ref is not None: say("| gate_top2_cf099 (recomputed, information) | " + " | ".join(f"{pct(ref, d):.2f} %" for d in DOMS) + " |")
    if NT is not None:
        say("| non-tie level (gate_top2_cf099's non-tie flips) | " + " | ".join(f"{NTL[d]:.2f} %" for d in DOMS) + " |")
        if ref is not None and all(len(ref[d]) == len(NT[d]) for d in DOMS):
            say("| gate_top2_cf099, non-tie rows (recomputed, information) | " + " | ".join(f"{npct(ref, NT, d):.2f} %" for d in DOMS) + " |")
    v, mc = [], []
    for d in DOMS:
        g, own = pct(A, d), pct(B, d)
        if NULL: r = "PASS" if g <= own + PLUS else "COST"
        else:
            ok = g <= LVL[d] and (PLUS is None or g <= own + PLUS)
            r = "PASS" if ok else "BETWEEN" if g <= 1.5 * LVL[d] else "KILL"
        v.append(f"{d} {r} {g:.2f} % (" + (f"{REF} {own:.2f} + {PLUS} = {own + PLUS:.2f}; " if PLUS is not None else "")
                 + f"level {LVL[d]:.2f}, x1.5 {1.5 * LVL[d]:.2f})")
        a, b = int((A[d] & ~B[d]).sum()), int((B[d] & ~A[d]).sum())
        mc.append(f"{d} {a} / {b}, z {(a - b) / max(np.sqrt(a + b), 1):+.2f}")
    say(f"VERDICT {ARM}: " + " / ".join(v))
    say(f"  McNemar vs {REF} (only-{ARM} / only-{REF} flips): " + "; ".join(mc))
    if NT is not None and not NULL and all(len(A[d]) == len(NT[d]) == len(B[d]) for d in DOMS):
        v2, n2, m2 = [], [], []
        for d, vd in zip(DOMS, v):
            g = npct(A, NT, d); okn = g <= NTL[d]; n2.append(f"{d} {'PASS' if okn else 'OVER'} {g:.2f} % (level {NTL[d]:.2f})")
            v2.append(f"{d} {'PASS' if (' PASS ' in ' ' + vd + ' ' and okn) else 'FAIL'}")
            a, b = int((A[d] & ~B[d] & NT[d]).sum()), int((B[d] & ~A[d] & NT[d]).sum())
            m2.append(f"{d} {a} / {b}, z {(a - b) / max(np.sqrt(a + b), 1):+.2f}")
        say(f"NON-TIE {ARM}: " + " / ".join(n2))
        say(f"  McNemar vs {REF} on non-tie rows: " + "; ".join(m2))
        say(f"VERDICT2 {ARM} (all-row PASS and non-tie PASS): " + " / ".join(v2))
    for o in ALSO:
        O, io = fv(o)
        if O is None: say(f"  McNemar vs {o}: {io}"); continue
        say(f"  McNemar vs {o} ({' / '.join(f'{pct(O, d):.2f} %' for d in DOMS)}; only-{ARM} / only-{o} flips): "
            + "; ".join(mcn(A, O, d) for d in DOMS))
open(os.path.join(os.environ.get("SCREEN_OUT", OUT), f"SCREEN_read_{ARM}.txt"), "w").write("\n".join(lines) + "\n")
