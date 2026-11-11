#!/usr/bin/env python3
"""Independent read of BigMoE's D2 drafter arms on the decision map (Breakthrough, Wed 11 Nov 2026 18:4x; BigMoE's
prereg 69b0cc8, Kamen wants agreement per step and kept per check with intervals on chat / IFEval / multilingual / code /
agent). CPU only; reads the arms' out/<arm>.json (measure()'s format: per window dom, a, off, tk [rows][8], lp) and never
writes into the map folder.
Reference = the exact 125B's top-1 per answer row from the stored D-02a references (the map's gate source: code
~/Leap/data/l118_tgt/<window>, the others ~/TrainLab/data/d02a/<window>), so windows outside baseline.json (agent, any
window beyond --max 12) are scored the same way; baseline.json, where it has the window, is a second reference.
Per row: agree = the arm's top-1 == the exact top-1 (row j predicts the token after its position).
Kept per check: 16-row checks (15 drafts + 1), kept = min(accepted + 1, rows left), accepted = the leading agreements;
the next check starts after the kept rows; from the first answer row. Tokens per check = rows / checks.
Per domain: rows pooled (each row weighs the same); chat = ifeval + multi pooled (no chat set with recorded 125B tokens
exists; Leap's chat runs used the multi prompts). Intervals: 95 % window bootstrap (resample windows, 10,000 draws).
Output: ~/Breakthrough/data/m1/d2_crosscheck.out. Usage: nice -n 15 ~/BigMoE/.venv/bin/python -I -B d2_crosscheck.py
[arm ...] (default: every out/bm_*.json; also accepts any out/<name>.json, e.g. baseline as a self-check)
"""
import glob
import json
import os
import sys

import numpy as np

OUT = os.path.expanduser("~/FableGod/decision_map/out")
REF_CODE = os.path.expanduser("~/Leap/data/l118_tgt")
REF_D02A = os.path.expanduser("~/TrainLab/data/d02a")
B, BLOCK = 10000, 16
rng = np.random.default_rng(20261111)
lines = []


def say(s=""):
    print(s, flush=True)
    lines.append(s)


_refc = {}


def exact_top1(nm, dom, off, a, rows):
    """the stored exact top-1 for this window's answer rows (absolute row = off + a + j), as dm_common.gate reads it"""
    if nm not in _refc:
        z = np.load(os.path.join(REF_CODE if dom == "code" else REF_D02A, nm))
        _refc[nm] = (z["tk"][:, 0] if "tk" in z else z["top1"], int(z["off"]) if "off" in z else 0)
    ref, roff = _refc[nm]
    pos = off + a + np.arange(rows) - roff
    ok = (pos >= 0) & (pos < len(ref))
    return np.where(ok, ref[np.clip(pos, 0, len(ref) - 1)], -1), ok


def walk(f):
    """kept tokens and checks over one window's agreement flags"""
    j, kept, checks = 0, 0, 0
    R = len(f)
    while j < R:
        span = f[j:j + BLOCK - 1]
        acc = int(np.argmin(span)) if not span.all() else len(span)
        k = min(acc + 1, R - j)
        kept += k; checks += 1; j += k
    return kept, checks


def ci(num, den):
    """pooled ratio and its 95 % window-bootstrap interval"""
    num, den = np.asarray(num, float), np.asarray(den, float)
    idx = rng.integers(0, len(num), size=(B, len(num)))
    r = num[idx].sum(1) / den[idx].sum(1)
    return num.sum() / den.sum(), np.percentile(r, 2.5), np.percentile(r, 97.5)


base = json.load(open(os.path.join(OUT, "baseline.json")))
arms = sys.argv[1:] or sorted(os.path.basename(p)[:-5] for p in glob.glob(os.path.join(OUT, "bm_*.json")))
say(f"D2 cross-check (independent read): {len(arms)} arm(s); reference = the stored exact top-1 (gate source); "
    f"{BLOCK}-row checks; 95 % window bootstrap ({B} draws)")
DOMS = ["chat (ifeval+multi)", "ifeval", "multi", "code", "agent"]
for arm in arms:
    res = json.load(open(os.path.join(OUT, arm + ".json")))
    per = {}                                        # dom -> list of (agree, rows, kept, checks, agree_vs_baseline, rows_b)
    miss_ref = 0
    for nm, r in res.items():
        dom = r["dom"]
        mine = np.array([t[0] for t in r["tk"]])
        ref, ok = exact_top1(nm, dom, r["off"], r["a"], len(mine))
        miss_ref += int((~ok).sum())
        f = (mine == ref) & ok
        kept, checks = walk(f[ok])
        if nm in base and base[nm]["off"] == r["off"] and base[nm]["a"] == r["a"]:
            bt = np.array([t[0] for t in base[nm]["tk"]])
            n = min(len(bt), len(mine)); ab, rb = int((bt[:n] == mine[:n]).sum()), n
        else:
            ab, rb = 0, 0
        row = (int(f.sum()), int(ok.sum()), kept, checks, ab, rb)
        per.setdefault(dom, []).append(row)
        if dom in ("ifeval", "multi"):
            per.setdefault("chat (ifeval+multi)", []).append(row)
    say("")
    say(f"== {arm}: {len(res)} windows" + (f"; {miss_ref} rows without a reference (skipped)" if miss_ref else ""))
    say("| domain | windows | rows | agreement per step [95 % CI] | tokens kept per 16-row check [95 % CI] | "
        "vs baseline.json (rows) |")
    say("|---|---|---|---|---|---|")
    for dom in DOMS:
        if dom not in per:
            continue
        X = np.array(per[dom], float)
        a, alo, ahi = ci(X[:, 0], X[:, 1])
        k, klo, khi = ci(X[:, 2], X[:, 3])
        vb = f"{100 * X[:, 4].sum() / X[:, 5].sum():.2f} % ({int(X[:, 5].sum())})" if X[:, 5].sum() else "n/a"
        say(f"| {dom} | {len(X)} | {int(X[:, 1].sum())} | {100 * a:.2f} % [{100 * alo:.2f}, {100 * ahi:.2f}] | "
            f"{k:.2f} [{klo:.2f}, {khi:.2f}] | {vb} |")
open(os.path.expanduser("~/Breakthrough/data/m1/d2_crosscheck.out"), "w").write("\n".join(lines) + "\n")

# Information lines (added Wed 19:4x after the agent gate failed: 11 of 12 agent windows start at off 5,877-8,555 and the
# map prefills only ids[off:], so the stored full-context references do not apply there).
ba_p = os.path.join(OUT, "bm_base_agent.json")
if os.path.exists(ba_p) and not sys.argv[1:]:
    ba = json.load(open(ba_p))
    say("")
    say("AGENT, information only (excluded by the gate): each arm vs bm_base_agent, the exact pass on the same cut "
        "context, 95 % window bootstrap")
    for arm in arms:
        if arm == "bm_base_agent":
            continue
        res = json.load(open(os.path.join(OUT, arm + ".json")))
        P = []
        for nm, b in ba.items():
            if nm not in res:
                continue
            x = np.array([t[0] for t in res[nm]["tk"]]); y = np.array([t[0] for t in b["tk"]]); m = min(len(x), len(y))
            f = x[:m] == y[:m]
            k, c = walk(f)
            P.append((int(f.sum()), m, k, c))
        if P:
            X = np.array(P, float)
            a, alo, ahi = ci(X[:, 0], X[:, 1]); k, klo, khi = ci(X[:, 2], X[:, 3])
            say(f"  {arm}: agreement {100 * a:.2f} % [{100 * alo:.2f}, {100 * ahi:.2f}], kept per check {k:.2f} "
                f"[{klo:.2f}, {khi:.2f}] ({int(X[:, 1].sum())} rows, {len(X)} windows)")
rp = os.path.join(OUT, "bm_d2_rep.json")
if os.path.exists(rp) and not sys.argv[1:]:
    rep, d2 = json.load(open(rp)), json.load(open(os.path.join(OUT, "bm_d2.json")))
    for nm in rep:
        same = rep[nm]["tk"] == d2[nm]["tk"] and rep[nm]["lp"] == d2[nm]["lp"]
        say(f"DETERMINISM bm_d2_rep vs bm_d2 on {nm}: top-8 ids and log-probs {'bit-identical' if same else 'DIFFER'}")
open(os.path.expanduser("~/Breakthrough/data/m1/d2_crosscheck.out"), "w").write("\n".join(lines) + "\n")
