"""RUNAHEAD-BUILD (4) (NewIdeas Sun 12:4x; lines in SCOREBOARD 'RUNAHEAD-BUILD LINES', written before the analysis; CPU):
MoonShots' M-16 simulated on MTP Head's D-23 dfl log (read-only ~/TrainLab/data/d23_runs.jsonl: the 45 GB default at C 224,
t 0.3; 18 prompts, checks in their logged order), with REF2's 16-row limit: B capped at 16 - |A| rows.
Pipeline: block i (the logged check i: rows r_i at t 0.3, kept k_i) is run ahead as pass i's B; pass i re-checks block
i-1's provisionally kept rows as A (exact) and emits them:
  a_i = |A_i| = min(k_{i-1}, rows of B_{i-1});  b_i = min(r_i, 16 - a_i);  tokens of pass i = a_i
  compute_i = C0 + 2.27 ms x ctx K + beta x (a_i + b_i) + replay (4 ms when today's check i-1 was a full accept: the commit
              replay that a full accept skips today always runs, since B is always discarded)
  ssd_i = 0.10 ms x B_i's misses at deep queue (13.8 GB/s; 0.149 ms per miss at today's per-layer bursts), with B_i's
          misses = the prompt's median misses per check x b_i / its mean rows; pass time = max(compute_i, ssd_i)
  C0 = FIXED-PART's constant 42.2 ms (dense padded to 16 rows, draft, syncs); the SSD term (0.559 x io) leaves the path.
  [12:4x, before reporting: the first run took the SSD out twice (via beta AND via C0) and left out the start-up excess that
  the logged times hold. Corrected: each prompt's own steady fixed part (median of check + draft - 7.82 x rows, checks 3..)
  stays, SSD leaves only through beta; each prompt's start-up excess (checks 1-2 over steady) is added to RUNAHEAD too.
  'optimistic' also drops 0.559 x the prompt's io from its fixed part (if some misses are per check, not per row).]
  Bubbles: with probability 1 - U2 a pass's provisional verdict is wrong; the next pass then emits 1 token (it re-checks
  only the exact bonus) at a full pass's cost.
Today = the logged decode time. beta = the pure compute per verified row (unknown until M-16 step 2):
  2.5 (MoonShots' optimistic, Leap's 3-bit kernel ~2.4 ms + overheads), 3.5, central = 7.82 - 0.149 x (median misses per
  row) (the slope less each row's own misses, fully on the path as R-69b's cycle ~ 50 + 1.1 x SSD), 5.7 (= the slope less
  only 0.559 of them, my FIXED-PART between-prompt share).
Usage: [BigMoE venv] python runahead_model.py -> stdout + data/trees/runahead_model.txt"""
import json, os
import numpy as np

R = [json.loads(l) for l in open(os.path.expanduser("~/TrainLab/data/d23_runs.jsonl"))]
P = [r for r in R if r["arm"] == "dfl" and not (r["paused"] or r["invalid"] or r["cut"])]
ES = os.path.expanduser("~/EvalSuite/data/items/")
NP = {}
for f in os.listdir(ES):
    if f.startswith("dec_"):
        for l in open(ES + f):
            it = json.loads(l)
            NP[it["key"]] = len(it["prompt_ids"])
C0, CTX, REPLAY, CAP = 42.2, 2.27, 4.0, 16
mpr = float(np.median([r["miss"] / np.mean([c[0] for c in r["pc"]]) for r in P]))
BETAS = {"2.5": 2.5, "3.5": 3.5, f"central {7.82 - 0.149 * mpr:.1f}": 7.82 - 0.149 * mpr, f"{7.82 - 0.559 * 0.149 * mpr:.1f}": 7.82 - 0.559 * 0.149 * mpr}
U2S = (1.0, 0.98, 0.95)
lines = []


def say(s=""):
    print(s, flush=True); lines.append(s)


def sim(r, beta, u2, cap=CAP):
    pc = np.array(r["pc"], float)                                     # rows, kept, check ms, draft ms
    mrow = r["miss"] / pc[:, 0].mean()                                # misses per row (the prompt's median per check / rows)
    F = pc[:, 2] + pc[:, 3] - 7.82 * pc[:, 0]
    fs = float(np.median(F[2:] if len(F) > 4 else F))                # the prompt's steady fixed part (FIXED-PART)
    start = float((F[:2] - fs).sum()) if len(F) > 4 else 0.0           # its start-up excess, paid by RUNAHEAD too
    fx = fs - (0.559 * r["io"] if OPT[0] else 0.0)
    tok, ms, a, full_prev, rows_tot = 0.0, start, 1.0, False, 0.0
    for i, (rw, kp, ck, dr) in enumerate(pc):
        b = min(rw, cap - a)
        comp = fx + beta * (a + b) + (REPLAY if full_prev else 0.0)
        ssd = 0.10 * mrow * b
        t = max(comp, ssd)
        good = a * u2 + 1.0 * (1 - u2)                                # a bubble: this pass's A was wrong -> 1 token
        tok += good if i > 0 else 0.0                                 # pass 0 only fills the pipeline (A = anchor)
        ms += t
        rows_tot += a + b
        full_prev = kp >= rw                                          # today's check was a full accept (no replay)
        a = min(kp, b)                                                # block i's provisional kept -> next pass's A
    tok += a                                                          # drain: one last pass re-checks the final A
    ms += fx + beta * a
    return tok, ms, rows_tot / len(pc)


say(f"RUNAHEAD model on D-23's dfl log ({len(P)} prompts, {sum(len(r['pc']) for r in P)} checks; C 224, t 0.3); median misses "
    f"per row {mpr:.1f}; B capped at {CAP} - |A|. tok/s gain vs the logged decode (today), by type and whole set:")
OPT = [False]
for bn, beta in list(BETAS.items()) + [("7.82 (no SSD saved)", 7.82)]:
    for u2 in U2S:
        cells = []
        for t in ("code", "math", "chat", "agent", "whole"):
            g = [r for r in P if t == "whole" or r["type"] == t]
            today_tok = sum(r["n"] for r in g); today_ms = sum(1000 * r["n"] / r["tok_s"] for r in g)
            s = [sim(r, beta, u2) for r in g]
            ra_tok = sum(x[0] for x in s); ra_ms = sum(x[1] for x in s)
            cells.append(f"{t} {100 * (ra_tok / ra_ms) / (today_tok / today_ms) - 100:+5.1f} %")
        say(f"  beta {bn:12s} U2 {u2:4.2f}: " + "  ".join(cells))
OPT[0] = True
for bn, beta in BETAS.items():
    cells = []
    for t in ("code", "math", "chat", "agent", "whole"):
        g = [r for r in P if t == "whole" or r["type"] == t]
        s = [sim(r, beta, 0.98) for r in g]
        cells.append(f"{t} {100 * (sum(x[0] for x in s) / sum(x[1] for x in s)) / (sum(r['n'] for r in g) / sum(1000 * r['n'] / r['tok_s'] for r in g)) - 100:+5.1f} %")
    say(f"  optimistic (per-check SSD also off the fixed part), beta {bn:12s} U2 0.98: " + "  ".join(cells))
OPT[0] = False
def whole(beta, cap=CAP):
    s = [sim(r, beta, 0.98, cap) for r in P]
    return 100 * (sum(x[0] for x in s) / sum(x[1] for x in s)) / (sum(r['n'] for r in P) / sum(1000 * r['n'] / r['tok_s'] for r in P)) - 100
be = {th: next((b for b in np.arange(1.0, 9.0, 0.05) if whole(b) < th), None) for th in (15.0, 0.0)}
be32 = {th: next((b for b in np.arange(1.0, 9.0, 0.05) if whole(b, 32) < th), None) for th in (15.0, 0.0)}
say(f"  whole set (U2 0.98): +15 % until beta {be[15.0]:.2f}, break-even at beta {be[0.0]:.2f} (uncapped, 32 rows: {be32[15.0]:.2f} / {be32[0.0]:.2f})")
# the cap's own cost: tokens per pass with and without it (beta central, U2 1)
bc = list(BETAS.values())[2]
for t in ("code", "math", "chat", "agent"):
    g = [r for r in P if r["type"] == t]
    k_today = np.mean([c[1] for r in g for c in r["pc"]])
    s16 = [sim(r, bc, 1.0) for r in g]; s32 = [sim(r, bc, 1.0, cap=32) for r in g]
    n_checks = sum(len(r["pc"]) for r in g)
    say(f"  {t:6s} tokens per pass: today {k_today:.2f}; RUNAHEAD capped 16 {sum(x[0] for x in s16) / n_checks:.2f}, "
        f"uncapped (32) {sum(x[0] for x in s32) / n_checks:.2f}; rows per pass capped {np.mean([x[2] for x in s16]):.1f}")
open(os.path.expanduser("~/Breakthrough/data/trees/runahead_model.txt"), "w").write("\n".join(lines) + "\n")
OPT[0] = True
beo = {th: next((b for b in np.arange(1.0, 9.0, 0.05) if whole(b) < th), None) for th in (15.0, 0.0)}
beo32 = {th: next((b for b in np.arange(1.0, 9.0, 0.05) if whole(b, 32) < th), None) for th in (15.0, 0.0)}
say(f"  optimistic, whole set (U2 0.98): +15 % until beta {beo[15.0]:.2f}, break-even {beo[0.0]:.2f} (uncapped 32 rows: "
    f"{beo32[15.0]:.2f} / {beo32[0.0]:.2f})")
for bn, beta in BETAS.items():
    cells = []
    for t in ("code", "math", "chat", "agent", "whole"):
        g = [r for r in P if t == "whole" or r["type"] == t]
        s = [sim(r, beta, 0.98, 32) for r in g]
        cells.append(f"{t} {100 * (sum(x[0] for x in s) / sum(x[1] for x in s)) / (sum(r['n'] for r in g) / sum(1000 * r['n'] / r['tok_s'] for r in g)) - 100:+5.1f} %")
    say(f"  optimistic + UNCAPPED (32-row kernels), beta {bn:12s} U2 0.98: " + "  ".join(cells))
open(os.path.expanduser("~/Breakthrough/data/trees/runahead_model.txt"), "w").write("\n".join(lines) + "\n")
# MoonShots 12:5x SHADOW CACHE, run as two back-to-back passes (no batch-2 kernels): B uncapped on its own sequence, but each
# cycle pays the prompt's fixed part twice (the exact pass and the shadow pass)
def sim2(r, beta, u2):
    pc = np.array(r["pc"], float)
    mrow = r["miss"] / pc[:, 0].mean()
    F = pc[:, 2] + pc[:, 3] - 7.82 * pc[:, 0]
    fs = float(np.median(F[2:] if len(F) > 4 else F))
    start = float((F[:2] - fs).sum()) if len(F) > 4 else 0.0
    fx = fs - (0.559 * r["io"] if OPT[0] else 0.0)
    tok, ms, a = 0.0, start, 1.0
    for i, (rw, kp, ck, dr) in enumerate(pc):
        b = min(rw, 16)
        t = max(2 * fx + beta * (a + b) + REPLAY, 0.10 * mrow * b)
        tok += (a * u2 + (1 - u2)) if i > 0 else 0.0
        ms += t
        a = min(kp, b)
    return tok + a, ms + fx + beta * a
for opt in (False, True):
    OPT[0] = opt
    for bn, beta in list(BETAS.items())[:3]:
        s = [sim2(r, beta, 0.98) for r in P]
        g = 100 * (sum(x[0] for x in s) / sum(x[1] for x in s)) / (sum(r['n'] for r in P) / sum(1000 * r['n'] / r['tok_s'] for r in P)) - 100
        say(f"  SHADOW as two passes ({'optimistic' if opt else 'conservative'}), beta {bn:12s} U2 0.98 (an upper bound: no drift cost): whole {g:+5.1f} %")
open(os.path.expanduser("~/Breakthrough/data/trees/runahead_model.txt"), "w").write("\n".join(lines) + "\n")
