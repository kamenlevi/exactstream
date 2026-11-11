"""M-16 RUNAHEAD step 2 post-analysis (MoonShots): reads data/runahead/ra_<C>_checks.jsonl (runahead_live.py's per-check
records) and asks what the measured numbers mean for the engine:
  1. beta and the constant from the all-resident exact re-run (ms vs rows), per prompt type and pooled;
  2. U2 by the runahead's own top-1 minus top-2 logit margin at its bonus row: a FALL-BACK rule (margin < m -> wait for the
     exact verdict for this check, i.e. today's serial check) trades bubbles for serial checks; best m per variant;
  3. modelled tok/s per type with and without the fall-back rule, at the SSD's deep-queue rate and at 10 GB/s.
usage: [BigMoE venv] python runahead_post.py [C=224] -> stdout + data/runahead/ra_<C>_post.txt"""
import json, os, sys
import numpy as np

C = int(sys.argv[1]) if len(sys.argv) > 1 else 224
D = os.environ.get("RA_DIR", os.path.expanduser("~/MoonShots/data/runahead"))
R = [json.loads(l) for l in open(os.path.join(D, f"ra_{C}_checks.jsonl"))]
TYPE = lambda k: "code" if k.startswith("humaneval") else "math" if k.startswith(("math", "gsm")) else "chat"
lines = []


def say(s):
    print(s, flush=True); lines.append(s)


rows = np.array([r["rows"] for r in R], float)
res = np.array([r["res_ms"] for r in R])
beta, alpha = np.polyfit(rows, res, 1)
say(f"M-16 step 2 post-analysis, C {C}: {len(R)} checks over {len({r['key'] for r in R})} prompts")
say(f"  all-resident exact re-run: {alpha:.1f} ms + {beta:.2f} ms x rows (pooled fit); per type:")
for t in ("code", "math", "chat"):
    s = [r for r in R if TYPE(r["key"]) == t]
    if len(s) > 3:
        b, a = np.polyfit([r["rows"] for r in s], [r["res_ms"] for r in s], 1)
        say(f"    {t}: {a:.1f} + {b:.2f} x rows ({len(s)} checks; median rows {np.median([r['rows'] for r in s]):.1f}, "
            f"exact check {np.median([r['ck_ms'] for r in s]):.1f} ms, re-run {np.median([r['res_ms'] for r in s]):.1f} ms)")
U = {"skip": ("u1", "u3"), "hybrid": ("u1h", "u3h"), "hybrid8": ("u1h8", "u3h8")}


def bonus_margin(r, var):
    k = r[var]["kR"]                                   # the runahead's own bonus row
    m = r["marg"][var]
    return m[k] if k < len(m) else 0.0


def tps(recs, var, mpm, fb=None, cap=None):
    """today vs modelled RUNAHEAD (runahead_live.py's model); fb = margin below which the check waits for its exact verdict
    (no bubble possible; cost = today's serial check for that check, the next block drafted from the exact bonus)"""
    u, u3k = U[var]
    tok_t = tok_p = t_t = t_p = 0.0
    prevk = None
    for r in recs:
        n = r["k"] + 1
        tok_t += n; t_t += r["draft_ms"] + r["ck_ms"]
        per = r["io_ms"] / max(r["misses"], 1.0)
        if fb is not None and bonus_margin(r, var) < fb:
            t_p += r["draft_ms"] + r["ck_ms"]            # serial check: no runahead for this block
            tok_p += n; prevk = None
            continue
        nA = (prevk + 1) if prevk is not None else 0
        rowsB, sc = r["rows"], 1.0
        if cap is not None:                              # A + B within REF2's row limit (NewIdeas 12:4x)
            rowsB = max(1, min(r["rows"], cap - nA))
            n = min(n, rowsB)
            sc = (rowsB / r["rows"]) ** 0.67
        comp = alpha + beta * (rowsB + nA)
        ssd = mpm * r[u]["req"] * sc
        stall = (r[u]["miss"] - r[u]["cov"]) * per + r[u].get("wait", 0) * per
        t_p += r["draft_ms"] + max(comp, ssd) + stall
        if not r[var]["agree"]:
            t_p += r["draft_ms"] + alpha + beta * r["rows"]
        x = r.get(u3k)
        if x is not None and not x["same"]:
            n = min(n, x["lcp"] + 1)
        tok_p += n
        prevk = n - 1
    return tok_t / t_t * 1e3, tok_p / t_p * 1e3


for var in U:
    ag = np.array([r[var]["agree"] for r in R])
    mg = np.array([bonus_margin(r, var) for r in R])
    say(f"  {var}: U2 {100 * ag.mean():.1f} %; by the runahead's bonus-row margin (logits):")
    for lo, hi in ((0, 0.5), (0.5, 1), (1, 2), (2, 4), (4, 1e9)):
        sel = (mg >= lo) & (mg < hi)
        if sel.any():
            say(f"    [{lo}, {hi if hi < 1e9 else 'inf'}): {sel.sum()} checks, U2 {100 * ag[sel].mean():.1f} %")
    for t in ("code", "math", "chat", "all"):
        s = [r for r in R if t == "all" or TYPE(r["key"]) == t]
        if not s:
            continue
        a0, b0 = tps(s, var, 2.16 / 13.8)
        _, b10 = tps(s, var, 2.16 / 10.0)
        best = max(((tps(s, var, 2.16 / 13.8, fb)[1], fb) for fb in (0.25, 0.5, 1.0, 2.0)), key=lambda z: z[0])
        _, b16 = tps(s, var, 2.16 / 13.8, cap=16)
        say(f"    {t}: today {a0:.1f} tok/s -> RUNAHEAD {b0:.1f} ({b0 / a0 - 1:+.1%}; SSD 10 GB/s {b10 / a0 - 1:+.1%}; within 16 "
            f"rows {b16 / a0 - 1:+.1%}); with the fall-back at margin < {best[1]}: {best[0]:.1f} ({best[0] / a0 - 1:+.1%})")
open(os.path.join(D, f"ra_{C}_post.txt"), "w").write("\n".join(lines) + "\n")
