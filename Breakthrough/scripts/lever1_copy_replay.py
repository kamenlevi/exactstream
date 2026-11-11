"""Lever 1 of ~/Breakthrough/PLAN_12GB_90.md (Breakthrough, Fri 2026-11-06 04:0x; pre-registered in SCOREBOARD): what a copy
draft (prompt lookup) adds to DFlash2's kept per check on code at 12 GB. CPU, read-only inputs, seconds.

Inputs (read-only): Fit12's calibration run c51 (~/Fit12/data/scr/t3cal): per check [rows, flag, kept, cf] (<tag>.json),
the 15 drafted tokens (<tag>_cal.json "d"), the output tokens (<tag>_toks.json, = the exact tokens, cap 256); the prompt
ids from ~/EvalSuite/data/items/dec_humaneval.jsonl.

A check from position q keeps 1 + L tokens, L = the drafts' matched prefix (<= 15; the end of the output caps it).
  - drafter at a logged start: its logged drafts (must reproduce the logged kept); elsewhere it runs to its next logged
    divergence (the first p_i + L_i >= q with L_i < 15), at most 15;
  - copy: prompt lookup, the latest earlier occurrence of the longest suffix n-gram (n = NMAX..1) in prompt + output so
    far, then the W tokens after it (W = 15, or 31 for a 32-row copy check).
Walks (each from q = 0 to the end): A today (drafter), B tree max(drafter, copy) (31 rows), C_N copy if its n-gram match
>= N else drafter, E as B with 31 copy drafts (a 32-row copy check).
Out: ~/Breakthrough/data/lever1/replay.json + the printed table."""
import json, os, sys
T3 = os.path.expanduser("~/Fit12/data/scr/t3cal"); TAG = "c51"; ARM = os.environ.get("L1_ARM", "g1k2")
ITEMS = os.path.expanduser("~/EvalSuite/data/items/dec_humaneval.jsonl"); NMAX = 8
OUT = os.path.expanduser("~/Breakthrough/data/lever1"); os.makedirs(OUT, exist_ok=True)
prompts = {}
for line in open(ITEMS):
    it = json.loads(line); prompts[it["key"]] = it["prompt_ids"]
keys = sorted({f.split("_")[3] for f in os.listdir(T3) if f.startswith(f"{TAG}_humaneval_HumanEval_") and f.endswith(f"_{ARM}_cal.json")}, key=int)

def matched(prop, toks, q, cap):
    L = 0
    while L < min(cap, len(prop)) and q + L < len(toks) and prop[L] == toks[q + L]: L += 1
    return L

def copy_props(prompt, toks, W):
    """for every q: (copy proposal's matched prefix L, n-gram match length n); n = 0: no match"""
    C = list(prompt); idx = [dict() for _ in range(NMAX + 1)]
    for e in range(1, len(C)):                    # n-grams ending before the prompt's end (exclusive end e)
        for n in range(1, NMAX + 1):
            if e - n >= 0: idx[n][tuple(C[e - n:e])] = e
    Lc, Nc = [], []
    for q in range(len(toks)):
        m = len(C); best = (0, 0)
        for n in range(min(NMAX, m), 0, -1):
            e = idx[n].get(tuple(C[m - n:m]))
            if e is not None:
                prop = C[e:e + W]; best = (matched(prop, toks, q, W), n); break
        Lc.append(best[0]); Nc.append(best[1])
        for n in range(1, NMAX + 1):              # the suffix ending at m joins the index after its own lookup
            if m - n >= 0: idx[n][tuple(C[m - n:m])] = m
        C.append(toks[q])
    return Lc, Nc

def walk(n_tok, pick, rows=16):
    """pick(q) -> kept, or (kept, rows of this check); returns (checks, rows)"""
    q = checks = rw = 0
    while q < n_tok:
        p = pick(q); k, r = p if isinstance(p, tuple) else (p, rows)
        k = max(1, min(k, n_tok - q)); q += k; checks += 1; rw += r
    return checks, rw

res = {"prompts": {}, "arm": ARM}; tot = {}
for kk in keys:
    key = f"humaneval/HumanEval/{kk}"; base = os.path.join(T3, f"{TAG}_humaneval_HumanEval_{kk}_{ARM}")
    lg = json.load(open(base + ".json")); cal = json.load(open(base + "_cal.json")); toks = json.load(open(base + "_toks.json"))
    assert len(lg) == len(cal), (key, len(lg), len(cal))
    n_tok = len(toks); q = 0; div = []; bad = 0
    for i, ((rows, flag, kept, cf), c) in enumerate(zip(lg, cal)):
        L = matched(c["d"], toks, q, 15); kc = min(1 + L, n_tok - q)
        last = i == len(lg) - 1                       # the output file stops at EOS / the cap; the last check may keep past it
        bad += (kc != kept) if not last else (kc != min(kept, n_tok - q))
        if L < 15 and q + L < n_tok: div.append(q + L)
        q += kept
    assert q >= n_tok and q - lg[-1][2] < n_tok, (key, q, n_tok)
    div.sort()
    def dr(q_, div=div, n_tok=n_tok):
        nxt = next((v for v in div if v >= q_), n_tok); return min(15, nxt - q_)
    L15, N15 = copy_props(prompts[key], toks, 15); L31, _ = copy_props(prompts[key], toks, 31)
    w = {"A": walk(n_tok, lambda q_: 1 + dr(q_)),
         "B": walk(n_tok, lambda q_: 1 + max(dr(q_), L15[q_]), rows=31),
         "E": walk(n_tok, lambda q_: max(1 + dr(q_), 1 + L31[q_]), rows=47)}
    for N in (2, 3, 4, 6):
        w[f"C{N}"] = walk(n_tok, lambda q_, N=N: 1 + (L15[q_] if N15[q_] >= N else dr(q_)))
    for N in (3, 4, 6, 8):   # post-hoc (Fri 04:0x): a 32-row copy check (31 copy drafts) when the n-gram match is >= N
        w[f"CW{N}"] = walk(n_tok, lambda q_, N=N: (1 + L31[q_], 32) if N15[q_] >= N else (1 + dr(q_), 16))
    cov = sum(1 for x in L15 if x > 0) / n_tok   # positions whose next token the copy proposal gets right
    res["prompts"][key] = dict(tokens=n_tok, logged_checks=len(lg), reproduced_bad=bad, walks=w, copy_hit=round(cov, 3),
                               prompt_len=len(prompts[key]))
    for a, (v, r) in w.items(): tot[a] = tot.get(a, 0) + v; tot[a + "_rows"] = tot.get(a + "_rows", 0) + r
    tot["tok"] = tot.get("tok", 0) + n_tok; tot["logged"] = tot.get("logged", 0) + len(lg); tot["bad"] = tot.get("bad", 0) + bad
    print(f"{key:24s} tok {n_tok:3d} prompt {len(prompts[key]):4d} | logged {len(lg):2d} (mismatch {bad}) | "
          + " ".join(f"{a} {n_tok / v:5.2f}" for a, (v, r) in w.items()) + f" | copy hit {cov:.2f}", flush=True)
T = tot["tok"]; A = T / tot["A"]; WALKS = ["A", "B", "C2", "C3", "C4", "C6", "E", "CW3", "CW4", "CW6", "CW8"]
print(f"\nTOTAL {len(keys)} prompts, {T} tokens; logged checks {tot['logged']} (kept {T / tot['logged']:.2f}), "
      f"check-level mismatches {tot['bad']}")
print("| walk | checks | kept per check | vs A | rows per kept token |\n|---|---|---|---|---|")
for a in WALKS:
    print(f"| {a} | {tot[a]} | {T / tot[a]:.2f} | x{(T / tot[a]) / A:.3f} | {tot[a + '_rows'] / T:.2f} |")
res["total"] = {a: dict(checks=tot[a], kept=round(T / tot[a], 3), vs_A=round((T / tot[a]) / A, 4),
                        rows_per_tok=round(tot[a + "_rows"] / T, 3)) for a in WALKS}
res["total"].update(tokens=T, logged_checks=tot["logged"], mismatches=tot["bad"])
json.dump(res, open(os.path.join(OUT, f"replay_{ARM}.json"), "w"), indent=1)
print(f"wrote {os.path.join(OUT, f'replay_{ARM}.json')}")
