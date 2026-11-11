"""Fit12 (Sat 7 Nov 08:2x; the orchestrator 08:1x): the report of a ONE-ARM fit_t3.py run (/Users/kamen/Fit12/scripts/c1u16.sh:
arm t80ck1u alone, the 12 GB FAST candidate), in the orchestrator's metrics since 05:0x
(/Users/kamen/FableGod/findings/2026-11-07_0500_steady-metric-overstates-whole-decode.md):
- WHOLE DECODE = n / decode wall time (prefill excluded; the harness's dec_s runs from after prefill to the last token, so it
  holds the drafter's setup, each prompt's first check and all host work): per prompt, geometric mean and pooled;
- FULL-PERIOD STEADY = the tokens after each prompt's first check / all wall time after that check (from the per-phase
  marks, F12_PHLOG=1: the end of check 0's last phase to the end of the last phase); geometric mean and pooled;
- END TO END = n / (prefill + decode);
- the process's lifetime peak phys_footprint (RUSAGE_INFO_V4, the harness's 'done' line);
- tokens identical to the verified-greedy loop's outputs (ex16, information): per prompt equal or the first divergence.
A PAUSE-voided run is replaced by its clean re-run (F12_RERUN=1); a prompt paused twice is excluded and named.
Usage: python -I -B c1u16_read.py [LOG] [TAG] [ARM] [REF_DIR] [REF_TAG] [REF_ARM]
  defaults /Users/kamen/Fit12/data/scr/c1u16/c1u16.txt c1u16 t80ck1u /Users/kamen/Fit12/data/scr/ex16 ex16 ex
Prints the report and writes <TAG>_report.json beside the log."""
import json, math, os, re, sys

A = sys.argv[1:] + [None] * 6
F = A[0] or "/Users/kamen/Fit12/data/scr/c1u16/c1u16.txt"
TAG, ARM = A[1] or "c1u16", A[2] or "t80ck1u"
RD, RT, RA = A[3] or "/Users/kamen/Fit12/data/scr/ex16", A[4] or "ex16", A[5] or "ex"
D = os.path.dirname(os.path.abspath(F))
T, AR = re.escape(TAG), re.escape(ARM)
RX = re.compile(rf"^\[{T}\] (\S+) {AR} theta (\S+) .*?\| tok (\d+) sha (\w+) .*?\| checks (\d+), kept ([\d.]+) \| pass2 (\d+) "
                r"exact (\d+) cfgate (\d+) skipped (\d+) .*?\| prefill ([\d.]+) s, decode ([\d.]+) s, ([\d.]+) tok/s \| lifetime "
                r"([\d.]+) GB \| per check ms: draft ([\d.]+) pass 1 ([\d.]+) pass 2 / exact ([\d.]+) commit ([\d.]+);")
RP = re.compile(rf"^\[{T}\] (\S+) {AR} PHASES (.*)$")
RPH = re.compile(r"(\w+): (\d+) runs, ([\d.]+) ms, (\d+) MB")
RPAIR = re.compile(rf"^\[{T}\] PAIR (\S+?)( \(re-run\))?: .*-> (.*)$")
geo = lambda xs: math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float("nan")
settings, nprompt, pend, cnt, void, done, total, pz = [], {}, {}, {}, [], None, None, None
for ln in open(F):
    s = ln.rstrip("\n")
    if s.startswith("[bigmoe] tier") or s.startswith("b12: ") or s.startswith(f"[{TAG}] bt_b12 "):
        settings.append(s)
        if s.startswith(f"[{TAG}] bt_b12 "):
            nprompt.update({k: int(n) for k, n in re.findall(r"(\S+) \(n (\d+)\)", s)})
        continue
    m = RP.match(s)
    if m:
        pend.setdefault(m.group(1), {})["phases"] = {k: (int(n), float(ms), int(mb)) for k, n, ms, mb in RPH.findall(m.group(2))}
        continue
    m = RX.match(s)
    if m:
        g = m.groups()
        pend.setdefault(g[0], {}).update(theta=g[1], tok=int(g[2]), sha=g[3], checks=int(g[4]), kept=float(g[5]),
                                          modes=dict(pass2=int(g[6]), exact=int(g[7]), cfgate=int(g[8]), skipped=int(g[9])),
                                          pf=float(g[10]), dec=float(g[11]), tps=float(g[12]), life=float(g[13]),
                                          ms=dict(draft=float(g[14]), p1=float(g[15]), p2=float(g[16]), commit=float(g[17])))
        continue
    m = RPAIR.match(s)
    if m:
        key, rerun, verdict = m.group(1), bool(m.group(2)), m.group(3)
        r = pend.pop(key, None)
        if r is not None and verdict.startswith("ok"):
            cnt[key] = dict(r, rerun=rerun)
        elif r is not None:
            void.append((key, "re-run" if rerun else "first run", verdict))
        continue
    if s.startswith(f"[{TAG}] done; lifetime "):
        done = float(re.search(r"lifetime ([\d.]+) GB", s).group(1))
    elif s.startswith(f"[{TAG}] TOTAL {ARM}:"):
        total = s
    elif s.startswith(f"[{TAG}] PAUSE summary:"):
        pz = s
rows = []
for k, r in cnt.items():
    kf = k.replace("/", "_")
    lg = json.load(open(os.path.join(D, f"{TAG}_{kf}_{ARM}.json")))            # per check: rows, path, committed, cf
    ph = json.load(open(os.path.join(D, f"{TAG}_{kf}_{ARM}_ph.json")))         # name, check, (uptime, seq, bytes, misses) x2
    toks = json.load(open(os.path.join(D, f"{TAG}_{kf}_{ARM}_toks.json")))
    rp = os.path.join(RD, f"{RT}_{kf}_{RA}_toks.json")
    ref = json.load(open(rp)) if os.path.exists(rp) else None
    assert len(toks) == r["tok"], (k, len(toks), r["tok"])
    e0 = max(p[6] for p in ph if p[1] == 0)                                     # the end of check 0's last phase
    eN = max(p[6] for p in ph)                                                  # the end of the last phase
    k0 = min(r["tok"], lg[0][2])
    ssd = sum(p[8] - p[4] for p in ph) / 1e6                                   # (Sun 8 Nov) MB read inside the decode phases
    full = (r["tok"] - k0) / (eN - e0) if len(lg) > 1 and eN > e0 else None
    div = None if ref is None else next((i for i, (a, b) in enumerate(zip(toks, ref)) if a != b),
                                        None if len(toks) == len(ref) else min(len(toks), len(ref)))
    rows.append(dict(key=k, n_prompt=nprompt.get(k), tok=r["tok"], pf=r["pf"], dec=r["dec"], whole=r["tps"],
                     e2e=r["tok"] / max(1e-9, r["pf"] + r["dec"]), full=full, first_ms=1e3 * (r["dec"] - (eN - e0)),
                     after_tok=r["tok"] - k0, after_s=eN - e0, checks=r["checks"], kept=r["kept"], modes=r["modes"],
                     ms=r["ms"], rerun=r["rerun"], sha=r["sha"], ssd_mb=ssd, ref_same=None if ref is None else toks == ref,
                     ref_div=div, ref_len=None if ref is None else len(ref), theta=r["theta"]))
out = []
P = out.append
P(f"{ARM} MEASURED, alone in its process ({F}; tag {TAG}); metrics: the orchestrator's since 05:0x (whole decode headline)")
P("Settings as printed by the engine:")
for s in settings:
    P("  " + s[:300])
P(f"Per prompt ({len(rows)} counted of {len(rows) + len({k for k, _, _ in void} - set(cnt))}):")
P("  key                     | prompt | tokens | prefill s | decode s | WHOLE tok/s | full-period | e2e tok/s | checks | kept "
  "| modes pass2/exact/cfgate | first check + setup ms | = ex16 tokens")
for x in rows:
    md = x["modes"]
    eq = "n/a" if x["ref_same"] is None else ("SAME" if x["ref_same"] else f"differs from token {x['ref_div']}")
    P(f"  {x['key']:23s} | {str(x['n_prompt']):>6s} | {x['tok']:6d} | {x['pf']:9.2f} | {x['dec']:8.2f} | {x['whole']:11.2f} | "
      f"{(x['full'] or float('nan')):11.2f} | {x['e2e']:9.2f} | {x['checks']:6d} | {x['kept']:4.2f} | "
      f"{md['pass2']:3d} / {md['exact']:3d} / {md['cfgate']:3d}          | {x['first_ms']:22.0f} | {eq}"
      + ("  (clean re-run)" if x["rerun"] else ""))
res = {}
if rows:
    tok, dec = sum(x["tok"] for x in rows), sum(x["dec"] for x in rows)
    fl = [x for x in rows if x["full"]]
    res = dict(whole_geo=geo([x["whole"] for x in rows]), whole_pooled=tok / dec,
               full_geo=geo([x["full"] for x in fl]), full_pooled=sum(x["after_tok"] for x in fl) / sum(x["after_s"] for x in fl),
               e2e_geo=geo([x["e2e"] for x in rows]), e2e_pooled=tok / sum(x["pf"] + x["dec"] for x in rows),
               tokens=tok, decode_s=dec, checks=sum(x["checks"] for x in rows),
               first_ms_mean=sum(x["first_ms"] for x in rows) / len(rows),
               same=sum(1 for x in rows if x["ref_same"]), compared=sum(1 for x in rows if x["ref_same"] is not None))
    ch = res["checks"]
    P(f"WHOLE DECODE (n / decode wall, prefill excluded) over {len(rows)} prompts: GEOMETRIC MEAN {res['whole_geo']:.2f}; "
      f"pooled {res['whole_pooled']:.2f} ({tok} tokens in {dec:.2f} s); min {min(x['whole'] for x in rows):.2f}, "
      f"max {max(x['whole'] for x in rows):.2f}")
    P(f"FULL-PERIOD STEADY (tokens after each prompt's first check / wall time after it, {len(fl)} prompts): geometric mean "
      f"{res['full_geo']:.2f}; pooled {res['full_pooled']:.2f}")
    P(f"END TO END (n / (prefill + decode)): geometric mean {res['e2e_geo']:.2f}; pooled {res['e2e_pooled']:.2f}")
    res["ssd_mb_per_tok"] = sum(x["ssd_mb"] for x in rows) / tok
    P(f"SSD in decode (the process's own reads inside the phases): {res['ssd_mb_per_tok']:.0f} MB per kept token (pooled), "
      f"{sum(x['ssd_mb'] for x in rows) / 1e3 / max(1, res['checks']):.2f} GB per check; per prompt "
      f"{min(x['ssd_mb'] / x['tok'] for x in rows):.0f}-{max(x['ssd_mb'] / x['tok'] for x in rows):.0f} MB per kept token")
    sets = sorted({x["key"].split("/")[0] for x in rows})      # (Sat 7 Nov, the table) per set when the run mixes sets
    for s_ in (sets if len(sets) > 1 else []):
        rs = [x for x in rows if x["key"].split("/")[0] == s_]
        fs = [x for x in rs if x["full"]]
        bs = dict(n=len(rs), whole_geo=geo([x["whole"] for x in rs]), whole_pooled=sum(x["tok"] for x in rs) / sum(x["dec"] for x in rs),
                  full_geo=geo([x["full"] for x in fs]), full_pooled=sum(x["after_tok"] for x in fs) / sum(x["after_s"] for x in fs),
                  e2e_geo=geo([x["e2e"] for x in rs]), ssd_mb_per_tok=sum(x["ssd_mb"] for x in rs) / sum(x["tok"] for x in rs))
        res[f"set_{s_}"] = bs
        P(f"  {s_} ({bs['n']} prompts): WHOLE DECODE geo {bs['whole_geo']:.2f} / pooled {bs['whole_pooled']:.2f}; full-period "
          f"geo {bs['full_geo']:.2f} / pooled {bs['full_pooled']:.2f}; end to end geo {bs['e2e_geo']:.2f}; SSD "
          f"{bs['ssd_mb_per_tok']:.0f} MB per kept token")
    P(f"Per check ({ch} checks, {tok / ch:.2f} tokens kept): ms draft "
      f"{sum(x['ms']['draft'] * x['checks'] for x in rows) / ch:.1f}, pass 1 {sum(x['ms']['p1'] * x['checks'] for x in rows) / ch:.1f}, "
      f"pass 2 / exact {sum(x['ms']['p2'] * x['checks'] for x in rows) / ch:.1f}, commit "
      f"{sum(x['ms']['commit'] * x['checks'] for x in rows) / ch:.1f}; modes (checks) pass2 "
      f"{sum(x['modes']['pass2'] for x in rows)}, exact {sum(x['modes']['exact'] for x in rows)}, cfgate "
      f"{sum(x['modes']['cfgate'] for x in rows)}; first check + setup {res['first_ms_mean']:.0f} ms per prompt (mean)")
    P(f"Tokens identical to the verified-greedy loop ({RT} {RA}, information): {res['same']} / {res['compared']} prompts; "
      "first divergences: " + (", ".join(f"{x['key'].replace('humaneval/HumanEval/', 'HE/')} @{x['ref_div']}" for x in rows
                                         if x["ref_same"] is False) or "none"))
P(f"Lifetime peak phys_footprint of this process (RUSAGE_INFO_V4): "
  + (f"{done:.3f} GB = {done * 1e9 / 2 ** 30:.3f} GiB" if done is not None else "MISSING (no 'done' line: the run did not finish)"))
P(f"PAUSE: {pz or 'no summary line'}; void runs: {void or 'none'}")
P(f"Harness's own total: {total or 'MISSING'}")
print("\n".join(out))
json.dump(dict(arm=ARM, prompts=rows, lifetime_gb=done, settings=settings, void=void, pause=pz, total=total, **res),
          open(os.path.join(D, f"{TAG}_report.json"), "w"), indent=1)
