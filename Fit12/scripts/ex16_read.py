"""Fit12 (Fri 6 Nov 21:4x; NewIdeas QUEUE f2243e8 '12GB-CLEAN', the orchestrator 21:3x): the report of
/Users/kamen/Fit12/scripts/ex16.sh's run, the 12 GB VERIFIED-GREEDY loop alone (fit_t3.py arm ex in its own process).
Per counted prompt (a PAUSE-voided run is replaced by its clean re-run; a prompt paused twice is excluded): prompt length,
tokens, prefill s, decode s, decode tok/s, end-to-end tok/s (prefill + decode), checks, tokens per check, ms per check
(draft / exact), SSD GB per check (the process's own reads, rusage ri_diskio_bytesread). Then the geometric mean and the
pooled decode tok/s over the counted prompts, the process's lifetime peak phys_footprint (RUSAGE_INFO_V4
ri_lifetime_max_phys_footprint, printed by the harness's 'done' line) and the engine's own settings lines, verbatim.
Usage: python -I -B ex16_read.py [/Users/kamen/Fit12/data/scr/ex16/ex16.txt] [TAG]   (prints; ex16.sh saves it as
ex16_report.txt) and writes <TAG>_report.json beside the log."""
import json, math, os, re, sys

F = sys.argv[1] if len(sys.argv) > 1 else "/Users/kamen/Fit12/data/scr/ex16/ex16.txt"
TAG = sys.argv[2] if len(sys.argv) > 2 else "ex16"
T = re.escape(TAG)
RX = re.compile(rf"^\[{T}\] (\S+) ex theta None .*?\| tok (\d+) sha (\w+) .*?\| checks (\d+), kept ([\d.]+) .*?"
                r"\| prefill ([\d.]+) s, decode ([\d.]+) s, ([\d.]+) tok/s \| lifetime ([\d.]+) GB \| per check ms: draft "
                r"([\d.]+) pass 1 ([\d.]+) pass 2 / exact ([\d.]+) commit ([\d.]+);")
RP = re.compile(rf"^\[{T}\] (\S+) ex PHASES (.*)$")
RPH = re.compile(r"(\w+): (\d+) runs, ([\d.]+) ms, (\d+) MB")
RPAIR = re.compile(rf"^\[{T}\] PAIR (\S+?)( \(re-run\))?: .*-> (.*)$")
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
        pend.setdefault(g[0], {}).update(tok=int(g[1]), sha=g[2], checks=int(g[3]), kept=float(g[4]), pf=float(g[5]),
                                          dec=float(g[6]), tps=float(g[7]), life=float(g[8]), draft=float(g[9]),
                                          exact=float(g[11]))
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
    elif s.startswith(f"[{TAG}] TOTAL ex:"):
        total = s
    elif s.startswith(f"[{TAG}] PAUSE summary:"):
        pz = s
keys = list(cnt)
out = []
P = out.append
P(f"12 GB VERIFIED-GREEDY loop, alone in its process ({F}; tag {TAG})")
P("  arm ex: DFlash2 L1-20 (4-bit) drafts 15 tokens per check; the 12 GB model verifies every token in <= 4-row calls")
P("  (verify rows wait for all ten ranks; ranks 0-1 at 4 bits, 2-9 at 3). NOT exact: 2-4-row verify calls round differently")
P("  from one-token decoding, and near-tie tokens depend on the draft pattern (keep-1 replays; PR1 not installed).")
P("  Workload: T3R2's 16 HumanEval prompts (EvalSuite b12llff4 keys), greedy, cap 256 new tokens, full outputs.")
P("Settings as printed by the engine:")
for s in settings:
    P("  " + s[:300])
P(f"Per prompt ({len(keys)} counted of {len(keys) + len({k for k, _, _ in void} - set(keys))}):")
P("  key                    | prompt | tokens | prefill s | decode s | decode tok/s | e2e tok/s | checks | tok/check | "
  "ms/check draft  exact | SSD GB/check")
rows = []
for k in keys:
    r = cnt[k]
    ph = r.get("phases", {})
    mb = sum(n * mbr for n, _, mbr in ph.values())
    e2e = r["tok"] / max(1e-9, r["pf"] + r["dec"])
    ssd = mb / 1e3 / max(1, r["checks"])
    rows.append(dict(key=k, n_prompt=nprompt.get(k), tok=r["tok"], pf=r["pf"], dec=r["dec"], tps=r["tok"] / max(1e-9, r["dec"]),
                     tps_printed=r["tps"], e2e=e2e, checks=r["checks"], kept=r["tok"] / max(1, r["checks"]),
                     draft=r["draft"], exact=r["exact"], ssd_gb=ssd, rerun=r["rerun"], sha=r["sha"]))
    x = rows[-1]
    P(f"  {k:22s} | {str(x['n_prompt']):>6s} | {x['tok']:6d} | {x['pf']:9.2f} | {x['dec']:8.2f} | {x['tps']:12.2f} | "
      f"{x['e2e']:9.2f} | {x['checks']:6d} | {x['kept']:9.2f} | {x['draft']:14.1f} {x['exact']:6.1f} | {x['ssd_gb']:6.2f}"
      + ("  (clean re-run)" if x["rerun"] else ""))
geo = math.exp(sum(math.log(x["tps"]) for x in rows) / len(rows)) if rows else float("nan")
geo_e = math.exp(sum(math.log(x["e2e"]) for x in rows) / len(rows)) if rows else float("nan")
tok, dec, ch = sum(x["tok"] for x in rows), sum(x["dec"] for x in rows), sum(x["checks"] for x in rows)
P(f"Decode tok/s over {len(rows)} prompts: GEOMETRIC MEAN {geo:.2f}; pooled {tok / max(1e-9, dec):.2f} ({tok} tokens in "
  f"{dec:.2f} s); min {min(x['tps'] for x in rows):.2f}, max {max(x['tps'] for x in rows):.2f}" if rows else "no counted prompts")
if rows:
    P(f"End-to-end tok/s (prefill + decode): geometric mean {geo_e:.2f}; tokens per check {tok / max(1, ch):.2f}; "
      f"ms per check draft {sum(x['draft'] * x['checks'] for x in rows) / ch:.1f}, exact {sum(x['exact'] * x['checks'] for x in rows) / ch:.1f}; "
      f"SSD {sum(x['ssd_gb'] * x['checks'] for x in rows) / ch:.2f} GB per check")
P(f"Lifetime peak phys_footprint of this process (RUSAGE_INFO_V4): "
  + (f"{done:.3f} GB = {done * 1e9 / 2 ** 30:.3f} GiB" if done is not None else "MISSING (no 'done' line: the run did not finish)"))
P(f"PAUSE: {pz or 'no summary line'}; void runs: {void or 'none'}")
P(f"Harness's own total: {total or 'MISSING'}")
print("\n".join(out))
json.dump(dict(prompts=rows, geo_tps=geo, pooled_tps=tok / max(1e-9, dec) if rows else None, geo_e2e=geo_e,
               lifetime_gb=done, settings=settings, void=void, pause=pz, total=total),
          open(os.path.join(os.path.dirname(os.path.abspath(F)), f"{TAG}_report.json"), "w"), indent=1)
