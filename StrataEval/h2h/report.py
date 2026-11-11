"""report.py - the Strata head-to-head report from run_arm.py's records (stdlib only).

Per arm (A default lossy HC_Q8, B A+MTP, C exact STRATA_HC_Q8=0, D C+MTP) and set (S5's 12 HumanEval / 12 MBPP, N45's 12 IFEval /
12 math), with N45's definitions (BigMoE scripts/n45.sh):
  WHOLE DECODE  geo mean over prompts of the prompt's mean over passes of generated / decode time (Strata's DONE decode_ms:
                first generated token to last, the number its server reports as predicted_per_second); per-pass geo means
  pooled        all generated tokens / all decode time
  END TO END    geo mean of generated / the request's wall time (GEN sent -> DONE: prompt + decode)
  prefill       the engine's prompt tokens read / prompt_ms (geo, pooled), and N45's P / TTFT
  TTFT          GEN sent -> first token, wall (median, mean)
  peak          /usr/bin/time -l's peak memory footprint (lifetime phys_footprint), max over the arm's processes
  identity      every run vs NAX6-N's one-token AR (EvalSuite results/ref2tnax), N45's rule: equal over the common prefix;
                information only (no MBPP AR exists). Also B vs A and D vs C (MTP must not change greedy tokens).
Requests that overlap a fanwatch freeze window (~/Library/Logs/fanwatch.log) are dropped, as in N45.
Writes <results>/REPORT.md and report.json.  Usage: python3 -I report.py [--results DIR]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import re
import statistics as st
from pathlib import Path

H = Path(__file__).resolve().parent
SETS = ("humaneval", "mbpp", "ifeval", "math")
SETNAME = {"humaneval": "12 HumanEval (S5)", "mbpp": "12 MBPP (S5)", "ifeval": "12 IFEval (N45)", "math": "12 math (N45)"}
ARMS = {"A": "default (HC Q8_0, lossy), no MTP", "B": "default + MTP", "C": "exact (STRATA_HC_Q8=0), no MTP", "D": "exact + MTP"}


def geo(xs):
    ys = [x for x in xs if x and x > 0]
    return math.exp(sum(map(math.log, ys)) / len(ys)) if ys else float("nan")


def fmt(x, nd=1):
    return "-" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"


def freeze_windows(path: Path):
    win, t0 = [], None
    try:
        for ln in open(path, errors="replace"):
            m = re.match(r"(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) (froze|cool)", ln)
            if not m:
                continue
            t = dt.datetime.fromisoformat(m.group(1)).timestamp()
            if m.group(2) == "froze":
                t0 = t
            elif t0 is not None:
                win.append((t0, t))
                t0 = None
    except OSError:
        pass
    if t0 is not None:
        win.append((t0, float("inf")))
    return win


def ours_reference():
    """Our engine's numbers for the same prompt sets, read-only from BigMoE's reports (information)."""
    out = []
    try:
        cur = None
        for ln in open(Path.home() / "BigMoE" / "data" / "s5p" / "s5p_report.txt"):
            if ln.startswith("12 HumanEval"):
                cur = "humaneval"
            elif ln.startswith("MBPP12"):
                cur = "mbpp"
            m = re.match(r"\s+(\w+) WHOLE DECODE\s+([\d.]+) tok/s.*END TO END\s+([\d.]+);.*tokens/task (\d+); lifetime peak ([\d.]+) GB", ln)
            if m and cur:
                cfg = {"B": "S5' B (cert config, lossy FAST)", "X": "S5' X (lossy FAST)"}.get(m.group(1), m.group(1))
                out.append(dict(set=cur, config=cfg, whole=float(m.group(2)), e2e=float(m.group(3)), tokens=int(m.group(4)),
                                peak=float(m.group(5)), ar=None, src="BigMoE data/s5p/s5p_report.txt"))
    except OSError:
        pass
    try:
        for ln in open(Path.home() / "BigMoE" / "data" / "n45" / "n45_report.txt"):
            m = re.match(r"\s+12 (IFEval|math)\s+(\w+)\s*: WHOLE DECODE\s+([\d.]+) tok/s.*END TO END\s+([\d.]+); tokens/task (\d+); "
                         r"lifetime peak ([\d.]+) GB; == NAX6-N AR (\d+) / (\d+)", ln)
            if m:
                cfg = {"EB": "N45 EB (exact)", "L": "N45 L (lossy)"}.get(m.group(2), m.group(2))
                out.append(dict(set=m.group(1).lower(), config=cfg, whole=float(m.group(3)), e2e=float(m.group(4)),
                                tokens=int(m.group(5)), peak=float(m.group(6)), ar=f"{m.group(7)} / {m.group(8)}",
                                src="BigMoE data/n45/n45_report.txt"))
    except OSError:
        pass
    return out


def prefix_equal(o, r):
    m = min(len(o), len(r))
    return m > 0 and o[:m] == r[:m]


def first_diff(o, r):
    for i, (x, y) in enumerate(zip(o, r)):
        if x != y:
            return i
    return min(len(o), len(r))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(H / "results"))
    ap.add_argument("--fanlog", default=str(Path.home() / "Library" / "Logs" / "fanwatch.log"))
    a = ap.parse_args()
    R = Path(a.results)
    ar = json.loads((H / "prompts" / "ar_ref.json").read_text())["refs"]
    recs, summ = [], {}
    for f in sorted(R.glob("*.requests.jsonl")):
        for ln in open(f):
            try:
                recs.append(json.loads(ln))
            except ValueError:
                pass
    for f in sorted(R.glob("*.summary.json")):
        s = json.loads(f.read_text())
        summ[s["tag"]] = s
    win = freeze_windows(Path(a.fanlog))
    dropped = {"error": 0, "freeze": 0, "incomplete_process": 0}
    done_tags = {p.name[:-5] for p in R.glob("*.done")}
    use = []
    for r in recs:
        if r.get("role") != "measure":
            continue
        if "error" in r or "decode_ms" not in r:
            dropped["error"] += 1
            continue
        if r["tag"] not in done_tags:
            dropped["incomplete_process"] += 1
            continue
        if any(r["t_send"] < b and r["t_done"] > a_ for a_, b in win):
            dropped["freeze"] += 1
            continue
        use.append(r)
    rows, tables = [], {}
    for arm in ARMS:
        for s in (*SETS, "all"):
            rs = [r for r in use if r["arm"] == arm and (s == "all" or r["set"] == s)]
            if not rs:
                continue
            keys = sorted({r["key"] for r in rs})
            tps = lambda r: r["generated"] / (r["decode_ms"] / 1e3) if r["generated"] >= 2 and r["decode_ms"] > 0 else None
            e2e = lambda r: r["generated"] / r["wall_total_s"] if r.get("wall_total_s") else None
            pfr = lambda r: (r.get("prompt_read", r["prompt"] - r["reused"])) / (r["prompt_ms"] / 1e3) if r["prompt_ms"] > 0 else None
            pft = lambda r: r["n_prompt"] / r["wall_ttft_s"] if r.get("wall_ttft_s") else None
            cdec = lambda r: (r["n_ids"] - 1) / (r["t_done"] - r["t_first"]) if r.get("t_first") and r["n_ids"] >= 2 and r["t_done"] > r["t_first"] else None

            def per_key(fn):
                v = {}
                for k in keys:
                    xs = [fn(r) for r in rs if r["key"] == k and fn(r) is not None]
                    if xs:
                        v[k] = sum(xs) / len(xs)
                return v
            passes = sorted({r["pass"] for r in rs})
            row = dict(arm=arm, set=s, n_runs=len(rs), n_keys=len(keys), passes=passes,
                       whole=geo(per_key(tps).values()),
                       whole_pass=[geo([tps(r) for r in rs if r["pass"] == p]) for p in passes],
                       pooled=sum(r["generated"] for r in rs) / max(1e-9, sum(r["decode_ms"] for r in rs) / 1e3),
                       client_decode=geo(per_key(cdec).values()),
                       e2e=geo(per_key(e2e).values()),
                       prefill_engine=geo(per_key(pfr).values()),
                       prefill_pooled=sum(r.get("prompt_read", r["prompt"] - r["reused"]) for r in rs) / max(1e-9, sum(r["prompt_ms"] for r in rs) / 1e3),
                       prefill_p_over_ttft=geo(per_key(pft).values()),
                       ttft_med=st.median([r["wall_ttft_s"] for r in rs if r.get("wall_ttft_s")] or [float("nan")]),
                       ttft_mean=st.mean([r["wall_ttft_s"] for r in rs if r.get("wall_ttft_s")] or [float("nan")]),
                       tokens_task=st.median(list(per_key(lambda r: r["generated"]).values()) or [float("nan")]),
                       stops=sum(1 for r in rs if r.get("finish") == "stop"),
                       drafts=(sum(r["drafts_accepted"] for r in rs), sum(r["drafts_offered"] for r in rs)))
            if s != "all":
                e = n = 0
                fd = []
                for r in rs:
                    ref = ar.get(r["key"], {}).get("completion_ids")
                    if ref is None:
                        continue
                    n += 1
                    e += prefix_equal(r["ids"], ref)
                    fd.append(first_diff(r["ids"], ref))
                row["ar"] = (e, n)
                row["ar_first_diff_median"] = st.median(fd) if fd else None
            rows.append(row)
    # process-level: peak footprint, arena, HC mode
    procs = {}
    def hc_q8_on(tag, s):   # the engine prints this line in both modes; the exact BF16 read converts 0 matrices
        log = R / f"{tag}.engine.log"
        m = re.search(r"HC read in Q8_0 \(.*?\): *(\d+) matrices", log.read_text(errors="replace")) if log.exists() else None
        return int(m.group(1)) > 0 if m else s.get("hc_q8")
    for tag, s in sorted(summ.items()):
        procs[tag] = dict(arm=s["arm"], pas=s["pass"], rc=s.get("rc"), done=tag in done_tags,
                          peak_gb=(s.get("peak_footprint_b") or 0) / 1e9 or None, rss_gb=(s.get("max_rss_b") or 0) / 1e9 or None,
                          arena_gb=s.get("arena_gb"), slots=s.get("arena_slots"), wired_gb=s.get("wired_gb"),
                          after_fill_gb=s.get("after_fill_footprint_gb"), planned_gb=s.get("planned_footprint_gb"),
                          cap_gb=s.get("footprint_cap_gb"), hc_q8=hc_q8_on(tag, s), ready_s=s.get("ready_s"), wall_s=s.get("wall_s"),
                          abort=s.get("abort"), env=s.get("env"))
    # cross-arm identity: B vs A, D vs C, C vs A (same key and pass)
    ids = {(r["arm"], r["pass"], r["key"]): r["ids"] for r in use}
    cross = {}
    for x, y in (("B", "A"), ("D", "C"), ("C", "A"), ("D", "B")):
        pairs = [(ids[(x, p, k)], ids[(y, p, k)]) for (ar_, p, k) in ids if ar_ == x and (y, p, k) in ids]
        cross[f"{x} vs {y}"] = (sum(1 for o, q in pairs if o == q), len(pairs))
    ours = ours_reference()
    # ---- REPORT.md
    L = [f"# Strata head-to-head ({dt.datetime.now():%Y-%m-%d %H:%M})", "",
         "Strata 0.1.58-metal (commit 146df3c), Qwen3.8-Flash-Next ISTA-DASLab IQ3_XXS, experts streamed from the SSD (64 GB Mac), "
         "greedy, 256 new tokens, EOS stop, one warm process per arm and pass (palindrome A B C D D C B A), the exact prompt ids "
         "S5 / N45 used. Arms: " + "; ".join(f"**{k}** {v}" for k, v in ARMS.items()) + ".", "",
         f"Records: {len(use)} measured requests used; dropped: {dropped}. Freeze windows in fanwatch.log: {len(win)}.", ""]
    hdr = ("| arm | set | WHOLE DECODE tok/s (passes) | pooled | END TO END | prefill tok/s (engine / P÷TTFT) | TTFT s (median) | "
           "tokens/task | stops | MTP accepted / offered | == NAX6-N AR |")
    L += ["## Speed and identity", "", hdr, "|" + "---|" * 11]
    for r in rows:
        arx = f"{r['ar'][0]} / {r['ar'][1]}" if r.get("ar") and r["ar"][1] else ("n/a (no AR ref)" if r["set"] == "mbpp" else "-")
        dr = f"{r['drafts'][0]} / {r['drafts'][1]}" if r["drafts"][1] else "-"
        L.append(f"| {r['arm']} | {SETNAME.get(r['set'], r['set'])} | **{fmt(r['whole'])}** ({', '.join(fmt(x) for x in r['whole_pass'])}) | "
                 f"{fmt(r['pooled'])} | {fmt(r['e2e'])} | {fmt(r['prefill_engine'], 0)} / {fmt(r['prefill_p_over_ttft'], 0)} | "
                 f"{fmt(r['ttft_med'], 2)} | {fmt(r['tokens_task'], 0)} | {r['stops']}/{r['n_runs']} | {dr} | {arx} |")
    L += ["", "## Processes (peak = /usr/bin/time -l peak memory footprint)", "",
          "| run | done | peak GB | max RSS GB | arena GB (slots) | wired GB | footprint after fill / cap GB | HC Q8 | ready s | wall s | note |",
          "|" + "---|" * 11]
    for tag, p in procs.items():
        hc_ok = (p["hc_q8"] is True) == (p["arm"] in ("A", "B"))
        L.append(f"| {tag} | {p['done']} | {fmt(p['peak_gb'], 2)} | {fmt(p['rss_gb'], 2)} | {fmt(p['arena_gb'], 2)} ({p['slots'] or '-'}) | "
                 f"{fmt(p['wired_gb'])} | {fmt(p['after_fill_gb'])} / {fmt(p['cap_gb'])} | {p['hc_q8']}{'' if hc_ok else ' (WRONG MODE)'} | "
                 f"{fmt(p['ready_s'], 1)} | {fmt(p['wall_s'], 0)} | {p['abort'] or ''} |")
    peak_arm = {arm: max([p["peak_gb"] or 0 for p in procs.values() if p["arm"] == arm] or [0]) for arm in ARMS}
    L += ["", "Peak footprint per arm (max over its processes): " + ", ".join(f"{k} {fmt(v, 2)} GB" for k, v in peak_arm.items() if v), "",
          "## Token identity across arms (same prompt and pass; information)", ""]
    L += [f"- {k}: {v[0]} / {v[1]} identical" for k, v in cross.items() if v[1]]
    if ours:
        L += ["", "## Our engine on the same prompt sets (read-only from BigMoE's reports; information)", "",
              "| set | config | WHOLE DECODE tok/s | END TO END | tokens/task | lifetime peak GB | == NAX6-N AR | source |", "|" + "---|" * 8]
        for o in ours:
            L.append(f"| {SETNAME[o['set']]} | {o['config']} | {o['whole']} | {o['e2e']} | {o['tokens']} | {o['peak']} | {o['ar'] or '-'} | {o['src']} |")
        L += ["", "Note: our END TO END includes the screen's prefill + cache / drafter setup; Strata's is the request's wall time."]
    text = "\n".join(L) + "\n"
    (R / "REPORT.md").write_text(text)
    (R / "report.json").write_text(json.dumps(dict(rows=rows, processes=procs, cross=cross, dropped=dropped, ours=ours), indent=1,
                                              default=str))
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
