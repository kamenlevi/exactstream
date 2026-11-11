"""Fit12 (Sun 8 Nov evening; SMALL_RAM paper, brief /Users/kamen/Papers/BRIEF.md rules 1-3): the 12 GB rows of the result
table, re-derived from the raw harness logs and per-prompt phase logs in one way for every run. READ-ONLY on the data
folders; it writes only /Users/kamen/Fit12/data/paper/small_ram_12gb_rows.{json,txt}.
Per run (only prompts the harness counted: a PAIR '-> ok' line when the log has PAIR lines, else every per-prompt line):
- WHOLE DECODE = n / dec_s (the harness's decode window, after prefill, holding the drafter setup, each prompt's first
  check and all host work): geometric mean over prompts and pooled (sum n / sum dec_s);
- WHOLE REQUEST = n / (pf_s + dec_s): geometric mean and pooled;
- FIRST TOKEN = pf_s + (the decode window up to the end of check 0's last phase) = pf_s + dec_s - (eN - e0), from the
  per-phase marks (F12_PHLOG=1; the c1u16_read.py definition). pf_s covers the prompt minus its last token; the last
  token goes through check 0, which commits the first output tokens. Mean, median, min, max;
- the process's lifetime peak phys_footprint (the harness's 'done; lifetime X GB' line, RUSAGE_INFO_V4);
- kept tokens per check, SSD MB per output token inside the decode phases (when the phase log has bytes).
Usage: python -I -B paper_rows.py"""
import json, math, os, re, statistics, sys

S = "/Users/kamen/Fit12/data/scr"
RUNS = [  # (row id, log, tag, arm, what it is)
    ("ex16", f"{S}/ex16/ex16.txt", "ex16", "ex", "12GB-CLEAN: verified greedy, 16 HumanEval, cap 256, quiet"),
    ("c1u16", f"{S}/c1u16/c1u16.txt", "c1u16", "t80ck1u", "two-pass FAST theta 0.8, 16 HumanEval, cap 256, quiet"),
    ("tab12e", f"{S}/tab12/tab12e.txt", "tab12e", "ex", "12E: verified greedy, 12 HumanEval + 12 MBPP, cap 256, quiet"),
    ("tab12f", f"{S}/tab12/tab12f.txt", "tab12f", "t95ck1u", "12F: two-pass FAST theta 0.95, 12 HumanEval + 12 MBPP, cap 256, quiet"),
    ("chat12", f"{S}/chat12/chat12.txt", "chat12", "ex", "chat: verified greedy (no PR1), 12 IFEval, cap 256"),
    ("chw13a", f"{S}/chw13/chw13a.txt", "chw13a", "ex", "chat A': ex + PR1 + tail, 4-row first call, 12 IFEval, cap 128, share lock"),
    ("chw14c2", f"{S}/chw14/chw14c2.txt", "chw14c2", "ex", "chat P2': width rule cf:0.9:2, 12 IFEval, cap 128, share lock"),
    ("chw14ha", f"{S}/chw14/chw14ha.txt", "chw14ha", "ex", "code A' (W1 4) + PR1 + tail, 16 HumanEval, cap 256, share lock"),
    ("chw14h2", f"{S}/chw14/chw14h2.txt", "chw14h2", "ex", "code P2', 16 HumanEval, cap 256, share lock"),
]
OUTP = "/Users/kamen/Fit12/data/paper/small_ram_12gb_rows"
if len(sys.argv) > 2:   # (chw19) ad-hoc rows: paper_rows.py OUT_BASE log|tag|arm|what ... (writes OUT_BASE.txt / .json only)
    OUTP = sys.argv[1]
    RUNS = [(x.split("|")[1], x.split("|")[0], x.split("|")[1], x.split("|")[2], x.split("|")[3]) for x in sys.argv[2:]]
geo = lambda xs: math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float("nan")


def run(rid, log, tag, arm, what):
    T, A = re.escape(tag), re.escape(arm)
    rx = re.compile(rf"^\[{T}\] (\S+) {A} theta \S+ .*?\| tok (\d+) .*?\| checks (\d+), kept ([\d.]+) .*?\| prefill ([\d.]+) s, "
                    rf"decode ([\d.]+) s")
    rpair = re.compile(rf"^\[{T}\] PAIR (\S+?)( \(re-run\))?: .*-> (.*)$")
    pend, cnt, any_pair, life, void = {}, {}, False, None, []
    for ln in open(log):
        s = ln.rstrip("\n")
        m = rx.match(s)
        if m:
            pend[m.group(1)] = dict(tok=int(m.group(2)), checks=int(m.group(3)), kept=float(m.group(4)),
                                    pf=float(m.group(5)), dec=float(m.group(6)))
            continue
        m = rpair.match(s)
        if m:
            any_pair = True
            r = pend.pop(m.group(1), None)
            if r is not None and m.group(3).startswith("ok"):
                cnt[m.group(1)] = r
            elif r is not None:
                void.append(m.group(1))
            continue
        m = re.match(rf"^\[{T}\] done; lifetime ([\d.]+) GB", s)
        if m:
            life = float(m.group(1))
    if not any_pair:
        cnt = pend
    D = os.path.dirname(log)
    rows = []
    for k, r in cnt.items():
        kf = k.replace("/", "_")
        php = os.path.join(D, f"{tag}_{kf}_{arm}_ph.json")
        first, mb = None, None
        if os.path.exists(php):
            ph = json.load(open(php))
            if ph:
                e0 = max(p[6] for p in ph if p[1] == 0)
                eN = max(p[6] for p in ph)
                first = r["pf"] + r["dec"] - (eN - e0)
                if len(ph[0]) > 8:
                    mb = sum(p[8] - p[4] for p in ph) / 1e6
        rows.append(dict(key=k, **r, first=first, mb=mb))
    if not rows:
        return dict(id=rid, what=what, log=log, n=0)
    n = sum(x["tok"] for x in rows)
    fs = [x["first"] for x in rows if x["first"] is not None]
    mbs = [x["mb"] for x in rows if x["mb"] is not None]
    return dict(id=rid, what=what, log=log, n=len(rows), tokens=n, void=void, counted_by="PAIR ok" if any_pair else "all lines",
                whole_geo=geo([x["tok"] / x["dec"] for x in rows]), whole_pooled=n / sum(x["dec"] for x in rows),
                req_geo=geo([x["tok"] / (x["pf"] + x["dec"]) for x in rows]),
                req_pooled=n / sum(x["pf"] + x["dec"] for x in rows),
                first_mean=statistics.mean(fs) if fs else None, first_median=statistics.median(fs) if fs else None,
                first_min=min(fs) if fs else None, first_max=max(fs) if fs else None, n_first=len(fs),
                pf_mean=statistics.mean(x["pf"] for x in rows),
                kept=sum(x["kept"] * x["checks"] for x in rows) / max(1, sum(x["checks"] for x in rows)),
                mb_per_tok=(sum(mbs) / n) if len(mbs) == len(rows) else None, peak_gb=life)


out = [run(*a) for a in RUNS]
os.makedirs(os.path.dirname(OUTP), exist_ok=True)
json.dump(out, open(OUTP + ".json", "w"), indent=1)
L = ["SMALL_RAM 12 GB rows (paper_rows.py; definitions in its docstring). tok/s geo / pooled; first token s mean (median,"
     " min-max); peak = lifetime phys_footprint GB"]
f2 = lambda v, d=2: "n/a" if v is None else f"{v:.{d}f}"
for o in out:
    if not o["n"]:
        L.append(f"{o['id']:8} NO ROWS ({o['log']})"); continue
    L.append(f"{o['id']:8} n {o['n']:2} ({o['counted_by']}, void {o['void']}) | whole decode {o['whole_geo']:.2f} / {o['whole_pooled']:.2f}"
             f" | whole request {o['req_geo']:.2f} / {o['req_pooled']:.2f} | first token {f2(o['first_mean'])} ({f2(o['first_median'])},"
             f" {f2(o['first_min'])}-{f2(o['first_max'])}; n {o['n_first']}) | prefill mean {o['pf_mean']:.2f} s | kept {o['kept']:.2f}"
             f" | MB/token {f2(o['mb_per_tok'], 0)} | peak {f2(o['peak_gb'], 3)} GB | {o['what']}")
open(OUTP + ".txt", "w").write("\n".join(L) + "\n")
print("\n".join(L))
