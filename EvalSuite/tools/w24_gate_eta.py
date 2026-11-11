#!/usr/bin/env python3
# The w24 card gate's projected finish, for the neighbours (PREREG_card_gates amendment 12b; NewIdeas 23:13). Every minute
# it writes results/w24_gate_eta.txt from runner 61's log and decode files; it never touches the runner. The finishes:
# the budget sim of amendment 12a on the keys still missing (central and slow rates), and the measured pace (the central
# sim scaled by wall time / sim time of the keys done, so memory waits, yields and a neighbour's slowdown count).
# "passes 13:30" uses the measured pace once a process has ended, the central sim before. On a change to "yes" it tells
# NewIdeas (peer send 21406). It ends when the runner has ended or the launcher started nothing.
# Tue 01:5x: it also reports the gate's lifetime footprint peak so far (the processes' runs files and live logs) and
# budgets the gate for neighbours at that peak + 2 GB, rounded up (w13b peaked at 24.813 GB on a long GPQA-D prefill).
# usage: python3 tools/w24_gate_eta.py [--once] [--no-send] [--tag w24] [--out PATH] [--line HH:MM] [--start "Y-m-d H:M:S"]
import glob, json, math, os, re, subprocess, sys, time
from datetime import datetime, timedelta

A = sys.argv[1:]
def opt(name, default):
    return A[A.index(name) + 1] if name in A else default
TAG, ONCE, SEND = opt("--tag", "w24"), "--once" in A, "--no-send" not in A
ES = opt("--es", os.path.expanduser("~/EvalSuite"))   # --es: a test tree
OUT = opt("--out", f"{ES}/results/w24_gate_eta.txt")
LINE = opt("--line", "13:30")
GSTART = datetime.strptime(opt("--start", "2026-11-10 02:10:00"), "%Y-%m-%d %H:%M:%S")   # the launcher's start
LOG, START_LOG = f"{ES}/results/day61_{TAG}_runner.log", f"{ES}/results/card_{TAG}_start.log"
DEC = {o: f"{ES}/results/card_{TAG}_{o}/decode.jsonl" for o in ("fwd", "rev")}
HARD, PERPROC, OV = 1200.0, 20, 65.0
RATES = {"central": (0.63, 0.75, 1.2), "slow": (0.5, 0.65, 1.5)}   # exact-read sets / lcb6 / first token (amendment 12a)

rows = {}                                   # lcoh's rows at 45 GB, as in the budget sim
for f in glob.glob(f"{ES}/results/card/es_cdlcoh_*_runs.json"):
    for r in json.load(open(f)):
        if not r.get("cut"):
            rows.setdefault(r["key"], r)
KEYS = sorted(rows)
assert len(KEYS) == 381, len(KEYS)

def t(k, rate):
    chat, code, pf = rate
    r = rows[k]
    return (r["pfs"] or 0) * pf + r["n"] / (r["tps"] * (code if k.startswith("lcb6/") else chat))

def sim(order, rate):                       # seconds for these keys in this order: PERPROC per process, cut at HARD + 40
    miss, wall, procs = list(order), 0.0, 0
    while miss and procs < 60:
        procs += 1; cum = 0.0; done = []
        for k in miss[:PERPROC]:
            if cum + t(k, rate) > HARD + 40:
                cum = HARD + 40; break
            cum += t(k, rate); done.append(k)
        wall += cum + OV
        miss = [k for k in miss if k not in done]
    return wall

def peak_so_far():                          # the gate's lifetime footprint peak (GB): ended processes' runs + the live logs
    pk = 0.0
    for f in glob.glob(f"{ES}/results/card/es_cd{TAG}_*_runs.json"):
        try:
            pk = max([pk] + [max(r["mem"][1], r["mem"][-1]) for r in json.load(open(f))])
        except (ValueError, KeyError, IndexError):
            pass
    for f in glob.glob(f"{ES}/results/card/es_cd{TAG}_*.log"):
        for m in re.finditer(r"lifetime peak after prefill ([\d.]+) -> after decode ([\d.]+) GB", open(f, errors="replace").read()):
            pk = max(pk, float(m.group(1)), float(m.group(2)))
    return pk

def done_keys(o):
    try:
        return {json.loads(l)["key"] for l in open(DEC[o]) if l.strip()}
    except FileNotFoundError:
        return set()

def stamp(hms, now):                        # a log's HH:MM:SS as a datetime (no date in the log; never in the future)
    d = datetime.combine(now.date(), datetime.strptime(hms, "%H:%M:%S").time())
    return d - timedelta(days=1) if d > now + timedelta(seconds=60) else d

def state(now):
    if not os.path.exists(LOG):
        return "not started", None, 0, None
    L = open(LOG).read().splitlines()
    starts = [(stamp(m.group(1), now), m.group(2), int(m.group(3))) for l in L
              if (m := re.match(rf"(\d\d:\d\d:\d\d) shared lock: evalsuite: MEM \d+ GB: card gate {TAG}, (fwd|rev) process (\d+)", l))]
    ends = [(stamp(m.group(1), now), m.group(2), int(m.group(3))) for l in L
            if (m := re.match(r"(\d\d:\d\d:\d\d) CHAT (fwd|rev) process (\d+): rc=", l))]
    first = starts[0][0] if starts else None
    last = L[-1] if L else ""
    i0 = max([i for i, l in enumerate(L) if " shared lock: evalsuite: MEM " in l] or [-1])
    stops = [l for l in L[i0 + 1:] if "runner stops" in l or "the card verdict:" in l]
    if stops:
        return f"stopped: {stops[-1][9:120]}", None, len(ends), first
    if starts and (not ends or starts[-1][0] > ends[-1][0]):
        s = starts[-1]
        return f"running {s[1]} process {s[2]} since {s[0]:%H:%M:%S}", s[0], len(ends), first
    if "waiting for memory" in last:
        return f"waiting for memory: {last[9:110]}", None, len(ends), first
    if "waiting (day rule)" in last:
        return "yielding to another session's MEM run", None, len(ends), first
    return "between processes", None, len(ends), first

def ended():
    try:
        S = open(START_LOG).read()
    except FileNotFoundError:
        return False
    tail = S[S.rfind("armed (pid"):]
    return f"runner 61 ({TAG}) ended rc=" in tail or "nothing started" in tail

sent = None
while True:
    now = datetime.now().replace(microsecond=0)
    st, run_since, nend, first = state(now)
    df, dr = done_keys("fwd"), done_keys("rev")
    rem = {"fwd": [k for k in KEYS if k not in df], "rev": [k for k in KEYS[::-1] if k not in dr]}
    left = {c: sim(rem["fwd"], r) + sim(rem["rev"], r) for c, r in RATES.items()}
    anchor = run_since or (max(now, GSTART) if st == "not started" else now)
    fin = {c: anchor + timedelta(seconds=s) for c, s in left.items()}
    pace = None
    if nend and first:
        last_end = now if run_since is None else run_since
        sim_done = sim([k for k in KEYS if k in df], RATES["central"]) + sim([k for k in KEYS[::-1] if k in dr], RATES["central"])
        if sim_done > 0:
            pace = (last_end - first).total_seconds() / sim_done
    meas = anchor + timedelta(seconds=left["central"] * pace) if pace else None
    base = first or max(now, GSTART)       # the line is the first LINE at or after the run's start
    line_dt = datetime.combine(base.date(), datetime.strptime(LINE, "%H:%M").time())
    if line_dt < base:
        line_dt += timedelta(days=1)
    basis = meas or fin["central"]
    passes = "yes" if basis > line_dt else "no"
    complete = len(df) == 381 and len(dr) == 381
    pk = peak_so_far()
    budget = max(24, math.ceil(pk + 2))     # longer prompts can raise the peak (amendment 12d's report line; 01:5x)
    may = "no"
    why = "the gate is not in a running process"
    if complete or st.startswith("stopped"):
        why = "the gate has ended"
    elif passes == "yes":
        why = f"the projected finish passes {LINE}"
    elif run_since and (now - run_since).total_seconds() >= 300:
        may, why = "yes", f"process past its load ({int((now - run_since).total_seconds() // 60)} min in); budget the gate at {budget} GB"
    elif run_since:
        why = "the running process is in its first 5 min (load)"
    txt = (f"updated: {now:%Y-%m-%d %H:%M:%S}\n"
           f"state: {'done (both orders complete)' if complete else st}\n"
           f"keys done: fwd {len(df)}/381, rev {len(dr)}/381; processes ended {nend}\n"
           f"finish central: {fin['central']:%H:%M}\n"
           f"finish slow: {fin['slow']:%H:%M}\n"
           f"finish measured: {f'{meas:%H:%M} (pace x{pace:.2f} of the central sim)' if meas else 'n/a (no process ended yet)'}\n"
           f"passes {LINE}: {passes} (on the {'measured pace' if meas else 'central sim'})\n"
           f"neighbour may start: {may} ({why})\n"
           f"gate lifetime peak so far: {pk:.3f} GB ({'over' if pk >= 24.0 else 'under'} the 24.0 GB line); budget the gate at {budget} GB\n"
           f"rule: neighbours stop when 'passes {LINE}' says yes; a file older than 30 min counts as 'no'.\n")
    tmp = OUT + ".tmp"
    open(tmp, "w").write(txt); os.replace(tmp, OUT)
    if passes != sent:
        print(f"{now:%m-%d %H:%M:%S} passes {LINE}: {passes}; {txt.splitlines()[1]}; measured "
              f"{f'{meas:%H:%M}' if meas else 'n/a'}, central {fin['central']:%H:%M}, slow {fin['slow']:%H:%M}", flush=True)
        if SEND and passes == "yes":
            msg = (f"EvalSuite {now:%H:%M} (automatic) -> NewIdeas: the w24 gate's projected finish passes {LINE}: "
                   f"{'measured' if meas else 'central'} {basis:%H:%M} (central {fin['central']:%H:%M}, slow "
                   f"{fin['slow']:%H:%M}; fwd {len(df)}/381, rev {len(dr)}/381). Neighbours stop (your 23:13 rule). "
                   f"File: ~/EvalSuite/results/w24_gate_eta.txt")
            subprocess.run(["peer", "send", "21406", "-"], input=msg, text=True, timeout=60)
        sent = passes
    if ONCE or complete or (ended() and not os.path.exists(f"{ES}/results/q61.pid")):
        break
    time.sleep(60)
