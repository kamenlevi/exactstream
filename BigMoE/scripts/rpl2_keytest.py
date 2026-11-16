"""rpl2 real-loop key test (BigMoE, Mon 16 Nov 2026; EvalSuite 05:0x: tier 0a's re-run needs a fixed snapshot, a lossy env on
a QOM-capable path and, its new rule, one scored key per env on the real loop before arming).

For each env (x45rpl2, p2x45rpl2, m1bx45rpl2, lossycalx45rpl2): snapshot rpl2's screen on the full 125B, the env built as
EvalSuite's runner 64 builds it (tools/r57_headtest.py: its unsets, the env file's last line, R57_PRESET=45 R57_REPS=1
R57_WARM=1, R57_TAG, R57_KEYS; argv C 2048) + R57_REPLAY on EvalSuite's tier-0a rows (the variants read x45rpl1's margins;
the exact env writes its own), keys mbpp/118 (18 reference tokens: the last check holds 1 draft) and mbpp/128 (36: 3
drafts), each replayed to the reference's end, where rpl1's FAST row gate crashed (IndexError, relax_core.gate_rows).

PASS per env: rc 0 and a replay summary for both keys; the exact env also accepts every row (A 1.0000, prefill pick ok).
Machine rules: one share lock (MEM 45 GB in the reason), NO-SWAP >= 47 GB on two reads 30 s apart before each env, swap
guard +2 GB, a 15 min wall cap per env; HOLD / the lid closed stop it between envs; a PAUSE is waited out up to 15 min.
Writes data/r57/rpl2_keytest.json and data/r57/rpl2_keytest.log; the runs: data/r57/rpl2kt_<env>_* (the screen's OUT).
  .venv/bin/python scripts/rpl2_keytest.py            the real test (needs a slot from the orchestrator: ~43 GB peak)
  .venv/bin/python scripts/rpl2_keytest.py --dry      EvalSuite's head test per env with the same extra env (no model)
  .venv/bin/python scripts/rpl2_keytest.py --stub S --nolock --out D   the whole runner on a stub job (simulation)
"""
import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time

BM = os.path.expanduser("~/BigMoE")
ES = os.path.expanduser("~/EvalSuite")
Q = os.path.expanduser("~/.local/share/quiet")
ENVS = ("x45rpl2", "p2x45rpl2", "m1bx45rpl2", "lossycalx45rpl2")
KEYS = ("mbpp/118", "mbpp/128")
ROWS = os.path.join(ES, "results/card/replay_rows_x45rpl1.jsonl")      # = the p2 / m1b / lossycal rows files (same md5)
MARGINS = os.path.join(ES, "results/card/replay_margins_x45rpl1.json")
UNSETS = ("EVALSUITE_BIGMOE_CODE", "EVALSUITE_RELAX_VARIANT", "EVALSUITE_STAGE_ENV", "R57_KEYS", "R57_TAG", "BIGMOE_TIER",
          "BIGMOE_TIER_GLU", "BIGMOE_TIER_CL", "BIGMOE_TIER_COMBINE", "BIGMOE_OOO", "BIGMOE_MAXK")   # runner 64's UNSETS
GATE_GB, SWAP_GB, WALL_S, PAUSE_S, MIN_ENV_S = 47, 2.0, 900, 900, 180


def env_pairs(path):
    """runner 64's parse_env (as tools/r57_headtest.py): the last non-comment line, comma-split; a token without '='
    continues the previous value"""
    line = [l for l in open(path).read().splitlines() if l and not l.startswith("#")][-1]
    out = []
    for t in line.split(","):
        if "=" in t or not out:
            out.append(t)
        else:
            out[-1] += "," + t
    assert all("=" in t for t in out), path
    return [t.split("=", 1) for t in out]


def screen_of(path):
    comment = " ".join(l for l in open(path).read().splitlines() if l.startswith("#"))
    m = re.search(r"data/r57/snap/([A-Za-z0-9_.-]+/)*r57_screen[A-Za-z0-9_.-]*\.py", comment)
    c = re.search(r"(Sat|Sun|Mon|C) ([12][0-9][0-9])", comment)
    return os.path.join(BM, m.group(0)), (c.group(2) if c else "256")


def run_env(name):
    """(env, argv tail) of one env's run, as runner 64 builds it, + R57_REPLAY"""
    fenv = os.path.join(BM, "data/stack", f"{name}_env.txt")
    screen, C = screen_of(fenv)
    env = {k: v for k, v in os.environ.items() if k not in UNSETS}
    env.update(dict(env_pairs(fenv)))
    env.update(R57_PRESET="45", R57_REPS="1", R57_WARM="1", R57_TAG=f"rpl2kt_{name}", R57_KEYS=",".join(KEYS),
               R57_REPLAY=ROWS if name.startswith("x45") else f"{ROWS}:{MARGINS}")
    return fenv, screen, env, [C, "2048"]


def noswap_gb():
    out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    ps = int(re.search(r"page size of (\d+)", out).group(1))
    pages = sum(int(m.group(2).rstrip(".")) for m in re.finditer(r"^(Pages free|Pages purgeable|File-backed pages):\s+(\d+\.?)", out, re.M))
    return pages * ps / 1e9


def swap_gb():
    s = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout
    return float(re.search(r"used = ([0-9.]+)M", s).group(1)) / 1024


def lid_closed():
    s = subprocess.run(["ioreg", "-r", "-k", "AppleClamshellState", "-d", "1"], capture_output=True, text=True).stdout
    return '"AppleClamshellState" = Yes' in s


class Stop(Exception):
    pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--stub", default=None)                 # a stub python file run in place of the screen (simulation)
    ap.add_argument("--nolock", action="store_true")        # simulation: no peer lock, no NO-SWAP gate
    ap.add_argument("--out", default=os.path.join(BM, "data/r57"))
    ap.add_argument("--gate", type=float, default=GATE_GB)  # NO-SWAP GB on two reads 30 s apart (orchestrator 06:53: 44)
    ap.add_argument("--start-by", default=None)             # "YYYY-mm-dd HH:MM": the first env's gate must open by then
    ap.add_argument("--hard", default=None)                 # "YYYY-mm-dd HH:MM": TERM the run and stop (clean) at this time
    ap.add_argument("--away-min", type=int, default=0)      # do not start if away_secs.sh < this (orchestrator: 900)
    a = ap.parse_args()
    ep = lambda s_: time.mktime(time.strptime(s_, "%Y-%m-%d %H:%M")) if s_ else None
    start_by, hard = ep(a.start_by), ep(a.hard)
    py = os.path.join(BM, ".venv/bin/python")
    if a.dry:                                                # EvalSuite's own head test, + this test's R57_REPLAY
        ok = True
        for n in ENVS:
            fenv, _, env, _ = run_env(n)
            r = subprocess.run(["python3", os.path.join(ES, "tools/r57_headtest.py"), fenv, f"rpl2kt_{n}",
                                f"R57_REPLAY={env['R57_REPLAY']}"], cwd=ES, capture_output=True, text=True)
            last = (r.stdout.strip().splitlines() or ["(no output)"])[-1]
            print(last)
            ok &= r.returncode == 0 and "PASS" in last
        return 0 if ok else 1
    os.makedirs(a.out, exist_ok=True)
    LOG = os.path.join(a.out, "rpl2_keytest.log")
    RES = os.path.join(a.out, "rpl2_keytest.json")

    def say(m):
        with open(LOG, "a") as f:
            f.write(f"{time.strftime('%m-%d %H:%M:%S')} {m}\n")

    res = dict(started=time.strftime("%Y-%m-%d %H:%M:%S"), keys=KEYS, rows=ROWS, margins=MARGINS, envs={}, status="running")
    locked, child = False, [None]

    def on_signal(sig, _f):
        raise Stop(f"signal {sig}")
    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)
    try:
        say(f"start: {', '.join(ENVS)}; keys {', '.join(KEYS)}; gate NO-SWAP >= {a.gate:g} GB; swap guard +{SWAP_GB} GB; "
            f"start by {a.start_by}; hard stop {a.hard}; away >= {a.away_min} s")
        if a.away_min:
            aw_ = subprocess.run(["bash", os.path.join(Q, "away_secs.sh")], capture_output=True, text=True).stdout.split()
            aw_ = int(aw_[-1]) if aw_ and aw_[-1].isdigit() else 0
            say(f"away_secs {aw_}")
            if aw_ < a.away_min:
> (a quote from the project owner, kept private)
        if not a.nolock:
            r = subprocess.run(["peer", "lock", "share", "60m", "--cpu", "600", "--gpu", "100",
                                "BigMoE rpl2 key test: MEM 45 GB, EvalSuite tier 0a, correctness only (no timing)"],
                               capture_output=True, text=True, timeout=600)
            say(f"share lock rc {r.returncode}: {(r.stdout + r.stderr).strip()[:200]}")
            if r.returncode != 0:
                raise Stop("no share lock")
            locked = True
        for n in ENVS:
            if os.path.exists(os.path.join(Q, "HOLD")):
                raise Stop("HOLD")
            if lid_closed():
                raise Stop("lid closed")
            t_p = time.time()
            while os.path.exists(os.path.join(Q, "PAUSE")):
                if time.time() - t_p > PAUSE_S:
                    raise Stop("PAUSE for 15 min")
                if os.path.exists(os.path.join(Q, "HOLD")):
                    raise Stop("HOLD during a PAUSE")
                time.sleep(20)
            if hard is not None and time.time() + MIN_ENV_S > hard:
                raise Stop(f"no time for {n} before the hard stop {a.hard}")
            if not a.nolock:                                 # two reads 30 s apart, up to 10 min / the start-by / hard stop
                lim_ = time.time() + 600
                if start_by is not None and n == ENVS[0]:
                    lim_ = min(lim_, start_by)
                if hard is not None:
                    lim_ = min(lim_, hard - MIN_ENV_S)
                while True:
                    r1 = noswap_gb(); time.sleep(30); r2 = noswap_gb()
                    if r1 >= a.gate and r2 >= a.gate:
                        say(f"  {n}: gate {a.gate:g}: reads {r1:.1f} / {r2:.1f} GB")
                        break
                    say(f"  {n}: NO-SWAP {r1:.1f} / {r2:.1f} GB < {a.gate:g}")
                    if time.time() + 30 > lim_:
                        raise Stop(f"NO-SWAP below {a.gate:g} GB until {time.strftime('%H:%M:%S', time.localtime(lim_))} before {n}")
            fenv, screen, env, tail = run_env(n)
            argv = [py, a.stub or screen] + tail
            lp = os.path.join(a.out, f"rpl2kt_{n}.log")
            sw0, t0 = swap_gb(), time.time()
            say(f"{n} start: {os.path.relpath(screen, BM)} C {tail[0]} (NO-SWAP {noswap_gb():.0f} GB, swap {sw0:.1f} GB)")
            with open(lp, "w") as lf:
                child[0] = subprocess.Popen(argv, env=env, cwd=BM, stdout=lf, stderr=subprocess.STDOUT, start_new_session=True)
                why = None
                while child[0].poll() is None:
                    time.sleep(5)
                    if swap_gb() - sw0 > SWAP_GB:
                        why = f"swap +{swap_gb() - sw0:.1f} GB"
                    elif time.time() - t0 > WALL_S:
                        why = f"wall {WALL_S} s"
                    elif hard is not None and time.time() >= hard:
                        why = f"hard stop {a.hard}"
                    elif os.path.exists(os.path.join(Q, "HOLD")) or lid_closed():
                        why = "HOLD or the lid closed"
                    if why:
                        os.killpg(child[0].pid, signal.SIGTERM)
                        try:
                            child[0].wait(60)
                        except subprocess.TimeoutExpired:
                            os.killpg(child[0].pid, signal.SIGKILL); child[0].wait()
                        break
                rc = child[0].returncode
                child[0] = None
            txt = open(lp).read()
            reps = {m.group(1): dict(acc=int(m.group(2)), n=int(m.group(3)), A=float(m.group(4)))
                    for m in re.finditer(r"REPLAY (\S+): drafted rows accepted (\d+) / (\d+) \(A ([0-9.]+)\)", txt)}
            alls = {m.group(1): (int(m.group(2)), int(m.group(3)), m.group(4))
                    for m in re.finditer(r"REPLAY (\S+): .*?all rows (\d+) / (\d+); prefill pick ok (\w+)", txt)}
            err = [l for l in txt.splitlines() if re.match(r"\w*(Error|Exception)\b", l)][-1:]
            pk_ = [float(x) for l_ in txt.splitlines() if "lifetime peak" in l_    # every lifetime-peak figure (warm-ups,
                   for x in re.findall(r"([0-9]+\.[0-9]+) GB", l_.split("lifetime peak", 1)[1])]   # per key, the final line)
            ok = rc == 0 and why is None and all(k in reps for k in KEYS)
            if ok and n.startswith("x45"):                   # the exact env's self-check on these keys
                ok = all(reps[k]["acc"] == reps[k]["n"] and k in alls and alls[k][0] == alls[k][1] and alls[k][2] == "True"
                         for k in KEYS)
            res["envs"][n] = dict(rc=rc, stopped=why, wall_s=round(time.time() - t0), verdict="PASS" if ok else "FAIL",
                                  replay=reps, all_rows={k: list(v) for k, v in alls.items()}, error=err[0] if err else None,
                                  log=os.path.relpath(lp, BM), swap_gb=round(swap_gb() - sw0, 2),
                                  peak_gb=max(pk_) if pk_ else None)
            if why and why.startswith(("hard stop", "HOLD")):
                json.dump(res, open(RES, "w"), indent=1)
                raise Stop(f"{why} during {n}")
            say(f"{n} rc {rc}{' (' + why + ')' if why else ''}: {'PASS' if ok else 'FAIL'}; peak {max(pk_) if pk_ else '?'} GB; "
                f"swap {swap_gb() - sw0:+.2f} GB; "
                + "; ".join(f"{k} {v['acc']} / {v['n']} (A {v['A']:.4f})" for k, v in reps.items())
                + (f"; {err[0][:160]}" if err else ""))
            json.dump(res, open(RES, "w"), indent=1)
        res["status"] = "PASS" if len(res["envs"]) == len(ENVS) and all(v["verdict"] == "PASS" for v in res["envs"].values()) else "FAIL"   # never PASS over a missing env (t1c_keytest 11-16)
    except Stop as e:
        res["status"] = f"STOPPED: {e}"
        if child[0] is not None and child[0].poll() is None:
            os.killpg(child[0].pid, signal.SIGTERM)
            try:
                child[0].wait(60)
            except subprocess.TimeoutExpired:
                os.killpg(child[0].pid, signal.SIGKILL)
    finally:
        if locked:
            subprocess.run(["peer", "unlock"], capture_output=True)
            say("share lock released")
        res["ended"] = time.strftime("%Y-%m-%d %H:%M:%S")
        json.dump(res, open(RES, "w"), indent=1)
        say(f"END: {res['status']} ({', '.join(f'{k} {v['verdict']}' for k, v in res['envs'].items())})")
    return 0 if res["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
