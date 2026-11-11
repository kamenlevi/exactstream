"""evalsuite: does a 125B variant behave like the 4-bit reference?

  evalsuite run --variant plan4433333333 [--decode] [--no-gpu]
      runs whatever is missing (reference outputs, reference cache, both anchors, the variant), each GPU stage under
      `peer lock quiet --queue` with a memory gate, then prints the pass/fail table against both anchors
  evalsuite report --variant V          table only (CPU)
  evalsuite gpu <stage> [variants...]   one GPU stage (stages.py) under the lock
  evalsuite smoke [variants...]         a 1-minute GPU sanity run

Anchors: floor = the same weights through a different summation order (two runs of the reference); ar4 = an accepted
build (Intel's AutoRound int4 routed experts). Margins are fixed in evalsuite/margins.json before any variant runs."""
from __future__ import annotations
import argparse, json, os, re, subprocess, sys, threading, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GPU_PY = os.path.expanduser("~/BigMoE/.venv/bin/python")
RESULTS = os.path.join(ROOT, "results")
CACHE = os.path.join(ROOT, "cache", "ref")
LOG = os.path.join(ROOT, "results", "gpu.log")
PAUSE = os.environ.get("EVALSUITE_PAUSE_FILE") or os.path.expanduser("~/.local/share/quiet/PAUSE")   # env: tests only


def say(*a):
    line = time.strftime("%H:%M:%S ") + " ".join(str(x) for x in a)
    print(line, flush=True)
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def lock_file():
    pid = os.environ.get("CLAUDE_PID")
    return os.path.expanduser(f"~/.claude-peers/locks/{pid}.json") if pid else None


def keepalive():
    """stamp active_at in our quiet / shared lock (peer counts a lock as active from max(taken, active_at)); False = gone"""
    p = lock_file()
    if not p:
        return True
    try:
        d = json.load(open(p))
    except Exception:
        return False
    if d.get("mode") not in ("quiet", "share") or d.get("until", 0) <= time.time():
        return False
    d["active_at"] = time.time()
    t = f"{p}.tmp{os.getpid()}"
    json.dump(d, open(t, "w"))
    os.replace(t, p)
    return True


def free_gb():
    """GB a load can take without swapping: NO-SWAP = free + speculative + purgeable + file-backed pages (vm_stat;
    orchestrator 10-30 23:5x, RULES.md). memory_pressure's free % misses wired Metal memory, and vm_stat's
    'inactive' holds anonymous pages that only swap or compression can free"""
    r = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    ps, s = 16384, 0
    for l in r.splitlines():
        if "page size of" in l:
            ps = int(l.split("page size of")[1].split()[0])
        elif l.startswith(("Pages free:", "Pages speculative:", "Pages purgeable:", "File-backed pages:")):
            s += int(l.split(":")[1].strip().rstrip("."))
    return s * ps / 1e9


def big_others():
    """other processes with > 8 GB resident (logged only: the quiet lock already keeps a second model out, and on
    2026-10-20 03:58 this check idled 6 min on our own 9 GB AutoRound converter while 58 GB were free)"""
    r = subprocess.run(["ps", "-axo", "rss=,pid=,command="], capture_output=True, text=True).stdout
    me = os.getpid()
    return [" ".join(l.split()[:3])[:80] for l in r.splitlines()
            if l.split() and int(l.split()[0]) > 8_000_000 and int(l.split()[1]) != me]


DEADLINE = float(os.environ.get("EVALSUITE_DEADLINE", "0") or 0)   # epoch s: no stage starts within 10 min of it
FRUGAL_GB = {"w2f_r10": 14, "w2f_r05": 11}       # Breakthrough's measured peaks (13.2 / 9.9 GB), rounded up
# Breakthrough's small-RAM configs (bt_b12.load_b12) as they ship, with MTP: footprints ~13 / ~10 GB, rounded up
B12_GB = {"b12": 14, "b12c192": 32, "b8": 11, "b8fast": 11, "b12l": 14, "b12ll": 14, "b12llb16": 14, "b12llb18": 14, "b12llb20": 14,
          "b12llugb": 14, "b12llugbx": 14, "b12llu2": 14, "b12llk8": 14, "b12llka80": 14, "b12llk10": 14, "b12llka100": 14,
          "b12llk8u2": 14, "b12llk10u2": 14, "b12llfoff": 14, "b12llfon": 14, "b12llox": 14, "b12llfx": 14, "b12llff": 14,
          "b12llfx2": 14, "b12llff2": 14, "b12llff3": 14, "b12llff4": 14, "b12llff4p": 14,
          "b12llff4t0c": 14, "b12llff4t0h": 14}
B12_ENV = {"b12": {"B12_NORM": "mass"}, "b8": {"B12_LATE3": "2,4", "B12_NORM": "mass"},   # mass: each used expert keeps
           "b8fast": {"B12_W": "3", "B12_NORM": "mass"},   # its exact top-10 weight (Breakthrough + NewIdeas, 10-22 07:47)
           "b12l": {"B12_LATE3": "2,4", "B12_NORM": "mass"},     # B12 + late cold ranks 2-3 (Breakthrough 10-23 02:42)
           "b12ll": {"B12_LATE3": "2,10", "B12_NORM": "mass",    # B12 + all cold ranks 2-9 late; PF_GROUP 64 + ASYNC_EVERY:
                     "BIGMOE_PF_GROUP": "64", "BIGMOE_ASYNC_EVERY": "1"}}   # its shipping env (Fit12 10-23: same shas, 11.28 GB)
B12_ENV["b12c192"] = {"B12_CM": "192", "B12_NORM": "mass",        # D1 at C_m 192 (C_l = bt_b12's default 77); B12LL's
                       "BIGMOE_PF_GROUP": "64", "BIGMOE_ASYNC_EVERY": "1"}   # exact memory env
for _b in ("b12llb16", "b12llb18", "b12llb20", "b12llugb", "b12llugbx", "b12llu2",   # Fit12's hook: B12LL's env
           "b12llk8", "b12llka80", "b12llk10", "b12llka100", "b12llk8u2", "b12llk10u2"):
    B12_ENV[_b] = dict(B12_ENV["b12ll"])
FOLD_SHIP = {"B12_TRIMS": "1", "B12_PFR": "1", "B12_ALLWAIT": "0", "B12_CACHE_MB": "64"}   # bt_b12's shipped switches,
B12_ENV["b12llfoff"] = dict(B12_ENV["b12ll"], **FOLD_SHIP, B12_ENGINE="v3l", B12_FOLD="0")  # pinned (= today's defaults;
B12_ENV["b12llfon"] = dict(B12_ENV["b12ll"], **FOLD_SHIP, B12_ENGINE="v3m", B12_FOLD="1",   # cache cap = fold_check's env)
                           B12_FOLD_CM="2700:64,8600:60,16500:56,32800:47", B12_FOLD_ROWS="2048")   # 0228898's defaults
# the fold confirmation's arms (R-41, Breakthrough 10-28 17:3x; bt_b12 31b63af, pinned in model.BT_B12_SHA): its envs
B12_ENV["b12llox"] = dict(B12_ENV["b12ll"], **FOLD_SHIP, B12_ENGINE="v3l", B12_FOLD="0",     # old shipped
                          B12_PFR_GRID="balanced")
B12_ENV["b12llfx"] = dict(B12_ENV["b12ll"], **FOLD_SHIP, B12_ENGINE="v3m", B12_FOLD="1",     # fold-exact (the new
                          B12_PFR_GRID="keep")                                               # defaults)
B12_ENV["b12llff"] = dict(B12_ENV["b12ll"], **FOLD_SHIP, B12_ENGINE="v3m", B12_FOLD="1",     # fold-fast (non-exact,
                          B12_FOLD_AW="1", B12_FOLD_CM="1", B12_PFR_GRID="balanced")         # Kamen's call) = old fold-on
# R-41D (docs/PREREG_r41d_2026-10-29.md): the fold arms + R-45's fix (B12_FRS_NOCYCLE frees each replaced FR-Spec
# draft head at once) on Breakthrough's swapped bt_b12 (model.BT_B12_SHA); b12llff2 = FFX: C_m capped at 60
B12_ENV["b12llfx2"] = dict(B12_ENV["b12llfx"], B12_FRS_NOCYCLE="1")
B12_ENV["b12llff2"] = dict(B12_ENV["b12llff"], B12_FOLD_CM="2700:60,8600:60,16500:56,32800:47", B12_FRS_NOCYCLE="1")
# b12llff3 (docs/PREREG_b12llff3_2026-10-31.md, decision 5): b12llff2 on the live bt_b12 (v3o, M-01 by default) +
# B12_PFR_MIN 512 (pools released while prompts over 512 tokens are read)
B12_ENV["b12llff3"] = dict(B12_ENV["b12llff2"], B12_ENGINE="v3o", B12_PFR_MIN="512")
# PMIN12 (docs/PREREG_pmin12_2026-11-01.md): fold-fast as b12llff3 on the live bt_b12 8db71cc537 (Breakthrough 3676db3:
# default engine v3p with R-53 part a off, exact vs v3o; opt-in B12_PMIN = spec125.generate's p_min when its caller passes
# none): the control, and + p_min 0.5
B12_ENV["b12llff4"] = dict(B12_ENV["b12llff3"], B12_ENGINE="v3p", F12_R53A="0")
B12_ENV["b12llff4p"] = dict(B12_ENV["b12llff4"], B12_PMIN="0.5")
B12_ENV["b12llff4t0c"] = B12_ENV["b12llff4t0h"] = {'MTPLX_PREFILL_CHUNK_SIZE': '512', 'BIGMOE_TIER_PF_GROUP': '64', 'BIGMOE_RESET_GEN': '0', 'B12_LATE3': '2,10', 'B12_NORM': 'mass', 'BIGMOE_PF_GROUP': '64', 'BIGMOE_ASYNC_EVERY': '1', 'B12_TRIMS': '1', 'B12_PFR': '1', 'B12_ALLWAIT': '0', 'B12_CACHE_MB': '64', 'BIGMOE_LOOP_GUARD': '0', 'MTPLX_NGRAM_PREWARM': 'off', 'MTPLX_COMPILED_VERIFY': '0', 'MTPLX_QWEN4EXP_COMPILE': '0', 'MTPLX_COMPILE_AR_FORWARD': '0', 'MTPLX_COMPILED_TARGET_PREFIX': '0', 'BIGMOE_SPARE': '0', 'BIGMOE_PF_NOPUBLISH': '1'}   # T0: Fit12's gate env (model.T0_ENV) less EVALSUITE_SHIP (set beside)
T0_POP_BASES = ("b12llff4t0c", "b12llff4t0h")   # their env drops T0_POP (model.T0_POP): the live file's defaults
T0_POP = ("B12_ENGINE", "B12_FOLD", "B12_PFR_GRID", "B12_CM", "B12_CL", "B12_FOLD_CM", "B12_FOLD_AW", "B12_W", "B12_PMIN")
SHIP_AR = {"MTPLX_PREFILL_CHUNK_SIZE": "512", "BIGMOE_TIER_PF_GROUP": "64",    # the hook's run env (04:43)
           "B12_ENGINE": "v3f", "B12_TRIMS": "0",  # the verdicts' engine (bt_b12 defaults to v3g + trims from a50d103)
           "BIGMOE_RESET_GEN": "0"}               # spec125 must not import ~/BigMoE's bigmoe (its bigmoe_ext clashes)
SHIP_MTP = {"BIGMOE_MTP_STREAM": "1", "BIGMOE_MTP_C": "64", "BIGMOE_MAXK": "80", "BIGMOE_FRSPEC": "1",   # bt_b12's MTP
            "MTPLX_FAMILY_CAPTURE_COMMIT": "1", "MTPLX_DROP_EVENTS": "1", "MTPLX_LAZY_MTP_HISTORY_APPEND": "1",  # path
            "MTPLX_DEFER_REPAIR_EVAL": "1"}
B12_MTP = {"b12": True, "b8": False, "b8fast": True, "b12c192": True, "b12l": True, "b12ll": True,   # = model.B12_MTP
           "b12llb16": True, "b12llb18": True, "b12llb20": True, "b12llugb": True, "b12llugbx": True, "b12llu2": True,
           "b12llk8": True, "b12llka80": True, "b12llk10": True, "b12llka100": True, "b12llk8u2": True, "b12llk10u2": True,
           "b12llfoff": True, "b12llfon": True, "b12llox": True, "b12llfx": True, "b12llff": True,
           "b12llfx2": True, "b12llff2": True, "b12llff3": True, "b12llff4": True, "b12llff4p": True,
           "b12llff4t0c": False, "b12llff4t0h": False}
B12_RE = re.compile(r"(b12llff4t0c|b12llff4t0h|b12llff4p|b12llff4|b12llff3|b12llfx2|b12llff2|b12c192|b12llfoff|b12llfon|b12llox|b12llfx|b12llff|b12llugbx|b12llugb|b12llu2|b12llk10u2|b12llk8u2|b12llka100|b12llka80|b12llk10|b12llk8|b12llb16|b12llb18|b12llb20|b12ll|b12l|b12|b8fast|b8)(g?)((?:\.r\d+)?)")   # = model


F24_RE = re.compile(r"f24(b16|b18|b20|nat)((?:\.r\d+)?)")   # = model.F24_RE
FAST24_COMMIT = "36c9cc5"                        # BigMoE's pin for scripts/fast24.py (engine = af0cdcb)


def ship_of(name):
    m = B12_RE.fullmatch(name)
    return (m.group(1), bool(m.group(2))) if m else None              # spec125 must not import ~/BigMoE's bigmoe (its bigmoe_ext clashes)


def swap_gb():
    """swap in use, GB (sysctl vm.swapusage)"""
    r = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout
    m = re.search(r"used = ([\d.]+)M", r)
    return float(m.group(1)) / 1024 if m else 0.0


def stop_tree(p):
    """stop a stage started with start_new_session=True: its whole process group (RULES, Tue 27 Oct 20:59), TERM then KILL"""
    import signal
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(p.pid, sig)
        except ProcessLookupError:
            return
        for _ in range(20):
            if p.poll() is not None:
                break
            time.sleep(0.5)


def held_until():
    """end of the quiet lock this session already holds (0 if none)"""
    p = lock_file()
    try:
        d = json.load(open(p))
    except Exception:
        return 0
    return d.get("until", 0) if d.get("mode") == "quiet" and d.get("until", 0) > time.time() else 0


SHARE = False                                    # --share: bits-only stages (exactness, references) under `peer lock share`
#                                                  (Kamen 10-30 18:45, FAST INTEGRATION): refused while a quiet lock runs
#                                                  (retried); the stage yields at a row boundary when a quiet ticket waits
LOW = False                                      # --low: '[low]' in the lock's why text: the queue ranks it behind every
#                                                  research ticket (orchestrator 10-24 06:10); background runs only


def research_waits():
    """a ticket without [low] waits for the quiet lock (ours is not queued while our stage runs)"""
    r = subprocess.run(["peer", "ls"], capture_output=True, text=True).stdout
    return any("waiting for a quiet window" in l and "[low]" not in l for l in r.splitlines())


def wait_pause(why):
    """wait while the quiet window's PAUSE file exists (a heat pause or the day hold): no stage starts inside one"""
    t0 = time.time()
    while os.path.exists(PAUSE):
        if int(time.time() - t0) % 600 < 30:
            say(f"PAUSE: waiting ({why})")
        time.sleep(30)


def gpu(args, why, need_gb, timeout_s):
    """run `python -m evalsuite.stages args` under a queued quiet lock with a memory gate and keepalive. If this
    session already holds a quiet lock (a ticket taken with `peer lock quiet --queue`), the stage runs inside it: a
    second --queue request would go to the back of the line while our own lock sits idle (2026-10-20 08:43)."""
    if DEADLINE and DEADLINE - time.time() < 600:
        raise SystemExit(f"past the deadline: not starting {why}")
    hold = os.path.join(RESULTS, "HOLD")             # one stage prefix per line: a running runner skips those jobs
    if os.path.exists(hold) and any(l.strip() and why.startswith(l.strip()) for l in open(hold)):
        raise SystemExit(f"held by results/HOLD: not starting {why}")
    until = held_until()
    if until:
        timeout_s = min(timeout_s, until - time.time() - 60)
        say(f"inside our held quiet lock (until {time.strftime('%H:%M', time.localtime(until))}): {why}")
        keepalive()
    elif SHARE:
        lease, t_ask = int(min(180, timeout_s / 60 + 15)), time.time()
        while True:                                      # exit 3 = a quiet lock runs: wait, then ask again
            wait_pause(why)
            r = subprocess.run(["peer", "lock", "share", f"{lease}m", "--cpu", "300", "--gpu", "60", f"evalsuite: MEM {need_gb} GB: {why}"],
                               capture_output=True, text=True)
            if r.returncode == 0 and os.path.exists(PAUSE):  # a PAUSE that began while we waited (11-08 18:32): release
                subprocess.run(["peer", "unlock"], capture_output=True, text=True)   # and wait (no ticket is queued here)
                say(f"PAUSE right after the shared lock: released it ({why})")
                continue
            if r.returncode == 0:
                say(f"shared lock: {why} (lease {lease} min): {(r.stdout + r.stderr).strip().splitlines()[-1:]}")
                break
            if r.returncode != 3 or time.time() - t_ask > 14400 or (DEADLINE and DEADLINE - time.time() < 900):
                raise SystemExit(f"no shared lock for {why}: {(r.stdout + r.stderr).strip()[-300:]}")
            if int(time.time() - t_ask) % 600 < 60:
                say(f"shared lock refused (a quiet lock runs): waiting ({why})")
            time.sleep(60)
    else:
        lease = int(min(180, timeout_s / 60 + 15))       # the lock's lease must outlast the stage (keepalive checks it)
        say(f"queue: {why} (lease {lease} min)")
        r = subprocess.run(["peer", "lock", "quiet", "--queue", f"{lease}m", f"evalsuite: {'[low] ' if LOW else ''}{why}"], capture_output=True,
                           text=True)
        say("lock:", (r.stdout + r.stderr).strip().splitlines()[-1:] or "")
        if r.returncode:
            raise SystemExit(f"no quiet lock for {why}")
    try:
        if DEADLINE:
            if DEADLINE - time.time() < 600:
                raise SystemExit(f"past the deadline after the lock wait: not starting {why}")
            timeout_s = min(timeout_s, DEADLINE - time.time())
        for i in range(120):
            fg = free_gb()
            if fg >= need_gb + 4:
                break
            if i % 12 == 0:
                say(f"memory gate: {fg:.0f} GB free (need {need_gb + 4}), big processes {big_others()}")
            if i % 40 == 39:
                keepalive()
            time.sleep(5)
        else:
            raise SystemExit("memory gate: not enough free memory after 10 min")
        say(f"run: {' '.join(args)} ({free_gb():.0f} GB free)")
        env = dict(os.environ, PYTHONPATH=ROOT)
        yflag = os.path.join(RESULTS, f".yield_{os.getpid()}")
        env["EVALSUITE_YIELD_FLAG"] = yflag        # the stage stops at a row boundary when this file appears: cli makes
                                                   # it for [low] stages when research waits; touch it to pause any stage
        for k in ("EVALSUITE_FRUGAL", "EVALSUITE_SHIP", "EVALSUITE_FAST24", "B12_LATE3", "B12_W", "B12_NORM", "BIGMOE_LOOP_GUARD",
                  "B12_CM", "B12_CL",
                  "B12_ENGINE", "B12_TRIMS", "B12_PFR", "B12_ALLWAIT", "B12_CACHE_MB", "B12_FOLD", "B12_FOLD_CM", "B12_FOLD_ROWS",
                  "B12_PFR_GRID", "B12_FOLD_AW", "B12_FOLD_PFR", "B12_FOLD_MOV", "B12_FRS_NOCYCLE", "B12_PFR_MIN", "B12_PMIN", "F12_R53A",
                  "BIGMOE_PF_GROUP", "BIGMOE_ASYNC_EVERY", *SHIP_MTP):
            env.pop(k, None)
        for kv in filter(None, os.environ.get("EVALSUITE_STAGE_ENV", "").split(",")):   # a stage's own switches, set
            k, _, val = kv.partition("=")                # after the pops above (they strip e.g. BIGMOE_MAXK, a SHIP_MTP
            env[k] = val                                 # key): R-56b's gate arm, BIGMOE_MAXK=160,BIGMOE_EXT_BUILD=...
        if any(x in FRUGAL_GB for x in args):      # Breakthrough's runtime: no PF1 staging flags (evalsuite.env)
            env["EVALSUITE_FRUGAL"] = "1"
        if any(F24_RE.fullmatch(x) for x in args):  # BigMoE's fast mode: its own env (load_fast24), pinned commit
            head = subprocess.run(["git", "-C", os.path.expanduser("~/BigMoE"), "rev-parse", "--short", "HEAD"],
                                  capture_output=True, text=True).stdout.strip()
            if head != FAST24_COMMIT:
                raise SystemExit(f"~/BigMoE is at {head}, the fast24 verdict pins {FAST24_COMMIT}")
            env["EVALSUITE_FAST24"] = "1"
            sys.path.insert(0, os.path.expanduser("~/BigMoE/scripts"))   # fast24.py imports only os / sys at load
            from fast24 import FAST24_ENV                # from process start (selftest imports mtplx before load_fast24)
            env.update(FAST24_ENV)
            budget = next(F24_RE.fullmatch(x).group(1) for x in args if F24_RE.fullmatch(x))
            if budget == "nat":                          # = load_fast24(budget=""): the v3 natural arm
                env.pop("BIGMOE_BUDGET", None); env.pop("BIGMOE_BUDGET_JIT", None)
            else:
                env["BIGMOE_BUDGET"] = budget
        ship = [sh for sh in map(ship_of, args) if sh]
        if ship:                                   # one small-RAM config per process: its switches go in the env
            base, guard = ship[0]
            env.update(SHIP_AR, EVALSUITE_SHIP="1", **B12_ENV[base], **(SHIP_MTP if B12_MTP[base] else {}))
            if base in T0_POP_BASES:              # T0 (PREREG_t0): Fit12's gate leaves these to the live file's defaults
                for k in T0_POP:
                    env.pop(k, None)
            if not guard:                          # the verdict: guard off on both sides (it derails long math)
                env["BIGMOE_LOOP_GUARD"] = "0"
        with open(os.path.join(RESULTS, "gpu_stage.log"), "a") as out:
            p = subprocess.Popen([GPU_PY, "-u", "-m", "evalsuite.stages", *args], cwd=ROOT, env=env, stdout=out,
                                 stderr=subprocess.STDOUT, start_new_session=True)   # its own group: a stop ends the tree
            t0, asked, sw0, swapped = time.time(), False, swap_gb(), False
            while p.poll() is None:
                time.sleep(5)
                if (LOW or SHARE) and not asked and int(time.time() - t0) % 60 < 5 and research_waits():
                    open(yflag, "w").close()      # research first (orchestrator 10-24 08:20): yield at the next row
                    asked = True
                    say("a research ticket waits: asking the [low] stage to stop at its next row")
                if time.time() - t0 > timeout_s:
                    say("timeout: stopping the stage"); stop_tree(p); break
                if int(time.time() - t0) % 240 < 5 and not keepalive():
                    say("lost the quiet lock: stopping the stage"); stop_tree(p); break
                if int(time.time() - t0) % 60 < 5 and swap_gb() > sw0 + 2:   # RULES: swap eats the disk
                    say(f"swap {swap_gb():.1f} GB > {sw0:.1f} + 2 at the stage start: stopping the stage"); stop_tree(p)
                    swapped = True
                    break
            rc = p.wait()
        if os.path.exists(yflag):
            os.remove(yflag)
        say(f"done: {why} rc={rc} in {time.time() - t0:.0f}s")
        if swapped:
            raise SystemExit(f"swap guard stopped the stage ({why})")
        if rc == 75:
            raise SystemExit(75)                  # yielded: the runner queues it again (as [low] if it was)
        if rc:
            raise SystemExit(f"stage failed ({why}): see results/gpu_stage.log")
    finally:
        subprocess.run(["peer", "unlock"], capture_output=True)


def have(path):
    return os.path.exists(os.path.join(ROOT, path))


def ensure(variant, decode=False, no_gpu=False):
    steps = []
    if not have("cache/ref/tf_gen.jsonl") or not tf_gen_complete():
        steps.append((["gen-tf"], "reference outputs for the teacher-forced corpus", 34, 90 * 60))
    # one locked stage per variant (~45 min each at today's ~220 tok/s on short items), so other sessions' jobs can
    # interleave; the variant right after the reference, the floor after it, the accepted build when its files exist
    order = ["ref"] + ([variant] if variant not in ("ref", "floor", "ar4") else []) + ["floor"]
    for v in order:
        if not have(f"results/{v}/run.json"):
            steps.append((["prefill", v], f"prefill tests {v}", 29, 70 * 60))
    if not have("results/ar4/run.json") and ar4_linked():
        steps.append((["prefill", "ar4"], "prefill tests ar4 (accepted build)", 27, 70 * 60))
    if decode:
        for v in ("ref", variant, "floor") + (("ar4",) if ar4_linked() else ()):
            if not have(f"results/{v}/decode.json"):
                steps.append((["decode", v], f"decode sets {v}", 34, 80 * 60))
    for args, why, need, to in steps:
        if no_gpu:
            say(f"missing (not run, --no-gpu): {why}")
            continue
        gpu(args, why, need, to)


def tf_gen_complete():
    sys.path.insert(0, ROOT)
    from evalsuite import data as D
    want = {r["key"] for r in D.load("tf_prompts")}
    got = {json.loads(l)["key"] for l in open(os.path.join(CACHE, "tf_gen.jsonl"))}
    return want <= got


def ar4_linked():
    p = os.path.join(ROOT, "models", "ar4", "model.safetensors.index.json")
    return os.path.exists(p) and "evalsuite_anchor" in json.load(open(p)).get("metadata", {})


def main():
    ap = argparse.ArgumentParser(prog="evalsuite")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run"); r.add_argument("--variant", required=True); r.add_argument("--decode", action="store_true")
    r.add_argument("--no-gpu", action="store_true")
    p = sub.add_parser("report"); p.add_argument("--variant", required=True); p.add_argument("--json", action="store_true")
    p.add_argument("--decode", action="store_true")
    p.add_argument("--ref", default="ref", help="decode run to compare with (default: the 4-bit reference)")
    g = sub.add_parser("gpu"); g.add_argument("args", nargs="+"); g.add_argument("--minutes", type=int, default=45)
    g.add_argument("--C", type=int, default=0, help="expert slots per layer for the stage (numerics do not depend on it)")
    g.add_argument("--need", type=int, default=0, help="GB the stage needs (memory gate); default from C")
    g.add_argument("--low", action="store_true", help="background run: queued behind research tickets ('[low]')")
    g.add_argument("--share", action="store_true", help="bits-only stage under `peer lock share` (FAST INTEGRATION)")
    s = sub.add_parser("smoke"); s.add_argument("variants", nargs="*", default=["ref", "floor", "plan4433333333"])
    a, rest = ap.parse_known_args()
    if rest and a.cmd != "gpu":
        ap.error(f"unrecognized arguments: {' '.join(rest)}")
    sys.path.insert(0, ROOT)
    if a.cmd == "gpu":                                                 # options it does not know go to the stage
        global LOW, SHARE
        LOW, SHARE = a.low, a.share
        extra = rest + (["--C", str(a.C)] if a.C else [])
        frugal = [FRUGAL_GB[x] for x in a.args if x in FRUGAL_GB] + [B12_GB[sh[0]] for sh in map(ship_of, a.args) if sh] + \
            [26 for x in a.args if F24_RE.fullmatch(x)]
        need = a.need or (max(frugal) if frugal else int(a.C * 0.133) + 9 if a.C else 34)   # pools (2.77 MB x 48
        gpu(a.args + extra, " ".join(a.args + extra), need, a.minutes * 60)                # per slot) + ~9 GB rest
    elif a.cmd == "smoke":
        gpu(["prefill", *a.variants, "--smoke"], "smoke test", 27, 1800)
    elif a.cmd == "run":
        ensure(a.variant, a.decode, a.no_gpu)
        from evalsuite import report
        report.main([a.variant] + (["--decode"] if a.decode else []))
    elif a.cmd == "report":
        from evalsuite import report
        report.main([a.variant] + (["--json"] if a.json else []) + (["--decode"] if a.decode else [])
                    + (["--ref", a.ref] if a.ref != "ref" else []))


if __name__ == "__main__":
    main()
