"""killtest.py (Fit12, Mon 16 Nov 2026 14:1x; pre-registered in /Users/kamen/Fit12/PLAN_12GB_90_fit12.md "RES1C INSTALLER" +
NewIdeas 13:4x / 13:5x): the unattended KILL TEST of 1c-tiers + comment-QOM on the gate's 4 runaway keys.
Real loop (45 GB), Fit12's COPY of BigMoE's t1c1 screen with the res1c installer (/Users/kamen/Fit12/res1c/screen/
r57_screen_f12.py; outputs in /Users/kamen/Fit12/data/res1c/killtest), env = BigMoE's data/stack/t1cx45t1c1_env.txt
(read-only) built as EvalSuite's runner builds it (BigMoE scripts/t1c_keytest.py run_env), cap 2048.
Arms, one process each, in this order, keys mbpp/122, mbpp/13, HumanEval/147, mbpp/147 (the gate's process order):
  OFF     the switches unset (= 1c-tiers as gated; the CONTROL)
  COM     R57_COMQOM_SETS=humaneval,humanevalx,mbpp ('#' comment rows on QOM cf 0)
  COMDOC  COM + R57_COMQOM_DOC_SETS=mbpp (MBPP docstrings too)
A key FAILS when it hits the cap (>= 2040 tokens) or loops (a periodic tail, P <= 400, >= 3 repeats: runaway4's test).
LINES: OFF must FAIL on >= 3 of 4 keys, else the test is VOID (the harness differs). Each switch arm: DIES at >= 3 of 4
failing, BETWEEN at 2, SURVIVES at <= 1 (weak; only a kill is decisive; EvalSuite's RUNAWAY guard is the real test).
Gate (every 30 s; the bit tests are NOT a gate: if Mon's run did not pass, they run first in the slot):
- Breakthrough's COMQOM_END marker newer than the arming;
- not before Tue 07:00, latest start 08:30, else SKIP;
- NO-SWAP >= 47 GB on two reads 30 s apart;
- no other session's MEM >= 16 GB lock or quiet lock; no PAUSE / HOLD; the lid open.
Guards: swap +1 GB (stop everything), 15 min per arm, HOLD / lid (stop). A share lock is held while the arms run.
A marker data/res1c/killtest/KILL_DONE is written on EVERY end; the verdict goes to KILL_RESULT.txt, PLAN + TRIED, and
NewIdeas (peer send)."""
import glob
import json
import os
import re
import subprocess
import sys
import time

F = "/Users/kamen/Fit12"
O = os.environ.get("F12KT_O") or os.path.join(F, "data/res1c/killtest")    # F12KT_O: a test-only output dir
SCREEN = os.path.join(F, "res1c/screen/r57_screen_f12.py")
ENVF = "/Users/kamen/BigMoE/data/stack/t1cx45t1c1_env.txt"
MARK = "/Users/kamen/Breakthrough/data/m1/comqom_map/COMQOM_END"
CHECK = os.path.join(F, "data/res1c")
Q = os.path.expanduser("~/.local/share/quiet")
PY = "/Users/kamen/BigMoE/.venv/bin/python"
FIXED = ("mbpp/122", "mbpp/147")                     # COUNTED (NewIdeas 14:1x, 5d (b) + (c)): 1c-tiers' row a 3a runaway,
PRINT_ONLY = ("mbpp/13", "humaneval/HumanEval/147")  # the base's row 'eos' and not looped; these two: run + printed only
VROWS = "/Users/kamen/EvalSuite/results/card_t1cx45t1c1_t1code/decode.jsonl"   # 1c-tiers' tier-1 rows (EvalSuite)
BROWS = "/Users/kamen/EvalSuite/results/card_lx45t1c1_t1code/decode.jsonl"     # the base's tier-1 rows
PANEL = "/Users/kamen/EvalSuite/tools/lossy_panel.json"                        # the gate's key order (tier1)
KEYS, COUNTED, KEYNOTE = list(FIXED + PRINT_ONLY), list(FIXED), "fixed 2 (discovery not run yet)"
UNSETS = ("EVALSUITE_BIGMOE_CODE", "EVALSUITE_RELAX_VARIANT", "EVALSUITE_STAGE_ENV", "R57_KEYS", "R57_TAG", "BIGMOE_TIER",
          "BIGMOE_TIER_GLU", "BIGMOE_TIER_CL", "BIGMOE_TIER_COMBINE", "BIGMOE_OOO", "BIGMOE_MAXK",
          "R57_COMQOM_SETS", "R57_COMQOM_DOC_SETS", "R57_RES_SETS")
ARMS = (("OFF", {}), ("COM", {"R57_COMQOM_SETS": "humaneval,humanevalx,mbpp"}),
        ("COMDOC", {"R57_COMQOM_SETS": "humaneval,humanevalx,mbpp", "R57_COMQOM_DOC_SETS": "mbpp"}))
T_OPEN = time.mktime(time.strptime("2026-11-17 07:00", "%Y-%m-%d %H:%M"))
T_LAST = time.mktime(time.strptime("2026-11-17 08:30", "%Y-%m-%d %H:%M"))
if os.environ.get("F12KT_LAST"):                    # test only: a latest start in seconds from now
    T_LAST = time.time() + float(os.environ["F12KT_LAST"])
GATE_GB, SWAP_GB, ARM_S, CAP_HIT = 47, 1.0, 900, 2040
os.makedirs(O, exist_ok=True)
LOG = os.path.join(O, "killtest.log")
# amendment 3a's scoring pieces, read from EvalSuite (read-only): looped's exact source (md5-checked), EOS, the items'
# max_tokens, and the base's own finish on these keys (card_lx45t1c1_t0code)
import ast as _ast
import hashlib as _hl
_drs = open("/Users/kamen/EvalSuite/evalsuite/decode_report.py").read()
_fn = next(n for n in _ast.parse(_drs).body if isinstance(n, _ast.FunctionDef) and n.name == "looped")
_src = "".join(_drs.splitlines(keepends=True)[_fn.lineno - 1:_fn.end_lineno])
assert _hl.md5(_src.encode()).hexdigest() == "621d07e246d758cc1ada8fe4c8f5a580", "EvalSuite's looped() changed"
_ns = {}
exec(_src, _ns)
looped = _ns["looped"]
EOS = {248044, 248046}
MAXTOK, BASE_FIN = {}, {}
CODE_SETS = ("humaneval", "humanevalx", "mbpp")


def _rows(path):
    out = {}
    if os.path.exists(path):
        for l in open(path):
            d = json.loads(l)
            out[d["key"]] = d
    return out


def runaway3a(r):
    return r["finish"] == "length" or bool(looped(r["completion_ids"]))


def build_keys():
    """at the slot's start: the COUNTED keys = FIXED + further discordant code keys in 1c-tiers' full tier-1 rows
    (variant row a 3a runaway, base row 'eos' and not looped), up to 6 in all, gate order; PRINT_ONLY run too. Any
    failure falls back to FIXED, and KEYNOTE says so. Then the items' max_tokens and the base's finish per key."""
    global KEYS, COUNTED, KEYNOTE
    try:
        V, B = _rows(VROWS), _rows(BROWS)
        Vr = _rows(VROWS.replace("_t1code/", "_t1code_rev/"))     # either order (EvalSuite 5d, 7498f37): the reverse pair
        Br = _rows(BROWS.replace("_t1code/", "_t1code_rev/"))
        order = json.load(open(PANEL))["tier1"]

        def disc(k, v_, b_):
            return k in v_ and k in b_ and runaway3a(v_[k]) and b_[k]["finish"] == "eos" and not looped(b_[k]["completion_ids"])
        found = [k for k in order if k not in FIXED + PRINT_ONLY and k.split("/")[0] in CODE_SETS
                 and (disc(k, V, B) or disc(k, Vr, Br))]
        extra = found[:6 - len(FIXED)]
        fixed_ok = {k: (k in V and k in B and runaway3a(V[k]) and B[k]["finish"] == "eos") for k in FIXED}
        COUNTED = list(FIXED) + extra
        want = set(COUNTED) | set(PRINT_ONLY)
        KEYS = [k for k in order if k in want] + [k for k in COUNTED + list(PRINT_ONLY) if k not in order]
        KEYNOTE = (f"counted {len(COUNTED)}: fixed {list(FIXED)} (rule holds on the full rows: {fixed_ok}) + discovered "
                   f"{extra} ({len(found)} candidates) from {os.path.basename(os.path.dirname(VROWS))} ({len(V)} variant / "
                   f"{len(B)} base rows at {time.strftime('%m-%d %H:%M')}); printed only {list(PRINT_ONLY)}")
    except Exception as e:
        KEYS, COUNTED = list(FIXED[:1]) + list(PRINT_ONLY) + list(FIXED[1:]), list(FIXED)
        KEYNOTE = f"counted the fixed 2 {list(FIXED)} (discovery failed: {e!r}); printed only {list(PRINT_ONLY)}"
    MAXTOK.clear(); BASE_FIN.clear()
    for f_ in ("dec_mbpp.jsonl", "dec_humaneval.jsonl", "dec_humanevalx.jsonl"):
        for l_ in open(os.path.join("/Users/kamen/EvalSuite/data/items", f_)):
            d_ = json.loads(l_)
            if d_["key"] in KEYS:
                MAXTOK[d_["key"]] = int(d_["max_tokens"])
    assert set(MAXTOK) == set(KEYS), (MAXTOK, KEYS)
    for path in (os.path.join(os.path.dirname(BROWS), "decode.jsonl"),
                 "/Users/kamen/EvalSuite/results/card_lx45t1c1_t0code/decode.jsonl"):
        for k, d in _rows(path).items():
            if k in KEYS:
                BASE_FIN.setdefault(k, d["finish"] + (" looped" if looped(d["completion_ids"]) else ""))


def say(s):
    line = f"{time.strftime('%a %H:%M:%S')} {s}"
    print(line, flush=True)
    open(LOG, "a").write(line + "\n")


def sh(*a):
    return subprocess.run(list(a), capture_output=True, text=True).stdout


def noswap_gb():
    out = sh("vm_stat")
    ps = int(re.search(r"page size of (\d+)", out).group(1))
    tot = 0
    for k in ("Pages free", "Pages purgeable", "File-backed pages"):
        m = re.search(rf"{k}:\s+(\d+)", out)
        tot += int(m.group(1)) if m else 0
    return tot * ps / 1e9


def swap_gb():
    m = re.search(r"used = ([0-9.]+)M", sh("sysctl", "-n", "vm.swapusage"))
    return float(m.group(1)) / 1024 if m else 0.0


def lid_closed():
    return "\"AppleClamshellState\" = Yes" in sh("ioreg", "-r", "-k", "AppleClamshellState", "-d", "1")


def others_busy():
    """another session holds a quiet lock, or a MEM >= 16 GB lock"""
    for ln in sh("peer", "ls").splitlines():
        if not ln.startswith("- "):
            continue
        if "· benchmarking" in ln:
            return "a quiet lock"
        for m in re.finditer(r"MEM [~]?(\d+) GB", ln):
            if int(m.group(1)) >= 16:
                return f"a MEM {m.group(1)} GB lock"
    return None


_DONE, _LOCKED = [False], [False]


def done(status):
    open(os.path.join(O, "KILL_DONE"), "w").write(f"KILL END {time.strftime('%m-%d %H:%M:%S')}: {status}\n")
    _DONE[0] = True
    say(f"KILL_DONE: {status}")


def _at_exit():                                       # the marker on EVERY end (an error, a TERM), and the lock released
    if _LOCKED[0]:
        subprocess.run(["peer", "unlock"], capture_output=True)
        _LOCKED[0] = False
    if not _DONE[0]:
        done("TECHNICAL: the runner ended without a verdict (see killtest.log)")


import atexit
import signal
atexit.register(_at_exit)
signal.signal(signal.SIGTERM, lambda *a: sys.exit(143))


def env_pairs(path):
    line = [l for l in open(path).read().splitlines() if l and not l.startswith("#")][-1]
    out = []
    for t in line.split(","):
        if "=" in t or not out:
            out.append(t)
        else:
            out[-1] += "," + t
    assert all("=" in t for t in out), path
    return [t.split("=", 1) for t in out]


def loop(t, maxp=400, reps=3):
    n = len(t)
    for P in range(1, maxp + 1):
        if n < reps * P:
            break
        s = n - P
        while s - 1 >= 0 and t[s - 1] == t[s - 1 + P]:
            s -= 1
        if n - s >= reps * P:
            return (P, s)
    return None


def bit_tests_passed():
    d = os.path.join(CHECK, "CHECK_DONE")
    r = os.path.join(CHECK, "res1c_check.txt")
    return (os.path.exists(d) and open(d).read().startswith("rc 0") and os.path.exists(r)
            and "RES1C BIT TESTS: ALL PASS" in open(r).read())


try:                                                 # detach from the launching session's process group: a session
    os.setsid()                                      # teardown (usage stop) must not TERM this overnight runner
except OSError:
    pass
ARMED = time.time()
open(os.path.join(O, "ARMED"), "w").write(f"armed {time.strftime('%m-%d %H:%M:%S')} pid {os.getpid()}\n")
if os.path.exists(os.path.join(O, "KILL_DONE")):
    os.remove(os.path.join(O, "KILL_DONE"))
say(f"armed (pid {os.getpid()}): waits for {MARK} newer than now + Tue 07:00 + NO-SWAP >= {GATE_GB} (the bit tests run first in the slot unless Mon passed them); "
    f"latest start 08:30")
last = ""
while True:
    now = time.time()
    if now >= T_LAST:
        done(f"SKIP: no start by 08:30 ({last})"); sys.exit(5)
    why = None
    if not (os.path.exists(MARK) and os.path.getmtime(MARK) > ARMED):
        why = "COMQOM_END not newer than the arming"
    elif now < T_OPEN:
        why = "before 07:00"
    elif os.path.exists(os.path.join(Q, "HOLD")) or os.path.exists(os.path.join(Q, "PAUSE")):
        why = "HOLD / PAUSE"
    elif lid_closed():
        why = "lid closed"
    elif (ob := others_busy()):
        why = ob
    elif noswap_gb() < GATE_GB:
        why = f"NO-SWAP {noswap_gb():.0f} GB"
    else:
        time.sleep(30)
        if noswap_gb() >= GATE_GB and not others_busy():
            break
        why = "a second read failed"
    if why != last:
        say(f"waiting: {why}")
        last = why
    time.sleep(30)

say(f"gate open: NO-SWAP {noswap_gb():.0f} GB, swap {swap_gb():.1f} GB; marker {open(MARK).read().strip()}")
build_keys()
SCREEN_CAP = str(max(MAXTOK.values()))               # the 3a rule cuts at the item caps; decoding past them is unused
say(f"keys ({len(KEYS)}, gate order, screen cap {SCREEN_CAP}): {', '.join(KEYS)}; {KEYNOTE}")
subprocess.run(["peer", "lock", "share", "60m", "--cpu", "600", "--gpu", "100",
                "MEM 45 GB Fit12 KILL TEST: 1c + comment-QOM on 4 runaway keys, 3 arms, ~15-20 min"], capture_output=True)
_LOCKED[0] = True
try:
    if not bit_tests_passed():                       # Mon's memory-gated run did not get its slot: the bit tests first, here
        say("bit tests not passed yet: running res1c_check.py first (MEM ~4 GB, <= 10 min)")
        with open(os.path.join(CHECK, "res1c_check_run_kt.txt"), "w") as cf:
            try:
                rc_ = subprocess.run([PY, "-I", os.path.join(F, "res1c/res1c_check.py")], cwd=os.path.join(F, "res1c"),
                                     stdout=cf, stderr=subprocess.STDOUT, timeout=600).returncode
            except subprocess.TimeoutExpired:
                rc_ = 124
        open(os.path.join(CHECK, "CHECK_DONE"), "w").write(f"rc {rc_} {time.strftime('%H:%M:%S')} (in the kill slot)\n")
        if not bit_tests_passed():
            raise RuntimeError(f"bit tests did not pass (rc {rc_}; see data/res1c/res1c_check.txt): no arm run")
        say("bit tests: ALL PASS")
except Exception as e_:
    subprocess.run(["peer", "unlock"], capture_output=True); _LOCKED[0] = False
    done(f"SKIP: {e_}")
    subprocess.run(["peer", "send", "ojox-28", f"Fit12 KILL TEST {time.strftime('%H:%M')}: SKIP: {e_}"], capture_output=True)
    sys.exit(6)
sw0 = swap_gb()
base_env = {k: v for k, v in os.environ.items() if k not in UNSETS}
base_env.update(dict(env_pairs(ENVF)))
base_env.update(R57_PRESET="45", R57_REPS="1", R57_WARM="1", R57_KEYS=",".join(KEYS),
                PYTHONPATH="/Users/kamen/BigMoE")
res = {}
stopped = None
for arm, extra in ARMS:
    env = dict(base_env); env.update(extra); env["R57_TAG"] = f"f12kt_{arm}"
    lf = open(os.path.join(O, f"f12kt_{arm}.log"), "w")
    say(f"{arm} start ({', '.join(f'{k}={v}' for k, v in extra.items()) or 'switches unset'}); NO-SWAP {noswap_gb():.0f} GB")
    t0 = time.time()
    p = subprocess.Popen([PY, SCREEN, "256", SCREEN_CAP], env=env, cwd=os.path.dirname(SCREEN), stdout=lf,
                         stderr=subprocess.STDOUT, start_new_session=True)
    rc = None
    while rc is None:
        time.sleep(5)
        rc = p.poll()
        if rc is not None:
            break
        if swap_gb() - sw0 > SWAP_GB:
            stopped = "swap +1 GB"
        elif time.time() - t0 > ARM_S:
            stopped = f"{arm}: 15 min"
        elif os.path.exists(os.path.join(Q, "HOLD")) or lid_closed():
            stopped = "HOLD / lid"
        if stopped:
            p.terminate()
            try:
                p.wait(30)
            except subprocess.TimeoutExpired:
                p.kill()
            rc = -1
    lf.close()
    say(f"{arm} end rc {rc} ({time.time() - t0:.0f} s)")
    res[arm] = dict(rc=rc)
    if stopped:
        break
subprocess.run(["peer", "unlock"], capture_output=True); _LOCKED[0] = False

# ---- read-out
lines = []
for arm, _ in ARMS:
    if arm not in res:
        continue
    tf = os.path.join(O, f"f12kt_{arm}_tokens.json")
    rf = os.path.join(O, f"f12kt_{arm}_runs.json")
    tok = json.load(open(tf)) if os.path.exists(tf) else {}
    runs = json.load(open(rf)) if os.path.exists(rf) else []
    per = {}
    cutrun = {r["key"]: bool(r.get("cut")) for r in runs if r.get("key") in KEYS}
    for k in KEYS:
        v = tok.get(k)
        if v is None or cutrun.get(k):                 # no row, or cut at the time budget (EvalSuite drops such runs)
            per[k] = None; continue
        x = list(v.values())[0]
        t = list(x[0] if (isinstance(x, list) and x and isinstance(x[0], list)) else x)
        lp = loop(t)                                    # Fit12's periodic-tail reading (information only)
        cap = MAXTOK.get(k, 2048)                       # amendment 3a (decides): EvalSuite's r57rows cut at the item's
        ci = next((i + 1 for i, tt in enumerate(t) if tt in EOS), None)   # max_tokens or the first EOS, finish, and
        ids3 = t[:ci] if ci is not None and ci <= cap else t[:cap]       # evalsuite.decode_report.looped
        fin = "eos" if ids3 and ids3[-1] in EOS else "length"
        lo3 = bool(looped(ids3))
        per[k] = dict(n=len(t), n3=len(ids3), finish=fin, looped3=lo3, fail=(fin == "length") or lo3,
                      base_finish=BASE_FIN.get(k), old_fail=(len(t) >= CAP_HIT) or lp is not None)
    lc = [r.get("lossy_check") for r in runs if r.get("key") in KEYS]
    nf = sum(1 for k in COUNTED if per.get(k) and per[k]["fail"])          # COUNTED keys only (5d, 7498f37)
    nk = sum(1 for k in COUNTED if per.get(k))
    res[arm].update(per=per, nfail=nf, nkeys=nk, share=(nf / nk if nk else None), lossy_check=lc)
    fmt = lambda k: (f"{k.split('/', 1)[-1]} {per[k]['n3']} {per[k]['finish']}{' LOOPED' if per[k]['looped3'] else ''}"
                     f" (base {per[k]['base_finish']})" if per.get(k) else f"{k.split('/', 1)[-1]} n/a")
    lines.append(f"{arm}: rc {res[arm]['rc']}; COUNTED {nk} / {len(COUNTED)} decoded, 3a runaway (length or looped) {nf}"
                 f"{f' = {100 * nf / nk:.0f} %' if nk else ''}: " + "; ".join(fmt(k) for k in COUNTED)
                 + " | printed only: " + "; ".join(fmt(k) for k in PRINT_ONLY))
    if lc:
        tot = {}
        for d in lc:
            for kk in ("com_rows", "com_opt", "opt_post", "opt_drop", "e3", "edrop"):
                if d and kk in d:
                    tot[kk] = tot.get(kk, 0) + d[kk]
        lines.append(f"   {arm} lossy_check (summed over the run's keys): {tot}")
off = res.get("OFF")
if stopped or not off or off.get("nkeys", 0) < len(COUNTED):
    verdict = f"TECHNICAL: {stopped or 'the OFF arm did not decode every counted key'}"
elif off["share"] < 0.75:
    verdict = (f"VOID: the OFF control reproduced {off['nfail']} of {off['nkeys']} counted runaways "
               f"({100 * off['share']:.0f} % < 75 %: the harness differs from the gate's)")
else:
    parts = []
    for arm in ("COM", "COMDOC"):
        r = res.get(arm)
        if not r or r.get("nkeys", 0) < len(COUNTED):
            parts.append(f"{arm} INCOMPLETE"); continue
        v = "KILL" if r["share"] >= 0.75 else ("SURVIVE (weak)" if r["share"] <= 0.25 else "BETWEEN")
        parts.append(f"{arm} {v} ({r['nfail']} of {r['nkeys']} still runaway = {100 * r['share']:.0f} %)")
    verdict = f"OFF control VALID ({off['nfail']} of {off['nkeys']}); " + "; ".join(parts)
lines.insert(0, f"KEYS: {KEYNOTE}")
lines.append(f"VERDICT (EvalSuite amendment 5d 7498f37 lines: KILL >= 75 %, SURVIVE <= 25 %, VOID if OFF < 75 %; 3a's "
             f"rule at the item caps): {verdict}")
txt = "\n".join(lines)
open(os.path.join(O, "KILL_RESULT.txt"), "w").write(txt + "\n")
json.dump(res, open(os.path.join(O, "KILL_RESULT.json"), "w"), indent=1, default=str)
say(txt)
stamp = time.strftime("%a %d %b %H:%M")
open(os.path.join(F, "PLAN_12GB_90_fit12.md"), "a").write(
    f"\n## KILL TEST RESULT (auto-recorded {stamp} by /Users/kamen/Fit12/res1c/killtest.py; lines pre-registered in "
    f"\"RES1C INSTALLER\")\n" + "\n".join(f"- {l}" for l in lines) + "\n")
open("/Users/kamen/TRIED/fit12.md", "a").write(
    f"- WEEK98 PILLAR 2 KILL TEST 1c + comment-QOM [K, auto-recorded {stamp}] (4 runaway keys, real loop, OFF / COM / COMDOC) | "
    f"{verdict} | /Users/kamen/Fit12/data/res1c/killtest/KILL_RESULT.txt | 2026-11-17 | [fit12]\n")
subprocess.run(["peer", "send", "ojox-28", f"Fit12 KILL TEST {time.strftime('%H:%M')}: {verdict}. "
                f"Per key: /Users/kamen/Fit12/data/res1c/killtest/KILL_RESULT.txt"], capture_output=True)
done(f"run complete: {verdict}")
