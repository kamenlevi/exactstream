"""G1 key test (BigMoE, Mon 16 Nov 2026; EvalSuite PREREG_lossygate amendments 5 / 5a: G1 = C6 + int8 dense, the arming
rule as for 1c-tiers: each env decodes one scored HumanEval key and one IFEval key end to end on the REAL loop before
EvalSuite arms; NewIdeas 12:3x: report the at-peak margins).

Envs (data/stack): lx45<snap> (the lossy-45 base, R57_TIERS and R57_INT8D unset) and g1x45<snap> (base + C6 = R57_TIERS=1c
at LAM 0.018 + R57_INT8D=1, both on the code sets), each built as EvalSuite's runner builds it (unsets, the env line,
R57_PRESET=45 R57_REPS=1 R57_WARM=1, R57_TAG, R57_KEYS; argv C 2048).
Keys: humaneval/HumanEval/45 (60 tokens in lx45cc1's run), humanevalx/HumanEval/23 (53), ifeval/2239 (11).
PASS per env: rc 0, the 3 keys decoded (tokens written), and
  - base: tokens == lx45cc1's rows (EvalSuite results/card_lx45cc1_t1code/decode.jsonl) on humanevalx + IFEval (the unset
    path); no lossy_check; pf.qom_opt > 0 on humaneval (a QOM set);
  - G1: lossy_check on humaneval + humanevalx with e3 + edrop > 0 (1c engaged) AND int8 > 0 (int8 dense engaged), absent
    on IFEval; IFEval tokens == the base's (unengaged, identical); pf.qom_opt == 0 on humaneval (1c replaces QOM).
  - amendment 5d (NewIdeas + EvalSuite, Mon 14:0x; snapshot g1b = g1a + Fit12's res1c): the variants of REQ, each with and
    without RES: g1x / g1rx (lambda 1.8e-2), g1cx / g1crx (COMQOM, lambda 1.6e-2), g1cdx / g1cdrx (COMQOM + MBPP docstrings),
    g1lx / g1lrx (lambda 0.8e-2). As G1 on every code key (1c + int8), plus their res1c counters > 0 summed over the code
    keys; the COMQOM envs also decode mbpp/115 ('#' comments) and mbpp/146 (a docstring only: tagged rows required on the
    DOC envs); a res1c env's pf.qom_opt is reported, not ruled (its optional uses go through QOM's bit 21).
Per env the at-peak evidence: the start gate's two NO-SWAP reads, the lifetime peak, the AT-PEAK MARGIN (the lower read
minus the peak), the swap delta and vm_stat's swapouts over the run (EvalSuite's gate = MEM + 4 GB; every peak <= gate - 0.5).
The run's own stops as rpl2_keytest.py / t1c_keytest.py (share lock "MEM 45 GB", swap guard +2 GB, HOLD / PAUSE / lid).
  .venv/bin/python scripts/g1_keytest.py --snap <name> [--gate 40 --start-by .. --hard .. --away-min 900]
  .venv/bin/python scripts/g1_keytest.py --snap <name> --dry      EvalSuite's head test per env (no model)
  .venv/bin/python scripts/g1_keytest.py --snap <name> --rescore  re-score the last run from its logs + records
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
SFX = os.environ.get("G1KT_SFX", "")                     # Tue 11-17 01:0x: a record suffix (the COMQOM probe re-run: "p"),
assert re.fullmatch(r"[a-z0-9_]*", SFX), SFX             # so a re-run never overwrites the first run's g1kt_* records / result
ES = os.path.expanduser("~/EvalSuite")
Q = os.path.expanduser("~/.local/share/quiet")
ENVS = ()                                                # set from --snap: (lx45<snap>, g1x45<snap>)
KEYS = ("humaneval/HumanEval/45", "humanevalx/HumanEval/23", "ifeval/2239")
REF = os.path.join(ES, "results/card_lx45cc1_t1code/decode.jsonl")     # lx45cc1's forward run (the old base)
UNSETS = ("EVALSUITE_BIGMOE_CODE", "EVALSUITE_RELAX_VARIANT", "EVALSUITE_STAGE_ENV", "R57_KEYS", "R57_TAG", "BIGMOE_TIER",
          "BIGMOE_TIER_GLU", "BIGMOE_TIER_CL", "BIGMOE_TIER_COMBINE", "BIGMOE_OOO", "BIGMOE_MAXK")   # runner 64's UNSETS
GATE_GB, SWAP_GB, WALL_S, PAUSE_S, MIN_ENV_S = 47, 2.0, 900, 900, 180
REQ = {"g1x": ("int8",), "g1rx": ("int8", "res_opt", "opt_post"), "g1cx": ("int8", "com_rows", "com_opt"),
       "g1crx": ("int8", "com_rows", "com_opt", "res_opt", "opt_post"), "g1cdx": ("int8", "com_rows", "com_opt"),
       "g1cdrx": ("int8", "com_rows", "com_opt", "res_opt", "opt_post"), "g1lx": ("int8",),
       "g1lrx": ("int8", "res_opt", "opt_post"),
       # NewIdeas 18:0x (EvalSuite's amendment): int8 dense DROPPED from G1 and every 5d branch: the same eight without
       # R57_INT8D; "noint8" = no int8 / int8_rows in lossy_check on any key (int8 absent), in place of "int8 > 0"
       "g1": ("noint8",), "g1r": ("noint8", "res_opt", "opt_post"), "g1c": ("noint8", "com_rows", "com_opt"),
       "g1cr": ("noint8", "com_rows", "com_opt", "res_opt", "opt_post"), "g1cd": ("noint8", "com_rows", "com_opt"),
       "g1cdr": ("noint8", "com_rows", "com_opt", "res_opt", "opt_post"), "g1l": ("noint8",),
       "g1lr": ("noint8", "res_opt", "opt_post")}   # amendment 5d's variants (NewIdeas + EvalSuite, Mon 14:0x): int8 > 0 per
# code key; Fit12 res1c's lossy_check counters (com_rows / com_opt: COMQOM's tagged rows and their optional uses; res_opt:
# RES's optional MSB-3-tier uses; opt_post: optional uses the ref2 kernel saw) > 0 summed over the env's code keys
F12C = ("f12_checks", "com_rows", "com_opt", "res_opt", "opt_post", "opt_drop")   # res1c.prompt_counts() (Fit12, 13:5x)
XKO = tuple(k_ for k_ in os.environ.get("G1KT_XKEYS", "").split(",") if k_)   # Tue 11-17 (NewIdeas 00:5x): the COMQOM
# probe re-run's own extra keys (mbpp/303,mbpp/237: '#' comment lines early in g1-like decodes); unset = the keys below
XKEYS = {v_: (XKO or ("mbpp/115", "mbpp/146")) for v_ in ("g1cx", "g1crx", "g1cdx", "g1cdrx", "g1c", "g1cr", "g1cd", "g1cdr")}   # the COMQOM envs' extra keys: the
# scored keys' lx45cc1 answers have docstrings but no '#' comment; mbpp/115's has two '#' comment lines (from its line 2),
# mbpp/146's only a docstring (76 / 70 tokens). DOCV: the DOC envs (R57_COMQOM_DOC_SETS=mbpp) must tag rows on mbpp/146.
DOCV = ("g1cdx", "g1cdrx", "g1cd", "g1cdr")
XTRA = {"inx": "inert"}   # --extra envs, run after the variants (any data/stack env: its screen from its header). inx45cc1
# (NewIdeas 18:1x: night1117 arm 2's INERT env on cc1, new on the real loop): rc 0 + every key decoded + its INERT path
# ran on each code key (runs.json inert rows > 0); the kept counts and token equality vs lx45cc1 are reported only
CODE = ("humaneval", "humanevalx", "mbpp")


def prefix(n):
    return n[:n.index("45")] if "45" in n else n            # the env's prefix: lx (the base) or a REQ key


def keys_of(n):
    return KEYS + XKEYS.get(prefix(n), ())


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
    """(env, argv tail) of one env's run, as EvalSuite's runner builds it"""
    fenv = os.path.join(BM, "data/stack", f"{name}_env.txt")
    screen, C = screen_of(fenv)
    env = {k: v for k, v in os.environ.items() if k not in UNSETS}
    env.update(dict(env_pairs(fenv)))
    env.update(R57_PRESET="45", R57_REPS="1", R57_WARM="1", R57_TAG=f"g1kt_{name}{SFX}", R57_KEYS=",".join(keys_of(name)))
    return fenv, screen, env, [C, "2048"]


def noswap_gb():
    out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    ps = int(re.search(r"page size of (\d+)", out).group(1))
    pages = sum(int(m.group(2).rstrip(".")) for m in re.finditer(r"^(Pages free|Pages purgeable|File-backed pages):\s+(\d+\.?)", out, re.M))
    return pages * ps / 1e9


def swapouts():
    m = re.search(r"^Swapouts:\s+(\d+)", subprocess.run(["vm_stat"], capture_output=True, text=True).stdout, re.M)
    return int(m.group(1)) if m else None


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
    ap.add_argument("--snap", required=True)
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--stub", default=None)                 # a stub python file run in place of the screen (simulation)
    ap.add_argument("--nolock", action="store_true")        # simulation: no peer lock, no NO-SWAP gate
    ap.add_argument("--out", default=os.path.join(BM, "data/r57"))
    ap.add_argument("--gate", type=float, default=GATE_GB)  # NO-SWAP GB on two reads 30 s apart (orchestrator 06:53: 44)
    ap.add_argument("--start-by", default=None)             # "YYYY-mm-dd HH:MM": the first env's gate must open by then
    ap.add_argument("--hard", default=None)                 # "YYYY-mm-dd HH:MM": TERM the run and stop (clean) at this time
    ap.add_argument("--away-min", type=int, default=0)      # do not start if away_secs.sh < this (orchestrator: 900)
    ap.add_argument("--rescore", action="store_true")       # no run: re-score the last run's envs from its logs + records
    ap.add_argument("--variants", default="g1x")            # the variant envs' prefixes, after the base: g1x (G1), g1rx (G1 +
                                                             # RES), g1crx (G1 + COMQOM + RES); each must show its REQ counters
    ap.add_argument("--extra", default="")                 # extra envs after the variants (XTRA prefixes), e.g. inx45cc1
    a = ap.parse_args()
    global ENVS
    assert all(v_ in REQ for v_ in a.variants.split(",")), (a.variants, sorted(REQ))
    xt_ = tuple(x_ for x_ in a.extra.split(",") if x_)
    assert all(prefix(x_) in XTRA and os.path.exists(os.path.join(BM, "data/stack", f"{x_}_env.txt")) for x_ in xt_), xt_
    ENVS = (f"lx45{a.snap}",) + tuple(f"{v_}45{a.snap}" for v_ in a.variants.split(",")) + xt_
    ep = lambda s_: time.mktime(time.strptime(s_, "%Y-%m-%d %H:%M")) if s_ else None
    start_by, hard = ep(a.start_by), ep(a.hard)
    py = os.path.join(BM, ".venv/bin/python")
    if a.dry:                                                # EvalSuite's own head test per env
        ok = True
        for n in ENVS:
            fenv, _, env, _ = run_env(n)
            r = subprocess.run(["python3", os.path.join(ES, "tools/r57_headtest.py"), fenv, f"g1kt_{n}{SFX}"], cwd=ES,
                               capture_output=True, text=True)
            last = (r.stdout.strip().splitlines() or ["(no output)"])[-1]
            print(last)
            ok &= r.returncode == 0 and "PASS" in last
        return 0 if ok else 1
    os.makedirs(a.out, exist_ok=True)
    LOG = os.path.join(a.out, f"g1_keytest_{a.snap}{SFX}.log")
    RES = os.path.join(a.out, f"g1_keytest_{a.snap}{SFX}.json")

    def say(m):
        with open(LOG, "a") as f:
            f.write(f"{time.strftime('%m-%d %H:%M:%S')} {m}\n")

    res = dict(started=time.strftime("%Y-%m-%d %H:%M:%S"), keys=KEYS, xkeys={k_: v_ for k_, v_ in XKEYS.items()
                                                                              if any(prefix(n_) == k_ for n_ in ENVS)},
               ref=REF, envs={}, status="running")
    allk_ = set(KEYS).union(*XKEYS.values())
    refrows = {d_["key"]: d_["completion_ids"] for d_ in (json.loads(l_) for l_ in open(REF)) if d_["key"] in allk_}
    locked, child = False, [None]

    def score(n, rc, why, lp, swap_d, wall_s, reads=None, swo=None):
        """one env's verdict from its run log + the screen's runs / tokens records, stored in res["envs"][n] (the store
        was missing until Mon 16 Nov 11:3x: that run's final PASS was computed over an empty list; --rescore refills it)"""
        txt = open(lp).read()
        err = [l for l in txt.splitlines() if re.match(r"\w*(Error|Exception)\b", l)][-1:]
        pk_ = [float(x) for l_ in txt.splitlines() if "lifetime peak" in l_    # every lifetime-peak figure (warm-ups,
               for x in re.findall(r"([0-9]+\.[0-9]+) GB", l_.split("lifetime peak", 1)[1])]   # per key, the final line)
        tag_ = f"g1kt_{n}{SFX}"; R_, T_ = {}, {}
        try:
            R_ = {x_["key"]: x_ for x_ in json.load(open(os.path.join(a.out, f"{tag_}_runs.json")))}
            T_ = json.load(open(os.path.join(a.out, f"{tag_}_tokens.json")))
        except (OSError, ValueError):
            pass
        per, fails, tk_ = {}, [], {}
        vp_ = prefix(n)
        variant = vp_ in REQ
        extra = vp_ in XTRA                                   # an --extra env: its own checks only
        f12v = variant and any(c_ in F12C for c_ in REQ[vp_])   # a res1c env (RES and / or COMQOM)
        for k in keys_of(n):
            r_ = R_.get(k)
            if r_ is None:
                fails.append(f"{k}: no record"); continue
            toks = (T_.get(k) or {}).get(r_["arm"], [[]])[0]
            tk_[k] = toks
            lc = r_.get("lossy_check"); qo = (r_.get("pf") or {}).get("qom_opt"); qs = (r_.get("pf") or {}).get("qom_skip")
            ck = r_.get("t1c_ck") or []
            per[k] = dict(arm=r_["arm"], n=r_["n"], tokens_eq_lx45cc1=toks == refrows.get(k), lossy_check=lc,
                          inert=r_.get("inert"), qom_opt=qo, qom_skip=qs, t1c_union_per_check_layer=round(sum(sum(c_[:3]) for c_ in ck) / max(sum(c_[3] for c_ in ck), 1), 2) if ck else None,
                          lifetime_peak_gb=round(max(r_["mem"][1], r_["mem"][-1]), 2) if isinstance(r_.get("mem"), list) and len(r_["mem"]) >= 7 else None)   # footprint peaks after prefill / at the key's end
            st_ = k.split("/")[0]
            if not toks:
                fails.append(f"{k}: no tokens")
            if st_ in ("humanevalx", "ifeval") and not variant and not extra and toks != refrows.get(k):
                fails.append(f"{k}: base tokens != lx45cc1's")
            if not variant and not extra and lc is not None:
                fails.append(f"{k}: lossy_check on the base")
            if not variant and not extra and st_ == "humaneval" and not (qo or 0) > 0:
                fails.append(f"{k}: base pf.qom_opt {qo} (want > 0)")
            if extra and XTRA[vp_] == "inert" and st_ in CODE and not ((r_.get("inert") or {}).get("rows", 0) > 0):
                fails.append(f"{k}: the INERT path did not run (inert {r_.get('inert')})")
            if variant and st_ in CODE and not (lc and lc.get("e3", 0) + lc.get("edrop", 0) + lc.get("lmsl_rows", 0) > 0):
                fails.append(f"{k}: 1c not engaged ({lc})")
            if variant and "int8" in REQ[vp_] and st_ in CODE and not (lc and lc.get("int8", 0) > 0):
                fails.append(f"{k}: int8 dense not engaged ({lc})")
            if variant and "noint8" in REQ[vp_] and lc and (lc.get("int8", 0) or lc.get("int8_rows", 0)):
                fails.append(f"{k}: int8 dense present on an int8-free env ({lc})")
            if variant and st_ == "ifeval" and lc is not None and any(lc.get(x_) for x_ in ("e3", "edrop", "lmsl_rows", "int8") + F12C):
                fails.append(f"{k}: G1 engaged on IFEval ({lc})")
            if variant and not f12v and st_ == "humaneval" and (qo or 0) != 0:   # a res1c env posts QOM's bit-21 optional
                fails.append(f"{k}: variant pf.qom_opt {qo} (want 0: 1c replaces QOM)")   # uses itself: reported, not ruled
            if variant and st_ == "ifeval":                  # never skipped: no base tokens to compare is a failure
                bt_ = res["envs"].get(ENVS[0], {}).get("tokens", {}).get(k)
                if bt_ is None or toks != bt_:
                    fails.append(f"{k}: IFEval tokens != the base's" if bt_ is not None else f"{k}: no base tokens to compare")
        for c_ in (REQ[vp_] if variant else ()):             # the res1c counters, summed over the env's code keys
            if c_ not in ("int8", "noint8") and not sum(((v_.get("lossy_check") or {}).get(c_, 0) for k_, v_ in per.items()
                                         if k_.split("/")[0] in CODE)) > 0:
                fails.append(f"{c_} not engaged on the code keys (sum 0)")
        if vp_ in DOCV and "mbpp/146" in keys_of(n) and not ((per.get("mbpp/146") or {}).get("lossy_check") or {}).get("com_rows", 0) > 0:
            fails.append("mbpp/146: no tagged rows on a DOC env (its answer's only comment is a docstring)")
        ok = rc == 0 and why is None and not fails and len(per) == len(keys_of(n))
        res["envs"][n] = dict(rc=rc, stopped=why, wall_s=wall_s, verdict="PASS" if ok else "FAIL", keys=per, fails=fails,
                              tokens=tk_, error=err[0] if err else None, log=os.path.relpath(lp, BM), swap_gb=swap_d,
                              peak_gb=max(pk_) if pk_ else None, start_noswap_gb=list(reads) if reads else None,
                              at_peak_margin_gb=round(min(reads) - max(pk_), 2) if reads and pk_ else None, swapouts=swo)
        return ok, per, fails, pk_, err

    def verdict_all():                                       # never PASS over a missing env
        return "PASS" if len(res["envs"]) == len(ENVS) and all(v["verdict"] == "PASS" for v in res["envs"].values()) else "FAIL"

    if a.rescore:                                            # the last run's per-env lines: rc, swap; its start / end times
        res = json.load(open(RES))
        res["envs"] = {}
        L_ = open(LOG).read().splitlines()
        for n in ENVS:
            ln = [l for l in L_ if re.search(rf" {n} rc -?\d+", l)][-1]
            st = [l for l in L_ if f" {n} start: " in l][-1]
            hm = lambda l_: time.mktime(time.strptime(time.strftime("%Y-") + l_[:14], "%Y-%m-%d %H:%M:%S"))
            score(n, int(re.search(r" rc (-?\d+)", ln).group(1)), None, os.path.join(a.out, f"g1kt_{n}{SFX}.log"),
                  float(re.search(r"swap ([+-][0-9.]+) GB", ln).group(1)), round(hm(ln) - hm(st)))
        res["status"] = verdict_all()
        res["rescored"] = (f"{time.strftime('%Y-%m-%d %H:%M:%S')}: envs refilled by --rescore from the run logs and the "
                           "g1kt_* runs / tokens records (the runner's per-env store was missing; its final status had "
                           "been computed over an empty list)")
        json.dump(res, open(RES, "w"), indent=1)
        print(json.dumps({"status": res["status"], "envs": {k_: {x_: v_[x_] for x_ in ("verdict", "peak_gb", "swap_gb", "fails")}
                                                              for k_, v_ in res["envs"].items()}}, indent=1))
        return 0 if res["status"] == "PASS" else 1

    def on_signal(sig, _f):
        raise Stop(f"signal {sig}")
    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)
    try:
        say(f"start: {', '.join(ENVS)}; keys {', '.join(KEYS)} (+ {', '.join(sorted(set().union(*XKEYS.values())))} on "
            f"{', '.join(sorted(XKEYS))}); gate NO-SWAP >= {a.gate:g} GB; swap guard +{SWAP_GB} GB; "
            f"start by {a.start_by}; hard stop {a.hard}; away >= {a.away_min} s")
        if a.away_min:
            aw_ = subprocess.run(["bash", os.path.join(Q, "away_secs.sh")], capture_output=True, text=True).stdout.split()
            aw_ = int(aw_[-1]) if aw_ and aw_[-1].isdigit() else 0
            say(f"away_secs {aw_}")
            if aw_ < a.away_min:
> (a quote from the project owner, kept private)
        if not a.nolock:                                     # retried while another session's quiet lock refuses it (up to
            t_l, last_l = time.time(), 0.0                   # 25 min: a 15-min quiet measurement + slack), as EvalSuite's
            while True:                                      # share_lock(); any other refusal stops the test
                r = subprocess.run(["peer", "lock", "share", "60m", "--cpu", "600", "--gpu", "100",
                                    f"MEM 45 GB BigMoE G1 key test, EvalSuite lossygate 5a / 5d / 5e ({a.snap}, {len(ENVS)} envs), correctness only (no timing)"],
                                   capture_output=True, text=True, timeout=600)
                out_ = (r.stdout + r.stderr).strip()
                if r.returncode == 0:
                    say(f"share lock rc 0: {out_[:200]}")
                    break
                if "quiet" not in out_.lower() or time.time() - t_l > 1500:
                    say(f"share lock rc {r.returncode}: {out_[:200]}")
                    raise Stop("no share lock" + (" (a quiet lock for 25 min)" if "quiet" in out_.lower() else ""))
                if time.time() - last_l >= 300:
                    say(f"share lock refused (a quiet lock runs): waiting; {out_[-160:]}"); last_l = time.time()
                if os.path.exists(os.path.join(Q, "HOLD")) or lid_closed():
                    raise Stop("HOLD or the lid closed while waiting for the share lock")
                if hard is not None and time.time() + MIN_ENV_S > hard:
                    raise Stop(f"no share lock before the hard stop {a.hard}")
                time.sleep(30)
            locked = True
        absent_ = os.environ.get("G1KT_ABSENT", "")         # a pgrep -f pattern (the wrapper's): EvalSuite's runner 65 / CHAT
        # process. Its memory loop does not re-check the shared-lock yield, so it could launch between two of these envs
        # (NewIdeas 17:2x order: this test before B2). Checked before each env's gate reads and again before its start.

        def absent_check(n_):
            if absent_ and subprocess.run(["pgrep", "-f", absent_], capture_output=True).returncode == 0:
                raise Stop(f"a process matching '{absent_}' is running before {n_} (EvalSuite's runner / CHAT process)")
        for n in ENVS:
            absent_check(n)
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
            rd_ = None                                       # the gate's two reads (the at-peak margin's start side)
            if not a.nolock:                                 # two reads 30 s apart, up to 10 min / the start-by / hard stop
                lim_ = time.time() + 600
                if start_by is not None and n == ENVS[0]:
                    lim_ = min(lim_, start_by)
                if hard is not None:
                    lim_ = min(lim_, hard - MIN_ENV_S)
                while True:
                    r1 = noswap_gb(); time.sleep(30); r2 = noswap_gb()
                    if r1 >= a.gate and r2 >= a.gate:
                        rd_ = (round(r1, 2), round(r2, 2))
                        say(f"  {n}: gate {a.gate:g}: reads {r1:.1f} / {r2:.1f} GB")
                        break
                    say(f"  {n}: NO-SWAP {r1:.1f} / {r2:.1f} GB < {a.gate:g}")
                    if time.time() + 30 > lim_:
                        raise Stop(f"NO-SWAP below {a.gate:g} GB until {time.strftime('%H:%M:%S', time.localtime(lim_))} before {n}")
            absent_check(n)
            fenv, screen, env, tail = run_env(n)
            argv = [py, a.stub or screen] + tail
            lp = os.path.join(a.out, f"g1kt_{n}{SFX}.log")
            sw0, t0, so0 = swap_gb(), time.time(), swapouts()
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
            so1 = swapouts()
            ok, per, fails, pk_, err = score(n, rc, why, lp, round(swap_gb() - sw0, 2), round(time.time() - t0), rd_,
                                             so1 - so0 if None not in (so0, so1) else None)
            if why and why.startswith(("hard stop", "HOLD")):
                json.dump(res, open(RES, "w"), indent=1)
                raise Stop(f"{why} during {n}")
            say(f"{n} rc {rc}{' (' + why + ')' if why else ''}: {'PASS' if ok else 'FAIL'}; peak {max(pk_) if pk_ else '?'} GB; "
                f"swap {swap_gb() - sw0:+.2f} GB; start NO-SWAP {rd_}; at-peak margin "
                f"{res['envs'][n]['at_peak_margin_gb']} GB; swapouts {res['envs'][n]['swapouts']}; "
                + "; ".join(f"{k} n {v['n']} eq_cc1 {v['tokens_eq_lx45cc1']} lc {v['lossy_check']} qom_opt {v['qom_opt']}" for k, v in per.items())
                + (f"; FAILS {fails}" if fails else "")
                + (f"; {err[0][:160]}" if err else ""))
            json.dump(res, open(RES, "w"), indent=1)
        res["status"] = verdict_all()
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
