"""FLAT-2's driver (zero session, Thu 5 Nov 2026): the passes, the read-out and the files. It gets the harness as a
namespace `ns` (the globals r57_screen.py's definitions were executed in), so flat2.py runs it on the engine and
dryrun_cpu.py runs the same code on a stub. run() reads `draft`, `CAP` and the hooks from that namespace, so those are
set there; nothing else of the harness is touched. See flat2.py for what FLAT-2 measures and its lines."""
import json
import os
import subprocess
import traceback

import flat2_core
import flat2_probe
import xpt                                               # STEPB_XPT=1: the Python thread's time per real check (Leap, Fri 23:5x)

XPT_ON = os.environ.get("STEPB_XPT") == "1"

TC = 1e-9                                                # the cut t: below any cf (top-1 of 16 candidates >= 1 / 16), so the
                                                         # block is never cut short of its first draft; maxd 1 then keeps one


def config(smoke=None):
    e, s = os.environ.get, (os.environ.get("FLAT2_SMOKE") == "1") if smoke is None else smoke
    here = os.path.dirname(os.path.abspath(__file__))
    return dict(smoke=s, cap=int(e("FLAT2_CAP", "24" if s else "100")), passes=int(e("FLAT2_PASSES", "1" if s else "5")),
                at=tuple(int(x) for x in e("FLAT2_AT", "4" if s else "8,16,24,32").split(",") if x),
                real=int(e("FLAT2_REAL", "1" if s else "2")), real_cap=int(e("FLAT2_REAL_CAP", "24" if s else "160")),
                key=e("FLAT2_KEY", "humaneval/HumanEval/68"), tag=e("FLAT2_TAG", "flat2"),
                out=os.path.expanduser(e("FLAT2_OUT", os.path.join(here, "out"))),
                screen=os.path.expanduser(e("FLAT2_SCREEN", "~/BigMoE/scripts/r57_screen.py")))


def head(repo, path):
    try:
        h = subprocess.run(["git", "-C", os.path.expanduser(repo), "rev-parse", "--short", "HEAD"], capture_output=True,
                           text=True, timeout=10).stdout.strip()
        d = subprocess.run(["git", "-C", os.path.expanduser(repo), "diff", "--quiet", "HEAD", "--", path], timeout=10).returncode
        return f"{h}{' +local changes in ' + path if d else ''}"
    except Exception as e:
        return f"? ({e})"


def one_pass(ns, H, ids, arm, fset, mode, label, at=(), real_cap=0):
    """one run() of the prompt; mode meas = the loop as it is, stage / probe = the per-check hooks on. Checks are held to
    2 rows (maxd 1, cut t ~0) unless real_cap: then the engine's own blocks (its t cut, up to 16 rows) for real_cap tokens"""
    say, clock = ns["say"], ns["time"].time
    H.new_pass(mode, at)
    H.real_pass = bool(real_cap)                         # S2c step 1b: the live per-piece sync runs in real-block passes only
    w0_ = clock()                                        # step B: wall stamps of the pass (GPU-utilization sampler join)
    ns["F2ON"][0] = mode != "meas"
    del ns["F2C"][:]                                     # run() stamps each cycle's start and the loop's end here
    t_, cap0 = clock(), ns["CAP"]
    xp_ = bool(XPT_ON and real_cap)
    prof = None
    if xp_:
        xpt.reset()
        xpt.ON[0] = True
        if os.environ.get("STEPB_XPT_PROF") == "1":     # cProfile of the real pass (the calling thread; ~2x slower: shares
            import cProfile                              # only, not times)
            prof = cProfile.Profile()
            prof.enable()
    try:
        if real_cap:
            ns["CAP"] = real_cap
            r = ns["run"](ids, arm, None, "dfl", False, 0, fset, None)
        else:
            r = ns["run"](ids, arm, None, "dfl", False, 1, fset, TC)
    finally:
        ns["F2ON"][0], ns["CAP"] = False, cap0
        xpt.ON[0] = False
        if prof is not None:
            prof.disable()
            try:
                import io
                import pstats
                s_ = io.StringIO()
                st_p = pstats.Stats(prof, stream=s_)
                st_p.sort_stats("tottime").print_stats(45)
                st_p.sort_stats("cumulative").print_stats(45)
                fn_ = os.path.join(ns["OUT"], os.environ.get("FLAT2_TAG", "flat2") + "_prof.txt")
                with open(fn_, "a") as f_:
                    f_.write(f"\n===== {label} ({mode}, real_cap {real_cap})\n" + s_.getvalue())
            except Exception as e:
                say(f"  XPT profile not written: {e!r}")
    try:
        pools = H.pool_snapshot()                        # what the next pass's prewarm puts back
    except Exception as e:
        H.pool_prev, pools = None, (0, 0)
        H.rec["errors"].append(f"pool snapshot: {e!r}")
    hk = H.take()
    pc = r["pc"]
    n = len(pc["ck"])
    st_ = list(ns["F2C"])                                # n starts + the end: check i's cycle = start i -> start i + 1
    period = [1e3 * (b_ - a_) for a_, b_ in zip(st_, st_[1:])] if len(st_) == n + 1 else None
    if period is None:
        hk["errors"].append(f"cycle stamps {len(st_)} for {n} checks: cycles fall back to draft + check + commit submit")
    if xp_:
        e_ = xpt.attach(pc, st_)
        if e_:
            hk["errors"].append(e_)
    z = sum(1 for i in range(n) if pc["miss"][i] == 0 and pc["stl"][i] == 0)
    cy = [period[i] if period else pc["drf"][i] + pc["ck"][i] for i in flat2_core.select(pc, rows=None if real_cap else flat2_core.ROWS)]
    pw = hk["prewarm"]
    say(f"  FLAT-2 {label} [{mode}]: {r['n']} tokens in {n} checks (kept {r['kept']:.2f}), rows per check "
        f"{sorted(set(pc['rows']))}; misses {sum(pc['miss']):.0f}, {z} of {n} checks with no miss and no stalled layer call; "
        f"steady zero-miss cycle median {flat2_core.med(cy):.2f} ms over {len(cy)} (all checks: draft {r['drf']:.2f}, check + "
        f"commit {r['ck']:.2f}); expert groups per check {flat2_core.med(pc['uses']):.0f}; "
        + (f"prewarm {pw['ids']} residents put back in {pw['ms']:.0f} ms ({pw['miss']:.0f} read); " if pw else "no prewarm; ")
        + f"pools at the end {pools[0]} base / {pools[1]} sidecar residents; {clock() - t_:.1f} s{r['bad']}"
        + ("; cut (time budget)" if r["cut"] else "") + (f"; hook errors {hk['errors']}" if hk["errors"] else ""))
    return dict(mode=mode, label=label, t_start=w0_, t_end=clock(), pc=pc, bad=r["bad"], cut=r["cut"], out=r["out"], n=r["n"], tps=r["tps"], tpsw=r["tpsw"],
                kept=r["kept"], checks=r["checks"], fast=r["fast"], pf=r["pf"], commit=hk["commit"], probes=hk["probes"],
                prewarm=pw, errors=hk["errors"], pools=pools, probed=sorted(at), real=bool(real_cap), period=period)


def main(ns, cfg, info, hooks=None, prewarm=True):
    """ns: the harness's globals (after its definitions ran); cfg: config(); info: flat2_patch.build's report.
    Returns the process's exit code; writes <out>/<tag>_screen.txt and _flat2.json."""
    say, el, HARD = ns["say"], ns["el"], ns["HARD"]
    H = hooks or flat2_probe.Hooks(ns)
    ns["F2H"] = H
    for w_ in info.get("warnings") or ():
        say(f"  PATCH WARNING: {w_}")
    ns["OUT"] = cfg["out"]                               # nothing of this run is written under ~/BigMoE
    heads = {r_: head(r_, p_) for r_, p_ in (("~/BigMoE", "scripts/r57_screen.py"), ("~/Leap", "scripts"),
                                             ("~/EvalSuite", "evalsuite"), ("~/Breakthrough", "keepset"),
                                             ("/Users/kamen/BigMoE/data/r57/snap/frozen_chs4/TrainLab_root", "sticky/draft17.py"))} if not cfg.get("no_git") else {}
    env = {k_: v_ for k_, v_ in sorted(os.environ.items())
           if k_.split("_")[0] in ("R57", "BIGMOE", "REF2", "K73", "K83", "GDNB", "FLAT2")}
    say(f"FLAT-2 on {cfg['screen']} (md5 {info.get('md5')}; hooks at source lines {info.get('anchors')}); HEADs {heads}")
    say(f"  levers (env): {env}")
    fset = ns["SETOF"][cfg["key"].split("/")[0]]
    item = next(x_ for x_ in ns["ED"].load("dec_" + fset) if x_["key"] == cfg["key"])
    ids = list(item["prompt_ids"])
    typ = ns["rule"](item)
    assert not (ns["CHATSW"] and typ == "chat"), f"{cfg['key']} is a chat prompt: FLAT-2 measures the DFlash2 loop, pick code"
    d4 = ns["DRAFT4"]
    ns["draft"] = d4 if d4 is not None and typ == "code" else ns["DRAFT16"]   # as the harness picks it (D7B=code: 4-bit)
    if XPT_ON:
        xpt.install(ns)
        say("  XPT on: real-block passes log the trunk's layer calls, the expert calls and mx.async_eval (wall + thread CPU)")
    arm = ns["ARMS_"][0]
    ns["CAP"] = cfg["cap"]
    say(f"  prompt {cfg['key']} ({len(ids)} tokens, type {typ}, drafter {'D7b 4-bit' if ns['draft'] is d4 else 'bf16'}), arm "
        f"{arm}, FAST {ns['FAST'] or 'off'}{' split ' + str(ns['QSPLIT']) if ns['QSPLIT'] is not None else ''}, C {ns['C']}; "
        f"{cfg['cap']} tokens per pass, {cfg['passes']} measured passes, probes at checks {list(cfg['at'])}, "
        f"{cfg.get('real', 0)} real-block passes of {cfg.get('real_cap', 0)} tokens"
        f"{' (SMOKE)' if cfg['smoke'] else ''}")
    runs, err = [], None
    try:
        runs.append(one_pass(ns, H, ids, arm, fset, "meas", "pass 1 (cold, discarded)"))
        TG_ = os.environ.get("STEPB_TOGGLE")             # step B's interleaved A/B: "CFG_KEY=A|B" alternates per measured pass
        tg_key, tg_vals = (TG_.split("=", 1)[0], [int(v) if v.lstrip("-").isdigit() else v for v in TG_.split("=", 1)[1].split("|")]) if TG_ else (None, None)
        for j in range(cfg["passes"]):
            if el() > HARD - 120:
                say("  time budget: no further measured pass")
                break
            if not prewarm:
                H.pool_prev = None                       # the control: passes start from the prefill's pools
            if tg_key:                                   # A B A B ...: pass j runs value j % 2 (read per call by ref2)
                if tg_key.startswith("ns:"):             # a harness variable held in a 1-element list (e.g. ns:STEPB_SBN)
                    ns[tg_key[3:]][0] = tg_vals[j % 2]
                else:
                    ns["ref2"].CFG[tg_key] = tg_vals[j % 2]
            r_ = one_pass(ns, H, ids, arm, fset, "meas", f"measured pass {j + 1}" + (f" [{tg_key}={tg_vals[j % 2]}]" if tg_key else ""))
            r_["toggle"] = tg_vals[j % 2] if tg_key else None
            runs.append(r_)
        if tg_key:                                       # the paired read-out: per value, the passes' check + commit and cycles
            if tg_key.startswith("ns:"):
                ns[tg_key[3:]][0] = tg_vals[0]
            else:
                ns["ref2"].CFG[tg_key] = tg_vals[0]
            by = {}
            for r_ in runs[1:]:
                if r_.get("toggle") is None:
                    continue
                pc = r_["pc"]
                ok_ = [i for i in range(len(pc["ck"])) if pc["miss"][i] == 0 and pc["stl"][i] == 0 and i > 0]
                ck = [pc["ck"][i] for i in ok_]
                cy = [r_["period"][i] for i in ok_] if r_.get("period") else []
                by.setdefault(r_["toggle"], []).append((flat2_core.med(ck), flat2_core.med(cy), len(ok_)))
            for v_, xs in by.items():
                say(f"  TOGGLE {tg_key}={v_}: passes {len(xs)}, check + commit (zero-miss) per pass {[round(x[0], 2) for x in xs]}, "
                    f"cycles {[round(x[1], 2) for x in xs]}, n {[x[2] for x in xs]}")
            if len(by) == 2:
                (va, xa), (vb, xb) = list(by.items())
                pairs = [(b_[0] - a_[0]) for a_, b_ in zip(xa, xb) if a_[0] == a_[0] and b_[0] == b_[0]]   # pairs with zero-miss checks
                say(f"  TOGGLE paired (B - A per pass pair, check + commit): {[round(d, 2) for d in pairs]} -> median "
                    f"{flat2_core.med(pairs):+.2f} ms ({tg_key}: {va} -> {vb})")
        side = [("stage", "stage pass", (), 0), ("probe", "probe pass", cfg["at"], 0)]
        if cfg.get("real"):                              # the engine's own blocks: one pass to load their rows, then measured
            side += [("meas", "real-block pass 1 (discarded)", (), cfg["real_cap"])]
            side += [("meas", f"real-block pass {j + 2}", (), cfg["real_cap"]) for j in range(cfg["real"])]
            side += [("stage", "real-block commit pass", (), cfg["real_cap"])]
            if flat2_probe.S2C:                          # S2c step 1 (Sat 00:5x): a real check's own block, alone vs alternating
                side += [("probe", "real-block probe pass (S2c)", (int(os.environ.get("STEPB_S2C_AT", "6")),), cfg["real_cap"])]
        for mode, label, at, rc_ in side:
            if el() > HARD - 60:
                say(f"  time budget: {label} skipped")
                continue
            if not prewarm:
                H.pool_prev = None
            try:                                         # a failed side pass must not cost the measured cycles
                runs.append(one_pass(ns, H, ids, arm, fset, mode, label, at, rc_))
            except Exception as e:
                say(f"  FLAT-2 {label} FAILED: {e!r}\n{traceback.format_exc()}")
    except Exception as e:                               # the report below still says what was measured
        err = f"{e!r}\n{traceback.format_exc()}"
        say(f"  FLAT-2 run FAILED: {err}")
    sync = flat2_probe.sync_ms(ns["mx"], H.clock)
    try:                                                 # step B: round trip, dispatch cost, read floor (same process)
        cal = flat2_probe.KB.calibrate(H.clock, big_mb=512)
        cal["sync_chains"] = flat2_probe.KB.sync_chains(H.clock)
        say("  step B t_sync chains (threads x tg: us per dependent kernel): " + ", ".join(
            f"{c_['threads']}x{c_['tg']}: {c_['us']:.2f}" for c_ in cal["sync_chains"]))
        say(f"  step B calibration: round trip {cal['sync_ms'] * 1e3:.0f} us (p10-p90 "
            f"{cal['sync_p10_p90'][0] * 1e3:.0f}-{cal['sync_p10_p90'][1] * 1e3:.0f}), dependent dispatch "
            f"{cal['dispatch_us']:.2f} us; read floor (KB: cold / warm GB/s) "
            + ", ".join(f"{r_['kb']}: {r_['cold_gbs']:.0f} / {r_['warm_gbs']:.0f}" for r_ in cal["read"]))
    except Exception as e:
        cal = {"error": repr(e), "sync_ms": sync}
        say(f"  step B calibration FAILED: {e!r}\n{traceback.format_exc()}")
    for p_ in [p_ for r_ in runs if r_["mode"] == "probe" for p_ in r_["probes"]]:
        for key_ in ("k_real", "k_16"):
            for j_, kr_ in enumerate(p_.get(key_) or []):
                try:
                    say("  " + flat2_probe.KB.brief(kr_, cal.get("sync_ms", sync), key_, p_["check"], j_))
                except Exception as e:
                    say(f"  step B brief failed: {e!r}")
    meas = [r_ for r_ in runs[1:] if r_["mode"] == "meas" and not r_["real"]]
    stage = next((r_ for r_ in runs if r_["mode"] == "stage" and not r_["real"]), None)
    real = [r_ for r_ in runs if r_["mode"] == "meas" and r_["real"]][1:]   # the first real-block pass loads their rows
    rstage = next((r_ for r_ in runs if r_["mode"] == "stage" and r_["real"]), None)
    probes = [p_ for r_ in runs if r_["mode"] == "probe" for p_ in r_["probes"]]
    say(f"  one eval sync costs {sync * 1e3:.0f} us (taken off every synced record)")
    for p_ in probes:                                    # C0's read-out per probe
        wm = p_.get("warm") or []
        say(f"  probe at check {p_['check']} ({p_['rows']} rows, live misses {p_['live_miss']:.0f}): warm repeats "
            + " / ".join(f"{w_['wall']:.2f}" for w_ in wm) + " ms (build " + " / ".join(f"{w_['build']:.2f}" for w_ in wm)
            + ", misses " + " / ".join(str(w_["miss"]) for w_ in wm) + f", logits == live {[w_['equal'] for w_ in wm]}); last "
            f"forward == live {p_.get('final_equal')}; {p_['wall_s']:.1f} s" + (f"; ERROR {p_['error']}" if p_.get("error") else ""))
        if p_.get("sweep"):
            say("    sweep (rows: warm wall ms [cold pass misses, warm misses]): " + "; ".join(
                f"{r_}: {flat2_core.med([x_['wall'] for x_ in v_[1:]]):.2f} [{v_[0]['miss']}, {v_[1]['miss']} / {v_[2]['miss']}]"
                for r_, v_ in sorted(p_["sweep"].items())))
    summ, rep = (flat2_core.summarize(meas, stage, probes, sync, real, rstage) if meas
                 else ({}, ["FLAT-2 VERDICT: NO RUN (no measured pass)"]))
    for x_ in rep:
        say(x_)
    cnt = ns["ref2"].counts()
    say("  engagement (ref2 counts at graph build, whole process): " + ", ".join(f"{k_} {v_}" for k_, v_ in sorted(cnt.items()) if v_))
    say(f"  footprint now {ns['footprint_now_gb']():.1f} GB, lifetime peak {ns['footprint_peak_gb']():.1f} GB; "
        f"{ns['time'].time() - ns['t0']:.0f} s")
    os.makedirs(cfg["out"], exist_ok=True)
    try:
        json.dump(dict(meta=dict(tag=cfg["tag"], key=cfg["key"], C=ns["C"], cap=cfg["cap"], passes=cfg["passes"], at=list(cfg["at"]),
                                 smoke=cfg["smoke"], arm=arm, fast=ns["FAST"], split=ns["QSPLIT"], heads=heads, env=env,
                                 patch=info, sync_ms=sync, error=err, prewarm=prewarm, stepb_cal=cal),
                       runs=runs, summary=summ), open(os.path.join(cfg["out"], f"{cfg['tag']}_flat2.json"), "w"),
                  default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    except Exception as e:
        say(f"  flat2.json not written: {e}")
    open(os.path.join(cfg["out"], f"{cfg['tag']}_screen.txt"), "w").write("\n".join(ns["lines"]) + "\n")
    return 1 if err else 0
