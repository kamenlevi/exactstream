"""FLAT-2 analysis (zero session, Thu 5 Nov 2026; ~/FableGod/zero/LOSSY350_ZERO.md section 4). Pure Python + numpy, no
engine imports: flat2.py feeds it the harness's per-check logs and the probes, dryrun_cpu.py feeds it a stub engine's.
The number: FLAT = the steady all-resident cycle [draft + 2-row check + commit] minus the 2 rows' expert work
(2 x the warm per-row slope of the check, measured in the same run by the row sweep). Lines (pre-registered in section 4,
repeated by the orchestrator Thu 18:0x): PASS FLAT <= 30 ms, BETWEEN 30-36, KILL > 36. 350 tok/s itself needs <= 23.5-25."""
import numpy as np

PASS_MS, KILL_MS = 30.0, 36.0
NEED_350 = 25.0                                          # flat at K 12 with e 0.77-0.9 ms per kept (LOSSY350_ZERO section 1)
ROWS = 2
MIN_CYCLES, TARGET_CYCLES = 100, 200
SKIP = 2                                                 # a pass's first checks: the drafter's window prefill, first-use costs
KINDS = ("hc_a", "gdn", "att", "hc_m", "moe_pre", "experts", "moe_post", "hc_w", "ple", "glue", "hc_f")


def med(v):
    v = np.asarray(v, dtype=float)
    return float(np.median(v)) if v.size else float("nan")


def pct(v, q):
    v = np.asarray(v, dtype=float)
    return float(np.percentile(v, q)) if v.size else float("nan")


def select(pc, skip=SKIP, rows=ROWS, probed=()):
    """a pass's steady all-resident checks: index >= skip, exactly `rows` rows, no miss, no stalled layer call, not probed"""
    n = len(pc["ck"])
    miss, stl = pc["miss"], pc.get("stl") or [0.0] * n
    return [i for i in range(skip, n) if (rows is None or pc["rows"][i] == rows) and miss[i] == 0 and stl[i] == 0
            and (i + 1) not in probed]


def period(p, i):
    """check i's cycle in ms: its start to the next cycle's start (the loop's end for the last one), as flat2's stamps in
    run() give it; without stamps, draft + check + commit submit. The commit replay is submitted alone (SB): its GPU work
    lands in the next cycle's draft, or in the loop's tail when the loop reads the GPU there (BigMoE's QOM counters,
    Thu 19:0x) -- either way one commit per steady period."""
    pc = p["pc"]
    return p["period"][i] if p.get("period") else pc["drf"][i] + pc["ck"][i]


def cycles(passes, skip=SKIP):
    """the measured passes' steady all-resident cycles (ms, start to start) and a line per pass"""
    acc = {k: [] for k in ("cyc", "drf", "ck", "bld", "wt", "uses", "kept", "tail")}
    per, n_all = [], 0
    for p in passes:
        pc = p["pc"]
        n = len(pc["ck"])
        bad = p.get("bad", "") or ""
        row = dict(checks=n, misses=float(np.sum(pc["miss"])) if n else 0.0, bad=bad.strip(),
                   rows_other=int(sum(1 for r in pc["rows"] if r != ROWS)), used=0, cyc=float("nan"))
        if "PAUSED" in bad:                              # a paused pass's clock is wrong: out, as the harness does
            per.append(row)
            continue
        n_all += max(n - skip, 0)
        idx = select(pc, skip)
        row["used"] = len(idx)
        c = [period(p, i) for i in idx]
        row["cyc"] = med(c)
        acc["cyc"] += c
        acc["tail"] += [period(p, i) - pc["drf"][i] - pc["ck"][i] for i in idx]
        for k in ("drf", "ck", "bld", "wt", "uses", "kept"):
            if pc.get(k) is not None and len(pc[k]) == n:
                acc[k] += [pc[k][i] for i in idx]
        per.append(row)
    return acc, per, n_all


def sweep_fit(pr):
    """one probe's row sweep: warm wall (median of its zero-miss warm passes) vs rows, rows >= 2 -> slope (ms per row),
    intercept, the largest residual. None with fewer than 3 row counts."""
    pts = []
    for r, recs in (pr.get("sweep") or {}).items():
        warm = [x["wall"] for x in recs[1:] if x["miss"] == 0]
        if int(r) >= 2 and warm:
            pts.append((int(r), med(warm)))
    if len(pts) < 3:
        return None
    pts.sort()
    x, y = np.array([a for a, _ in pts], float), np.array([b for _, b in pts], float)
    b, a = np.polyfit(x, y, 1)
    return dict(slope=float(b), icpt=float(a), resid=float(np.abs(a + b * x - y).max()), pts=pts)


def kinds(seg, sync):
    """Leap's C0 split (c0_timeline.py kinds): exclusive ms per piece kind summed over the layers, one eval sync taken off
    every record; seg = [(kind, layer or None, build ms, eval ms)]. Returns (ms by kind, CPU build ms by kind)."""
    by = {}
    for kind, L, b, e in seg:
        by.setdefault(L, {}).setdefault(kind, []).append((b, e))
    tot, cpu = {k: 0.0 for k in KINDS}, {k: 0.0 for k in KINDS}
    inc = lambda v: sum(b + e for b, e in v)
    for L, d in by.items():
        if L is None:
            for b, e in d.get("hc_f", []):
                tot["hc_f"] += b + e - sync; cpu["hc_f"] += b
            continue
        for k in ("hc_a", "gdn", "att", "hc_m", "moe_pre", "experts", "hc_w", "ple"):
            for b, e in d.get(k, []):
                tot[k] += b + e - sync; cpu[k] += b
        moe = inc(d.get("moe", [])) - inc(d.get("moe_pre", [])) - inc(d.get("experts", []))
        tot["moe_post"] += moe - sync * len(d.get("moe", []))
        kids = sum(inc(d.get(k, [])) for k in ("hc_a", "gdn", "att", "hc_m", "moe", "hc_w", "ple"))
        tot["glue"] += inc(d.get("layer", [])) - kids - sync * len(d.get("layer", []))
    return tot, cpu


def split(x, sync):
    """a synced pass -> the named terms of the check (ms): the C0 pieces grouped as LOSSY350_ZERO's budget names them"""
    t, c = kinds(x["seg"], sync)
    rest = x["wall"] - sum(b + e for kind, L, b, e in x["seg"] if kind in ("layer", "hc_f")) - sync
    return {"GDN": t["gdn"], "HC reads": t["hc_a"] + t["hc_m"] + t["hc_f"], "attention": t["att"],
            "router + shared": t["moe_pre"], "experts": t["experts"], "MoE tail": t["moe_post"], "HC write": t["hc_w"],
            "PLE": t["ple"], "layer glue": t["glue"], "embed + norm + head": rest, "_cpu_build": sum(c.values()),
            "_wall": x["wall"], "_rows": x["rows"], "_miss": x["miss"]}


def verdict(flat):
    return "PASS" if flat <= PASS_MS else "BETWEEN" if flat <= KILL_MS else "KILL"


def real_blocks(real, rstage, S):
    """The engine's own blocks (its t cut, up to 16 rows), all-resident: the cycle the loop would run at with no SSD
    wait, and the commit of a real block (it replays the kept rows; a 2-row check's replays at most one). Not part of
    the verdict: it checks FLAT's arithmetic on the real loop and gives the no-wait tok/s of today's levers."""
    cyc, rows, kept, n_all = [], [], [], 0
    for p in real:
        pc = p["pc"]
        if "PAUSED" in (p.get("bad") or ""):
            continue
        n_all += max(len(pc["ck"]) - SKIP, 0)
        for i in select(pc, rows=None):
            cyc.append(period(p, i)); rows.append(pc["rows"][i]); kept.append(pc["kept"][i])
    if not cyc:
        return None, []
    R = dict(n=len(cyc), n_steady=n_all, cycle=med(cyc), cycle_mean=float(np.mean(cyc)), rows=float(np.mean(rows)),
             kept=float(np.mean(kept)), tps=1e3 * float(np.sum(kept)) / float(np.sum(cyc)))
    L = [f"  the engine's own blocks, all-resident (its t cut, <= 16 rows; {R['n']} zero-miss steady cycles of {n_all}): cycle "
         f"median {R['cycle']:.2f} ms (mean {R['cycle_mean']:.2f}) at {R['rows']:.1f} rows, kept {R['kept']:.2f} per check = "
         f"{R['tps']:.1f} tok/s with no SSD wait; 350 tok/s at this kept needs a cycle of {R['kept'] * 1e3 / 350:.1f} ms"]
    if rstage is not None and rstage.get("commit"):
        cm = {int(k): v for k, v in rstage["commit"].items()}
        cs = [cm[i + 1] for i in select(rstage["pc"], rows=None) if (i + 1) in cm]
        if cs:
            R.update(commit=med(cs), commit_mean=float(np.mean(cs)), commit_n=len(cs))
            c2 = (S.get("stage") or {}).get("commit")
            txt = f"  a real block's commit ({len(cs)} zero-miss checks, waited for): median {R['commit']:.2f} ms (mean {R['commit_mean']:.2f})"
            if c2 is not None:
                txt += f" vs the 2-row check's {c2:.2f}"
                if S.get("flat") == S.get("flat") and S.get("e_row") == S.get("e_row"):   # both measured (not nan)
                    R["predicted"] = S["flat"] + S["e_row"] * R["rows"] + R["commit_mean"] - c2
                    txt += (f"; FLAT + rows x slope + the commit difference = {R['predicted']:.2f} ms vs the measured mean cycle "
                            f"{R['cycle_mean']:.2f} ({R['cycle_mean'] - R['predicted']:+.2f}: what the 2-row form does not see)")
            L.append(txt)
    return R, L


def summarize(meas, stage=None, probes=(), sync=0.0, real=(), rstage=None):
    """meas: the measured passes [dict(pc, bad, out)]; stage: the stage pass (pc + commit ms per check: draft / check /
    commit serialized) or None; probes: the probe pass's records; sync: one eval sync in ms; real / rstage: the passes
    on the engine's own blocks (measured / commit waited for). -> (summary, report lines)"""
    S, L = {}, []
    acc, per, n_all = cycles(meas)
    n = len(acc["cyc"])
    S.update(n=n, n_steady=n_all, per_pass=per, cycle=med(acc["cyc"]), cycle_iqr=[pct(acc["cyc"], 25), pct(acc["cyc"], 75)],
             cycle_p10_p90=[pct(acc["cyc"], 10), pct(acc["cyc"], 90)], draft=med(acc["drf"]), check_commit=med(acc["ck"]),
             build=med(acc["bld"]), wait=med(acc["wt"]), uses=med(acc["uses"]), tail=med(acc["tail"]),
             kept=float(np.mean(acc["kept"])) if acc["kept"] else float("nan"))
    same = [a["out"] == b["out"] for a, b in zip(meas, meas[1:]) if a.get("out") is not None and b.get("out") is not None]
    S["passes_same_tokens"] = same
    L.append("FLAT-2 (zero session; LOSSY350_ZERO.md section 4): the steady cycle [draft + 2-row check + commit], all-resident")
    for i, r in enumerate(per):
        L.append(f"  measured pass {i + 1}: {r['checks']} checks, {r['misses']:.0f} misses, {r['rows_other']} checks not at {ROWS} rows, "
                 f"{r['used']} steady zero-miss cycles used, median cycle {r['cyc']:.2f} ms"
                 + (f"; tokens == the pass before: {same[i - 1]}" if 0 < i <= len(same) else "") + (f"; {r['bad']}" if r["bad"] else ""))
    L.append(f"  cycles used: {n} steady zero-miss of {n_all} steady (target >= {TARGET_CYCLES}, minimum {MIN_CYCLES})")
    L.append(f"  cycle: median {S['cycle']:.2f} ms (IQR {S['cycle_iqr'][0]:.2f}-{S['cycle_iqr'][1]:.2f}, p10-p90 "
             f"{S['cycle_p10_p90'][0]:.2f}-{S['cycle_p10_p90'][1]:.2f}), start to start; in it (medians) draft {S['draft']:.2f}, "
             f"check + commit submit {S['check_commit']:.2f} (graph build {S['build']:.2f}, eval wait {S['wait']:.2f}), the loop's "
             f"tail before the next draft {S['tail']:.2f}; kept per cycle {S['kept']:.2f}; expert groups per check {S['uses']:.0f}")
    # ---- the expert term of the 2-row check: the row sweep's warm slope (primary), the synced split (cross-check / fallback)
    fits = [f for f in (sweep_fit(p) for p in probes) if f]
    e_row = med([f["slope"] for f in fits]) if fits else float("nan")
    S.update(e_row=e_row, sweep=fits, sweep_icpt=med([f["icpt"] for f in fits]) if fits else float("nan"),
             sweep_resid=max([f["resid"] for f in fits]) if fits else float("nan"))
    sp2 = [split(p["synced_real"], sync) for p in probes if p.get("synced_real") and p["synced_real"]["rows"] == ROWS
           and p["synced_real"]["miss"] == 0]
    sp16 = [split(p["synced_16"], sync) for p in probes if p.get("synced_16") and p["synced_16"]["miss"] == 0]
    e_split = med([s["experts"] for s in sp2]) if sp2 else float("nan")
    S.update(e_split=e_split, split2={k: med([s[k] for s in sp2]) for k in sp2[0]} if sp2 else None,
             split16={k: med([s[k] for s in sp16]) for k in sp16[0]} if sp16 else None)
    how = None
    if fits:
        e2, how = ROWS * e_row, f"{ROWS} x the row sweep's warm slope {e_row:.2f} ms per row (median of {len(fits)} probes; warm "\
                                 f"intercept {S['sweep_icpt']:.2f} ms, largest fit residual {S['sweep_resid']:.2f} ms)"
    elif sp2:
        e2, how = e_split, f"FALLBACK: the synced split's experts piece at {ROWS} rows (median of {len(sp2)} probes; no valid row sweep)"
    else:
        e2 = float("nan")
    S["experts2"] = e2
    S["flat"] = S["cycle"] - e2 if how else float("nan")
    if how:
        L.append(f"  the {ROWS} rows' expert work: {e2:.2f} ms = {how}")
        L.append(f"  FLAT = cycle - the {ROWS} rows' expert work = {S['cycle']:.2f} - {e2:.2f} = {S['flat']:.2f} ms")
    else:
        L.append("  the 2 rows' expert work: NOT MEASURED (no valid probe): FLAT has only its upper bound, the cycle")
    # ---- the stages one after the other (the stage pass waits for each commit replay before the next draft)
    if stage is not None and stage.get("commit"):
        pc, cm = stage["pc"], {int(k): v for k, v in stage["commit"].items()}
        idx = [i for i in select(pc) if (i + 1) in cm and (i + 1) not in (stage.get("probed") or ())]
        if idx:
            d_ = med([pc["drf"][i] for i in idx]); m_ = med([cm[i + 1] for i in idx])
            c_ = med([pc["ck"][i] - cm[i + 1] for i in idx])
            t_ = med([period(stage, i) - pc["drf"][i] - pc["ck"][i] for i in idx])
            S["stage"] = dict(n=len(idx), draft=d_, check=c_, commit=m_, tail=t_, total=d_ + c_ + m_ + t_)
            L.append(f"  stages run one after the other ({len(idx)} zero-miss checks of the stage pass): draft {d_:.2f} + check "
                     f"{c_:.2f} + commit {m_:.2f} + loop tail {t_:.2f} = {d_ + c_ + m_ + t_:.2f} ms (live cycle {S['cycle']:.2f}: "
                     f"the loop's overlap is worth {d_ + c_ + m_ + t_ - S['cycle']:+.2f} ms)")
    # ---- the split, for the next cut
    names = ("GDN", "HC reads", "attention", "router + shared", "experts", "MoE tail", "HC write", "PLE", "layer glue",
             "embed + norm + head")
    for tag, sp, m in ((f"{ROWS}-row check", S["split2"], len(sp2)), ("16-row check", S["split16"], len(sp16))):
        if sp:
            L.append(f"  split of the warm {tag} (synced pass, one sync off every piece, median of {m} probes; wall "
                     f"{sp['_wall']:.1f} ms synced): " + ", ".join(f"{k} {sp[k]:.2f}" for k in names)
                     + f"; CPU graph build in it {sp['_cpu_build']:.2f}")
    if S["split2"] and S["split16"] and S["split16"]["_rows"] > ROWS:
        dr_ = S["split16"]["_rows"] - ROWS
        L.append("  per added row (synced 16 vs 2 rows): " + ", ".join(
            f"{k} {(S['split16'][k] - S['split2'][k]) / dr_:+.3f}" for k in names if abs(S["split16"][k] - S["split2"][k]) >= 0.05))
    # ---- cross-checks
    wr = [w["wall"] for p in probes for w in p.get("warm", []) if w["miss"] == 0 and w["rows"] == ROWS]
    eq = [bool(w["equal"]) for p in probes for w in p.get("warm", []) if w["miss"] == 0 and p.get("live_miss", 0) == 0]
    fe = [bool(p["final_equal"]) for p in probes if "final_equal" in p]
    S.update(warm2=med(wr), warm_equal=eq, final_equal=fe)
    if probes:
        L.append(f"  cross-checks ({len(probes)} probes): experts by the synced split {e_split:.2f} ms (FLAT would be "
                 f"{S['cycle'] - e_split:.2f}); the warm {ROWS}-row check alone {S['warm2']:.2f} ms vs the fit's "
                 f"{S['sweep_icpt'] + ROWS * e_row:.2f}; warm repeats bitwise == the live check (zero-miss checks): "
                 f"{sum(eq)} of {len(eq)}; last forward == the live check: {sum(fe)} of {len(fe)}")
    # ---- the engine's own blocks (not in the verdict)
    S["real"], rl = real_blocks(real, rstage, S)
    L += rl
    # ---- validity and the verdict
    notes = []
    if n < MIN_CYCLES:
        notes.append(f"only {n} steady zero-miss cycles (< {MIN_CYCLES}): not all-resident, shorten FLAT2_CAP or add passes")
    if any("INVALID" in r["bad"] for r in per):
        notes.append("the harness flagged a measured pass INVALID (gate check)")
    if sum(r["rows_other"] for r in per) > 0:
        notes.append(f"{sum(r['rows_other'] for r in per)} measured checks were not {ROWS}-row (left out)")
    if eq and not all(eq):
        notes.append("a zero-miss warm repeat was not bitwise == its live check: the probe's forward is not the loop's")
    valid = n >= MIN_CYCLES and not any("INVALID" in r["bad"] for r in per)
    if not valid:
        v = "INVALID"
    elif not how:
        v = "PASS" if S["cycle"] <= PASS_MS else "INCOMPLETE"   # the cycle bounds FLAT from above
    else:
        v = verdict(S["flat"])
        if not fits:
            notes.append("expert term from the synced split (fallback)")
    S.update(verdict=v, valid=valid, notes=notes, n_target_met=n >= TARGET_CYCLES)
    shown = S["flat"] if how else S["cycle"]
    L.append(f"FLAT-2 VERDICT: {v} (FLAT {'' if how else '<= '}{shown:.2f} ms; lines: PASS <= {PASS_MS:.0f}, BETWEEN "
             f"{PASS_MS:.0f}-{KILL_MS:.0f}, KILL > {KILL_MS:.0f}; 350 tok/s itself needs <= {NEED_350:.0f} at K 12: "
             f"{shown - NEED_350:+.2f} ms from it)" + ("; " + "; ".join(notes) if notes else ""))
    return S, L
