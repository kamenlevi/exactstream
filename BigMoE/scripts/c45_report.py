"""C45 report (BigMoE, Sun 2026-11-01): reads tonight's data/c45/<tag>.txt (r57_screen's output per run, arm gate+t) and
prints the per-C table and the pre-registered verdicts (HANDOVER 'TONIGHT'S C45 TURN'):
- OOO A-B-A at the first C: B c45o_C vs the mean of A c45_C and A2 c45_Cr, steady tok/s (+ per-prompt paired ratios);
  every token == REF2-T in all three.
- prior arm P c45p_C vs B: PASS = median misses per check -7 % or more, tokens identical, steady tok/s not lower.
- L arm c45l_C vs B (NewIdeas 15:3x, MoonShots' M-17 LAYSZ45 per-layer pool sizes, the same pool bytes): PASS = SSD MB per
  kept token -10 % or more with steady tok/s +3 % or more; KILL = bytes above -3 %; every token == REF2-T.
- sizing: per C on the default (c45o_C, incl. C45_BIG's ~50 GB arms): tok/s, steady, cycle, check, misses, SSD busy (share
  of the check), gate service on misses, draft ms, SSD MB + misses per kept token (pooled), lifetime peak.
usage: .venv/bin/python scripts/c45_report.py [DIR=data/c45] -> stdout + DIR/c45_report.txt"""
import glob, os, re, statistics, sys

D = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/BigMoE/data/c45")
NUM = r"([-\d.]+)"


def grab(pat, s, cast=float):
    m = re.search(pat, s)
    return cast(m.group(1)) if m else None


def parse(path):
    s = open(path).read()
    agg = next((l for l in s.splitlines() if l.startswith("  gate+t: tok/s")), "")
    tg = next((l for l in s.splitlines() if l.startswith("  gate+t where the time goes")), "")
    fl = next((l for l in s.splitlines() if "tier flags:" in l), "")
    pk = next((l for l in s.splitlines() if l.startswith("  gate+t per kept token")), "")
    r = dict(mbk=grab(r"SSD " + NUM + r" MB, misses", pk), mpk=grab(r"MB, misses " + NUM, pk),tps=grab(r"tok/s " + NUM, agg), tpsw=grab(r"steady state " + NUM, agg), cyc=grab(r"cycle " + NUM, agg),
             ck=grab(r"check \+ commit " + NUM, agg), miss=grab(r"misses " + NUM, agg), io=grab(r"demand SSD " + NUM, agg),
             svc=grab(r"gate service on misses " + NUM, tg), drf=grab(r"draft " + NUM, tg),
             peak=grab(r"footprint now [\d.]+ GB, lifetime peak " + NUM, s), ooo=grab(r"OOO (True|False)", fl, str),
             calls=grab(r"\((\d+) OOO calls\)", fl, int), eq=grab(r"every token equal (True|False)", fl, str),
             k73=grab(r"K-73 engagement: (.*)", s, str), prompts={})
    for l in s.splitlines():                      # per prompt: steady tok/s, misses per check, REFTOK equality
        m = re.match(r"  (\S+) \[gate\+t\]: .*?steady state, checks >= 2: " + NUM + r"\).*?per check \(medians\) "
                     + NUM + r" misses", l)
        if m:
            t = re.search(r"tokens == REFTOK (\d+) / (\d+)", l)
            r["prompts"][m.group(1)] = (float(m.group(2)), float(m.group(3)), bool(t) and int(t.group(1)) >= int(t.group(2)),
                                        re.search(r"in (\d+) checks \(kept ([\d.]+)\)", l).groups())
    r["ok"] = r["eq"] == "True" and bool(r["prompts"]) and all(v[2] for v in r["prompts"].values())
    r["warm"] = grab(r"warm-up \(discarded\).*lifetime peak [\d.]+ -> ([\d.]+) GB", s) or 0.0
    r["mem"] = []                                  # per prompt: prefill / decode MLX peak, end cache, lifetime peak rise
    for l in s.splitlines():
        m = re.search(r"MLX peak prefill ([\d.]+) / decode ([\d.]+) GB; at the end active ([\d.]+), cache ([\d.]+), footprint "
                      r"([\d.]+) GB; lifetime peak after prefill ([\d.]+) -> after decode ([\d.]+) GB", l)
        if m:
            r["mem"].append(tuple(float(x) for x in m.groups()))
    return r


R = {os.path.basename(p)[:-4]: parse(p) for p in sorted(glob.glob(os.path.join(D, "c45*_*.txt")))}
R = {k: v for k, v in R.items() if v["tps"] is not None}
out = []


def say(x=""):
    print(x); out.append(x)


def f(x, n=1):
    return "-" if x is None else f"{x:.{n}f}"


say(f"C45 report ({D}): {len(R)} runs with results")
say("  tag          OOO  tok/s steady  cycle  check  misses  SSD ms (share)  gate svc  draft  MB/kept  miss/kept  peak GB  "
    "tokens==REF2-T")
for k, r in sorted(R.items(), key=lambda kv: (int(re.search(r"_(\d+)", kv[0]).group(1)), kv[0])):
    sh = f"({r['io'] / r['ck']:.0%})" if r["io"] and r["ck"] else ""
    say(f"  {k:12s} {str(r['ooo'])[:1]:>3s} {f(r['tps']):>6s} {f(r['tpsw']):>6s} {f(r['cyc']):>6s} {f(r['ck']):>6s} "
        f"{f(r['miss'], 0):>7s} {f(r['io']):>7s} {sh:>7s} {f(r['svc']):>9s} {f(r['drf']):>6s} {f(r['mbk']):>8s} {f(r['mpk']):>10s} "
        f"{f(r['peak']):>8s}  {'yes' if r['ok'] else 'NO'} ({r['calls'] or 0} OOO calls)")
for k, r in sorted(R.items()):
    if r["mem"]:
        M = r["mem"]
        say(f"  {k:12s} mem: MLX peak prefill max {max(m[0] for m in M):.1f} / decode max {max(m[1] for m in M):.1f} GB; cache at "
            f"the end max {max(m[3] for m in M):.1f} GB; lifetime peak rose in prefill {sum(m[5] > p for m, p in zip(M, [r['warm']] + [x[6] for x in M]))} "
            f"/ decode {sum(m[6] > m[5] for m in M)} of {len(M)} prompts (end {M[-1][6]:.1f} GB)")
firsts = sorted({int(m.group(1)) for k in R for m in [re.match(r"c45_(\d+)r?$", k)] if m})
if firsts:
    C = firsts[0]
    A, B, A2, P = (R.get(t) for t in (f"c45_{C}", f"c45o_{C}", f"c45_{C}r", f"c45p_{C}"))
    if A and B:
        As = [x for x in (A, A2) if x]
        a_w = sum(x["tpsw"] for x in As) / len(As)
        rat = sorted(B["prompts"][p][0] / (sum(x["prompts"][p][0] for x in As) / len(As))
                     for p in B["prompts"] if all(p in x["prompts"] for x in As))
        pp = (f"per-prompt steady ratio median {statistics.median(rat) - 1:+.1%} (range {rat[0] - 1:+.1%} .. {rat[-1] - 1:+.1%}, "
              f"{len(rat)} prompts)" if rat else "no per-prompt pairs")
        drift = f"; A2 vs A {A2['tpsw'] / A['tpsw'] - 1:+.1%}" if A2 else "; no A2"
        say(f"OOO A-B-A at C {C}: steady tok/s {a_w:.1f} -> {B['tpsw']:.1f} ({B['tpsw'] / a_w - 1:+.1%}); whole run "
            f"{sum(x['tps'] for x in As) / len(As):.1f} -> {B['tps']:.1f}; cycle {sum(x['cyc'] for x in As) / len(As):.1f} -> "
            f"{B['cyc']:.1f} ms; {pp}{drift}; every token == REF2-T: {all(x['ok'] for x in As + [B])}; OOO calls {B['calls']}")
    noise = f" (run-to-run: A2 vs A {A2['tpsw'] / A['tpsw'] - 1:+.1%})" if A and A2 else ""
    K = R.get(f"c45k_{C}")
    if K and B:                                    # Leap's K-73 exact-safe kernel stack on the default (its every-token check)
        rk = sorted(K["prompts"][p][0] / B["prompts"][p][0] for p in K["prompts"] if p in B["prompts"])
        pk = (f"per-prompt steady ratio median {statistics.median(rk) - 1:+.1%} (range {rk[0] - 1:+.1%} .. {rk[-1] - 1:+.1%})"
              if rk else "no per-prompt pairs")
        say(f"K-73 stack at C {C} (K c45k vs B, both the default): every token == REF2-T: {K['ok']}; steady tok/s "
            f"{B['tpsw']:.1f} -> {K['tpsw']:.1f} ({K['tpsw'] / B['tpsw'] - 1:+.1%}){noise}; check + commit {B['ck']:.1f} -> "
            f"{K['ck']:.1f} ms; cycle {B['cyc']:.1f} -> {K['cyc']:.1f} ms; draft {f(B['drf'])} -> {f(K['drf'])} ms; drafts "
            f"identical (checks + kept per prompt == B): {all(K['prompts'][p][3] == B['prompts'][p][3] for p in K['prompts'] if p in B['prompts'])}; "
            f"{pk}; {K['k73'] or 'NO engagement line'}")
    Lr = R.get(f"c45l_{C}")
    if Lr and B and Lr["mbk"] and B["mbk"]:        # MoonShots' M-17 per-layer pool sizes (same pool bytes), on the default
        db, dw = Lr["mbk"] / B["mbk"] - 1, Lr["tpsw"] / B["tpsw"] - 1
        v = ("FAIL (a token left REF2-T)" if not (Lr["ok"] and B["ok"])
             else "PASS" if db <= -0.10 and dw >= 0.03 else "KILL" if db > -0.03 else "BETWEEN")
        say(f"L per-layer pool sizes at C {C} (vs B, the same pool bytes; MoonShots' lines: PASS = SSD MB per kept token -10 % "
            f"or more with steady tok/s +3 % or more, KILL = bytes above -3 %; the replay said -20.7 %): {v} (SSD MB per kept "
            f"token {B['mbk']:.1f} -> {Lr['mbk']:.1f}, {db:+.1%}; misses per kept token {f(B['mpk'])} -> {f(Lr['mpk'])}; steady "
            f"tok/s {B['tpsw']:.1f} -> {Lr['tpsw']:.1f}, {dw:+.1%}{noise}; whole run {B['tps']:.1f} -> {Lr['tps']:.1f}; SSD busy "
            f"{f(B['io'])} -> {f(Lr['io'])} ms per check; lifetime peak {f(B['peak'])} -> {f(Lr['peak'])} GB; every token == "
            f"REF2-T: {Lr['ok']})")
    if P and B:
        dm, dw = P["miss"] / B["miss"] - 1, P["tpsw"] / B["tpsw"] - 1
        v = ("FAIL (a token left REF2-T)" if not (P["ok"] and B["ok"])
             else "PASS" if dm <= -0.07 and dw >= 0 else "KILL" if dm > -0.03 else "BETWEEN")
        say(f"Prior arm at C {C} (vs B, both on OOO; PASS = misses per check -7 % or more, tokens identical, steady tok/s "
            f"not lower): {v} (misses {B['miss']:.0f} -> {P['miss']:.0f}, {dm:+.1%}; steady tok/s {B['tpsw']:.1f} -> "
            f"{P['tpsw']:.1f}, {dw:+.1%}; SSD {f(B['io'])} -> {f(P['io'])} ms)")
siz = sorted((int(k[5:]), r) for k, r in R.items() if re.match(r"c45o_\d+$", k))
if siz:
    say("Sizing on the default (OOO on): " + "; ".join(
        f"C {c}: {r['tpsw']:.1f} steady ({r['tps']:.1f} whole run), {f(r['mpk'])} misses + {f(r['mbk'])} MB per kept token, "
        f"lifetime peak {f(r['peak'])} GB, SSD {f(r['io'])} ms of a {f(r['ck'])} ms check" for c, r in siz))
LOG = os.path.join(D, "c45.log")                   # the runner's log, tonight's turn (after its last "c45 start"): what the
if os.path.exists(LOG):                            # NO-SWAP gate skipped (the orchestrator 15:3x: say if C 300 / 320 were), what
    ll = open(LOG).read().splitlines()             # failed, and pauses / guards
    ll = ll[max((i for i, l in enumerate(ll) if "c45 start" in l), default=0):]
    notes = []
    for l in ll:
        if m := re.search(r"(\S+) skipped: NO-SWAP (\d+) GB < (\d+)", l):
            notes.append(f"{m.group(1)} skipped by the NO-SWAP gate ({m.group(2)} GB free < {m.group(3)} needed)")
        elif m := re.search(r"done (\S+) rc=(\d+)", l):
            if m.group(2) != "0":
                notes.append(f"{m.group(1)} ended rc={m.group(2)}")
        elif re.search(r"SWAP GUARD|PAUSE|lost the lock|L arm skipped|SSD hot", l):
            notes.append(l[9:].strip())
    say("Not run, cut short or paused: " + ("; ".join(notes) if notes else "none"))
open(os.path.join(D, "c45_report.txt"), "w").write("\n".join(out) + "\n")
