#!/bin/bash
# LEVER 2, STAGE 1 at C 160 by day (BUILD_DAY #2; the orchestrator Wed 14:46 / 15:0x): forecast reads into the volatile tier
# (BIGMOE_FC_F / FS, ext/build_l2, e8b7b6d) vs the same build with them off, on the adopted default (R57_PRESET=45, 0365ada).
#   0. smoke  l2s: lever 2 on 2 prompts, CAP 64; it must issue forecast reads, use some, and keep every token == REF2-T
#             (prefix of ref14), else the arms below do not run
#   1. speed, untraced, A-B-B-A, the 14 prompts, CAP 256: l2c / l2f / l2f2 / l2c2   (c = FC off; f = FC_F 4 + FC_FS 2)
#   2. exposed SSD, traced (BIGMOE_TRACE + R57_STALL -> stall_report.py), W1B's six: l2ct / l2ft
# Line (pre-registered Wed ~15:0x, sent to the orchestrator before any run): KEEP if every token == REF2-T AND the traced
# stall per check (stall_report's cadence excess, the exposed SSD) <= -20 % AND steady tok/s >= +3 % geo (both passes
# per side); KILL if < +1 % or the next-demand penalty (traced demand SSD ms per burst, l2ft - l2ct) > 0.05 ms; else BETWEEN.
# usage: inside turn45 (HAVE_LOCK=1 KEEP_LOCK=1) -> data/l2/l2.log, data/l2/l2_report.txt (~25-30 min quiet, MEM ~38 GB)
export PATH="$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:$PATH"
cd "$HOME/BigMoE" || exit 1
mkdir -p data/l2
LOG="$HOME/BigMoE/data/l2/l2.log"
say() { echo "$(date '+%m-%d %H:%M:%S') $*" >> "$LOG"; }
swap_gb() { sysctl -n vm.swapusage | awk '{gsub("M","",$6); print $6 / 1024}'; }
cp scripts/r57_screen.py data/r57/snap/r57_screen_l2.py && cp scripts/c45_runner.sh data/r57/snap/c45_runner_l2.sh || exit 2
RUN="/bin/bash data/r57/snap/c45_runner_l2.sh"
export C45_SCREEN=data/r57/snap/r57_screen_l2.py C45_DEADLINE=${L2_DEADLINE:-2250} C45_OUT="$HOME/BigMoE/data/l2"
export C45_SWAP0=$(swap_gb)
CC=${L2_C:-160}
B="BIGMOE_EXT_BUILD=build_l2,R57_ARMS=gate+t+pf,BIGMOE_PRED_LINES=4"   # +pf: the arm that posts L+1's ranked forecast (set_arm; Wed 15:4x: plain
# gate+t posts none, so lever 2 read nothing in the by-day all-in arm); the control posts it too, its reads stay off
F="$B,BIGMOE_FC_F=${L2_F:-4},BIGMOE_FC_FS=${L2_FS:-2}"
REF="$HOME/BigMoE/data/d45/ref14_tokens.json:gate+t"
K="humaneval/HumanEval/68,humaneval/HumanEval/91,humaneval/HumanEval/16,humaneval/HumanEval/6,math500/test/prealgebra/1640.json"
K="$K,math500/test/algebra/694.json,math500/test/algebra/1282.json,gsm8k/708,ifeval/1220,ifeval/1561,multi/de3,multi/zh15"
K="$K,agent/bash18,agent/text23"
K6="humaneval/HumanEval/68,humaneval/HumanEval/91,math500/test/prealgebra/1640.json,math500/test/algebra/694.json"
K6="$K6,math500/test/algebra/1282.json,gsm8k/708"
say "lever 2 stage 1 start (pid $$, C $CC, F ${L2_F:-4} / FS ${L2_FS:-2}, swap $C45_SWAP0 GB)"
R57_KEYS="humaneval/HumanEval/68,agent/bash18" MG=${L2_MG:-6} C45_ALARM=600 C45_CAP=64 C45_REFTOK="$REF" \
  C45_ARMS="l2s_$CC:$CC:$F" $RUN
SM=$(.venv/bin/python -I - "$CC" 2>&1 <<'EOF'
import json, sys
C = sys.argv[1]
try:
    rs = json.load(open(f"data/r57/l2s_{C}_runs.json")); tk = json.load(open(f"data/r57/l2s_{C}_tokens.json"))
except OSError as e:
    print(f"FAIL no run files ({e})"); sys.exit()
ref = json.load(open("data/d45/ref14_tokens.json"))
pf = lambda k: sum(r.get("pf", {}).get(k, 0) for r in rs)
tok_ = lambda v: v.get("gate+t+pf", v.get("gate+t"))
diff = [k for k, v in tk.items() if tok_(v)[0] != ref[k]["gate+t"][0][:len(tok_(v)[0])]]   # [run][token]: prefix of run 0
ok = len(rs) == 2 and pf("fc_issued") > 0 and pf("fc_hits") > 0 and not diff
print(("OK" if ok else "FAIL") + f": runs {len(rs)}, forecast reads {pf('fc_issued')}, used {pf('fc_hits')} "
      f"({pf('fc_late')} landing), dropped {pf('fc_drops')}; tokens differ {diff}")
EOF
)
say "smoke: $SM"
if [ "${SM:0:2}" != "OK" ]; then say "smoke failed: arms not run"; say "lever 2 stage 1 end"; exit 3; fi
R57_KEYS=$K MG=${L2_MG:-6} C45_ALARM=900 C45_REFTOK="$REF" \
  C45_ARMS="l2c_$CC:$CC:$B l2f_$CC:$CC:$F l2f2_$CC:$CC:$F l2c2_$CC:$CC:$B" $RUN
say "speed arms rc=$?"
R57_KEYS=$K6 MG=${L2_MG:-6} C45_ALARM=900 C45_REFTOK="$REF" \
  C45_ARMS="l2ct_$CC:$CC:$B,BIGMOE_TRACE=1,R57_STALL=1 l2ft_$CC:$CC:$F,BIGMOE_TRACE=1,R57_STALL=1" $RUN
say "traced arms rc=$?"
.venv/bin/python scripts/stall_report.py data/r57 "l2ct_$CC,l2ft_$CC" > data/l2/l2_stall.txt 2>&1
say "stall_report rc=$? (data/l2/l2_stall.txt)"
.venv/bin/python -I - "$CC" >> "$LOG" 2>&1 <<'EOF'
import json, math, re, sys
C = sys.argv[1]
ref = json.load(open("data/d45/ref14_tokens.json"))
dom = lambda k: "code" if k.startswith("humaneval") else "math" if k.startswith(("math500", "gsm8k")) else \
    "agent" if k.startswith("agent") else "chat"
g = lambda xs: math.exp(sum(map(math.log, xs)) / len(xs)) if xs else float("nan")
T = ("l2c", "l2f", "l2f2", "l2c2", "l2ct", "l2ft")
runs, toks = {}, {}
for t in T:
    try:
        runs[t] = {r["key"]: r for r in json.load(open(f"data/r57/{t}_{C}_runs.json"))}
        toks[t] = json.load(open(f"data/r57/{t}_{C}_tokens.json"))
    except OSError:
        runs[t], toks[t] = {}, {}
ok = lambda t, k: k in runs[t] and "PAUSED" not in runs[t][k].get("bad", "")
L = [f"LEVER 2 STAGE 1 at C {C} (data/l2): forecast reads into the volatile tier (FC_F 4 / FS 2, ext/build_l2 e8b7b6d) vs "
     f"FC off, same build; R57_PRESET=45"]
bad = {}
for t in T:
    rs = runs[t]
    if not rs:
        L.append(f"{t:5s} MISSING"); bad[t] = ["missing"]; continue
    bad[t] = [k for k in toks[t] if toks[t][k].get("gate+t+pf", toks[t][k].get("gate+t")) != ref[k]["gate+t"]]
    pf = lambda x: sum(r.get("pf", {}).get(x, 0) for r in rs.values())
    tw = [r["tpsw"] for k, r in rs.items() if ok(t, k)]
    L.append(f"{t:5s} tokens {len(toks[t]) - len(bad[t])} / {len(toks[t])} == REF2-T" + (f" DIFFER {bad[t]}" if bad[t] else "") +
             f"; geo steady {g(tw):.1f} tok/s over {len(tw)}; SSD MB per kept {g([r['mb'] for r in rs.values()]):.1f}; lifetime "
             f"max {max(r['mem'][-1] for r in rs.values()):.1f} GB" + (f"; forecast reads {pf('fc_issued')}, used {pf('fc_hits')} "
             f"({pf('fc_late')} landing), dropped {pf('fc_drops')}, demand bursts meeting them {pf('fc_busy_bursts')}"
             if pf("fc_issued") else ""))
rat, per = [], {}
for k in ref:
    a = [runs[t][k]["tpsw"] for t in ("l2c", "l2c2") if ok(t, k)]
    b = [runs[t][k]["tpsw"] for t in ("l2f", "l2f2") if ok(t, k)]
    if a and b:
        x = (sum(b) / len(b)) / (sum(a) / len(a)); rat.append(x); per.setdefault(dom(k), []).append(x)
        L.append(f"  {k}: off {sum(a) / len(a):.1f} -> on {sum(b) / len(b):.1f} tok/s steady ({100 * (x - 1):+.1f} %)")
G = g(rat)
L.append("geo-mean steady tok/s on vs off: %+.1f %% over %d prompts; by domain: %s" % (100 * (G - 1), len(rat), ", ".join(
    f"{d} {100 * (g(v) - 1):+.1f} %" for d, v in sorted(per.items()))))
st = open("data/l2/l2_stall.txt").read() if True else ""
sec = {m.group(1): m.start() for m in re.finditer(r"== (l2[cf]t)_\d+:", st)}
def grab(tag, pat):
    if tag not in sec:
        return None
    a = sec[tag]; b = min([v for v in sec.values() if v > a] + [len(st)])
    m = re.search(pat, st[a:b])
    return [float(x) for x in m.groups()] if m else None
sc, sf = grab("l2ct", r"stall \(cadence excess\) ([\d.]+) ms"), grab("l2ft", r"stall \(cadence excess\) ([\d.]+) ms")
dc, df = grab("l2ct", r"demand SSD time per check ([\d.]+) ms in ([\d.]+) bursts"), \
    grab("l2ft", r"demand SSD time per check ([\d.]+) ms in ([\d.]+) bursts")
ex = (sf[0] / sc[0] - 1) if sc and sf and sc[0] > 0 else None
pen = (df[0] / df[1] - dc[0] / dc[1]) if dc and df and dc[1] > 0 and df[1] > 0 else None
L.append("traced (W1B's six): stall per check " + (f"{sc[0]:.1f} -> {sf[0]:.1f} ms ({100 * ex:+.1f} %)" if ex is not None
         else "n/a") + "; demand SSD per burst " + (f"{dc[0] / dc[1]:.3f} -> {df[0] / df[1]:.3f} ms ({pen:+.3f} ms, the "
         f"next-demand penalty)" if pen is not None else "n/a"))
exact = not any(bad[t] for t in T)
if not exact:
    v = "LOOK: a token differs or an arm is missing (exact by construction: find the cause before anything else)"
elif len(rat) < len(ref) - 2 or ex is None or pen is None:
    v = f"INCOMPLETE (timed {len(rat)} of {len(ref)}, traced {'yes' if ex is not None else 'no'})"
elif G < 1.01 or pen > 0.05:
    v = "KILL"
elif G >= 1.03 and ex <= -0.20:
    v = "KEEP"
else:
    v = "BETWEEN"
L.append(f"Verdict (pre-registered Wed ~15:0x): {v} (KEEP: every token == REF2-T AND stall <= -20 % AND >= +3 % geo; KILL: "
         f"< +1 % or penalty > 0.05 ms per burst)")
open("data/l2/l2_report.txt", "w").write("\n".join(L) + "\n"); print("\n".join(L))
EOF
say "report rc=$? (data/l2/l2_report.txt)"
say "lever 2 stage 1 end"
