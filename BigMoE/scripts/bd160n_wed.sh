#!/bin/bash
# Wed ~20:20-21:00 BY DAY (the orchestrator 17:5x: after MTP Head's P-CAP; numbers before the 23:00 C 224 ticket): C 160
# uniform pools (R57_LAYSZ=0, C_l 64: every 16-row check on the FAST path), one quiet ticket, MEM ~36 GB.
#   1. frozen screen (scripts/r57_screen.py, smoke-tested): exact lever 2 off / on (l2cn / l2fn, 14 prompts, gate+t+pf,
#      PRED_LINES 4, every token vs ref14) and the QOM all-in (u3cn U3C / qalln = QOM + D17 + D7b-NAX + HC8 + DNR + hc_b +
#      SMALL_NATIVE + OOO-1C / qallfn = qalln + lever 2; 12 HumanEval)
#   2. MoonShots' relaxed acceptance (32edb12) on its own screen copy (scripts/r57_screen_rlx.py, 7 asserted edits): rx0n =
#      flag off, exact, 2 prompts CAP 64 (must equal REF2-T: the gate for the 23:00 arm), then qallrn = qalln +
#      R57_RELAX_DELTA=0.5 (+ R57_RELAX_LOG) on the 12 HumanEval
# usage: inside turn45 -> data/bdn/bdn.log, data/bdn/bdn_report.txt
export PATH="$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:$PATH"
cd "$HOME/BigMoE" || exit 1
mkdir -p data/bdn
LOG="$HOME/BigMoE/data/bdn/bdn.log"
say() { echo "$(date '+%m-%d %H:%M:%S') $*" >> "$LOG"; }
swap_gb() { sysctl -n vm.swapusage | awk '{gsub("M","",$6); print $6 / 1024}'; }
cp scripts/r57_screen.py data/r57/snap/r57_screen_bdn.py && cp scripts/r57_screen_rlx.py data/r57/snap/r57_screen_bdnr.py \
  && cp scripts/c45_runner.sh data/r57/snap/c45_runner_bdn.sh || exit 2
RUN="/bin/bash data/r57/snap/c45_runner_bdn.sh"
export C45_DEADLINE=${BDN_DEADLINE:-2058} C45_OUT="$HOME/BigMoE/data/bdn" C45_SWAP0=$(swap_gb)
CC=160
P="R57_LAYSZ=0,BIGMOE_TIER_CL=64"
K="humaneval/HumanEval/68,humaneval/HumanEval/91,humaneval/HumanEval/16,humaneval/HumanEval/6,math500/test/prealgebra/1640.json"
K="$K,math500/test/algebra/694.json,math500/test/algebra/1282.json,gsm8k/708,ifeval/1220,ifeval/1561,multi/de3,multi/zh15"
K="$K,agent/bash18,agent/text23"
H="humaneval/HumanEval/68,humaneval/HumanEval/16,humaneval/HumanEval/91,humaneval/HumanEval/6,humaneval/HumanEval/79"
H="$H,humaneval/HumanEval/19,humaneval/HumanEval/161,humaneval/HumanEval/17,humaneval/HumanEval/56,humaneval/HumanEval/141"
H="$H,humaneval/HumanEval/101,humaneval/HumanEval/63"
L2="$P,BIGMOE_EXT_BUILD=build_l2,R57_ARMS=gate+t+pf,BIGMOE_PRED_LINES=4"
U="$P,BIGMOE_EXT_BUILD=build_qom,R57_FAST=U3C,R57_U3C_LEN=1,R57_REFTOK="
QA="$P,BIGMOE_EXT_BUILD=build_qom,R57_REFTOK=,R57_FAST=QOM,R57_D17=1,R57_D7B=code,R57_HC8=1,REF2_HC8_FAST=1,R57_DNR=1"
QA="$QA,R57_HCB=1,REF2_SMALL_NATIVE=1,R57_D7BNAX=1,REF2_OOO_1C=1"
QF="$QA,BIGMOE_FC_F=4,BIGMOE_FC_FS=2,R57_ARMS=gate+t+pf,BIGMOE_PRED_LINES=4"
REF="$HOME/BigMoE/data/d45/ref14_tokens.json:gate+t"
say "by-day C $CC start (pid $$, swap $C45_SWAP0 GB)"
C45_SCREEN=data/r57/snap/r57_screen_bdn.py R57_KEYS=$K MG=3 C45_ALARM=900 C45_REFTOK="$REF" \
  C45_ARMS="l2cn_$CC:$CC:$L2 l2fn_$CC:$CC:$L2,BIGMOE_FC_F=4,BIGMOE_FC_FS=2" $RUN
say "lever 2 arms rc=$?"
C45_SCREEN=data/r57/snap/r57_screen_bdn.py R57_KEYS=$H MG=3 C45_ALARM=900 \
  C45_ARMS="u3cn_$CC:$CC:$U qalln_$CC:$CC:$QA qallfn_$CC:$CC:$QF" $RUN
say "QOM all-in arms rc=$?"
C45_SCREEN=data/r57/snap/r57_screen_bdnr.py R57_KEYS=humaneval/HumanEval/91,gsm8k/708 MG=3 C45_ALARM=600 C45_CAP=64 \
  C45_REFTOK="$REF" C45_ARMS="rx0n_$CC:$CC:$P" $RUN
say "relax flag-off check rc=$?"
C45_SCREEN=data/r57/snap/r57_screen_bdnr.py R57_KEYS=$H MG=3 C45_ALARM=900 \
  C45_ARMS="qallrn_$CC:$CC:$QA,R57_RELAX_DELTA=0.5,R57_RELAX_LOG=$HOME/BigMoE/data/bdn/relax_accepts.jsonl" $RUN
say "relaxed arm rc=$?"
.venv/bin/python -I - "$CC" >> "$LOG" 2>&1 <<'EOF'
import json, math, sys
C = sys.argv[1]
g = lambda xs: math.exp(sum(map(math.log, xs)) / len(xs)) if xs else float("nan")
ref = json.load(open("data/d45/ref14_tokens.json"))
T = ("l2cn", "l2fn", "u3cn", "qalln", "qallfn", "rx0n", "qallrn")
runs, toks = {}, {}
for t in T:
    try:
        runs[t] = {r["key"]: r for r in json.load(open(f"data/r57/{t}_{C}_runs.json"))}
        toks[t] = json.load(open(f"data/r57/{t}_{C}_tokens.json"))
    except OSError:
        runs[t], toks[t] = {}, {}
ok = lambda t, k: k in runs[t] and "PAUSED" not in runs[t][k].get("bad", "")
tk = lambda v: (v.get("gate+t+pf") or v.get("gate+t") or [[]])[0]
L = [f"BY DAY C {C} uniform (data/bdn; quiet ticket, apps open): exact lever 2, the QOM all-in, relaxed acceptance (0.5 nat)"]
for t in T:
    rs = runs[t]
    if not rs:
        L.append(f"{t:7s} MISSING"); continue
    tw = [r["tpsw"] for k, r in rs.items() if ok(t, k)]
    pf = lambda x: sum(r.get("pf", {}).get(x, 0) for r in rs.values())
    ex = ""
    if t in ("l2cn", "l2fn", "rx0n"):
        d = [k for k, v in toks[t].items() if tk(v) != ref[k]["gate+t"][0][:len(tk(v))]]
        ex = f" tokens == REF2-T {len(toks[t]) - len(d)} / {len(toks[t])}{' DIFFER ' + str(d) if d else ''};"
    rl = [r.get("rlx") for r in rs.values() if r.get("rlx")]
    L.append(f"{t:7s}{ex} geo steady {g(tw):6.1f} tok/s over {len(tw)}; kept per check {g([r['kept'] for r in rs.values()]):.2f}; "
             f"SSD MB per kept {g([r['mb'] for r in rs.values()]):.1f}" +
             (f"; forecast reads {pf('fc_issued')}, used {pf('fc_hits')}" if pf("fc_issued") else "") +
             (f"; relaxed accepts {sum(x.get('relaxed', 0) for x in rl)}" if rl else ""))
def ratio(a, b):
    r = [runs[b][k]["tpsw"] / runs[a][k]["tpsw"] for k in runs[a] if ok(a, k) and ok(b, k)]
    return f"{100 * (g(r) - 1):+.1f} % over {len(r)}" if r else "n/a"
L.append(f"lever 2 (exact): {ratio('l2cn', 'l2fn')}; QOM all-in vs U3C: {ratio('u3cn', 'qalln')}; + lever 2: "
         f"{ratio('qalln', 'qallfn')}; relaxed 0.5 on the all-in: {ratio('qalln', 'qallrn')} (geo steady, per prompt)")
open("data/bdn/bdn_report.txt", "w").write("\n".join(L) + "\n"); print("\n".join(L))
EOF
say "report rc=$? (data/bdn/bdn_report.txt)"
say "by-day C $CC end"
