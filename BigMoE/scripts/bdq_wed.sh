#!/bin/bash
# QOM BY DAY, bare minimum (Kamen's build rules 15:1x; the orchestrator's order item 2): QOM in the engine (R57_FAST=QOM,
# ext/build_qom: Breakthrough's keep-set rule 6f1ca44 + the loader's optional uses, bit 21) at C 128 (45 preset, uniform
# pools, C_l 64; the smallest by-day config), share lock (scripts/guarded.sh). FAST: tokens differ by design.
#   0. smoke qs: QOM on 2 HumanEval, CAP 64: it must post optional uses and leave some unread, else stop
#   1. A-B-C, 12 HumanEval, CAP 256: u3cq (U3C len) / qomq (QOM) / qalq (QOM + D7b code + HC8 v4 + DENSE-N-RAM + hc_b;
#      no REF2_OOO_1C: QOM's -2 base slots need the two-phase OOO)
# usage: NEED=27 WAITUNTIL=1800 LOCKWHY="BigMoE: ... MEM 26 GB" bash scripts/guarded.sh 2400 data/bdq/bdq.out /bin/bash scripts/bdq_wed.sh
export PATH="$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:$PATH"
cd "$HOME/BigMoE" || exit 1
mkdir -p data/bdq
LOG="$HOME/BigMoE/data/bdq/bdq.log"
say() { echo "$(date '+%m-%d %H:%M:%S') $*" >> "$LOG"; }
swap_gb() { sysctl -n vm.swapusage | awk '{gsub("M","",$6); print $6 / 1024}'; }
cp scripts/r57_screen.py data/r57/snap/r57_screen_bdq.py && cp scripts/c45_runner.sh data/r57/snap/c45_runner_bdq.sh || exit 2
RUN="/bin/bash data/r57/snap/c45_runner_bdq.sh"
export C45_SCREEN=data/r57/snap/r57_screen_bdq.py C45_DEADLINE=${BDQ_DEADLINE:-1900} C45_OUT="$HOME/BigMoE/data/bdq" KEEP_LOCK=1
export C45_SWAP0=$(swap_gb)
CC=${BDQ_C:-128}
K="humaneval/HumanEval/68,humaneval/HumanEval/16,humaneval/HumanEval/91,humaneval/HumanEval/6,humaneval/HumanEval/79"
K="$K,humaneval/HumanEval/19,humaneval/HumanEval/161,humaneval/HumanEval/17,humaneval/HumanEval/56,humaneval/HumanEval/141"
K="$K,humaneval/HumanEval/101,humaneval/HumanEval/63"
B="BIGMOE_EXT_BUILD=build_qom,R57_LAYSZ=0,BIGMOE_TIER_CL=64,R57_REFTOK="
U="$B,R57_FAST=U3C,R57_U3C_LEN=1"
Q="$B,R57_FAST=QOM"
QA="$Q,R57_D7B=code,R57_HC8=1,REF2_HC8_FAST=1,R57_DNR=1,R57_HCB=1"
say "QOM by day C $CC start (pid $$, swap $C45_SWAP0 GB)"
R57_KEYS="humaneval/HumanEval/68,humaneval/HumanEval/91" MG=${BDQ_MG:-2} C45_ALARM=600 C45_CAP=64 C45_ARMS="qs_$CC:$CC:$Q" $RUN
SM=$(.venv/bin/python -I - "$CC" 2>&1 <<'EOF'
import json, sys
try:
    rs = json.load(open(f"data/r57/qs_{sys.argv[1]}_runs.json"))
except OSError as e:
    print(f"FAIL no runs ({e})"); sys.exit()
pf = lambda k: sum(r.get("pf", {}).get(k, 0) for r in rs)
print(("OK" if len(rs) == 2 and pf("qom_opt") > 0 else "FAIL") + f": runs {len(rs)}, optional uses {pf('qom_opt')}, "
      f"optional-only misses not read {pf('qom_skip')}, tok/s {[round(r['tps'], 1) for r in rs]}")
EOF
)
say "smoke: $SM"
if [ "${SM:0:2}" != "OK" ]; then say "smoke failed: arms not run"; say "QOM by day end"; exit 3; fi
R57_KEYS=$K MG=${BDQ_MG:-2} C45_ALARM=900 C45_ARMS="u3cq_$CC:$CC:$U qomq_$CC:$CC:$Q qalq_$CC:$CC:$QA" $RUN
say "arms rc=$?"
.venv/bin/python -I - "$CC" >> "$LOG" 2>&1 <<'EOF'
import json, math, statistics as S, sys
C = sys.argv[1]
g = lambda xs: math.exp(sum(map(math.log, xs)) / len(xs)) if xs else float("nan")
T = ("u3cq", "qomq", "qalq")
runs, chk = {}, {}
for t in T:
    try:
        runs[t] = {r["key"]: r for r in json.load(open(f"data/r57/{t}_{C}_runs.json"))}
        chk[t] = json.load(open(f"data/r57/{t}_{C}_checks.json"))
    except OSError:
        runs[t], chk[t] = {}, []
ok = lambda t, k: k in runs[t] and "PAUSED" not in runs[t][k].get("bad", "")
F = ("ck", "drf", "io", "svc", "wt", "bld")
L = [f"QOM BY DAY, C {C} (45 preset, uniform pools, C_l 64; ext/build_qom; shared GPU, apps open: rough, relative within this run)"]
for t in T:
    rs = runs[t]
    if not rs:
        L.append(f"{t:5s} MISSING"); continue
    tw = [r["tpsw"] for k, r in rs.items() if ok(t, k)]
    md = {f: S.median([x for e in chk[t] if not e.get("paused") for x in e.get(f, [])[1:]] or [float("nan")]) for f in F}
    pf = lambda x: sum(r.get("pf", {}).get(x, 0) for r in rs.values())
    L.append(f"{t:5s} geo steady {g(tw):6.1f} tok/s over {len(tw)}; kept per check {g([r['kept'] for r in rs.values()]):.2f}; SSD MB per "
             f"kept {g([r['mb'] for r in rs.values()]):.1f}; lifetime max {max(r['mem'][-1] for r in rs.values()):.1f} GB; medians per check "
             "(ms): " + ", ".join(f"{f} {md[f]:.1f}" for f in F) +
             (f"; QOM optional uses {pf('qom_opt')}, not read {pf('qom_skip')}" if pf("qom_opt") else ""))
def ratio(a, b):
    r = [runs[b][k]["tpsw"] / runs[a][k]["tpsw"] for k in runs[a] if ok(a, k) and ok(b, k)]
    return f"{100 * (g(r) - 1):+.1f} % over {len(r)}" if r else "n/a"
def mbr(a, b):
    r = [runs[b][k]["mb"] / runs[a][k]["mb"] for k in runs[a] if k in runs[b]]
    return f"x{g(r):.3f}" if r else "n/a"
L.append(f"QOM vs U3C: {ratio('u3cq', 'qomq')} steady, SSD MB per kept {mbr('u3cq', 'qomq')}; QOM all-in vs U3C: "
         f"{ratio('u3cq', 'qalq')}, MB {mbr('u3cq', 'qalq')}; all-in pieces on QOM: {ratio('qomq', 'qalq')}")
open("data/bdq/bdq_report.txt", "w").write("\n".join(L) + "\n"); print("\n".join(L))
EOF
say "report rc=$? (data/bdq/bdq_report.txt)"
say "QOM by day end"
