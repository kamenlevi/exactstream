#!/bin/bash
# Thu ~14:15 (the orchestrator Thu 10:5x): a GPU pre-flight of the Fri 00:32 ticket's new runtime paths, one HumanEval prompt
# at CAP 32 per arm type, C 160 uniform, under a share lock "MEM 36 GB" in the gap between EvalSuite's runner 43 (pid 62942)
# and runner 44 (held: the orchestrator renamed q44/GO). Arms: pu3c / pqall / ps99 (R57_QOM_SPLIT=0.99) on the frozen screen,
# pqallr (R57_RELAX_DELTA 0.5) / pspan (R57_RELAX_SPAN) on the relaxed copy (both modules copied beside its snapshot, as
# fri_night.sh does), pexr (the exact default, R57_VTIER=). Checks per arm: it ran (runs.json), checks.json carries uses / stl,
# the relaxed arms report rlx / spn, no Traceback / AssertionError in its log.
# Start only when pid 62942 is gone and no other process holds > 5 GB (and no peer lock line says MEM >= 5 GB), and only by
# 14:35; on EVERY exit path touch ~/EvalSuite/results/q44/GO (runner 44 follows at once; Part 2 must end before 16:30).
# usage (a harness background task): bash scripts/preflight_fri.sh; touch ~/EvalSuite/results/q44/GO
export PATH="$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:$PATH"
cd "${PF_ROOT:-$HOME/BigMoE}" || exit 1
mkdir -p data/pf data/r57/snap
LOG="$PWD/data/pf/pf.log"
say() { echo "$(date '+%m-%d %H:%M:%S') $*" >> "$LOG"; }
GO="${PF_GO:-$HOME/EvalSuite/results/q44/GO}"
LOCKED=
finish() { [ -n "$LOCKED" ] && peer unlock >> "$LOG" 2>&1; touch "$GO"; say "q44/GO touched (exit path: $1)"; }
trap 'finish "signal or error"' EXIT
W43=${PF_PID:-62942}; LAST=${PF_LAST:-1435}
big_other() { [ -n "$PF_NOBIG" ] && return 1; [ "$(ps -axo rss= | awk '$1 > 5000000' | wc -l | tr -d ' ')" -gt 0 ] \
  || peer ls 2>/dev/null | grep -E 'shared run|benchmarking: quiet lock' | grep -qE 'MEM ([5-9]|[1-9][0-9]+) GB'; }
say "pre-flight armed: waiting for pid $W43 to exit and no other > 5 GB job, start by $LAST"
while kill -0 "$W43" 2>/dev/null || big_other; do
  [ "$(date +%H%M)" -ge "$LAST" ] && { say "SKIPPED: not clear by $LAST (pid $W43 alive: $(kill -0 $W43 2>/dev/null && echo yes || echo no))"; trap - EXIT; finish "skip"; exit 0; }
  sleep 20
done
[ -n "$PF_NOLOCK" ] || peer lock share 20m --cpu 300 --gpu 100 "BigMoE pre-flight for Fri: 1 HumanEval prompt per arm type, CAP 32, C 160, MEM 36 GB" >> "$LOG" 2>&1 \
  && LOCKED=$([ -n "$PF_NOLOCK" ] || echo 1) || { say "share lock refused: skipped"; trap - EXIT; finish "lock refused"; exit 0; }
cp scripts/r57_screen.py data/r57/snap/r57_screen_pf.py && cp scripts/r57_screen_rlx.py data/r57/snap/r57_screen_pfr.py \
  && cp scripts/relax_delta.py scripts/relax_span.py data/r57/snap/ && cp scripts/c45_runner.sh data/r57/snap/c45_runner_pf.sh || exit 2
RUN="/bin/bash data/r57/snap/c45_runner_pf.sh"; FS=data/r57/snap/r57_screen_pf.py; RS=data/r57/snap/r57_screen_pfr.py
export C45_OUT="$PWD/data/pf" C45_SWAP0=$(sysctl -n vm.swapusage | awk '{gsub("M","",$6); print $6 / 1024}') C45_ALARM=600 MG=4 \
  C45_CAP=32 KEEP_LOCK=1 R57_KEYS=humaneval/HumanEval/68 C45_DEADLINE=$(date -v+25M +%H%M)
P="R57_LAYSZ=0,BIGMOE_TIER_CL=64"
U="$P,BIGMOE_EXT_BUILD=build_qom,R57_FAST=U3C,R57_U3C_LEN=1,R57_REFTOK="
QA="$P,BIGMOE_EXT_BUILD=build_qom,R57_REFTOK=,R57_FAST=QOM,R57_D17=1,R57_D7B=code,R57_HC8=1,REF2_HC8_FAST=1,R57_DNR=1,R57_HCB=1"
QA="$QA,REF2_SMALL_NATIVE=1,R57_D7BNAX=1,REF2_OOO_1C=1"
say "pre-flight start (pid $$, NO-SWAP $(vm_stat | awk '/page size of/ {ps=$8} /^Pages (free|purgeable):/ || /^File-backed pages:/ {v=$NF; sub("[.]","",v); s+=v} END {printf "%d", s*ps/1e9}') GB)"
for a in "pu3c $FS $U" "pqall $FS $QA" "ps99 $FS $QA,R57_QOM_SPLIT=0.99" \
         "pqallr $RS $QA,R57_RELAX_DELTA=0.5,R57_RELAX_LOG=$PWD/data/pf/relax.jsonl" \
         "pspan $RS $QA,R57_RELAX_SPAN=1,R57_RELAX_LOG=$PWD/data/pf/span.jsonl" "pexr $FS $P,R57_VTIER="; do
  set -- $a
  C45_SCREEN=$2 C45_REFTOK= C45_ARMS="$1_160:160:$3" $RUN
  say "$1 rc=$?"
done
"${PF_PY:-$HOME/BigMoE/.venv/bin/python}" -I - >> "$LOG" 2>&1 <<'EOF'
import json, os
L = ["PRE-FLIGHT (data/pf): one HumanEval prompt, CAP 32, C 160 uniform, per arm type"]
for t, extra in (("pu3c", None), ("pqall", None), ("ps99", None), ("pqallr", "rlx"), ("pspan", "spn"), ("pexr", None)):
    v = []
    try:
        rs = json.load(open(f"data/r57/{t}_160_runs.json")); v.append(f"ran {len(rs)}")
        if extra:
            v.append(f"{extra} {'present' if any(r.get(extra) for r in rs) else 'MISSING'}")
    except OSError:
        v.append("NO runs.json")
    try:
        ck = json.load(open(f"data/r57/{t}_160_checks.json"))
        v.append("uses / stl " + ("present" if ck and "uses" in ck[0] and "stl" in ck[0] else "MISSING"))
        if ck and "uses" in ck[0]:
            v.append(f"uses per check {sum(sum(e['uses']) for e in ck) / max(1, sum(len(e['uses']) for e in ck)):.0f}")
    except OSError:
        v.append("NO checks.json")
    try:
        txt = open(f"data/pf/{t}_160.txt").read()
        bad = [x for x in ("Traceback", "AssertionError", "Error:") if x in txt]
        v.append("log clean" if not bad else "LOG HAS " + "/".join(bad))
    except OSError:
        v.append("no log")
    ok = all(("MISSING" not in x and "NO " not in x and "LOG HAS" not in x and x != "no log") for x in v)
    L.append(f"{t:7s} {'OK  ' if ok else 'FAIL'} " + "; ".join(v))
open("data/pf/pf_report.txt", "w").write("\n".join(L) + "\n"); print("\n".join(L))
EOF
say "pre-flight end"
trap - EXIT; finish "done"
