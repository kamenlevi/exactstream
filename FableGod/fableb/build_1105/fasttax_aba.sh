#!/bin/bash
# FAST-TAX A-B-A (Fable B, ~/FableGod/fableb/build_1105, Thu 5 Nov; the orchestrator: "build everything so we get actual
# numbers"). Needs fasttax_r57.patch applied to scripts/r57_screen.py (this script checks). One run on fri_night.sh's 12
# HumanEval keys, FAST (no token check), one process per arm, ~2 min per arm (Thu 07:38 at C 208: 97-109 s):
#   A  fta1 = the QOM all-in FAST arm, exactly fri_night.sh's qall
#   B  ftb  = A + R57_FASTTAX=1 (ref2 att_single on; gdn_inproj is skipped under R57_DNR=1 and the log line says so)
#   A  fta2 = A again                      (FTA_PAL=1: the palindrome A B B A, arms fta1 ftb ftb2 fta2)
# C = the largest of 224 / 208 / 192 whose peak + 4 fits NO-SWAP at the start (fri_night.sh's rule); FTA_C overrides.
# Line (pre-registered Thu 5 Nov ~18:4x, before any run; the metric is the median check ms over all checks but each
# prompt's first, as c224_thu.sh reports it):
#   PASS    = B's median is >= 2.0 ms under the mean of the two A medians, the two A medians are within 2.0 ms of each
#             other, B's kept per check is within 1 % of A's, and ref2 counted att_single calls in B and none in A.
#   KILL    = the saving is < 1.0 ms with the two A medians within 2.0 ms.
#   VOID    = the two A medians differ by more than 2.0 ms (drift): rerun as the palindrome.
#   BETWEEN = anything else.
# Quality is not judged here (map arm + EvalSuite decision 4 judge a FAST arm); the report only says how many prompts kept
# identical tokens (att_single is bitwise == REF2-T's chunked calls below 512 keys).
# usage (a harness background task, as fri_night.sh): NOTBEFORE=hh:mm NOTAFTER=hh:mm WAITMEM=40 GAP0=0 GAP=0 MARGIN=4 \
#   bash scripts/turn45.sh 37 20 "BigMoE: FAST-TAX A-B-A" /bin/bash scripts/fasttax_aba.sh
#   -> data/fta/fta.log, data/fta/fta_report.txt (report only: FTA_REPORT_ONLY=1 FTA_C=<C> bash scripts/fasttax_aba.sh)
export PATH="$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:$PATH"
cd "${FTA_ROOT:-$HOME/BigMoE}" || exit 1
mkdir -p data/fta data/r57/snap
LOG="$PWD/data/fta/fta.log"
PY="${FTA_PY:-$HOME/BigMoE/.venv/bin/python}"
say() { echo "$(date '+%m-%d %H:%M:%S') $*" >> "$LOG"; }
swap_gb() { sysctl -n vm.swapusage | awk '{gsub("M","",$6); print $6 / 1024}'; }
avail_gb() { vm_stat | awk '/page size of/ {ps=$8} /^Pages (free|speculative|purgeable):/ || /^File-backed pages:/ {v=$NF; sub("[.]","",v); s+=v} END {printf "%.1f", s*ps/1e9}'; }
grep -q 'os.environ.get("R57_FASTTAX")' scripts/r57_screen.py || { say "scripts/r57_screen.py has no R57_FASTTAX: apply fasttax_r57.patch first"; echo "apply fasttax_r57.patch first" >&2; exit 2; }
[ -f scripts/fasttax_report.py ] || { say "scripts/fasttax_report.py missing"; exit 2; }
A=${FTA_AVAIL:-$(avail_gb)}
fits() { awk -v a="$A" -v n="$1" 'BEGIN {exit !(n <= a)}'; }
CC=${FTA_C:-}
[ -z "$CC" ] && for c in 224:44.3 208:42.2 192:40.0; do fits ${c#*:} && { CC=${c%%:*}; break; }; done
[ -z "$CC" ] && { say "NO-SWAP $A GB < 40.0: nothing fits, nothing runs"; exit 0; }
TAGS="fta1 ftb fta2"; [ -n "$FTA_PAL" ] && TAGS="fta1 ftb ftb2 fta2"
if [ -z "$FTA_REPORT_ONLY" ]; then
  cp scripts/r57_screen.py data/r57/snap/r57_screen_fta.py && cp scripts/c45_runner.sh data/r57/snap/c45_runner_fta.sh || exit 2
  RUN="/bin/bash data/r57/snap/c45_runner_fta.sh"
  FS=data/r57/snap/r57_screen_fta.py
  export C45_OUT="$PWD/data/fta" C45_SWAP0=$(swap_gb) C45_ALARM=900 MG=${FTA_MG:-4}
  export C45_DEADLINE=${FTA_DEADLINE:-$(date -v+20M +%H%M)}
  K="humaneval/HumanEval/68,humaneval/HumanEval/16,humaneval/HumanEval/91,humaneval/HumanEval/6,humaneval/HumanEval/79"
  K="$K,humaneval/HumanEval/19,humaneval/HumanEval/161,humaneval/HumanEval/17,humaneval/HumanEval/56,humaneval/HumanEval/141"
  K="$K,humaneval/HumanEval/101,humaneval/HumanEval/63"
  QA="BIGMOE_EXT_BUILD=build_qom,R57_REFTOK=,R57_FAST=QOM,R57_D17=1,R57_D7B=code,R57_HC8=1,REF2_HC8_FAST=1,R57_DNR=1,R57_HCB=1"
  QA="$QA,REF2_SMALL_NATIVE=1,R57_D7BNAX=1,REF2_OOO_1C=1"
  FT="$QA,R57_FASTTAX=${FTA_PIECES:-1}"
  say "FAST-TAX run start (pid $$, NO-SWAP $A GB -> C $CC, arms $TAGS, deadline $C45_DEADLINE, swap $C45_SWAP0 GB)"
  arm() {   # arm TAG ENV: one runner call
    R57_KEYS=$K C45_SCREEN=$FS C45_REFTOK= C45_ARMS="$1_$CC:$CC:$2" $RUN
    say "$1 rc=$? $(grep -h 'FAST-TAX' "data/fta/$1_$CC.txt" 2>/dev/null | head -2 | tr '\n' ' ' | cut -c1-220)"
  }
  for t in $TAGS; do
    case $t in ftb*) arm $t "$FT";; *) arm $t "$QA";; esac
  done
fi
"$PY" -I scripts/fasttax_report.py data/r57 "$CC" $TAGS > data/fta/fta_report.txt 2>> "$LOG"
say "report rc=$? (data/fta/fta_report.txt): $(tail -1 data/fta/fta_report.txt | cut -c1-200)"
say "FAST-TAX run end"
