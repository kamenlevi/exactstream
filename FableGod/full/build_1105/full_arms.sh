#!/bin/bash
# full_arms.sh (Fable FULL, LOSSY350 build_1105; INSTALL.md beside this file): the two tests ON THEIR OWN, for a ticket that
# is not scripts/fri_night.sh (if Friday's ticket has no slack). Same layout as fri_night.sh at BigMoE 6adad93: memory-adaptive
# C, frozen snapshots, c45_runner's arm lists, the 12 HumanEval prompts, CAP 256.
#   T-B1  fqall / fqallp / fqallpb / fqallb   A-P-P-A (pair means cancel linear drift); P = qall + R57_NOCOM=1
#   T-A1  fcprobe, after the timed arms (qall + the split switch + R57_COPYPROBE=g+x+s); FULL_CPROBE_T=1 adds fcprobet (g+t)
# then the two CPU reports -> data/full/nocom_report.txt, data/full/copyprobe_report.txt (and the log data/full/full.log).
# usage (inside a quiet GPU turn, as fri_night.sh):  NOTBEFORE=.. NOTAFTER=.. WAITMEM=40 GAP0=0 GAP=0 MARGIN=4 bash scripts/turn45.sh \
#   37 30 "BigMoE: FULL's T-B1 + T-A1" /bin/bash ~/FableGod/full/build_1105/full_arms.sh
# env: FULL_ROOT (~/BigMoE); FULL_TB1=0 / FULL_TA1=0 skip a test; FULL_SPLIT90 (default R57_QOM_SPLIT=0.9; empty = QOM's rows);
#      FULL_QA (the qall env list; default = fri_night.sh's at 6adad93); FULL_AVAIL / FULL_PY / FULL_MG / FULL_DEADLINE.
export PATH="$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:$PATH"
B1105="$(cd "$(dirname "$0")" && pwd)"
cd "${FULL_ROOT:-$HOME/BigMoE}" || exit 1
mkdir -p data/full data/r57/snap
LOG="$PWD/data/full/full.log"
PY="${FULL_PY:-$HOME/BigMoE/.venv/bin/python}"
say() { echo "$(date '+%m-%d %H:%M:%S') $*" >> "$LOG"; }
swap_gb() { sysctl -n vm.swapusage | awk '{gsub("M","",$6); print $6 / 1024}'; }
avail_gb() { vm_stat | awk '/page size of/ {ps=$8} /^Pages (free|speculative|purgeable):/ || /^File-backed pages:/ {v=$NF; sub("[.]","",v); s+=v} END {printf "%.1f", s*ps/1e9}'; }
A=${FULL_AVAIL:-$(avail_gb)}
fits() { awk -v a="$A" -v n="$1" 'BEGIN {exit !(n <= a)}'; }
CC=; for c in 224:44.3 208:42.2 192:40.0; do fits ${c#*:} && { CC=${c%%:*}; break; }; done
[ -z "$CC" ] && { say "NO-SWAP $A GB < 40.0: nothing fits, nothing runs"; exit 0; }
cp scripts/r57_screen.py data/r57/snap/r57_screen_full0.py && cp scripts/c45_runner.sh data/r57/snap/c45_runner_full.sh || exit 2
RUN="/bin/bash data/r57/snap/c45_runner_full.sh"
FS=data/r57/snap/r57_screen_full0.py; XS=data/r57/snap/r57_screen_fullx.py
"$PY" -I "$B1105/apply_screen.py" "$FS" "$XS" >> "$LOG" 2>&1 || { say "apply_screen failed (an anchor moved): nothing runs"; exit 3; }
export C45_OUT="$PWD/data/full" C45_SWAP0=$(swap_gb) C45_ALARM=900 MG=${FULL_MG:-4}
export C45_DEADLINE=${FULL_DEADLINE:-$(date -v+25M +%H%M)}
K="humaneval/HumanEval/68,humaneval/HumanEval/16,humaneval/HumanEval/91,humaneval/HumanEval/6,humaneval/HumanEval/79"
K="$K,humaneval/HumanEval/19,humaneval/HumanEval/161,humaneval/HumanEval/17,humaneval/HumanEval/56,humaneval/HumanEval/141"
K="$K,humaneval/HumanEval/101,humaneval/HumanEval/63"
QA="BIGMOE_EXT_BUILD=build_qom,R57_REFTOK=,R57_FAST=QOM,R57_D17=1,R57_D7B=code,R57_HC8=1,REF2_HC8_FAST=1,R57_DNR=1,R57_HCB=1"
QA="$QA,REF2_SMALL_NATIVE=1,R57_D7BNAX=1,REF2_OOO_1C=1"
QA="${FULL_QA:-$QA}"
S90=${FULL_SPLIT90-R57_QOM_SPLIT=0.9}
say "FULL arms start (pid $$, NO-SWAP $A GB -> C $CC, deadline $C45_DEADLINE, swap $C45_SWAP0 GB, split '${S90}', build $B1105)"
arm() {   # arm TAG SCREEN ENV: one runner call
  R57_KEYS=$K C45_SCREEN=$2 C45_REFTOK= C45_ARMS="$1_$CC:$CC:$3" $RUN
  say "$1 rc=$?"
}
if [ "${FULL_TB1:-1}" = 1 ]; then
  arm fqall $FS "$QA";   arm fqallp $XS "$QA,R57_NOCOM=1";   arm fqallpb $XS "$QA,R57_NOCOM=1";   arm fqallb $FS "$QA"
fi
if [ "${FULL_TA1:-1}" = 1 ]; then
  arm fcprobe $XS "$QA${S90:+,$S90},R57_COPYPROBE=g+x+s"
  [ -n "$FULL_CPROBE_T" ] && arm fcprobet $XS "$QA${S90:+,$S90},R57_COPYPROBE=g+t"
fi
[ "${FULL_TB1:-1}" = 1 ] && { "$PY" -I "$B1105/nocom_report.py" "$CC" data/full/nocom_report.txt >> "$LOG" 2>&1; say "nocom report rc=$?"; }
[ "${FULL_TA1:-1}" = 1 ] && { "$PY" -I "$B1105/copyprobe_report.py" "data/r57/fcprobe_${CC}_copyprobe.jsonl" data/full/copyprobe_report.txt >> "$LOG" 2>&1; say "copy probe report rc=$?"; }
say "FULL arms end"
