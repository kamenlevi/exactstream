#!/bin/bash
# Fri 6 Nov ~08:5x (the orchestrator): a ~5-min share pre-flight of tonight's STACK and STACKR with Leap's D7BNAX_32=1 (d7b_nax
# dd9066d3) on r57_screen_fast.py: 2 prompts (HumanEval/68 + ifeval/1220), CAP 32, C 160 uniform, no volatile tier, once the Mac
# is clear (no quiet lock, no other > 5 GB job, NO-SWAP >= 40 GB for 3 polls), start by 14:15. Clean = both prompts ran and the
# arm's log has no Traceback / AssertionError / Error:.
export PATH="$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:$PATH"
cd "$HOME/BigMoE" || exit 1
mkdir -p data/pf3 data/r57/snap
LOG="$PWD/data/pf3/pf3.log"; say() { echo "$(date '+%m-%d %H:%M:%S') $*" >> "$LOG"; }
nsw() { vm_stat | awk '/page size of/ {ps=$8} /^Pages (free|speculative|purgeable):/ || /^File-backed pages:/ {v=$NF; sub("[.]","",v); s+=v} END {printf "%d", s*ps/1e9}'; }
busy() { [ "$(ps -axo rss= | awk '$1 > 5000000' | wc -l | tr -d ' ')" -gt 0 ] || [ "$(nsw)" -lt 40 ] \
  || peer ls 2>/dev/null | grep -qE 'benchmarking: quiet lock|waiting for a quiet window' \
  || peer ls 2>/dev/null | grep -E 'shared run' | grep -qE 'MEM ([5-9]|[1-9][0-9]+) GB'; }
big() { [ "$(ps -axo rss= | awk '$1 > 5000000' | wc -l | tr -d ' ')" -gt 0 ] || [ "$(nsw)" -lt 40 ]; }
while :; do nc=0; while [ $nc -lt 3 ]; do if busy; then nc=0; else nc=$((nc + 1)); fi
    [ "$(date +%H%M)" -ge 1415 ] && { say "SKIPPED: not clear by 14:15"; exit 0; }; [ $nc -lt 3 ] && sleep 30; done
  peer lock share 12m --cpu 300 --gpu 100 "BigMoE pre-flight: stack + D7BNAX_32 (2 prompts, CAP 32, C 160), MEM 36 GB" >> "$LOG" 2>&1 \
    || { say "share lock refused"; exit 0; }
  sleep 20; big || break   # Fri 11:25: two waiters on one clear Mac started together (2 x 31 GB, swap +3 GB): look again
  peer unlock >> "$LOG" 2>&1; say "a big job started beside us after the lock: back to waiting"; done
trap 'peer unlock >> "$LOG" 2>&1' EXIT
cp scripts/r57_screen_fast.py data/r57/snap/r57_screen_pf3.py && cp scripts/relax_delta.py scripts/relax_span.py scripts/inert_span.py \
  data/r57/snap/ && cp scripts/c45_runner.sh data/r57/snap/c45_runner_pf3.sh || exit 2
export C45_OUT="$PWD/data/pf3" C45_SWAP0=$(sysctl -n vm.swapusage | awk '{gsub("M","",$6); print $6 / 1024}') C45_ALARM=600 MG=4 \
  C45_CAP=32 KEEP_LOCK=1 R57_KEYS=humaneval/HumanEval/68,ifeval/1220 C45_DEADLINE=$(date -v+12M +%H%M) \
  C45_SCREEN=data/r57/snap/r57_screen_pf3.py C45_REFTOK=
QA="R57_LAYSZ=0,BIGMOE_TIER_CL=64,BIGMOE_EXT_BUILD=build_qom,R57_REFTOK=,R57_FAST=QOM,R57_D17=1,R57_D7B=code,R57_HC8=1"
QA="$QA,REF2_HC8_FAST=1,R57_DNR=1,R57_HCB=1,REF2_SMALL_NATIVE=1,R57_D7BNAX=1,REF2_OOO_1C=1,R57_VTIER="
L1="R57_NAX=1,REF2_NAX_1CQ=1,REF2_DN_KS=1,REF2_GDN_LAZY=1,D7BNAX_32=${PD_D7:-1},R57_FASTTAX=1"   # PD_D7=2: Leap d9e32317
rm -f data/r57/f3*_160_* data/pf3/f3?_160.txt; say "D7BNAX_32=${PD_D7:-1}"
say "start (pid $$, NO-SWAP $(nsw) GB)"
for a in "f3k:$QA,$L1,R57_INERT=CD16" "f3r:$QA,$L1,R57_RELAX_DELTA=0.5,R57_RELAX_SPAN=1,R57_RELAX_LOG=$PWD/data/pf3/r.jsonl"; do
  C45_ARMS="${a%%:*}_160:160:${a#*:}" /bin/bash data/r57/snap/c45_runner_pf3.sh
  n=$(grep -o '"key"' "data/r57/${a%%:*}_160_runs.json" 2>/dev/null | wc -l | tr -d ' ')
  bad=$(grep -cE 'Traceback|AssertionError|Error:' "data/pf3/${a%%:*}_160.txt" 2>/dev/null)
  say "${a%%:*}: ran ${n:-0} / 2; error lines ${bad:-none}; $([ "${n:-0}" = 2 ] && [ "${bad:-0}" = 0 ] && echo CLEAN || echo NOT CLEAN)"
done
say "end"
