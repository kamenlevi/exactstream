#!/bin/bash
# Fri 6 Nov 21:1x: GOAL's guarded CD16 forms G16 and D16 (inert_span.py) on the stack, never run on the GPU before: 2 prompts (HumanEval/68, mbpp/16),
# and on the stack, 2 prompts CAP 32 C 160; clean + the nw counters (late groups > 0 cold, layer 47 ends empty). (was: d7b_nax
# dd9066d3) on r57_screen_fast.py: 2 prompts (HumanEval/68 + ifeval/1220), CAP 32, C 160 uniform, no volatile tier, once the Mac
# is clear (no quiet lock, no other > 5 GB job, NO-SWAP >= 40 GB for 3 polls), start by 14:15. Clean = both prompts ran and the
# arm's log has no Traceback / AssertionError / Error:.
export PATH="$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:$PATH"
cd "$HOME/BigMoE" || exit 1
mkdir -p data/pf7 data/r57/snap
LOG="$PWD/data/pf7/pf7.log"; say() { echo "$(date '+%m-%d %H:%M:%S') $*" >> "$LOG"; }
nsw() { vm_stat | awk '/page size of/ {ps=$8} /^Pages (free|speculative|purgeable):/ || /^File-backed pages:/ {v=$NF; sub("[.]","",v); s+=v} END {printf "%d", s*ps/1e9}'; }
busy() { [ "$(ps -axo rss= | awk '$1 > 5000000' | wc -l | tr -d ' ')" -gt 0 ] || [ "$(nsw)" -lt 40 ] \
  || peer ls 2>/dev/null | grep -qE 'benchmarking: quiet lock|waiting for a quiet window' \
  || peer ls 2>/dev/null | grep -E 'shared run' | grep -qE 'MEM ([5-9]|[1-9][0-9]+) GB'; }
big() { [ "$(ps -axo rss= | awk '$1 > 5000000' | wc -l | tr -d ' ')" -gt 0 ] || [ "$(nsw)" -lt 40 ]; }
while :; do nc=0; while [ $nc -lt 3 ]; do if busy; then nc=0; else nc=$((nc + 1)); fi
    [ "$(date +%H%M)" -ge 2359 ] && { say "SKIPPED: not clear by 23:59"; exit 0; }; [ $nc -lt 3 ] && sleep 30; done
  peer lock share 12m --cpu 300 --gpu 100 "BigMoE pre-flight: G16 + D16 guarded CD16 (2 prompts, CAP 32, C 160), MEM 36 GB" >> "$LOG" 2>&1 \
    || { say "share lock refused"; exit 0; }
  sleep 20; big || break   # Fri 11:25: two waiters on one clear Mac started together (2 x 31 GB, swap +3 GB): look again
  peer unlock >> "$LOG" 2>&1; say "a big job started beside us after the lock: back to waiting"; done
trap 'peer unlock >> "$LOG" 2>&1' EXIT
cp scripts/r57_screen_fast.py data/r57/snap/r57_screen_pf7.py && cp scripts/relax_delta.py scripts/relax_span.py scripts/inert_span.py \
  data/r57/snap/ && cp scripts/c45_runner.sh data/r57/snap/c45_runner_pf7.sh || exit 2
export C45_OUT="$PWD/data/pf7" C45_SWAP0=$(sysctl -n vm.swapusage | awk '{gsub("M","",$6); print $6 / 1024}') C45_ALARM=600 MG=4 \
  C45_CAP=32 KEEP_LOCK=1 R57_KEYS=humaneval/HumanEval/68,mbpp/16 R57_QOM_SETS=humaneval,mbpp C45_DEADLINE=$(date -v+12M +%H%M) \
  C45_SCREEN=data/r57/snap/r57_screen_pf7.py C45_REFTOK=
QA="R57_LAYSZ=0,BIGMOE_TIER_CL=64,BIGMOE_EXT_BUILD=build_qom,R57_REFTOK=,R57_FAST=QOM,R57_D17=1,R57_D7B=code,R57_HC8=1"
QA="$QA,REF2_HC8_FAST=1,R57_DNR=1,R57_HCB=1,REF2_SMALL_NATIVE=1,R57_D7BNAX=1,REF2_OOO_1C=1,R57_VTIER="
L1="R57_NAX=1,REF2_NAX_1CQ=1,REF2_DN_KS=1,REF2_GDN_LAZY=1,D7BNAX_32=${PD_D7:-1},R57_FASTTAX=1"   # PD_D7=2: Leap d9e32317
rm -f data/r57/f7*_160_* data/pf7/f7?_160.txt
say "start (pid $$, NO-SWAP $(nsw) GB)"
NWE="R57_NEVERWAIT=1,REF2_NEVERWAIT=1"
for a in "f7g:$QA,$L1,R57_INERT=G16,R57_INERT_LOG=$PWD/data/pf7/g16.jsonl" "f7d:$QA,$L1,R57_INERT=D16,R57_INERT_LOG=$PWD/data/pf7/d16.jsonl"; do
  C45_ARMS="${a%%:*}_160:160:${a#*:}" /bin/bash data/r57/snap/c45_runner_pf7.sh
  n=$(grep -o '"key"' "data/r57/${a%%:*}_160_runs.json" 2>/dev/null | wc -l | tr -d ' ')
  bad=$(grep -cE 'Traceback|AssertionError|Error:' "data/pf7/${a%%:*}_160.txt" 2>/dev/null)
  say "${a%%:*}: ran ${n:-0} / 2; error lines ${bad:-none}; $([ "${n:-0}" = 2 ] && [ "${bad:-0}" = 0 ] && echo CLEAN || echo NOT CLEAN)"
  nwc=$(.venv/bin/python -I -c "import json,sys; c=json.load(open(sys.argv[1])); v=[x for e in c for x in e.get(\"nw\", [])]; s=[sum(x[i] for x in v) for i in range(6)] if v else []; print(len(v), \"checks; late groups\", s[0] if s else 0, \"late uses\", s[1] if s else 0, \"L47 drops\", s[3] if s else 0, \"L47 stalls\", s[4] if s else 0)" "data/r57/${a%%:*}_160_checks.json" 2>&1 | tail -1)
  say "${a%%:*} nw: $nwc"
done
say "end"
