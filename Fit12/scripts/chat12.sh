#!/bin/sh
> (a quote from the project owner, kept private)
# 24 and 12 gb"): Fit12 owns 12 GB chat + prefill. Three processes back to back under one quiet lock, AFTER CERT:
#  1) chat12: the 12 GB verified-greedy loop (arm ex, 12GB-CLEAN / tab12's footing) on BigMoE N45's 12 IFEval prompts (1220,
#     3280, 1561, 2853, 374, 3407, 2239, 2590, 2870, 2439, 1219, 1262), CAP 256 as N45: whole decode, full period, end to end,
#     tokens kept per check, SSD MB per kept token (the process's own reads inside the decode phases), lifetime peak; ids in
#     r57rows form for EvalSuite (chat12_ids.json).
#  2) pf12a / pf12b: the 12 GB PREFILL on BigMoE's PFX prompts (agent/edit10 1,565, agent/write13 4,730, agent/write11 7,932
#     tokens, ascending), arm ex, CAP 32: A = today's b12_prefill (F12_SMOE=4096: experts read once per layer per 4,096 rows),
#     B = one MoE call per prompt (F12_SMOE=8192: each expert read once per prompt, BigMoE's PFX form). TTFT, prefill tok/s,
#     the prefill's SSD reads, the lifetime peak per prompt (ascending: running max), decoded tokens A = B.
# GATE: /Users/kamen/EvalSuite/results/CERT_DONE exists (CERT's end; read-only), then 10 CLEAR minutes (1 once $O/GO exists; held
# while $O/WAIT exists): no other quiet lock / ticket, no other lock with MEM > 5 GB, no PAUSE / HOLD / ACT / ramp / closed lid,
# >= 16 GB free. A 30-min quiet lock (queue <= 60 min, else back to the gate), an idle GPU. Starts by DL (arg 1, default
# "2026-11-08 09:00") or not at all. Stops on swap +1 GB or 40 min. Writes $O/DONE at the end.
# Reports: c1u16_read.py -> $O/chat12_report.txt; pf12_read.py -> $O/pf12_report.txt. Out: /Users/kamen/Fit12/data/scr/chat12.
F=/Users/kamen/Fit12; PY=/Users/kamen/BigMoE/.venv/bin/python; LOG=$F/logs/summary.txt; O=$F/data/scr/chat12
Q=/Users/kamen/.local/share/quiet; PAUSEF=$Q/PAUSE; HOLDF=$Q/HOLD; HEALTH=$Q/HEALTH.md; CERTD=/Users/kamen/EvalSuite/results/CERT_DONE
IK=ifeval/1220,ifeval/3280,ifeval/1561,ifeval/2853,ifeval/374,ifeval/3407,ifeval/2239,ifeval/2590,ifeval/2870,ifeval/2439,ifeval/1219,ifeval/1262
PK=agent/edit10,agent/write13,agent/write11
DL=$(date -j -f "%Y-%m-%d %H:%M" "${1:-2026-11-08 09:00}" +%s) || exit 9
log() { echo "[$(date '+%a %H:%M:%S')] $*"; }
swapmb() { sysctl -n vm.swapusage | sed -E 's/.*used = ([0-9.]+)M.*/\1/' | cut -d. -f1; }
freegb() { vm_stat | awk '/Pages free|Pages inactive|Pages speculative|Pages purgeable/ {gsub(/\./,"",$NF); s+=$NF} END {printf "%d", s*16384/1e9}'; }
others() { peer ls 2>/dev/null | grep "^- "; }
locks() { grep -oE "· (shared run \(others may work\)|benchmarking|waiting for a quiet window)[^·]*"; }
nquiet() { others | locks | grep -cE "^· (benchmarking|waiting for a quiet window)"; }
nbig() { others | locks | grep -oE "MEM [~]?[0-9]+ GB" | tr -d '~' | awk '$2 > 5' | wc -l | tr -d " "; }
gpu() { peer top 2>/dev/null | head -1 | grep -o 'GPU [0-9]*%' | grep -o '[0-9]*'; }
act() { grep -q '^\*\*Verdict: ACT' "$HEALTH" 2>/dev/null; }
ramp() { pgrep -f '^/bin/zsh [^ ]*soft_end\.sh' >/dev/null; }
lid_closed() { ioreg -r -k AppleClamshellState -d 4 2>/dev/null | grep -m1 '"AppleClamshellState"' | grep -q Yes; }
blocked() { [ -e "$PAUSEF" ] || [ -e "$HOLDF" ] || [ -e "$O/WAIT" ] || act || ramp || lid_closed; }
cd $F || exit 9
mkdir -p $O
rm -f $O/DONE
log "chat12 armed: after $CERTD, 10 clear minutes (1 with $O/GO; held while $O/WAIT exists), start by $(date -r $DL '+%a %H:%M')"
while :; do
  clear=0
  while [ "$(date +%s)" -lt "$DL" ]; do
    if [ -e "$CERTD" ]; then
      q=$(nquiet); b=$(nbig); a=$(freegb)
      if [ "$q" -eq 0 ] && [ "$b" -eq 0 ] && ! blocked && [ "$a" -ge 16 ]; then clear=$((clear + 1)); else clear=0; fi
      need=10; [ -e $O/GO ] && need=1
      [ $clear -ge $((need * 2)) ] && break
    fi
    sleep 30
  done
  if [ "$(date +%s)" -ge "$DL" ]; then
    log "chat12: no clear slot by the deadline (CERT_DONE $([ -e $CERTD ] && echo yes || echo no)): NOT STARTED"
    echo "$(date +%H:%M:%S) SCR chat12 NOT STARTED: no clear slot by the deadline" >> $LOG
    echo "rc 5 not started $(date '+%a %H:%M:%S')" > $O/DONE
    exit 5
  fi
  log "chat12: CERT done, clear for $((clear / 2)) min ($a GB free): asking for the quiet lock"
  if ! PEER_QUEUE_TIMEOUT=3600 peer lock quiet --queue 30m "Fit12 12 GB chat (12 IFEval, verified greedy) + prefill A/B (1.6K / 4.7K / 7.9K, s_moe 4096 vs 8192), MEM 12 GB, ~20 min (NewIdeas 9c65385)" >/dev/null 2>&1; then
    log "chat12: no quiet lock within 60 min: back to the gate"; continue
  fi
  sleep 3; g=$(gpu); gp=100; gn=1                       # the previous holder's exit: two samples <= 10 % in a row, <= 2 min
  while { [ "${g:-0}" -gt 10 ] || [ "$gp" -gt 10 ]; } && [ $gn -lt 24 ]; do gp=${g:-0}; sleep 5; g=$(gpu); gn=$((gn + 1)); done
  if blocked || [ "$(freegb)" -lt 16 ]; then
    log "chat12: granted, but blocked or < 16 GB free ($(freegb) GB, GPU ${g}%): lock given back"; peer unlock >/dev/null 2>&1; continue
  fi
  break
done
SW0=$(swapmb); T0=$(date +%s); rc=0
echo "$(date +%H:%M:%S) SCR chat12 start: 12 GB chat (ex, 12 IFEval, CAP 256) + prefill A/B (ex, PFX prompts, s_moe 4096 / 8192), quiet lock ($(freegb) GB free, swap $SW0 MB, GPU ${g}%)" >> $LOG
{ echo "chat12 manifest $(date '+%a %d %b %H:%M:%S')"; shasum -a 256 $F/scripts/fit_t3.py $F/scripts/t3_twopass.py $F/scripts/t2_twopass.py \
    $F/pinned/8db7/bt_b12.py /Users/kamen/Leap/scripts/d7b_live.py; sed -n '/^ENV = /,/^for k in/p' $F/scripts/fit_t3.py; } > $O/MANIFEST.txt 2>&1
for J in "chat12|256|4096|$IK" "pf12a|32|4096|$PK" "pf12b|32|8192|$PK"; do
  T=$(echo "$J" | cut -d'|' -f1); C=$(echo "$J" | cut -d'|' -f2); S=$(echo "$J" | cut -d'|' -f3); KS=$(echo "$J" | cut -d'|' -f4)
  [ $rc -ne 0 ] && break
  log "chat12: $T (CAP $C, s_moe $S) start"
  env F12_OUT=$O F12_CAP=$C F12_SMOE=$S F12_PHLOG=1 F12_RERUN=1 F12_T3ARMS=ex F12_CMCL=47,32 F12_IDS_OUT=$O/${T}_ids.json F12_KEYS=$KS \
    $PY -B $F/scripts/fit_t3.py $T > $O/$T.txt 2>&1 &
  P=$!
  while kill -0 $P 2>/dev/null; do
    sleep 5
    [ $(( $(swapmb) - SW0 )) -gt 1024 ] && { rc=127; log "chat12: swap +1 GB: stopping"; }
    [ $(( $(date +%s) - T0 )) -ge 2400 ] && { rc=124; log "chat12: 40 min: stopping"; }
    [ $rc -ne 0 ] && { kill -TERM $P; sleep 5; kill -KILL $P 2>/dev/null; break; }
  done
  wait $P; w=$?; [ $rc -eq 0 ] && rc=$w
  log "chat12: $T end rc $w"
  echo "$(date +%H:%M:%S) SCR chat12 $T end (rc $w)" >> $LOG
done
peer unlock >/dev/null 2>&1
echo "$(date +%H:%M:%S) SCR chat12 end (rc $rc, $(( $(date +%s) - T0 )) s, swap $(swapmb) MB)" >> $LOG
echo "rc $rc started $(date -r $T0 '+%a %H:%M:%S') ended $(date '+%a %H:%M:%S')" > $O/DONE
grep -E 'Error|Traceback' $O/chat12.txt $O/pf12a.txt $O/pf12b.txt 2>/dev/null | head -3
$PY -I -B $F/scripts/c1u16_read.py $O/chat12.txt chat12 ex /nonexistent none ex > $O/chat12_report.txt 2>&1
$PY -I -B $F/scripts/pf12_read.py $O pf12a pf12b > $O/pf12_report.txt 2>&1
grep -E "^(WHOLE|FULL|END TO END|SSD|Per check|Lifetime|PAUSE|Harness)" $O/chat12_report.txt | cut -c1-250
cat $O/pf12_report.txt | cut -c1-250
exit $rc
