#!/bin/sh
# Fit12 (Sat 7 Nov 10:5x; the orchestrator 10:4x "optional, only if cheap; measure it separately, not inside the gate", 09:0x
# "when the hold lifts"): the WIRED-LIMIT A/B (Leap's FIRSTCHK fix, BigMoE's R57_WIRED form: mx.set_wired_limit(
# max_recommended_working_set_size) once after load) on the 12 GB FAST candidate t80ck1u, on c1u16's footing (16 HumanEval,
# cap 256, C_m 47 / C_l 32, F12_RERUN=1, F12_PHLOG=1). Two processes back to back under one quiet lock: W (F12_WIRED=1) first,
# then B (any warm-start advantage goes to the base). [P] from c1u16's phase logs: check 0 costs 3.6 s more than the later
# same-mode checks over the 16 prompts (4.7 % of decode; ~500 ms when check 0 goes exact), so W <= +4.9 % whole decode.
# Tokens W = B expected (wiring does not change the arithmetic).
# GATE: $O/GO must exist (touched once the usage hold lifts), then 10 clear minutes (no other quiet lock / ticket, no other
# lock with MEM > 5 GB, no PAUSE / HOLD / ACT / ramp / closed lid, >= 16 GB free), a 20-min quiet lock (queue <= 60 min) and an
# idle GPU. Starts by DL (arg 1, default "2026-11-08 12:00") or not at all. Stops on swap +1 GB or 25 min.
# Report: c1u16_read.py per run + /Users/kamen/Fit12/scripts/wab_cmp.py. Out: /Users/kamen/Fit12/data/scr/wab. Run ATTACHED.
F=/Users/kamen/Fit12; PY=/Users/kamen/BigMoE/.venv/bin/python; LOG=$F/logs/summary.txt; O=$F/data/scr/wab
K=humaneval/HumanEval; Q=/Users/kamen/.local/share/quiet; PAUSEF=$Q/PAUSE; HOLDF=$Q/HOLD; HEALTH=$Q/HEALTH.md
DL=$(date -j -f "%Y-%m-%d %H:%M" "${1:-2026-11-08 12:00}" +%s) || exit 9
KEYS=$K/156,$K/42,$K/68,$K/91,$K/16,$K/6,$K/79,$K/19,$K/161,$K/17,$K/56,$K/141,$K/101,$K/63,$K/114,$K/77
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
blocked() { [ -e "$PAUSEF" ] || [ -e "$HOLDF" ] || act || ramp || lid_closed; }
cd $F || exit 9
mkdir -p $O
rm -f $O/DONE
log "wab armed: needs $O/GO, then 10 clear minutes; start by $(date -r $DL '+%a %H:%M')"
while :; do
  clear=0
  while [ "$(date +%s)" -lt "$DL" ]; do
    if [ -e $O/GO ]; then
      q=$(nquiet); b=$(nbig); a=$(freegb)
      if [ "$q" -eq 0 ] && [ "$b" -eq 0 ] && ! blocked && [ "$a" -ge 16 ]; then clear=$((clear + 1)); else clear=0; fi
      [ $clear -ge 20 ] && break
    fi
    sleep 30
  done
  if [ "$(date +%s)" -ge "$DL" ]; then
    log "wab: no clear slot by the deadline (GO $([ -e $O/GO ] && echo yes || echo no)): NOT STARTED"
    echo "$(date +%H:%M:%S) SCR wab NOT STARTED: no slot by the deadline" >> $LOG
    echo "rc 5 not started $(date '+%a %H:%M:%S')" > $O/DONE
    exit 5
  fi
  log "wab: clear for 10 min ($a GB free): asking for the quiet lock"
  if ! PEER_QUEUE_TIMEOUT=3600 peer lock quiet --queue 20m "Fit12 wired-limit A/B on the 12 GB loop (t80ck1u W then B, 16 HumanEval), MEM 12 GB, ~7 min (orchestrator 10:4x)" >/dev/null 2>&1; then
    log "wab: no quiet lock within 60 min: back to the gate"; continue
  fi
  sleep 3; g=$(gpu); gp=100; gn=1
  while { [ "${g:-0}" -gt 10 ] || [ "$gp" -gt 10 ]; } && [ $gn -lt 24 ]; do gp=${g:-0}; sleep 5; g=$(gpu); gn=$((gn + 1)); done
  if blocked || [ "$(freegb)" -lt 16 ]; then
    log "wab: granted, but blocked or < 16 GB free ($(freegb) GB, GPU ${g}%): lock given back"; peer unlock >/dev/null 2>&1; continue
  fi
  break
done
SW0=$(swapmb); T0=$(date +%s); rc=0
echo "$(date +%H:%M:%S) SCR wab start: fit_t3.py t80ck1u W (F12_WIRED=1) then B, 16 HumanEval prompts, quiet lock ($(freegb) GB free, swap $SW0 MB, GPU ${g}%)" >> $LOG
{ echo "wab manifest $(date '+%a %d %b %H:%M:%S')"; shasum -a 256 $F/scripts/fit_t3.py $F/scripts/t3_twopass.py $F/scripts/t2_twopass.py \
    $F/pinned/8db7/bt_b12.py /Users/kamen/Leap/scripts/d7b_live.py; sed -n '/^ENV = /,/^for k in/p' $F/scripts/fit_t3.py; } > $O/MANIFEST.txt 2>&1
for TW in wabw:1 wabb:0; do
  T=${TW%%:*}; W=${TW##*:}
  [ $rc -ne 0 ] && break
  log "wab: $T (F12_WIRED=$W) start"
  env F12_OUT=$O F12_CAP=256 F12_PHLOG=1 F12_RERUN=1 F12_T3ARMS=t80ck1u F12_CMCL=47,32 F12_WIRED=$W F12_KEYS=$KEYS \
    $PY -B $F/scripts/fit_t3.py $T > $O/$T.txt 2>&1 &
  P=$!
  while kill -0 $P 2>/dev/null; do
    sleep 5
    [ $(( $(swapmb) - SW0 )) -gt 1024 ] && { rc=127; log "wab: swap +1 GB: stopping"; }
    [ $(( $(date +%s) - T0 )) -ge 1500 ] && { rc=124; log "wab: 25 min: stopping"; }
    [ $rc -ne 0 ] && { kill -TERM $P; sleep 5; kill -KILL $P 2>/dev/null; break; }
  done
  wait $P; w=$?; [ $rc -eq 0 ] && rc=$w
  log "wab: $T end rc $w"
done
peer unlock >/dev/null 2>&1
echo "$(date +%H:%M:%S) SCR wab end (rc $rc, $(( $(date +%s) - T0 )) s, swap $(swapmb) MB)" >> $LOG
echo "rc $rc started $(date -r $T0 '+%a %H:%M:%S') ended $(date '+%a %H:%M:%S')" > $O/DONE
grep -E 'Error|Traceback' $O/wabw.txt $O/wabb.txt 2>/dev/null | head -3
grep -h -E '^\[wab[wb]\] (WIRED|TOTAL|done; lifetime|PAUSE summary)' $O/wabw.txt $O/wabb.txt 2>/dev/null | cut -c1-240
for T in wabw wabb; do $PY -I -B $F/scripts/c1u16_read.py $O/$T.txt $T t80ck1u $F/data/scr/ex16 ex16 ex > $O/${T}_report.txt 2>&1; done
$PY -I -B $F/scripts/wab_cmp.py $O wabw wabb 2>&1 | tee $O/wab_cmp.txt
exit $rc
