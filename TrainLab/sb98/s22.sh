#!/bin/bash
# WEEK98 S22 QATC2 runner (MTP Head; REPORT 'WEEK98 S22 QATC2'; orchestrator 14:3x placement).
#   s22.sh smoke        : dd_qat.py --smoke 5, then the harness gate (sb98/m3_tree.py --cut none on HE+ 24 == BigMoE's dump)
#   s22.sh qa MARKER    : waits for MARKER (the Wed-night capture's END file), then t30qa (1,479 QAT steps), its M3 read-out
#                         at cut C2 on both sets, and the score vs the full t30b (sb98/m3_score.py).
# Gates (the orchestrator): Kamen away >= 900 s; NO-SWAP >= 24 GB (peak 16 + 8); no quiet lock of others; no PAUSE; no other
# >= 16 GB lock (one > 16 GB job at a time). Training yields (SIGTERM: dd_qat saves, exits 3) on NO-SWAP < 5 GB twice, a quiet
# lock of others or a PAUSE, and resumes when the gates pass. qa starts no later than Thu 05:00 (else exit 5).
set -u
MODE=${1:?smoke or qa}; MARK=${2:-}
D=$HOME/TrainLab/data/s22; S=$HOME/TrainLab/sb98; PY=$HOME/BigMoE/.venv/bin/python
source "$HOME/Sandbox98/scripts/guard_lib.sh"      # quiet_others, big_others, noswap, swap_mb (read-only use)
AWAY=$HOME/.local/share/quiet/away_secs.sh; PAUSEF=$HOME/.local/share/quiet/PAUSE
log() { echo "[$(date '+%a %H:%M:%S')] $*"; }
gate() { [ "$($AWAY)" -ge 900 ] && [ "$(noswap)" -ge 24 ] && [ "$(quiet_others)" -eq 0 ] && [ ! -f "$PAUSEF" ] \
         && [ "$(GUARD_BIG_GB=16 big_others)" -eq 0 ]; }
wait_gate() { local n=0; until gate; do
  [ $((n % 20)) -eq 0 ] && log "$1 waits: away $($AWAY) s, NO-SWAP $(noswap) GB, quiet $(quiet_others), big $(GUARD_BIG_GB=16 big_others)"
  n=$((n + 1)); sleep 30; done; }
lock() { peer lock share $1m --cpu 150 --gpu 60 "MTP Head: S22 $2, MEM $3 GB" 2>&1 | tr '\n' ' ' | cut -c1-140; }
mkdir -p $D
run_guarded() {   # $1 = name, $2 = log, $3.. = command; yields per the rules above; returns the command's exit
  local name=$1 lg=$2; shift 2
  local P low=0 why=""
  "$@" >> "$lg" 2>&1 &
  P=$!
  while kill -0 $P 2>/dev/null; do
    sleep 20
    if [ "$(noswap)" -lt 5 ]; then low=$((low + 1)); else low=0; fi
    if [ $low -ge 2 ]; then why="NO-SWAP < 5 GB twice"; elif [ "$(quiet_others)" -gt 0 ]; then why="a quiet lock"
    elif [ -f "$PAUSEF" ]; then why="PAUSE"; fi
    [ -n "$why" ] && { log "$name YIELD ($why): SIGTERM"; kill -TERM $P; break; }
  done
  wait $P; return $?
}
if [ "$MODE" = smoke ]; then
  wait_gate "smoke"; log "lock: $(lock 25 smoke 16)"
  run_guarded smoke_train $D/smoke_train.log $PY -I -B $S/dd_qat.py --run t30qa --smoke 5; rc=$?
  log "dd_qat smoke exit $rc: $(grep -E 'SMOKE|DONE|step 5/' $D/smoke_train.log | tail -2 | tr '\n' ' ' | cut -c1-260)"
  run_guarded smoke_gate $D/smoke_m3.log $PY -I -B $S/m3_tree.py --set heplus24 --cut none; rc2=$?
  $PY -I -B $S/m3_tree.py --set heplus24 --cut none --gate >> $D/smoke_m3.log 2>&1; rc3=$?
  peer unlock >/dev/null 2>&1
  log "harness dump exit $rc2, gate exit $rc3: $(tail -2 $D/smoke_m3.log | tr '\n' ' ' | cut -c1-260)"
  exit $(( rc != 0 || rc2 != 0 || rc3 != 0 ))
fi
# qa
[ -n "$MARK" ] || { log "qa needs the END marker path"; exit 2; }
until [ -e "$MARK" ]; do
  [ "$(date +%a%H%M)" \> "Thu0500" ] && [ "$(date +%a)" = Thu ] && { log "no marker by Thu 05:00: exit 5"; exit 5; }
  sleep 60
done
log "marker $MARK seen"
while ! grep -q '"status": "done"' $HOME/datasets/deepdraft/ckpt/dd_t30qa.json 2>/dev/null; do
  wait_gate "qa train"; log "lock: $(lock 180 'qa train' 16)"
  run_guarded qa_train $D/qa_train.log $PY -I -B $S/dd_qat.py --run t30qa; rc=$?
  peer unlock >/dev/null 2>&1
  log "dd_qat exit $rc: $(grep -E '\[qat\] (DONE|stopped|MLX)' $D/qa_train.log | tail -1)"
  case $rc in 0|3) ;; *) log "training failed: stop"; exit $rc;; esac
done
for st in heplus24 mbppbank; do
  [ -s $D/m3/tree_t30b_${st}_C2_qa.npz ] && continue
  wait_gate "read $st"; log "lock: $(lock 40 "read $st" 12)"
  run_guarded qa_read_$st $D/qa_read_$st.log $PY -I -B $S/m3_tree.py --set $st --cut C2 \
    --weights $HOME/datasets/deepdraft/ckpt/dd_t30qa.safetensors --tag qa; rc=$?
  peer unlock >/dev/null 2>&1
  log "read $st exit $rc"
done
$PY -I -B $S/m3_score.py --cuts C2_qa --tag s22 > $D/qa_score.log 2>&1
python3 -I -c "
import json
r = json.load(open('$D/m3/s22_scores.json'))
print(json.dumps(r)[:600])
" >> $D/qa_score.log 2>&1
tail -12 $D/qa_score.log
