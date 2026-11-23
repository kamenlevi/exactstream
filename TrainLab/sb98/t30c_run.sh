#!/bin/bash
# WEEK98 T30C runner (MTP Head; REPORT 'WEEK98 T30C'): BigMoE's scripts/dd_ttt.py, UNMODIFIED (read-only use), trains
# t30c = t30b + one more pass with a fresh Adam and a fresh cosine (the trainer's own: 20-step warm-up, 1e-5 cosine to
# 1e-6), rows 192, depth 30, wdecay 0.97, pass seed 1117, then evaluates t30b and t30c together (dd_ttt.py eval:
# HE+ 24 / MBPP 100 bands + kept per 32).
# Start = t30b's final weights through the trainer's own resume path: a seeded state dd_t30c.json at step 0 (step 0
# restores no optimizer state) next to a bitwise copy of dd_t30b.safetensors. Seeded once; a resume never re-seeds.
# Gates (the orchestrator 22:3x): start only when Kamen is away (away_secs >= 900), NO-SWAP >= 22 GB, no quiet lock and
# no other > 5 GB MEM lock of others, no PAUSE, before the night cutoff CUT (22:30: the 45 GB chain); a share lock
# 'MEM 16 GB'. Yield (SIGTERM: the trainer saves weights + Adam, exits 3) on NO-SWAP < 5 GB at two checks in a row, a
# quiet lock of others, a PAUSE, or the cutoff; then wait for the gates and continue from the saved step. At the cutoff
# the runner exits 5; re-run the same command (Tuesday by day) to continue.
# Usage: t30c_run.sh A|B   (A = split_A.json, unseen only, 1,479 steps; B = split_B.json, mixed, 5,685 steps)
set -u
M=${1:?A or B}
D=$HOME/TrainLab/data/t30c; CKD=$HOME/datasets/deepdraft/ckpt
PY=$HOME/BigMoE/.venv/bin/python
source "$HOME/Sandbox98/scripts/guard_lib.sh"      # quiet_others, mem_others, noswap, swap_mb (read-only use)
AWAY=$HOME/.local/share/quiet/away_secs.sh; PAUSEF=$HOME/.local/share/quiet/PAUSE
CUT=${CUT:-2230}
case $M in A) SAVE=740; LOCKM=150;; B) SAVE=1500,3000; LOCKM=540;; *) echo "A or B"; exit 2;; esac
log() { echo "[$(date '+%a %H:%M:%S')] $*"; }
now() { date +%H%M | sed 's/^0*//'; }
gate() { [ "$($AWAY)" -ge 900 ] && [ "$(noswap)" -ge 22 ] && [ "$(quiet_others)" -eq 0 ] && [ "$(mem_others)" -eq 0 ] \
         && [ ! -f "$PAUSEF" ] && [ "$(now)" -lt $CUT ]; }
wait_gate() {   # $1 = what waits; exits 5 at the cutoff
  local n=0
  until gate; do
    [ "$(now)" -ge $CUT ] && { log "night cutoff $CUT while waiting ($1): exit 5, re-run to continue"; exit 5; }
    [ $((n % 20)) -eq 0 ] && log "$1 waits: away $($AWAY) s, NO-SWAP $(noswap) GB, quiet $(quiet_others), mem $(mem_others), PAUSE $([ -f "$PAUSEF" ] && echo on || echo off)"
    n=$((n + 1)); sleep 30
  done
}
lock() { peer lock share $1m --cpu 150 --gpu 60 "MTP Head: $2, MEM $3 GB" 2>&1 | tr '\n' ' ' | cut -c1-160; }

if [ ! -e $CKD/dd_t30c.json ] && [ ! -e $CKD/dd_t30c.safetensors ]; then
  cp $CKD/dd_t30b.safetensors $CKD/dd_t30c.safetensors && cmp -s $CKD/dd_t30b.safetensors $CKD/dd_t30c.safetensors \
    || { log "seed copy failed"; exit 1; }
  python3 -I -c "
import json, sys
a = {'cmd': 'train', 'run': 't30c', 'runs': 'b5,t30a', 'frac': 0.03, 'depth': 30, 'rows': 192, 'lr': 1e-05, 'wdecay': 0.97,
     'steps': 0, 'peak_max': 18.0, 'sets': 'code,mbpp', 'manifest': '$D/split_$M.json', 'pass_seed': 1117,
     'save_at': '$SAVE', 'out': None, 'feat': False, 'xb': 0, 'xb_lr_mult': 10.0, 'smoke': False}
json.dump({'step': 0, 'loss': [], 'args': a, 'status': 'seeded: t30b final weights, step 0 (MTP Head T30C)'}, open(sys.argv[1], 'w'))
" $CKD/dd_t30c.json || exit 1
  log "seeded dd_t30c from dd_t30b (bitwise copy; step 0, no optimizer state; manifest $M)"
fi
while ! grep -q '"status": "done"' $CKD/dd_t30c.json; do
  wait_gate "t30c train"
  log "lock: $(lock $LOCKM "t30c train ($M)" 16)"
  log "start train (away $($AWAY) s, NO-SWAP $(noswap) GB, swap $(swap_mb) MB)"
  (cd $HOME/BigMoE && exec $PY -B scripts/dd_ttt.py train --run t30c --manifest $D/split_$M.json --pass-seed 1117 \
     --save-at $SAVE) >> $D/train_$M.log 2>&1 &
  P=$!; low=0; why=""
  while kill -0 $P 2>/dev/null; do
    sleep 30
    ns=$(noswap); if [ "$ns" -lt 5 ]; then low=$((low + 1)); else low=0; fi
    if [ $low -ge 2 ]; then why="NO-SWAP $ns GB at two checks"
    elif [ "$(quiet_others)" -gt 0 ]; then why="a quiet lock of others"
    elif [ -f "$PAUSEF" ]; then why="PAUSE"
    elif [ "$(now)" -ge $CUT ]; then why="night cutoff $CUT"; fi
    if [ -n "$why" ]; then log "YIELD ($why): SIGTERM, the trainer saves and exits 3"; kill -TERM $P; break; fi
  done
  wait $P; rc=$?
  peer unlock >/dev/null 2>&1
  log "trainer exit $rc: $(grep -E '\[dd\] (DONE|stopped|MLX peak)' $D/train_$M.log | tail -1)"
  case $rc in 0|3) ;; *) log "trainer failed (exit $rc): stop"; exit $rc;; esac
  [ "$why" = "night cutoff $CUT" ] && { log "exit 5 at the cutoff; re-run to continue"; exit 5; }
done
if [ ! -s $D/eval_$M/eval.json ]; then
  wait_gate "eval"
  log "lock: $(lock 60 "t30c eval ($M)" 11)"
  (cd $HOME/BigMoE && exec $PY -B scripts/dd_ttt.py eval --runs t30b,t30c --out $D/eval_$M) > $D/eval_$M.log 2>&1
  rc=$?; peer unlock >/dev/null 2>&1
  log "eval exit $rc: $(tail -1 $D/eval_$M.log | cut -c1-200)"
fi
log "T30C ($M) DONE"
