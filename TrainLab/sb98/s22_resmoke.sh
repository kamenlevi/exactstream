#!/bin/bash
# S22 trainer re-smoke (MTP Head): dd_qat.py --smoke 5 after the per-step STE fix (dada1b2), not before 16:30 (the
# orchestrator's placement), same gates as s22.sh (Kamen away >= 900 s, NO-SWAP >= 24, no quiet lock, no other >= 16 GB lock).
D=$HOME/TrainLab/data/s22; PY=$HOME/BigMoE/.venv/bin/python
source "$HOME/Sandbox98/scripts/guard_lib.sh"; AWAY=$HOME/.local/share/quiet/away_secs.sh
until [ "$(date +%H%M)" -ge 1630 ]; do sleep 60; done
until [ "$($AWAY)" -ge 900 ] && [ "$(noswap)" -ge 24 ] && [ "$(quiet_others)" -eq 0 ] && [ ! -f $HOME/.local/share/quiet/PAUSE ] \
      && [ "$(GUARD_BIG_GB=16 big_others)" -eq 0 ]; do sleep 30; done
peer lock share 15m --cpu 150 --gpu 60 "MTP Head: S22 trainer re-smoke, MEM 16 GB" > /dev/null 2>&1
echo "[$(date '+%H:%M:%S')] start (NO-SWAP $(noswap) GB)"
$PY -I -B $HOME/TrainLab/sb98/dd_qat.py --run t30qa --smoke 5 > $D/resmoke_train.log 2>&1; rc=$?
peer unlock > /dev/null 2>&1
echo "[$(date '+%H:%M:%S')] exit $rc: $(grep -E 'step [1-5]/|SMOKE|MLX peak' $D/resmoke_train.log | tr '\n' ' ' | cut -c1-500)"
