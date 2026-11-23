#!/bin/bash
# CHAN decode (results/LEVERS_A.md section CHAN): the HE+ and MBPP halves of the panel in two processes, each into its
# own file (resumable: done keys are skipped). With 'plain' (the identity check failed), each half then also runs
# J(CC)'s exact arm, our own coupled arm. A TERM stops both halves.
cd "$(dirname "$0")/.."
PY=${PY:-$HOME/BigMoE/.venv/bin/python}
CPL=${1:-arm0}
run_half() {   # SETS TAG
  $PY -I -B sb/chan.py --mode chan --sets $1 --tag $2 --mem 3 || return $?
  if [ "$CPL" = plain ]; then $PY -I -B sb/chan.py --mode plain --sets $1 --tag $2 --mem 3 || return $?; fi
  return 0
}
run_half humaneval,humanevalx he > logs/chan_decode_he.log 2>&1 & P1=$!
run_half mbpp mbpp > logs/chan_decode_mbpp.log 2>&1 & P2=$!
trap 'kill -TERM $P1 $P2 2>/dev/null; pkill -TERM -P $P1; pkill -TERM -P $P2; wait; exit 143' TERM INT
wait $P1; R1=$?; wait $P2; R2=$?
grep -h "DONE" logs/chan_decode_he.log logs/chan_decode_mbpp.log
echo "DONE chan decode: he rc $R1, mbpp rc $R2"
[ $R1 = 0 ] && [ $R2 = 0 ]
