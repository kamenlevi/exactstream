#!/bin/sh
# TrainLab night runner (only with the orchestrator's OK for a silent night): runs the job lines of turns/night.txt one
# after another, duty-cycled (sleep between steps, --duty) so the fans stay quiet; the thermostat may SIGSTOP/CONT us
# at any time (pause-safe: checkpoints at every eval). Stops at END (HH:MM, default 08:30) or when turns/stop exists.
# Each job line: an argument line for train35.py or mtp35.py (first word = script), with its own --time-limit added.
T=$HOME/TrainLab; PY=$HOME/BigMoE/.venv/bin/python; END=${1:-08:30}; DUTY=${2:-0.4}
cd $T
F="grep --line-buffered -v Token.indices"
endsec() { now=$(date +%s); e=$(date -j -f "%Y-%m-%d %H:%M" "$(date +%Y-%m-%d) $END" +%s); [ $e -le $now ] && e=$((e + 86400)); echo $((e - now)); }
echo "== $(date '+%m-%d %H:%M:%S') night start (until $END, duty $DUTY)" >> logs/night.log
while read -r script args; do
  [ -z "$script" ] && continue
  [ -f turns/stop ] && break
  left=$(endsec); [ $left -lt 900 ] && break
  echo "== $(date '+%m-%d %H:%M:%S') $script $args (time left $left s)" >> logs/night.log
  $PY $script $args --duty $DUTY --time-limit $((left - 300)) 2>&1 | $F >> logs/night.log
done < turns/night.txt
echo "== $(date '+%m-%d %H:%M:%S') night end" >> logs/night.log
