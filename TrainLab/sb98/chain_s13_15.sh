#!/bin/bash
# S13 -> S14 -> S15 in order, one job of mine at a time (MTP Head, Mon 16 Nov). S13 / S14 are retried up to 3 times when
# run_paused's memory guard killed them (13:26: another session's ~42 GB load took NO-SWAP 46 -> 4 GB in 20 s).
O=$HOME/TrainLab/data/sb98; S=$HOME/TrainLab/sb98
try() {   # $1 = script, $2 = its runner log
  local i
  for i in 1 2 3; do
    $S/$1 > $O/$2 2>&1 && return 0
    grep -q "MEMORY GUARD" $O/$2 || { echo "$(date +%H:%M:%S) $1 failed (not the memory guard): see $2"; return 1; }
    echo "$(date +%H:%M:%S) $1 killed by the memory guard (try $i): retry after the gate"
    cp $O/$2 $O/${2%.log}_try$i.log; sleep 120
  done
  return 1
}
try s13.sh s13_runner.log
try s14.sh s14_runner.log
$S/s15.sh > $O/s15_runner.log 2>&1
echo "$(date +%H:%M:%S) chain done"
