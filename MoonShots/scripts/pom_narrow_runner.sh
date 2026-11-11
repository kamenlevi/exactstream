#!/bin/sh
# MoonShots, detached CPU runner before a pacing sleep (no lock: a light single-core replay, PAUSE-aware): M-03 POM re-priced
# for the shipped narrow MTP path after BigMoE's R-43 KILL (DFlash2 16-row checks at 24 GB): checks of 4 rows accepting 3,
# and of 8 accepting 5, at the 24 GB stack (159 / 44). Output data/pom24_r4a3_159_44.txt, data/pom24_r8a5_159_44.txt.
D=$HOME/MoonShots; PY=$HOME/BigMoE/.venv/bin/python; LOG=$D/data/pom_narrow_runner.log
echo "$(date +%H:%M:%S) runner started" >> $LOG
for ra in "4 3" "8 5"; do
  [ -f $HOME/.local/share/quiet/PAUSE ] && { echo "$(date +%H:%M:%S) PAUSE set: stopping" >> $LOG; break; }
  set -- $ra
  POM_R=$1 POM_A=$2 taskpolicy -b $PY $D/scripts/pom24.py > $D/data/pom_narrow_r$1a$2.log 2>&1
  echo "$(date +%H:%M:%S) R $1 A $2: $(grep -E '^    R2 (10|6|4|2)|Traceback|Error' $D/data/pom_narrow_r$1a$2.log | cut -c1-230 | tr '\n' ' ')" >> $LOG
done
echo "$(date +%H:%M:%S) runner finished" >> $LOG
