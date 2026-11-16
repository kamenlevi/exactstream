#!/bin/sh
# WEEK98 MODULE 2 (MoonShots): one diagnostic measurement window (NewIdeas 22:0x: one quiet ticket <= 6 min, MEM 3 GB;
# the session takes the lock in its own call first, checks away_secs >= 300, then launches this). Order = value:
# the per-stage profile (Leap's "fixed cost or rate?"), every form's bits + timing (kernels, whole call, exp32's way),
# the ablations, then round / cap variants of the full form. No step starts after DEADLINE (HHMM, env); each run under
# the memory guard (5 GB cap, swap +1 GB kill). Drops the lock at the end.
cd "$HOME/MoonShots/week98/m2" || exit 1
PY="$HOME/BigMoE/.venv/bin/python"
TAG=${M2_TAG:-w1}
DL=${DEADLINE:-2400}
LOG=m2_window_$TAG.log
ok() { [ "$(date +%H%M)" -lt "$DL" ] || { echo "[m2] deadline $DL: $1 skipped $(date '+%H:%M:%S')" >> $LOG; return 1; }; }
echo "[m2] window $TAG start $(date '+%a %d %b %H:%M:%S'), deadline $DL, away $(sh $HOME/.local/share/quiet/away_secs.sh) s" >> $LOG
if ok prof; then
  MAXGB=5 SWAPGB=1 LOG=m2_prof_$TAG.log sh ../jcc_guard.sh "$PY" -I -B m2_prof.py --blocks 24 --reps 9 --out "$PWD/m2_prof_$TAG"
  echo "[m2] prof rc $? $(date '+%H:%M:%S')" >> $LOG
fi
if ok bits; then
  MAXGB=5 SWAPGB=1 LOG=m2_bits_$TAG.log sh ../jcc_guard.sh "$PY" -I -B m2_bits.py --forms ${M2_FORMS:-C1,C1d2,C1x,C2,C1C2,C3,C1C3C2,C3B,C1C3C3BC2,C1xC3C3BC2,C4,C1C3C3BC4C2,C1xC3C3BC4C2} --blocks 24 --reps 5 --out "$PWD/m2_bits_$TAG"
  echo "[m2] bits rc $? $(date '+%H:%M:%S')" >> $LOG
fi
if ok abl; then
  MAXGB=5 SWAPGB=1 LOG=m2_abl_$TAG.log sh ../jcc_guard.sh "$PY" -I -B m2_abl.py --blocks 8 --reps 7 --out "$PWD/m2_abl_$TAG"
  echo "[m2] abl rc $? $(date '+%H:%M:%S')" >> $LOG
fi
for v in "M2_TPB=2" "M2_TPB=8" "M2_TPT=4" "M2_GY=128"; do
  vt=$(echo $v | tr -d '=')
  if ok "$v"; then
    env $v MAXGB=5 SWAPGB=1 LOG=m2_bits_${TAG}_$vt.log sh ../jcc_guard.sh "$PY" -I -B m2_bits.py --forms C1C3C3BC4C2,C1xC3C3BC4C2 --blocks 24 --reps 5 --out "$PWD/m2_bits_${TAG}_$vt"
    echo "[m2] bits $v rc $? $(date '+%H:%M:%S')" >> $LOG
  fi
done
peer unlock
echo "[m2] window $TAG done, lock released $(date '+%H:%M:%S')" >> $LOG
touch M2_WINDOW_${TAG}_DONE
