#!/bin/sh
# M-47 S22ENGINE bit test (MoonShots, Tue 17 Nov): sf_tree.py --real --fmt c2 (t30b's weights in S22's C2 format), SF vs HS
# bitwise on the Q8 real test's 120 trees at the 8K screen + 48 trees at the 32K screen. One process at a time under the
# memory guard (14 GB cap, swap +1 GB kill). Share lock "MEM 12 GB" held by the session. Outputs: s22_bit_<tag>.json.
# S22's checkpoint, when it lands: W=<path> sh s22_bittest.sh he16   (adds --weights)
cd "$HOME/MoonShots/week98/draft" || exit 1
PY="$HOME/BigMoE/.venv/bin/python"
LOG=s22_bittest.log
WARG=""; [ -n "$W" ] && WARG="--weights $W"
run() {
  tag=$1; shift
  echo "[s22] $tag start $(date '+%H:%M:%S')" >> $LOG
  MAXGB=14 SWAPGB=1 LOG=$LOG sh ../jcc_guard.sh "$PY" -I -B sf_tree.py --real --fmt c2 $WARG "$@" --out "s22_bit_$tag.json" >> $LOG 2>&1
  echo "[s22] $tag rc $? $(date '+%H:%M:%S')" >> $LOG
}
ONLY=${1:-all}
[ "$ONLY" = all ] || [ "$ONLY" = he16 ] && run he16 --set heplus24 --windows 0,1 --nodes 16 --max-anchors 48
[ "$ONLY" = all ] || [ "$ONLY" = he32 ] && run he32 --set heplus24 --windows 2,3 --nodes 32 --max-anchors 24
[ "$ONLY" = all ] || [ "$ONLY" = mb16 ] && run mb16 --set mbppbank --windows 0,1,2,3 --nodes 16 --max-anchors 48
[ "$ONLY" = all ] || [ "$ONLY" = he16s32 ] && run he16s32 --set heplus24 --windows 0,1 --nodes 16 --max-anchors 48 --screen 32768
echo "[s22] DONE $(date '+%H:%M:%S')" >> $LOG
