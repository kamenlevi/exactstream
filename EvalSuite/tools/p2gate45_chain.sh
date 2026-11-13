#!/bin/bash
# p2gate45_chain.sh: the 45 GB P2 gate (docs/PREREG_p2gate45_2026-11-13.md). Waits for the start rule (FROM, then the
# pids given), runs runner 63's control arm (c45, P2 off), then, only if its 222 rows are complete, the P2 arm (p2_45)
# scored vs card_c45_fwd. CHAIN_DRY=1: the two runner calls as dry runs, no wait, no GPU.
# usage: p2gate45_chain.sh "<FROM YYYY-mm-dd HH:MM>" "<CUTOFF YYYY-mm-dd HH:MM>" <ctl_ident 0|1> [after pid ...]
cd "$HOME/EvalSuite" || exit 1
FROMS=${1:?from}; CUTS=${2:?cutoff}; IDENT=${3:?control identity 0 or 1}; shift 3; AFTER="$*"
case "$IDENT" in 0|1) ;; *) echo "control identity is 0 or 1"; exit 2;; esac
BM=$HOME/BigMoE; LOG=results/p2gate45_chain.log; DRY=${CHAIN_DRY:-0}
CB=40eee33e5e2cf21d3396738e715c050fbf0283d1; PB=e4012e8073f7ce272ad6a647dbd205cc81d337a3   # PREREG_p2gate45: the blobs
FROM=$(date -j -f "%Y-%m-%d %H:%M:%S" "$FROMS:00" +%s) || { echo "FROM does not parse"; exit 2; }
date -j -f "%Y-%m-%d %H:%M:%S" "$CUTS:00" +%s > /dev/null || { echo "CUTOFF does not parse"; exit 2; }
say() { echo "$(date '+%m-%d %H:%M:%S') $*" >> "$LOG"; }
say "chain (pid $$$([ "$DRY" = 1 ] && echo ', DRY')): from $FROMS, cutoff $CUTS, control identity $IDENT, after [$AFTER]"
if [ "$DRY" != 1 ]; then
  while [ "$(date +%s)" -lt "$FROM" ]; do sleep 30; done
  for p in $AFTER; do while kill -0 "$p" 2>/dev/null; do sleep 30; done; done
fi
pinned() {      # PREREG_p2gate45 amendment 1: the registered code, not just whatever BigMoE's HEAD holds at the start
  [ "$(git -C $BM rev-parse HEAD:data/r57/snap/p3)" = 311eb59a2435c990b0968657ef7ddcbe0c257ad4 ] \
    && [ "$(git -C $BM rev-parse HEAD:data/r57/snap/frozen_p3)" = cfb43f2e34223f40a2dde877c996de39dc81cc14 ] \
    && [ "$(git -C $BM rev-parse HEAD:bigmoe)" = b21c35631bd9c29862f83f230a4524d7ea50e8f8 ] \
    && [ "$(shasum -a 256 $BM/data/r57/snap/p3/r57_screen_p3.py | cut -c1-12)" = e410133cfe69 ] \
    && [ "$(shasum -a 256 $BM/ext/build_qom/*.so | cut -c1-12)" = ff69fa24a745 ]
}
pinned || { say "the P3 snapshot, frozen_p3, bigmoe/, the screen or the .so is not the registered one: no run"; exit 4; }
say "pinned code OK (snapshot trees, screen, .so as registered)"
say "the control arm (c45)"
CARD_DRY=$DRY CARD_TAG=c45 CARD_P2=off CARD_FENV=$BM/data/stack/chat45_env.txt CARD_FBLOB=$CB CARD_MEM=45 \
  CARD_CUTOFF="$CUTS" tools/day_runner63.sh; rc=$?
say "control arm rc $rc: $(cat results/CARD_c45_DONE 2>/dev/null)"
if [ "$DRY" != 1 ]; then
  grep -q "^CONTROL c45: 222 of 222 rows" results/CARD_c45_DONE 2>/dev/null \
    || { say "the control arm is not complete: no P2 arm (it resumes in a later window)"; exit 3; }
fi
say "the P2 arm (p2_45)"
CARD_DRY=$DRY CARD_TAG=p2_45 CARD_P2=on CARD_REF=card_c45_fwd CARD_REF_LATER=$DRY CARD_CTL_IDENT=$IDENT \
  CARD_FENV=$BM/data/stack/chat45_p2_env.txt CARD_FBLOB=$PB CARD_MEM=45 CARD_CUTOFF="$CUTS" tools/day_runner63.sh; rc=$?
say "P2 arm rc $rc: $(cat results/CARD_p2_45_DONE 2>/dev/null)"
