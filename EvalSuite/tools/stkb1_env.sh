#!/bin/bash
# stkb1_env.sh: EvalSuite's env file for blob 1's tier 1 #1 (PREREG_lossygate amendment 6), from BigMoE's
# data/stack/blob1_env.txt. The env line is BigMoE's byte for byte. The comment names the run's tag, the snapshot's screen
# and C, which runner 69 reads: it stops on a comment that says "snapshot" without a data/r57/snap/<dir>/r57_screen*.py
# (BigMoE's own comment does), and in a real run on one that does not name its tag; with no C it would take 256 silently.
#   bash tools/stkb1_env.sh data/r57/snap/s98q/r57_screen_s98q.py
# Writes results/lossygate/stkb1_env.txt; prints the source's and the output's git blob.
set -eu
cd "$HOME/EvalSuite"
SCR=${1:?the snapshot screen, data/r57/snap/DIR/r57_screen_NAME.py}
echo "$SCR" | grep -qxE 'data/r57/snap/([A-Za-z0-9_.-]+/)*r57_screen[A-Za-z0-9_.-]*\.py' && [ -f "$HOME/BigMoE/$SCR" ] \
  || { echo "no BigMoE screen $SCR"; exit 2; }
S=$HOME/BigMoE/data/stack/blob1_env.txt; O=$PWD/results/lossygate/stkb1_env.txt   # absolute: read with git -C BigMoE
bl() { git -C "$HOME/BigMoE" hash-object "$1"; }
L=$(grep -v '^#' "$S" | grep .)
[ "$(printf '%s\n' "$L" | wc -l | tr -d ' ')" = 1 ] || { echo "blob1_env.txt: not one env line"; exit 2; }
B=$(bl "$S"); BC=$(git -C "$HOME/BigMoE" log -1 --format=%h -- data/stack/blob1_env.txt)
printf '%s\n%s\n' "# STKB1 = blob 1 (Saturday's lossy stack, PREREG_lossygate amendment 6), tier 1 #1: BigMoE's data/stack/blob1_env.txt env line byte for byte (blob $B, BigMoE $BC). Snapshot s98q, screen $SCR; C 256 (the 45 GB release C, as ccx45cc1 and lx45cc1). Written by EvalSuite's tools/stkb1_env.sh for runner 69." "$L" > "$O"
[ "$(grep -v '^#' "$O" | grep .)" = "$L" ] || { echo "the copy's env line differs from BigMoE's"; exit 2; }
echo "source: $S blob $B (BigMoE $BC)"
echo "$O: blob $(bl "$O")"
