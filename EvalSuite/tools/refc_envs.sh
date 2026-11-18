#!/bin/bash
# refc_envs.sh: EvalSuite's three env files for the contract's Part A (docs/PREREG_refc_2026-11-18.md), from BigMoE's two
# (data/stack/c98kar_env.txt, c98kch_env.txt). Each env line is BigMoE's byte for byte; the control's is c98kar's minus
# REF2_REFC=1, nothing else changed. The comments name the run's tag and the snapshot's screen, which runner 70 reads:
# it stops on a comment that names no data/r57/snap/<dir>/r57_screen*.py, and on one that does not name its tag.
#   bash tools/refc_envs.sh data/r57/snap/s98n/r57_screen_s98n.py
# Writes results/refc/c98kctl_env.txt, c98kar_env.txt, c98kch_env.txt; prints each source's and output's git blob.
set -eu
cd "$HOME/EvalSuite"
SCR=${1:?the snapshot screen, data/r57/snap/DIR/r57_screen_NAME.py}
echo "$SCR" | grep -qxE 'data/r57/snap/([A-Za-z0-9_.-]+/)*r57_screen[A-Za-z0-9_.-]*\.py' && [ -f "$HOME/BigMoE/$SCR" ] \
  || { echo "no BigMoE screen $SCR"; exit 2; }
S=$HOME/BigMoE/data/stack; O=$PWD/results/refc   # absolute: the blobs are read with git -C BigMoE, as runner 70 reads them
bl() { git -C "$HOME/BigMoE" hash-object "$1"; }
line() { grep -v '^#' "$1" | grep .; }
for f in c98kar c98kch; do [ "$(line "$S/${f}_env.txt" | wc -l | tr -d ' ')" = 1 ] || { echo "$f: not one env line"; exit 2; }; done
AR=$(line "$S/c98kar_env.txt"); CH=$(line "$S/c98kch_env.txt")
[ "$(echo "$AR" | tr ',' '\n' | grep -cx 'REF2_REFC=1')" = 1 ] || { echo "c98kar: not one REF2_REFC=1"; exit 2; }
[ "$(echo "$AR" | sed 's/,R57_MTP_K=0,/,R57_MTP_K=15,/; s/,R57_MTP_K=0$/,R57_MTP_K=15/')" = "$CH" ] \
  || { echo "c98kch is not c98kar with R57_MTP_K=15"; exit 2; }
CTL=$(echo "$AR" | tr ',' '\n' | grep -vx 'REF2_REFC=1' | paste -sd, -)
BA=$(bl "$S/c98kar_env.txt"); BC=$(bl "$S/c98kch_env.txt")
TAIL="screen $SCR; argv C 192. Written by EvalSuite's tools/refc_envs.sh for runner 70."
printf '%s\n%s\n' "# C98KCTL = the contract's switch-off control (step 0): BigMoE's c98kar env line (data/stack/c98kar_env.txt, blob $BA) with REF2_REFC=1 removed, nothing else changed. $TAIL" "$CTL" > "$O/c98kctl_env.txt"
printf '%s\n%s\n' "# C98KAR = the contract's exact AR path (R57_MTP_K=0): BigMoE's data/stack/c98kar_env.txt env line byte for byte (blob $BA). $TAIL" "$AR" > "$O/c98kar_env.txt"
printf '%s\n%s\n' "# C98KCH = the contract's exact CHAIN-16 path (R57_MTP_K=15): BigMoE's data/stack/c98kch_env.txt env line byte for byte (blob $BC). $TAIL" "$CH" > "$O/c98kch_env.txt"
echo "sources: c98kar $BA, c98kch $BC"
for f in c98kctl c98kar c98kch; do echo "$O/${f}_env.txt: blob $(bl "$O/${f}_env.txt")"; done
