#!/bin/bash
# TREE16 snapshot builder (BigMoE, Sun 15 Nov 21:0x; generalises scratchpad snap_t16b.sh + patch_t16b.py, which built t16b).
# usage: bash scripts/snap_tree16.sh <name> <base> [qsa]   e.g.  bash scripts/snap_tree16.sh t16c t16b qsa
# - a fresh deep freeze (scripts/deep_freeze.sh <name>: Leap scripts incl. week98, Leap third_party, TrainLab, Breakthrough
#   keepset) + Breakthrough's tree16 dir (B1 tree_meta, B2 ple_tree, B3 qsa_tree) as frozen_<name>/Breakthrough_tree16;
# - the screen = the base snapshot's screen (its hooks: t16a's M1 test, t16b's R57_TREE16) on the new freeze;
# - frozen copies of scripts/tree16_draft.py, tree16_commit.py, tree16_loop.py;
# - every live ~/Breakthrough/scripts/m1/tree16 path in ANY frozen .py rewritten to the frozen copy (Leap's files may import
#   it); Breakthrough's own test files (test_*.py, tree16_ple_e2e.py) dropped from the freeze;
# - with "qsa": the screen's T16L.install(ref2) becomes T16L.install(ref2, qsa=True) (B3 in Leap's QSA_TREE slot, so the
#   loop's chain fallback past a + T > 2044 goes away).
# Prints the leftover live paths (must be 0) and the owners' HEADs. Commits nothing: head-test + stub-simulate the runner,
# then commit the snapshot before any gated run.
set -u
N=${1:?name}; B=${2:?base snapshot}; QSA=${3:-}
cd ~/BigMoE || exit 1
[ -e data/r57/snap/$N ] && { echo "data/r57/snap/$N exists: pick a new name"; exit 3; }
[ -f data/r57/snap/$B/r57_screen_$B.py ] || { echo "no base screen data/r57/snap/$B/r57_screen_$B.py"; exit 3; }
bash scripts/deep_freeze.sh $N || { echo "deep_freeze failed"; exit 2; }
FZ="$PWD/data/r57/snap/frozen_$N"; SD=data/r57/snap/$N
cp -R "$HOME/Breakthrough/scripts/m1/tree16" "$FZ/Breakthrough_tree16" || exit 2
find "$FZ" -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null
for f in "$FZ"/Breakthrough_tree16/test_*.py "$FZ"/Breakthrough_tree16/tree16_ple_e2e.py; do [ -f "$f" ] && rm -f "$f"; done
sed -e "s#frozen_$B#frozen_$N#g" -e "s#/snap/$B/#/snap/$N/#g" data/r57/snap/$B/r57_screen_$B.py > "$SD/r57_screen_$N.py" || exit 2
cp scripts/tree16_draft.py scripts/tree16_commit.py scripts/tree16_loop.py "$SD/" || exit 2
find "$FZ" "$SD" -name '*.py' -exec sed -i '' -e "s#~/Leap/scripts#$FZ/Leap_scripts#g" -e "s#/Users/kamen/Leap/scripts#$FZ/Leap_scripts#g" \
     -e "s#~/Breakthrough/scripts/m1/tree16#$FZ/Breakthrough_tree16#g" -e "s#/Users/kamen/Breakthrough/scripts/m1/tree16#$FZ/Breakthrough_tree16#g" {} +
if [ "$QSA" = qsa ]; then
  n=$(grep -c "        T16L.install(ref2)  " "$SD/r57_screen_$N.py")
  [ "$n" = 1 ] || { echo "the screen's T16L.install(ref2) anchor: $n matches (want 1)"; exit 4; }
  sed -i '' 's#        T16L.install(ref2)  #        T16L.install(ref2, qsa=True)  #' "$SD/r57_screen_$N.py"
  echo "qsa: $(grep -n 'T16L.install(ref2, qsa=True)' "$SD/r57_screen_$N.py" | cut -c1-80)"
fi
.venv/bin/python -m py_compile "$SD/r57_screen_$N.py" "$SD/tree16_loop.py" || exit 5
find "$SD" -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null
LEFT=$(grep -rnE --include='*.py' "(~|/Users/kamen)/Leap/(scripts|third_party)|(~|/Users/kamen)/TrainLab/sticky|[\"']~/TrainLab[\"']|(~|/Users/kamen)/Breakthrough/(keepset|scripts)|frozen_$B|/snap/$B/" "$FZ" "$SD" | grep -v "^[^:]*:[0-9]*: *#" | wc -l | tr -d ' ')
echo "live / $B paths left (code lines): $LEFT"
[ "$LEFT" = 0 ] || grep -rnE --include='*.py' "(~|/Users/kamen)/Leap/(scripts|third_party)|(~|/Users/kamen)/Breakthrough/(keepset|scripts)|frozen_$B|/snap/$B/" "$FZ" "$SD" | grep -v "^[^:]*:[0-9]*: *#" | head -8
echo "Leap $(git -C ~/Leap rev-parse --short HEAD) (scripts dirty: $(git -C ~/Leap status --short -- scripts | grep -v pycache | grep -c '^ M')); Breakthrough $(git -C ~/Breakthrough rev-parse --short HEAD) (tree16 dirty: $(git -C ~/Breakthrough status --short -- scripts/m1/tree16 | grep -v pycache | grep -c .)); TrainLab $(git -C ~/TrainLab rev-parse --short HEAD 2>/dev/null)"
du -sh "$FZ" "$SD"
