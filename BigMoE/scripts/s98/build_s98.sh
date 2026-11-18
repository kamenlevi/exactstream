#!/bin/bash
# Build an s98 snapshot (BigMoE, Wed 18 Nov). usage: bash scripts/s98/build_s98.sh <name> [leap commit, default b4331e4]
# - Leap scripts + third_party: frozen_g1v3's trees (== Leap b4331e4, verified file by file by scratchpad build_g1v3.sh),
#   with <commit>'s changed / added files overlaid when <commit> is later (deletions abort);
# - TrainLab_root / TrainLab_sticky / Breakthrough_keepset / Fit12_res1c: frozen_g1b's; Breakthrough_tree16: frozen_t16b's;
# - Breakthrough_xrw_knob_v2: ~/Breakthrough/data/xrw/knob_v2 + its md5 file (the screen checks every file's md5);
# - MoonShots_draft: ~/MoonShots/week98/draft/{r57_draft.py, ra_sf.py, sf_tree.py, s22_format.py} when present;
# - the screen: scripts/s98/r57_screen_s98_X.py with X = <name>; g1v3's aux files; scripts/tree16_{draft,commit,loop}.py;
# - every frozen .py's paths into frozen_<name>; exits 2 if a live or foreign-snapshot path is left. Commits nothing.
set -u
N=${1:?name}; C=${2:-b4331e4}; L=$HOME/Leap
cd ~/BigMoE || exit 1
S=data/r57/snap; FZ="$PWD/$S/frozen_$N"; SD=$S/$N
{ [ -e "$FZ" ] || [ -e "$SD" ]; } && { echo "$N exists"; exit 3; }
[ -z "$(git -C $L diff --name-only --diff-filter=DR b4331e4 $C -- scripts third_party)" ] || { echo "deletions / renames in b4331e4..$C"; exit 3; }
mkdir -p "$FZ" "$SD" || exit 2
cp -R $S/frozen_g1v3/Leap_scripts $S/frozen_g1v3/Leap_third_party "$FZ/" && \
cp -R $S/frozen_g1b/TrainLab_root $S/frozen_g1b/TrainLab_sticky $S/frozen_g1b/Breakthrough_keepset $S/frozen_g1b/Fit12_res1c "$FZ/" && \
cp -R $S/frozen_t16b/Breakthrough_tree16 "$FZ/" && \
( cd "$HOME/Breakthrough/data/xrw/knob_v2" && md5 -r $(awk '{print $2}' ../knob_v2.md5) | diff -q - <(awk '{print $1" "$2}' ../knob_v2.md5) > /dev/null ) || { echo "live knob_v2 != knob_v2.md5"; exit 2; }
cp -R "$HOME/Breakthrough/data/xrw/knob_v2" "$FZ/Breakthrough_xrw_knob_v2" && cp "$HOME/Breakthrough/data/xrw/knob_v2.md5" "$FZ/Breakthrough_xrw_knob_v2/" || exit 2
if [ -f "$HOME/MoonShots/week98/draft/r57_draft.py" ]; then mkdir -p "$FZ/MoonShots_draft"
  for f in r57_draft.py ra_sf.py sf_tree.py s22_format.py; do [ -f "$HOME/MoonShots/week98/draft/$f" ] && cp "$HOME/MoonShots/week98/draft/$f" "$FZ/MoonShots_draft/"; done; fi
if [ "$C" != b4331e4 ]; then n=0
  for f in $(git -C $L diff --name-only --diff-filter=AM b4331e4 $C -- scripts third_party | grep -v "__pycache__/"); do
    case "$f" in scripts/*) t="$FZ/Leap_scripts/${f#scripts/}";; third_party/*) t="$FZ/Leap_third_party/${f#third_party/}";; esac
    mkdir -p "$(dirname "$t")"; git -C $L show "$C:$f" > "$t" || exit 2; n=$((n+1)); done
  echo "overlaid $n files from b4331e4..$C"; fi
sed -e "s#frozen_X#frozen_$N#g" -e "s#/snap/X/#/snap/$N/#g" scripts/s98/r57_screen_s98_X.py > "$SD/r57_screen_$N.py" || exit 2
cp $S/g1v3/inert_span.py $S/g1v3/pflm.py $S/g1v3/relax_delta.py $S/g1v3/relax_span.py scripts/tree16_draft.py scripts/tree16_commit.py scripts/tree16_loop.py "$SD/" || exit 2
find "$FZ" "$SD" -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null
RW=(-e "s#~/Leap/scripts#$FZ/Leap_scripts#g" -e "s#/Users/kamen/Leap/scripts#$FZ/Leap_scripts#g"
    -e "s#~/Leap/third_party#$FZ/Leap_third_party#g" -e "s#/Users/kamen/Leap/third_party#$FZ/Leap_third_party#g"
    -e "s#~/TrainLab/sticky#$FZ/TrainLab_sticky#g" -e "s#/Users/kamen/TrainLab/sticky#$FZ/TrainLab_sticky#g"
    -e "s#\"~/TrainLab\"#\"$FZ/TrainLab_root\"#g" -e "s#'~/TrainLab'#'$FZ/TrainLab_root'#g"
    -e "s#~/Breakthrough/keepset#$FZ/Breakthrough_keepset#g" -e "s#/Users/kamen/Breakthrough/keepset#$FZ/Breakthrough_keepset#g"
    -e "s#~/Breakthrough/scripts/m1/tree16#$FZ/Breakthrough_tree16#g" -e "s#/Users/kamen/Breakthrough/scripts/m1/tree16#$FZ/Breakthrough_tree16#g"
    -e "s#/snap/frozen_g1v3/#/snap/frozen_$N/#g" -e "s#/snap/frozen_g1b/#/snap/frozen_$N/#g" -e "s#/snap/frozen_t16b/#/snap/frozen_$N/#g")
find "$FZ" "$SD" -name '*.py' -exec sed -i '' "${RW[@]}" {} +
LEFT=$(grep -rnE --include='*.py' "(~|/Users/kamen)/Leap/(scripts|third_party)|(~|/Users/kamen)/TrainLab/sticky|[\"']~/TrainLab[\"']|(~|/Users/kamen)/Breakthrough/(keepset|scripts/m1/tree16)|frozen_(X|g1v3|g1b|t16b)/" "$FZ" "$SD" | grep -v '^\S*:[0-9]*:\s*#' | wc -l | tr -d ' ')
echo "live / foreign paths left: $LEFT"; [ "$LEFT" = 0 ] || { grep -rnE --include='*.py' "frozen_(X|g1v3|g1b|t16b)/|(~|/Users/kamen)/(Leap|TrainLab|Breakthrough)/" "$FZ" "$SD" | head -5; exit 2; }
( cd "$FZ/Breakthrough_xrw_knob_v2" && md5 -r $(awk '{print $2}' knob_v2.md5) > frozen.md5 ) || exit 2
.venv/bin/python -m py_compile "$SD"/*.py && echo "BUILT $N (Leap $C; $(find "$FZ" -type f | wc -l | tr -d ' ') frozen files; MoonShots_draft: $( [ -d "$FZ/MoonShots_draft" ] && echo yes || echo no))"
find "$FZ" "$SD" -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null
