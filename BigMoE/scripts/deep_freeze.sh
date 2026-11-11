#!/bin/bash
# Sat 7 Nov 15:xx: the S5 deep freeze as a helper (s5_final.sh's block). usage: bash scripts/deep_freeze.sh <name>
# -> data/r57/snap/frozen_<name>/{Leap_scripts,Leap_third_party,TrainLab_sticky,TrainLab_root,Breakthrough_keepset},
#    data/r57/snap/<name>/r57_screen_<name>.py (+ relax_delta / relax_span / inert_span), data/r57/snap/c45_runner_<name>.sh;
#    every copied .py's code paths rewritten into the snapshot; exits 2 if any live code path is left. Prints "FROZEN <files>".
#    It does not commit: the caller commits the snapshot before any gated run (EvalSuite's provenance rule).
N=${1:?name}
cd "${DF_ROOT:-$HOME/BigMoE}" || exit 1
FZ="$PWD/data/r57/snap/frozen_$N"; SD=data/r57/snap/$N
rm -rf "$FZ"; mkdir -p "$FZ/TrainLab_root" "$SD"
cp scripts/r57_screen_fast.py "$SD/r57_screen_$N.py" && cp scripts/relax_delta.py scripts/relax_span.py scripts/inert_span.py scripts/pflm.py "$SD/" \
  && cp scripts/c45_runner.sh "data/r57/snap/c45_runner_$N.sh" || exit 2
cp -R "$HOME/Leap/scripts" "$FZ/Leap_scripts" && cp -R "$HOME/Leap/third_party" "$FZ/Leap_third_party" \
  && cp -R "$HOME/TrainLab/sticky" "$FZ/TrainLab_sticky" && cp "$HOME"/TrainLab/*.py "$FZ/TrainLab_root/" \
  && cp -R "$HOME/Breakthrough/keepset" "$FZ/Breakthrough_keepset" || exit 2
find "$FZ" "$SD" -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null
RW=(-e "s#~/Leap/scripts#$FZ/Leap_scripts#g" -e "s#/Users/kamen/Leap/scripts#$FZ/Leap_scripts#g"
    -e "s#~/Leap/third_party#$FZ/Leap_third_party#g" -e "s#/Users/kamen/Leap/third_party#$FZ/Leap_third_party#g"
    -e "s#~/TrainLab/sticky#$FZ/TrainLab_sticky#g" -e "s#/Users/kamen/TrainLab/sticky#$FZ/TrainLab_sticky#g"
    -e "s#\"~/TrainLab\"#\"$FZ/TrainLab_root\"#g" -e "s#'~/TrainLab'#'$FZ/TrainLab_root'#g"
    -e "s#~/Breakthrough/keepset#$FZ/Breakthrough_keepset#g" -e "s#/Users/kamen/Breakthrough/keepset#$FZ/Breakthrough_keepset#g")
find "$FZ" "$SD" -name '*.py' -exec sed -i '' "${RW[@]}" {} +
LEFT=$(grep -rnE --include='*.py' "(~|/Users/kamen)/Leap/(scripts|third_party)|(~|/Users/kamen)/TrainLab/sticky|[\"']~/TrainLab[\"']|(~|/Users/kamen)/Breakthrough/keepset" "$FZ" "$SD" | wc -l | tr -d ' ')
[ "$LEFT" = 0 ] || { echo "LIVE $LEFT code paths left in $N"; exit 2; }
echo "FROZEN $(grep -rl --include='*.py' "frozen_$N" "$FZ" "$SD" | wc -l | tr -d ' ')"
