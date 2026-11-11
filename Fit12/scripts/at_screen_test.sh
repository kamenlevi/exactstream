#!/bin/sh
# Dry tests of /Users/kamen/Fit12/scripts/at_screen.sh with dummy runners (no lock, no GPU, scratch files only), ~40 s:
# A slot over at STOP (marker absent), B normal end, C hard stop at BACK, D woke after LATE, E another Fit12 runner, F GO later.
D=${1:-/private/tmp/claude-501/-Users-kamen-OjoX/7dd61b7e-2c45-43ec-b373-7768d2c96a18/scratchpad/at_s}
rm -rf $D; mkdir -p $D/Fit12/scripts
A=/Users/kamen/Fit12/scripts/at_screen.sh
ts() { date -v"$1"S +%Y%m%d%H%M%S; }
printf 'sleep 1234 & wait\n' > $D/run_wait.sh
printf 'touch %s/mk_B.txt; sleep 3; exit 0\n' $D > $D/run_ok.sh
printf 'touch %s/mk_C.txt; sleep 1235 & wait\n' $D > $D/run_long.sh
printf 'touch %s/mk_F.txt; sleep 2; exit 0\n' $D > $D/run_okF.sh
t() { n=$1 rn=$2; shift 2; env CLAUDE_PID=18778 F12_AT_LOG=$D/$n.log F12_AT_Q=$D/$n.q F12_AT_UNLOCK="touch $D/$n.unlocked" \
  F12_AT_RUNNER=$D/$rn sh $A "$@"; echo "$n rc $?" >> $D/rc.txt; }
s0=$(date +%s)
t A run_wait.sh t3r2 $(ts -1) $(ts +30) $(ts +8) $(ts +60) $D/mk_A.txt &
t B run_ok.sh t3r2 $(ts -1) $(ts +30) $(ts +30) $(ts +60) $D/mk_B.txt &
t C run_long.sh t3r2 $(ts -1) $(ts +30) $(ts +6) $(ts +16) $D/mk_C.txt &
t D run_ok.sh t3r2 $(ts -100) $(ts -50) $(ts +30) $(ts +60) $D/mk_D.txt &
t F run_okF.sh t3r2 $(ts +3) $(ts +60) $(ts +90) $(ts +120) $D/mk_F.txt &
wait
echo "A-D,F took $(( $(date +%s) - s0 )) s"
printf 'sleep 1236\n' > $D/Fit12/scripts/runq.sh; sh $D/Fit12/scripts/runq.sh & P=$!; sleep 1
t E run_ok.sh t3r2 $(ts -1) $(ts +30) $(ts +30) $(ts +60) $D/mk_E.txt
kill $P; pkill -f 'sleep 1236'
for n in A B C D E F; do echo "== $n: $(grep "^$n rc" $D/rc.txt) | unlocked $([ -e $D/$n.unlocked ] && echo yes || echo no) | queue '$(cat $D/$n.q 2>/dev/null | tr '\n' ' ')'"; sed 's/^[0-9:]* SCR //' $D/$n.log 2>/dev/null | cut -c1-150; done
echo "left-over dummy sleeps: $(pgrep -f 'sleep 123[456]' | wc -l | tr -d ' ')"
