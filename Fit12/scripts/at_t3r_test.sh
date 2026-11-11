#!/bin/sh
# Dry tests of /Users/kamen/Fit12/scripts/at_t3r.sh with dummy runners (no lock, no GPU, scratch files only)
D=/private/tmp/claude-501/-Users-kamen-OjoX/7dd61b7e-2c45-43ec-b373-7768d2c96a18/scratchpad/at_t
A=/Users/kamen/Fit12/scripts/at_t3r.sh
ts() { date -v"$1"S +%Y%m%d%H%M%S; }
printf 'sleep 1234 & wait\n' > $D/run_wait.sh
printf 'touch %s/r47_B.txt; sleep 3; exit 0\n' $D > $D/run_ok.sh
printf 'touch %s/r47_C.txt; sleep 1235 & wait\n' $D > $D/run_long.sh
printf 'touch %s/r47_F.txt; sleep 2; exit 0\n' $D > $D/run_okF.sh
t() { n=$1; shift; env CLAUDE_PID=18778 F12_AT_LOG=$D/$n.log F12_AT_Q=$D/$n.q F12_AT_UNLOCK="touch $D/$n.unlocked" "$@" sh $A; echo "$n rc $?" >> $D/rc.txt; }
s0=$(date +%s)
t A F12_AT_GO=$(ts -1) F12_AT_LATE=$(ts +30) F12_AT_STOP=$(ts +8) F12_AT_BACK=$(ts +60) F12_AT_RUNNER=$D/run_wait.sh F12_AT_R47=$D/r47_A.txt &
t B F12_AT_GO=$(ts -1) F12_AT_LATE=$(ts +30) F12_AT_STOP=$(ts +30) F12_AT_BACK=$(ts +60) F12_AT_RUNNER=$D/run_ok.sh F12_AT_R47=$D/r47_B.txt &
t C F12_AT_GO=$(ts -1) F12_AT_LATE=$(ts +30) F12_AT_STOP=$(ts +6) F12_AT_BACK=$(ts +16) F12_AT_RUNNER=$D/run_long.sh F12_AT_R47=$D/r47_C.txt &
t D F12_AT_GO=$(ts -100) F12_AT_LATE=$(ts -50) F12_AT_RUNNER=$D/run_ok.sh F12_AT_R47=$D/r47_D.txt &
t F F12_AT_GO=$(ts +3) F12_AT_LATE=$(ts +60) F12_AT_STOP=$(ts +90) F12_AT_BACK=$(ts +120) F12_AT_RUNNER=$D/run_okF.sh F12_AT_R47=$D/r47_F.txt &
wait
echo "A-D,F took $(( $(date +%s) - s0 )) s"
printf 'sleep 1236\n' > $D/Fit12/scripts/runq.sh; sh $D/Fit12/scripts/runq.sh & P=$!; sleep 1
t E F12_AT_GO=$(ts -1) F12_AT_LATE=$(ts +30) F12_AT_RUNNER=$D/run_ok.sh F12_AT_R47=$D/r47_E.txt
kill $P; pkill -f 'sleep 1236'
for n in A B C D E F; do echo "== $n: $(grep "^$n rc" $D/rc.txt) | unlocked $([ -e $D/$n.unlocked ] && echo yes || echo no) | queue '$(cat $D/$n.q 2>/dev/null | tr '\n' ' ')'"; cat $D/$n.log 2>/dev/null; done
echo "left-over dummy sleeps: $(pgrep -f 'sleep 123[456]' | wc -l | tr -d ' ')"
