#!/bin/sh
# After after_night2.sh (MTP Head, Tue 11-03): Breakthrough's two walks (agent + MBPP, ~10 min, shared) if before 14:30; then
# tell Breakthrough (peer send, pid looked up from its session file).
cd $HOME/TrainLab
LOG=logs/turns.log
say() { echo "$(date '+%m-%d %H:%M:%S') after_night3: $*" >> $LOG; }
hm() { date +%H%M | sed 's/^0*//;s/^$/0/'; }
say "start: waiting for after_night2.sh, then Breakthrough's walks"
while pgrep -f "turns/after_night2.sh" > /dev/null; do sleep 60; done
while pgrep -f "turns/(et_job|dd_job|dv_job|kw_job|d23h_job|turn75)" > /dev/null; do sleep 20; done
[ "$(hm)" -lt 1430 ] || { say "past 14:30: Breakthrough's walks wait (Wed after 15:00 as the one > 5 GB job, or the next window)"; exit 0; }
./run_shared.sh 25 200 40 turns/bt_walks.sh "Breakthrough's CF walks: agent + MBPP (L1-20, ~5 GB, no timing)"
if [ -f data/d02b_walk_hpd0_agent.npz ] && [ -f data/d02b_walk_hpd0_mbpp.npz ]; then
  B=$(for f in $(grep -l '"/Users/kamen/Breakthrough"' $HOME/.claude-peers/sessions/*.json 2>/dev/null); do p=$(basename $f .json); kill -0 $p 2>/dev/null && peer show $p 2>/dev/null | grep -q "^Breakthrough" && echo $p; done | head -1)
  M="MTP Head (auto): your three HPD-0 CF walks are ready in ~/TrainLab/data: d02b_walk_hpd0_math.npz (it already existed: the same 60 tf_gsm windows, all fields), d02b_walk_hpd0_agent.npz (12 agent windows), d02b_walk_hpd0_mbpp.npz (100 MBPP answers of d1102mbpp, h format through identity fc / hidden_norm). Fields as asked (names, ids, off, gen, checks, dtok, dcf, ttok, tcf); read-only for you, please."
  [ -n "$B" ] && echo "$M" | peer send $B - > /dev/null 2>&1
  say "Breakthrough's walks done and sent"
else
  say "Breakthrough's walks incomplete"
fi
