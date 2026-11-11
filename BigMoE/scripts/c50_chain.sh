#!/bin/bash
# C50 chain (NewIdeas 11-02 01:3x + 01:5x; MTP Head 23:2x): wait until ~/TrainLab/data/E15_SLOT_DONE exists (MTP Head's
# E1 + E5 training slot ended; C50 then goes AHEAD of Leap's runner 62, which fits by day), then ONE quiet memory turn with
# the C50 arms: per-layer pool sizes
# (L, md5-pinned data/seed45/laysz45_<C>.json) vs uniform pools (R57_LAYSZ=0) at C 256 (the likely real ~42 GB setting:
# Kamen's number), 300 and 320, all on the adopted default (REF2-T + gate + t 0.3 + OOO + K-73), every token vs REF2-T,
# each arm only if NO-SWAP >= peak + 4 GB. Nothing may run past 12:00 (Chrome reopens): no queueing after 11:30, no run
# starts after 11:50 (C45_DEADLINE). The warm-replay split waits for Monday night (usage pacing).
# usage (a harness background task, not nohup): bash scripts/c50_chain.sh
export PATH="$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:$PATH"
LOG="$HOME/BigMoE/data/c50_chain.log"
say() { echo "$(date '+%m-%d %H:%M:%S') $*" >> "$LOG"; }
late() { local t; t=$(date +%H%M); [ "$t" -ge 1130 ] && [ "$t" -lt 2200 ]; }
say "chain start (pid $$): waiting for ~/TrainLab/data/E15_SLOT_DONE (NewIdeas 01:5x: C50 goes ahead of runner 62)"
while [ ! -e "$HOME/TrainLab/data/E15_SLOT_DONE" ]; do
  late && { say "11:30 and no E15_SLOT_DONE yet: C50 skipped (Chrome reopens at 12:00)"; exit 3; }
  sleep 30
done
late && { say "E15_SLOT_DONE came after 11:30: too late for C50 before 12:00"; exit 3; }
say "E15_SLOT_DONE seen: start the C50 turn"
cd "$HOME/BigMoE" || exit 1
CLAUDE_PID=65439 MARGIN=4 GAP0=0 GAP=0 NOWAIT45=1 MAXFAILS=2 WAITMEM=46 C45_OUT="$HOME/BigMoE/data/c50" C45_DEADLINE=1150 \
  C45_ARMS="c45l_256:256 c45u_256:256:R57_LAYSZ=0 c45u_300:300:R57_LAYSZ=0 c45l_300:300 c45l_320:320 c45u_320:320:R57_LAYSZ=0" \
  /bin/bash scripts/turn45.sh 42 25 "BigMoE C50: per-layer sizes vs uniform at C 256 / 300 / 320 (default + K-73)" \
  /bin/bash scripts/c45_runner.sh
say "C50 turn done rc=$?"
