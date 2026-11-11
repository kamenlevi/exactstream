#!/bin/bash
> (a quote from the project owner, kept private)
# (~/FableGod/DAY9_2026-11-02.md). ONE quiet ticket after night_apps.sh closes Chrome / Spotify / Steam (~01:xx, expect
# NO-SWAP ~51-52 GB), done before ~07:00 (no run starts after 06:50). Each arm runs in its own process. Each runs only if
# NO-SWAP >= its peak + MG (c45_runner one()). Groups:
#   1. condition 9 B2: C 300 / 320 with per-layer sizes (gates 52 / 55 GB); every token vs REF2-T
> (a quote from the project owner, kept private)
#      prompts + 2 agent. (a) every token == D-23d's (REF2-T loop), (b) B's lifetime peak with both drafters <= 45 GB,
#      (c) B vs A chat tok/s at C 256
#   3. C1 = NAX6 live (DAY9 #4; Leap 101573d): default vs NAX vs NAX again (determinism: NAX is its own reference). The warm
#      split (NewIdeas' Monday-night 1) on both: per-row GPU slope, cold-warm. P on L (Monday-night 2): the LRU + prior arm
#      on the C 256 default. Leap's K-83 (the second exact stack, K83_EXACT=1) as its own arm (adopt if exact and not
#      slower). Then touch ~/EvalSuite/results/ifx/GO (runner 38 queues after this ticket)
#   4. A6 (DAY9 #7) at C 256: ctl / rep / keep / keep+dummy+dwin / pools on agent bash18 -> text23 -> bash18
# usage: inside turn45 (HAVE_LOCK=1 KEEP_LOCK=1), with frozen copies of the runner and the harness (see HANDOVER)
export PATH="$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:$PATH"
cd "$HOME/BigMoE" || exit 1
mkdir -p data/day9
LOG="$HOME/BigMoE/data/day9/night.log"
say() { echo "$(date '+%m-%d %H:%M:%S') $*" >> "$LOG"; }
RUN="/bin/bash data/r57/snap/c45_runner_n9.sh"
export C45_SCREEN=data/r57/snap/r57_screen_n9.py C45_DEADLINE=0650 C45_OUT="$HOME/BigMoE/data/day9"
CHREF="$HOME/BigMoE/data/r57/chat23_ref_tokens.json:gate+t"
CHAT="ifeval/1220,ifeval/1561,ifeval/3280,multi/ar21,multi/ar22,multi/bg0,multi/bg2,multi/de3,multi/de4,multi/es10,multi/es11"
CHAT="$CHAT,multi/fr7,multi/fr8,multi/ja18,multi/ja19,multi/ru12,multi/ru13,multi/zh15,multi/zh16,agent/text23,agent/bash17"
PRIOR="$HOME/BigMoE/data/seed45/prior45.npz"
say "night start (pid $$, HAVE_LOCK=$HAVE_LOCK, NO-SWAP $(vm_stat | awk '/page size of/ {ps=$8} /^Pages (free|speculative|purgeable):/ || /^File-backed pages:/ {v=$NF; sub("[.]","",v); s+=v} END {printf "%d", s*ps/1e9}') GB)"
MG=4 C45_ARMS="c45l_300:300 c45l_320:320" $RUN
say "group 1 (C 300 / 320) rc=$?"
R57_KEYS=$CHAT MG=6 C45_ALARM=900 C45_REFTOK=$CHREF C45_ARMS="chA_256:256 chB_256:256:R57_CHATSW=1" $RUN
say "group 2 (chat switch A / B) rc=$?"
MG=4 C45_ARMS="c1d_256:256 c1k_256:256:K83_EXACT=1 c1n_256:256:R57_NAX=1,R57_REFTOK= c1n_256r:256:R57_NAX=1,R57_REFTOK= ws_256:256:R57_WARMSPLIT=3 wsn_256:256:R57_NAX=1,R57_REFTOK=,R57_WARMSPLIT=3 c45p_256:256:BIGMOE_EXT_BUILD=build_prior,BIGMOE_EVICT_PRIOR=30,R57_PRIOR=$PRIOR" $RUN
say "group 3 (C1 NAX6 + warm split + P on L) rc=$?"
touch "$HOME/EvalSuite/results/ifx/GO" && say "touched ~/EvalSuite/results/ifx/GO (warm split + P arm done or skipped)"
R57_KEYS=agent/bash18,agent/text23,agent/bash18 MG=7 C45_CAP=64 C45_ALARM=300 C45_REFTOK=$CHREF C45_ARMS="a6c_256:256:R57_A6=ctl a6r_256:256:R57_A6=rep a6k_256:256:R57_A6=keep a6w_256:256:R57_A6=keep+dummy+dwin a6p_256:256:R57_A6=pools" $RUN
say "group 4 (A6) rc=$?"
say "night end"
