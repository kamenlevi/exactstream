#!/bin/bash
# WALL 1 SPRINT, tonight's relative measurements at ONE C (the orchestrator 11-03 19:2x: NO-SWAP ~30-35 GB expected at 00:30,
# so C 160, peak ~29.7 GB, instead of 224; every A/B below inside one ticket at the same C). W1B's prompts (S45's six,
# docs/WALL1_BENCH.md), every token vs REF2-T (s45t_B), uniform pools (no M-17 file at 160):
#   c45t / c45tj               traced (BIGMOE_TRACE + R57_STALL), without / with JOINT     -> S1's line (stall_report)
#   ka0 / ka1 / ka0r / ka1r    the SSD keepalive A-B-A-B on build_la (KEEPALIVE_US 50)     -> S2's line + la45's line
#   vt0 / vt1                  the volatile tier smoke on build_vt (64 / 32 slots)         -> la45's pre-registered line
#   pr                         the pre-route capture (18 prompts, CAP 128, vs chat23)       -> lever 2's recall line
#   chA / chB                  DAY9's chat switch checks (21 chat prompts vs chat23)        -> the combination's chat switch
# usage: inside turn45 (HAVE_LOCK=1 KEEP_LOCK=1); frozen copies data/r57/snap/*_s160.* (r57_screen 2df492c+ with R57_VTIER "64/32")
export PATH="$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:$PATH"
cd "$HOME/BigMoE" || exit 1
mkdir -p data/s160
LOG="$HOME/BigMoE/data/s160/s160.log"
say() { echo "$(date '+%m-%d %H:%M:%S') $*" >> "$LOG"; }
RUN="/bin/bash data/r57/snap/c45_runner_s160.sh"
export C45_SCREEN=data/r57/snap/r57_screen_s160.py C45_DEADLINE=${S_DEADLINE:-0300} C45_OUT="$HOME/BigMoE/data/s160"
CC=${S_C:-160}; MGC=${S_MG:-3}
CHREF="$HOME/BigMoE/data/r57/chat23_ref_tokens.json:gate+t"
say "sprint start (pid $$, C $CC, MG $MGC, HAVE_LOCK=$HAVE_LOCK)"
MG=$MGC C45_ARMS="c45t_$CC:$CC:BIGMOE_TRACE=1,R57_STALL=1 c45tj_$CC:$CC:BIGMOE_TRACE=1,R57_STALL=1,BIGMOE_JOINT=1" $RUN
say "S1 JOINT, traced (c45t / c45tj) rc=$?"
B="BIGMOE_EXT_BUILD=build_la"; K="$B,BIGMOE_SSD_KEEPALIVE_US=50"
MG=$MGC C45_ARMS="ka0_$CC:$CC:$B ka1_$CC:$CC:$K ka0r_$CC:$CC:$B ka1r_$CC:$CC:$K" $RUN
say "S2 keepalive A-B-A-B rc=$?"
V="BIGMOE_EXT_BUILD=build_vt"
MG=$MGC C45_ARMS="vt0_$CC:$CC:$V vt1_$CC:$CC:$V,R57_VTIER=64/32" $RUN
say "volatile tier smoke rc=$?"
PRK="humaneval/HumanEval/68,humaneval/HumanEval/16,humaneval/HumanEval/91,humaneval/HumanEval/6,humaneval/HumanEval/79"
PRK="$PRK,humaneval/HumanEval/19,humaneval/HumanEval/161,humaneval/HumanEval/17,humaneval/HumanEval/56,humaneval/HumanEval/141"
PRK="$PRK,humaneval/HumanEval/101,humaneval/HumanEval/63,ifeval/1220,ifeval/1561,ifeval/3280,multi/ar21,multi/de3,multi/es10"
R57_KEYS=$PRK MG=$MGC C45_CAP=128 C45_ALARM=600 C45_REFTOK=$CHREF C45_ARMS="pr_$CC:$CC:R57_PREROUTE=1" $RUN
say "pre-route capture rc=$?"
CHAT="ifeval/1220,ifeval/1561,ifeval/3280,multi/ar21,multi/ar22,multi/bg0,multi/bg2,multi/de3,multi/de4,multi/es10,multi/es11"
CHAT="$CHAT,multi/fr7,multi/fr8,multi/ja18,multi/ja19,multi/ru12,multi/ru13,multi/zh15,multi/zh16,agent/text23,agent/bash17"
R57_KEYS=$CHAT MG=$MGC C45_ALARM=900 C45_REFTOK=$CHREF C45_ARMS="chA_$CC:$CC chB_$CC:$CC:R57_CHATSW=1" $RUN
say "chat switch checks (chA / chB) rc=$?"
say "sprint end"
