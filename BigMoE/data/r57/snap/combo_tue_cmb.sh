#!/bin/bash
# THE COMBINATION, Tuesday by day (the orchestrator 11-02 21:1x: "turn on, one switch group at a time, with a bitwise check
# after each"; BigMoE = the integrator, the only writer of R57_PRESET=45). One ticket, arms at C 256 in order, each the
# previous + one switch group (cumulative, so each delta is measured on the stack it joins):
#   cmb0 the default (REF2-T, R57_PRESET=45)       -> its tokens = the reference of every later arm (+ chat23 where stored)
#   cmb1 + the chat switch (R57_CHATSW=1)           cmb1c + chat at 8 rows (R57_CHATROWS=8; the orchestrator 21:2x)
#   cmb1t = cmb1c + chat's confidence cut at t 0.4 (R57_CHATTRUNC; NewIdeas 23:31): a variant on cmb1c's stack. Line: chat
#        steady tok/s (tpsw) >= +1.5 % vs cmb1c = PASS (it joins the stack), <= 0 = KILL, else BETWEEN
#   cmb2 + A6 keep (R57_A6=keep)
#   cmb3 + COPYGATE (R57_COPY=g2)                   cmb4 + D7b on code (R57_D7B=code)
#   cmb5 + JOINT (BIGMOE_JOINT=1)                   cmb6 + K-83 (K83_EXACT=1; its drafter forms stay on the bf16 drafter)
#   FAST: fastBE / fastCE = cmb6 + R57_FAST (EvalSuite's RELAX arms B + E / C + E), tokens not checked; fastBEn = fastBE +
#   NAX6 (R57_NAX=1), fastBEh = + Leap's nax_hybrid (R57_H1C=1; it fails at start if Leap's ref2 lacks the key); fastBEhw
#   = fastBEh + the warm split (gated warm passes at 1 / 4 / 8 / 16 rows: Fable C's gated expert term; its tok/s is not a
#   speed figure); judge cmbsc by SSD MB per kept token on code (MoonShots' M-25: the joint optimum shrinks the sidecar)
# then variants on cmb4's stack: cmb3t (COPYGATE with the drafter's own t cut) and cmb32 (R57_ROWS=32, MATCH 2.5's rule),
# and cmbsc on cmb6's: the sidecar x1.5 at equal pool bytes (data/seed45/laysz45_256_sc15.json, md5 5d6d6ec0; replay
# 11-02: reads per check -7..-25 %, MB code no-gate +4.6 %, chat -0.4..-3.3 %; exact: pool sizes only).
# A lever whose arm differs in any token, or that does not speed up the stack, stays off (the arm list is edited before
# arming, from the night's verdicts). MoonShots' M-25b joint-fit pool sizes (the orchestrator 23:4x; copies in data/seed45/,
# md5 in the commit) on the default's own stack, judged by SSD MB per kept token on code vs cmb0: cmbj0 = joint45_256_0
# (equal RAM), cmbj2 = joint45_256_2070 (+2.07 GB, lifetime peak ~43.8 GB, under the ~45 GB rule). Prompts: code x4 (HumanEval 68 / 91 / 16 / 6), math, chat x4, agent x2; CAP 256.
# Shared GPU by day: rough deltas; the quiet measurement of the combined default runs first each night from 00:30
# (NewIdeas' GPU order 11-02 21:3x), Wed before FLOOR.
# usage: inside turn45 (HAVE_LOCK=1 KEEP_LOCK=1), with frozen copies (data/r57/snap/*_cmb.*) like the night's
export PATH="$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:$PATH"
cd "$HOME/BigMoE" || exit 1
mkdir -p data/cmb
LOG="$HOME/BigMoE/data/cmb/cmb.log"
say() { echo "$(date '+%m-%d %H:%M:%S') $*" >> "$LOG"; }
RUN="/bin/bash data/r57/snap/c45_runner_cmb.sh"
export C45_SCREEN=data/r57/snap/r57_screen_cmb.py C45_DEADLINE=${CMB_DEADLINE:-1420} C45_OUT="$HOME/BigMoE/data/cmb"
KEYS="humaneval/HumanEval/68,humaneval/HumanEval/91,humaneval/HumanEval/16,humaneval/HumanEval/6"
KEYS="$KEYS,math500/test/prealgebra/1640.json,ifeval/1220,ifeval/1561,multi/de3,multi/zh15,agent/bash18,agent/text23"
S1="R57_CHATSW=1"; S1C="$S1,R57_CHATROWS=8"; S2="$S1C,R57_A6=keep"; S3="$S2,R57_COPY=g2"; S4="$S3,R57_D7B=code"; S5="$S4,BIGMOE_JOINT=1"
S6="$S5,K83_EXACT=1"
# CMB_C (11-03, the orchestrator: idle NO-SWAP is 45-47 GB with every app closed, so turns must peak <= ~41 GB): the C of
# every arm (224 = M-17's sizes, peak ~38 GB + the drafters). At 224 the joint-size arms (C 256 files) drop out and cmbsc
# loads laysz45_224_sc15.json. CMB_M2=1 first runs measurement 2 (the stall trace, without / with JOINT) and the pre-route
# capture at the same C (their C 320 / 256 night slots never ran), then DAY9's chat-switch checks (chA / chB, D-23d's 21
# prompts vs chat23; (b)'s <= 45 GB peak line projected to C 256 at +0.1325 GB per C unit).
CC=${CMB_C:-256}; MGC=${CMB_MG:-6}
CHREF="$HOME/BigMoE/data/r57/chat23_ref_tokens.json:gate+t"
say "combination start (pid $$, HAVE_LOCK=$HAVE_LOCK, C $CC, MG $MGC)"
if [ "${CMB_M2:-1}" = 1 ]; then
  MG=$MGC C45_ARMS="c45t_$CC:$CC:BIGMOE_TRACE=1,R57_STALL=1 c45tj_$CC:$CC:BIGMOE_TRACE=1,R57_STALL=1,BIGMOE_JOINT=1" $RUN
  say "measurement 2 (stall trace at C $CC, without / with JOINT) rc=$?"
  PRK="humaneval/HumanEval/68,humaneval/HumanEval/16,humaneval/HumanEval/91,humaneval/HumanEval/6,humaneval/HumanEval/79"
  PRK="$PRK,humaneval/HumanEval/19,humaneval/HumanEval/161,humaneval/HumanEval/17,humaneval/HumanEval/56,humaneval/HumanEval/141"
  PRK="$PRK,humaneval/HumanEval/101,humaneval/HumanEval/63,ifeval/1220,ifeval/1561,ifeval/3280,multi/ar21,multi/de3,multi/es10"
  R57_KEYS=$PRK MG=$MGC C45_CAP=128 C45_ALARM=600 C45_REFTOK=$CHREF C45_ARMS="pr_$CC:$CC:R57_PREROUTE=1" $RUN
  say "pre-route capture rc=$?"
  CHAT="ifeval/1220,ifeval/1561,ifeval/3280,multi/ar21,multi/ar22,multi/bg0,multi/bg2,multi/de3,multi/de4,multi/es10,multi/es11"
  CHAT="$CHAT,multi/fr7,multi/fr8,multi/ja18,multi/ja19,multi/ru12,multi/ru13,multi/zh15,multi/zh16,agent/text23,agent/bash17"
  R57_KEYS=$CHAT MG=$MGC C45_ALARM=900 C45_REFTOK=$CHREF C45_ARMS="chA_$CC:$CC chB_$CC:$CC:R57_CHATSW=1" $RUN
  say "chat switch checks (chA / chB) rc=$?"
fi
R57_KEYS=$KEYS MG=$MGC C45_ALARM=900 C45_REFTOK=$CHREF C45_ARMS="cmb0_$CC:$CC" $RUN
say "cmb0 (the default) rc=$?"
REF0="$HOME/BigMoE/data/r57/cmb0_${CC}_tokens.json:gate+t"   # the default's own tokens: every later arm against them
[ -s "${REF0%%:*}" ] || { say "no cmb0 tokens: stop"; exit 1; }
J=""
[ "$CC" = 256 ] && J="cmbj0_256:256:BIGMOE_TIER_SIZES=$HOME/BigMoE/data/seed45/joint45_256_0.json cmbj2_256:256:BIGMOE_TIER_SIZES=$HOME/BigMoE/data/seed45/joint45_256_2070.json "
R57_KEYS=$KEYS MG=$MGC C45_ALARM=900 C45_REFTOK=$REF0 C45_ARMS="${CMB_ARMS:-${J}cmb1_$CC:$CC:$S1 cmb1c_$CC:$CC:$S1C cmb1t_$CC:$CC:$S1C,R57_CHATTRUNC=0.4 cmb2_$CC:$CC:$S2 cmb3_$CC:$CC:$S3 cmb4_$CC:$CC:$S4 cmb5_$CC:$CC:$S5 cmb6_$CC:$CC:$S6 cmb3t_$CC:$CC:$S4,R57_COPY=g2t cmb32_$CC:$CC:$S4,R57_ROWS=32 cmbsc_$CC:$CC:$S6,BIGMOE_TIER_SIZES=$HOME/BigMoE/data/seed45/laysz45_${CC}_sc15.json}" $RUN
say "arms rc=$?"
# the FAST preset (Kamen 21:31: two first-class tracks): EvalSuite's RELAX rules on the full exact stack, per prompt set as
# its arms B + E (code mode: gate 0.99 + copy gate + the comment loophole; chat R1) and C + E (+ R1 on code / math);
# quality-defined, so no REF2-T token check (EvalSuite's same-scores test judges them): tok/s, kept and the gated share
R57_KEYS=$KEYS MG=$MGC C45_ALARM=900 C45_ARMS="${CMB_FAST:-fastBE_$CC:$CC:$S6,R57_FAST=BE,R57_REFTOK= fastCE_$CC:$CC:$S6,R57_FAST=CE,R57_REFTOK= fastBEn_$CC:$CC:$S6,R57_FAST=BE,R57_NAX=1,R57_REFTOK= fastBEh_$CC:$CC:$S6,R57_FAST=BE,R57_NAX=1,R57_H1C=1,R57_REFTOK= fastBEhw_$CC:$CC:$S6,R57_FAST=BE,R57_NAX=1,R57_H1C=1,R57_REFTOK=,R57_WARMSPLIT=3}" $RUN
say "FAST arms rc=$?"
CMB_C=$CC "$HOME/BigMoE/.venv/bin/python" scripts/combo_report.py >> "$LOG" 2>&1; say "report rc=$? (data/cmb/combo_report.txt)"
say "combination end"
