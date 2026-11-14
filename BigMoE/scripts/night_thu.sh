#!/bin/bash
# Thu early, one quiet ticket of ~95 min (the orchestrator Wed 10:1x fixes the slot in the Wed-night order ~22:00: at
# RELAX_DONE ~06:40 or after RELAX's quiet arms, before T0; apps closed: C 256 swapped +2.3 GB by day, Wed 09:15):
#   1. the adopted default's headline at C 256, 11-02's footing (scripts/d256_wed.sh, old vs new A-B-B-A, 14 prompts; ~25 min)
#   2. (the orchestrator Wed 21:3x) the C 224 FAST trio u3ct / qallt / qallft (scripts/c224_thu.sh, 12 HumanEval; ~25 min):
#      the Wed 23:00 ticket could not run C 224 with Kamen's apps open (NO-SWAP ~41 GB, scripts/eve_2300b.sh)
#   3. (the orchestrator Wed 22:1x) Leap's DKV token check (scripts/dkv_thu.sh: R57_DKV=1 + R57_DKV_ADD=1 in the exact
#      default, 14 ref14 prompts at C 224, every token vs REF2-T; ~7 min, taken from the LOO's time)
#   4. which lever costs agent at C 224 (scripts/agent_loo.sh, leave-one-out on 5 agent prompts, 9 arms; ~67 min: 7.4 min
#      per arm, measured Wed 13:49; the day try stopped on the swap rule in its 2nd arm at 14:02, only old ran). LOO arms
#      are cut from the tail: none starts after 08:35 (the orchestrator Wed 21:5x; the script caps ALOO_DEADLINE), so the
#      ticket releases and touches q42/GO by ~08:45, runner 42's original start. From a 07:10 grant: old, new, -vtier,
#      -chat (the priority order) run; -lever1, -K83, -keepalive, new2, old2 are cut.
# Both lines are pre-registered in their headers. Swap: the 2 GB rule per run (each script reads its own base). The new
# default holds its footprint + the 7.8 GB purgeable volatile tier (~50 GB at C 224, ~54 at C 256; the day try grew swap
# +1.8 GB in the new arm from NO-SWAP 51 GB, old +0.3), so MG is raised: an arm below 52.7 GB (C 256) / 50.5 GB (C 224)
# NO-SWAP at its start is skipped, not swapped.
# usage (a harness background task; runner 42 waits for q42/GO, so touch it after the ticket however it ends; Thu 00:1x armed as#   turn45.sh 32 120 with WAITMEM=36: the script itself picks what fits, see the 00:1x block below):
#   GAP0=0 GAP=0 MARGIN=4 NOTBEFORE=hh:mm NOTAFTER=hh:mm ALOO_DEADLINE=0900 bash scripts/turn45.sh 48 120 \
#   "BigMoE: C 256 headline + C 224 trio + agent LOO, MEM ~52 GB" /bin/bash scripts/night_thu.sh; touch ~/EvalSuite/results/q42/GO
# D256_DEADLINE / C224T_DEADLINE / DKVT_DEADLINE (hhmm) default to start + 27 / + 54 / + 66 min, so a late grant does not
# skip steps 1-3.
export PATH="$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:$PATH"
cd "$HOME/BigMoE" || exit 1
LOG="$HOME/BigMoE/data/night_thu.log"
say() { echo "$(date '+%m-%d %H:%M:%S') $*" >> "$LOG"; }
export D256_MG=${D256_MG:-10} ALOO_MG=${ALOO_MG:-12}
say "night_thu start (pid $$, MG $D256_MG / $ALOO_MG)"
say "builds: BigMoE $(git rev-parse --short HEAD), Leap $(git -C "$HOME/Leap" rev-parse --short HEAD 2>/dev/null) (Leap's _rows16 change moves the -K83 arm only if it lands first)"
export D256_DEADLINE=${D256_DEADLINE:-$(date -v+27M +%H%M)} C224T_DEADLINE=${C224T_DEADLINE:-$(date -v+54M +%H%M)}
export DKVT_DEADLINE=${DKVT_DEADLINE:-$(date -v+66M +%H%M)}
# The orchestrator Wed 21:5x: no new agent-LOO arm after 08:35, so the ticket releases and touches q42/GO by ~08:45 (runner 42
# keeps its ~08:45 start; runner 43 must stay inside its 14:00 no-new-chunk rule). A morning start also caps steps 1-2 at
# 08:10 / 08:30, so a late grant cannot push past ~08:45. (Only for a 04:00-09:59 start: an evening test keeps its own.)
H=$(date +%H%M)
if [ "$H" -ge 0400 ] && [ "$H" -lt 1000 ]; then
  [ "$D256_DEADLINE" -gt 0810 ] && D256_DEADLINE=0810
  [ "$C224T_DEADLINE" -gt 0830 ] && C224T_DEADLINE=0830
  [ "$DKVT_DEADLINE" -gt 0830 ] && DKVT_DEADLINE=0830
  [ "${ALOO_DEADLINE:-2400}" -gt 0835 ] && ALOO_DEADLINE=0835
fi
export ALOO_DEADLINE
say "deadlines: C 256 headline $D256_DEADLINE, C 224 trio $C224T_DEADLINE, DKV check $DKVT_DEADLINE, agent LOO ${ALOO_DEADLINE:-none}"
# Thu 00:1x (the orchestrator: NO-SWAP was 36 GB at 22:51 with nothing big running, the sessions' memory, and nobody closes
# apps tonight): memory-adaptive from NO-SWAP once at the start (RULES: lifetime peak + 4 GB; each arm's runner gate still
# re-checks at its own start). The C 256 headline only if its MG 10 gate fits (52.7), else skipped. The trio at the largest
# C whose qallft peak 34.8 + 0.134 (C - 160) + 4 fits (224: 47.4, 208: 45.2, 192: 43.1, 176: 40.9), else without qallft
# (qall 31.7 + 0.134 (C - 160) + 4: 224: 44.3, 208: 42.1, 192: 40.0, 176: 37.8), MG 4. DKV at C 224 (the exact default on
# 14 prompts: 42.1 + 4.5 = 46.6) else C 160 (31.8 + 4.5 = 36.3; tokens do not depend on C), MG 4. The LOO as before (its
# arms skip themselves below 50.5).
avail_gb() { vm_stat | awk '/page size of/ {ps=$8} /^Pages (free|purgeable):/ || /^File-backed pages:/ {v=$NF; sub("[.]","",v); s+=v} END {printf "%.1f", s*ps/1e9}'; }
A=${NT_AVAIL:-$(avail_gb)}
fits() { awk -v a="$A" -v n="$1" 'BEGIN {exit !(n <= a)}'; }
need() { awk -v b="$1" -v c="$2" 'BEGIN {printf "%.1f", b + 0.134 * (c - 160)}'; }
say "NO-SWAP $A GB at the start"
if fits 52.7; then
  D256_C=256 bash scripts/d256_wed.sh; say "1 C 256 headline rc=$? ($(grep Verdict data/d256/d256_report_256.txt 2>/dev/null | cut -c1-110))"
else say "1 C 256 headline SKIPPED: NO-SWAP $A GB < 52.7"; fi
C2=; NQ=
for c in 224 208 192 176; do fits $(need 38.8 $c) && { C2=$c; break; }; done
[ -z "$C2" ] && for c in 224 208 192 176; do fits $(need 35.7 $c) && { C2=$c; NQ=1; break; }; done
if [ -n "$C2" ]; then
  C224T_C=$C2 C224T_NOQF=$NQ C224T_MG=4 bash scripts/c224_thu.sh
  say "2 C $C2 FAST trio${NQ:+ without qallft} rc=$? ($(grep -E '^qallt vs' data/c224t/c224t_report.txt 2>/dev/null | cut -c1-140))"
else say "2 FAST trio SKIPPED: NO-SWAP $A GB < 37.8 (C 176 without qallft)"; fi
DC=; fits 46.6 && DC=224; [ -z "$DC" ] && fits 36.3 && DC=160
if [ -n "$DC" ]; then
  DKVT_C=$DC DKVT_MG=4 bash scripts/dkv_thu.sh
  say "3 DKV token check (C $DC) rc=$? ($(grep Verdict data/dkvt/dkvt_report.txt 2>/dev/null | cut -c1-60))"
else say "3 DKV token check SKIPPED: NO-SWAP $A GB < 36.3"; fi
bash scripts/agent_loo.sh; say "4 agent LOO rc=$? ($(grep Verdict data/aloo/aloo_report.txt 2>/dev/null | cut -c1-140))"
say "night_thu end"
