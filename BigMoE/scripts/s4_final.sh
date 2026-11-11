#!/bin/bash
# Sat 7 Nov (NewIdeas 01:2x: final v1 composition; EvalSuite's S4 file contract): freeze the final config's code into
# data/r57/snap/frozen_s4 + data/r57/snap/r57_screen_s4.py, commit that snapshot, then write data/stack/final_env.txt (code_env
# format; its comment carries 'Sat 256', 'MEM 47 GB' and 'screen data/r57/snap/r57_screen_s4.py') and commit it. S4 starts when
# the file lands, so run this ONLY on NewIdeas' word (after runner 49's code verdicts).
# usage: S4_VARIANT=stackb_s2 | stack_s2  [S4_NW=1]  bash scripts/s4_final.sh
#   stackb_s2 = the stack + ship B span + MATCH g2 at 16 rows + S2 (v1 if runner 49 passes B's code part)
#   stack_s2  = the stack + S2 (v1 if B rolls back or its same-scores fails)
#   S4_NW=1   adds NW v3 (only if S1's engine A/B PASSed in time)
#   S4_KEEP=1 adds REF2_KEEP_CACHES=1 (only with S2c's EXACT proof PASS + speed PASS)
export PATH="$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:$PATH"
cd "${S4_ROOT:-$HOME/BigMoE}" || exit 1
V=${S4_VARIANT:?set S4_VARIANT=stackb_s2 or stack_s2}
QA="BIGMOE_EXT_BUILD=build_qom,R57_REFTOK=,R57_FAST=QOM,R57_D17=1,R57_D7B=code,R57_HC8=1,REF2_HC8_FAST=1,R57_DNR=1,R57_HCB=1"
QA="$QA,REF2_SMALL_NATIVE=1,R57_D7BNAX=1,REF2_OOO_1C=1"
ST="R57_NAX=1,REF2_NAX_1CQ=1,REF2_DN_KS=1,REF2_GDN_LAZY=1,D7BNAX_32=2,R57_FASTTAX=1,R57_SBN=2,R57_DSUB=1,R57_SVH=1"
case "$V" in
  stackb_s2) ENV="$QA,$ST,R57_RELAX_SPAN=1,R57_RELAX_SPAN_RULE=R2:2:0.25,R57_COPY=g2" ;;
  stack_s2)  ENV="$QA,$ST" ;;
  *) echo "S4_VARIANT=$V: unknown" >&2; exit 2 ;;
esac
[ "${S4_NW:-0}" = 1 ] && ENV="$ENV,R57_NEVERWAIT=1,REF2_NEVERWAIT=1"
[ "${S4_NW:-0}" = 1 ] && [ "${S4_NWCNT0:-0}" = 1 ] && ENV="$ENV,R57_NWCNT=0"   # S1b (Sat 05:1x): NW without its counter graph
[ "${S4_KEEP:-0}" = 1 ] && ENV="$ENV,REF2_KEEP_CACHES=1"   # S2c (only with its EXACT proof PASS + speed PASS)
ENV="$ENV,R57_QOM_SETS=humaneval,mbpp"
D=${S4_DIR:-s4}                          # S4_DIR (Sat 05:5x): v1' goes to s4b/ so S4's PASSed s4/ snapshot stays
FZ="$PWD/data/r57/snap/frozen_$D"; rm -rf "$FZ"; mkdir -p "$FZ"
SD=data/r57/snap/$D; mkdir -p "$SD"     # own dir: never touch data/r57/snap/*.py while EvalSuite's runs load them
cp scripts/r57_screen_fast.py "$SD/r57_screen_$D.py" && cp scripts/relax_delta.py scripts/relax_span.py scripts/inert_span.py \
  "$SD/" || exit 2
[ -n "$S4_DRYRUN" ] || { cp -R "$HOME/Leap/scripts" "$FZ/Leap_scripts" && cp -R "$HOME/Leap/third_party" "$FZ/Leap_third_party" \
  && cp -R "$HOME/TrainLab/sticky" "$FZ/TrainLab_sticky" && cp -R "$HOME/Breakthrough/keepset" "$FZ/Breakthrough_keepset" || exit 2; }
find "$FZ" -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null
sed -i '' -e "s#\"~/Leap/scripts\"#\"$FZ/Leap_scripts\"#g" -e "s#\"~/Leap/third_party\"#\"$FZ/Leap_third_party\"#g" \
  -e "s#\"~/TrainLab/sticky\"#\"$FZ/TrainLab_sticky\"#g" -e "s#\"/Users/kamen/Breakthrough/keepset\"#\"$FZ/Breakthrough_keepset\"#g" \
  "$SD/r57_screen_$D.py"
NP=$(grep -c "frozen_$D" "$SD/r57_screen_$D.py")
MD=$(md5 -q "$FZ/Leap_scripts/ref2.py" "$FZ/Leap_scripts/session_vocab.py" "$FZ/Breakthrough_keepset/qom_keep.py" 2>/dev/null | cut -c1-8 | tr '\n' ' ')
[ -n "$S4_DRYRUN" ] && { echo "dry run: $NP screen paths; env $(echo "$ENV" | tr ',' '\n' | grep -c =) K=V"; exit 0; }
git add "data/r57/snap/frozen_$D" ":!data/r57/snap/frozen_$D/**/__pycache__/**" "$SD" ":!data/r57/snap/$D/__pycache__/**" && \
> (a quote from the project owner, kept private)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_011DvdV6eUB4NgBbVkyVFPrT" || exit 3
SNAP=$(git rev-parse --short HEAD)
printf '%s\n' "# FINAL ${S4_LABEL:-v1} for EvalSuite's S4 (BigMoE s4_final.sh, snapshot $SNAP): Sat 256, MEM 47 GB, screen data/r57/snap/$D/r57_screen_$D.py; $V${S4_NW:+ + NW v3}${S4_KEEP:+ + KEEP}; NewIdeas' composition. One comma-separated env line:" \
  "$ENV" > data/stack/final_env.txt
> (a quote from the project owner, kept private)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_011DvdV6eUB4NgBbVkyVFPrT" || exit 4
echo "final_env.txt written: $V${S4_NW:+ + NW v3}${S4_KEEP:+ + KEEP}, snapshot $SNAP, HEAD $(git rev-parse --short HEAD)"
