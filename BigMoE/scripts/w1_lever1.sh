#!/bin/bash
# WALL 1, lever 1 at 45 GB (docs/PLAN_wall1_lever1.md, docs/WED_TESTS.md item 2): W1B's six prompts, every token vs REF2-T,
# all arms on ext/build_w1 (5565a56: Leap's W1-PICKUP + S0 instrumentation + the traced direct path; every new path off by
# default), cumulative, traced (BIGMOE_TRACE + R57_STALL -> {tag}_stall.json for stall_report.py and s0_report.py):
#   w1t    the default                                   w1tj   + JOINT (one burst per layer: base + sidecar)
#   w1tk   + JOINT + keepalive (KEEPALIVE_US 50)         w1td   + the copy split over 3 threads (BIGMOE_COPY_PAR)
#   w1tc   + coherent one-row spin (SPIN_COHERENT)       w1tp   + system-scope routing post (POST_SYS)
# then the default and the full stack untraced (w1u / w1uf) for steady tok/s and SSD MB per kept token.
# Lines (pre-registered in PLAN_wall1_lever1.md): overall = exact AND the fixed part <= 10 ms per check AND steady >= +5 %
# (w1uf vs w1u) = PASS, steady < +1 % = KILL; per step: JOINT busy <= -8 ms and bursts <= 50; keepalive d -0.04 ms and busy
# -1 ms; pickup: cadence a - burst a falls >= 0.03 ms. A step that differs in any token stays off.
# usage: inside turn45 (HAVE_LOCK=1 KEEP_LOCK=1); W1_C (default 160; 224 when NO-SWAP >= 44), W1_MG (default 3)
export PATH="$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:$PATH"
cd "$HOME/BigMoE" || exit 1
mkdir -p data/w1
LOG="$HOME/BigMoE/data/w1/w1.log"
say() { echo "$(date '+%m-%d %H:%M:%S') $*" >> "$LOG"; }
RUN="/bin/bash data/r57/snap/c45_runner_w1.sh"
export C45_SCREEN=data/r57/snap/r57_screen_w1.py C45_DEADLINE=${W1_DEADLINE:-2400} C45_OUT="$HOME/BigMoE/data/w1"
CC=${W1_C:-160}; MGC=${W1_MG:-3}
[ -f ext/build_w1/bigmoe_ext.cpython-312-darwin.so ] || { say "no ext/build_w1: STOP"; exit 2; }
T="BIGMOE_EXT_BUILD=build_w1,BIGMOE_TRACE=1,R57_STALL=1"
J="$T,BIGMOE_JOINT=1"; K="$J,BIGMOE_SSD_KEEPALIVE_US=50"; D="$K,BIGMOE_COPY_PAR=1"   # (M-29: direct reads into today's UNALIGNED slots are +843 us: out; the copy split over 3 threads is the interim tail cut)
CO="$D,BIGMOE_SPIN_COHERENT=1"; P="$CO,BIGMOE_POST_SYS=1"
say "lever 1 start (pid $$, C $CC, MG $MGC)"
MG=$MGC C45_ARMS="w1t_$CC:$CC:$T w1tj_$CC:$CC:$J w1tk_$CC:$CC:$K w1td_$CC:$CC:$D w1tc_$CC:$CC:$CO w1tp_$CC:$CC:$P" $RUN
say "traced arms rc=$?"
U="BIGMOE_EXT_BUILD=build_w1"
MG=$MGC C45_ARMS="w1u_$CC:$CC:$U w1uf_$CC:$CC:$U,BIGMOE_JOINT=1,BIGMOE_SSD_KEEPALIVE_US=50,BIGMOE_COPY_PAR=1,BIGMOE_SPIN_COHERENT=1,BIGMOE_POST_SYS=1" $RUN
say "untraced default / full stack rc=$?"
.venv/bin/python scripts/stall_report.py data/r57 "w1t_$CC,w1tj_$CC" >> "$LOG" 2>&1
.venv/bin/python scripts/s0_report.py data/r57 "w1t_$CC,w1tj_$CC,w1tk_$CC,w1td_$CC,w1tc_$CC,w1tp_$CC" >> "$LOG" 2>&1
say "reports rc=$? (data/r57/stall_report.txt, data/r57/s0_report.txt)"
say "lever 1 end"
