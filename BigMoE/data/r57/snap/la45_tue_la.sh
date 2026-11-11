#!/bin/bash
# R-57's look-ahead at 45 GB (the orchestrator 11-02 21:4x: cloud task 15 makes the SSD stack -23 of the ladder's -48 ms; "move
# R-57's look-ahead up right behind the combination"). One ticket, S45's six prompts at C 256 (every token vs REF2-T: reads
# only, so exact), on ext/build_la (BIGMOE_PF_SIDE; defaults == ext/build):
#   la0  the default (gate+t)                                   la1  gate+t+pf: the ranked L+1 list (PF_B 96, PRED_LINES 4)
#        into 16 spare slots per layer (R57_SPARE; non-JIT: reads issue at the base row's release)
#   la2  la1 + BIGMOE_PF_SIDE=24: L+1's sidecar planes for the 24 most likely ids too (the sidecars are 60-70 % of reads)
#   la3  la2 + Fit12's held-out router correction (R57_CORR, R-57b's maps)
#   la4  la2 + BIGMOE_JOINT=1 (if tonight's c45tj_320 is exact)
# Line (pre-registered 11-02 21:5x, cloud task 12's requirement): PASS = demand misses per check <= 60 with the look-ahead's
# used / (used + demand misses) >= 0.69 and waste (issued - used) <= 0.5x used, and steady tok/s >= +5 % vs la0, every token
# equal; KILL = steady < +2 % (R-57b died at C 176 on late reads); else BETWEEN. Shared GPU by day = rough; a PASS gets a
# quiet confirmation at night. Added 11-02 23:3x before any run (the orchestrator: cloud task 18 / Fable C, code's SSD is
# bandwidth-bound and the look-ahead wastes ~31 % of what it prefetches): PASS also needs SSD MB per kept token (the
# process's disk bytes: demand + prefetch) not up > 3 % vs la0 on code; the report gives demand GB/s per code check too.
# then M-20 SSDWAKE (MoonShots, the orchestrator 11-02 22:1x): BIGMOE_SSD_KEEPALIVE_US=50 (a 4 KB F_NOCACHE read every 50 us
# while decoding) on the default, A-B-A-B: ka0 / ka1 / ka0r / ka1r. Line (pre-registered 22:2x, the orchestrator's): PASS =
# steady tok/s on code (HumanEval 68 / 91) >= +2 % (B vs A, both pairs pooled) with every token equal; KILL <= +0.5 %; else
# BETWEEN. Every run also logs its demand bursts and how many followed >= 0.25 / >= 1 ms of SSD idle (the ext's counters).
# usage: inside turn45 (HAVE_LOCK=1 KEEP_LOCK=1), with frozen copies (data/r57/snap/*_la.*)
export PATH="$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:$PATH"
cd "$HOME/BigMoE" || exit 1
mkdir -p data/la45
LOG="$HOME/BigMoE/data/la45/la45.log"
say() { echo "$(date '+%m-%d %H:%M:%S') $*" >> "$LOG"; }
RUN="/bin/bash data/r57/snap/c45_runner_la.sh"
export C45_SCREEN=data/r57/snap/r57_screen_la.py C45_DEADLINE=${LA_DEADLINE:-1420} C45_OUT="$HOME/BigMoE/data/la45"
B="BIGMOE_EXT_BUILD=build_la"
P="$B,R57_ARMS=gate+t+pf,BIGMOE_PRED_LINES=4,R57_SPARE=16"
CC=${LA_C:-256}; MGC=${LA_MG:-5}                       # LA_C (11-03: 224 when 256 does not fit the ~45 GB idle NO-SWAP)
say "look-ahead start (pid $$, HAVE_LOCK=$HAVE_LOCK)"
MG=$MGC C45_ARMS="${LA_ARMS:-la0_$CC:$CC:$B la1_$CC:$CC:$P la2_$CC:$CC:$P,BIGMOE_PF_SIDE=24 la3_$CC:$CC:$P,BIGMOE_PF_SIDE=24,R57_CORR=$HOME/Fit12/data/r57b la4_$CC:$CC:$P,BIGMOE_PF_SIDE=24,BIGMOE_JOINT=1}" $RUN
say "arms rc=$?"
K="$B,BIGMOE_SSD_KEEPALIVE_US=50"
MG=$MGC C45_ARMS="${KA_ARMS:-ka0_$CC:$CC:$B ka1_$CC:$CC:$K ka0r_$CC:$CC:$B ka1r_$CC:$CC:$K}" $RUN
say "keepalive arms rc=$?"
# the volatile expert tier smoke (BIGMOE_VTIER on ext/build_vt; added 11-03 15:3x, unit-tested: scripts/test_vtier.py PASS):
# vt0 = the default on build_vt, vt1 = + a 64 / 32-slot tier per base / sidecar pool (~7.8 GB volatile at C 224, about C 288
# of base slots). Line (pre-registered before any run): exact (every token == REF2-T); PASS = SSD MB per kept token <= -10 %
# vs vt0 with steady tok/s not down; KILL = MB down < 3 % or tok/s down > 2 %; else BETWEEN. The tier's GB beside the peak.
V="BIGMOE_EXT_BUILD=build_vt"
MG=$MGC C45_ARMS="${VT_ARMS:-vt0_$CC:$CC:$V vt1_$CC:$CC:$V,R57_VTIER=64,32}" $RUN
say "volatile tier smoke rc=$?"
say "look-ahead end"
