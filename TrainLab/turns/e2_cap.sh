#!/bin/sh
# E2 capture job (MTP Head, Mon 11-02; REPORT 'E2 / D2'), inside a held SHARE lock (run_shared.sh; a data run, no timing): the
# 125B's answers to MBPP (sticky/dlog_capture.py DLOG_SRC=mbpp, resumable) until 52K code answer tokens. C by NO-SWAP with the
# shared rule's +8 GB: C 160 >= 41, 128 >= 37, 96 >= 33, 64 >= 29, 48 >= 27 (peak ~19 GB), else not starting. Target 17K code
# tokens until x2 is walked (x2 trains sooner), then 52K. Budget from TL_LIMIT; swap guard +2 GB.
cd $HOME/TrainLab
PY=$HOME/BigMoE/.venv/bin/python
trap 'pkill -f dlog_capture.py; pkill -P $$; exit 143' TERM
swap() { sysctl -n vm.swapusage | awk '{v = $6; sub("M", "", v); print int(v)}'; }
noswap() { vm_stat | awk '/page size of/ {ps=$8} /^Pages (free|speculative|purgeable):/ || /^File-backed pages:/ {v=$NF; sub("[.]","",v); s+=v} END {printf "%d", s*ps/1e9}'; }
sleep 20
i=0; while [ "$(noswap)" -lt 25 ] && [ $i -lt 12 ]; do i=$((i + 1)); sleep 30; done   # a short dip: wait up to 6 min in the slot
SW0=$(swap); F=$(noswap)
# rows per check so that rows x top-10 fits the base pool (the engine aborts past C_m), with the LSB sidecar pool scaled
# with C (C_l = 112 C / 192; at a fixed 112 it would leave C 48 only ~37 base slots): C_m ~ (C - 0.14 C_l) / 0.86 = 51 /
# 68 / 102 / 137 / 171 at C 48 / 64 / 96 / 128 / 160 -> 4 / 6 / 9 / 13 / 16 rows (C 32: ~34, 3 rows, at >= 25 GB);
# BIGMOE_MAXK = 10 x rows. The 125B's
# tokens depend on neither (exact; pool sizes and rows only change speed).
if [ "$F" -ge 41 ]; then C=160; RW=16; elif [ "$F" -ge 37 ]; then C=128; RW=13; elif [ "$F" -ge 33 ]; then C=96; RW=9; elif [ "$F" -ge 29 ]; then C=64; RW=6; elif [ "$F" -ge 27 ]; then C=48; RW=4; elif [ "$F" -ge 25 ]; then C=32; RW=3; else
  echo "== E2 capture $(date +%H:%M:%S): NO-SWAP ${F} GB < 25: not starting"; exit 1; fi
if [ -f data/d02b_walk_e2x2.txt ]; then TG=52000; else TG=17000; fi
LIM=${TL_LIMIT:-6000}; B=$((LIM - 400)); HD=$((LIM - 250))
echo "== E2 MBPP capture $(date +%H:%M:%S): ${F} GB NO-SWAP -> C $C; swap ${SW0} MB; $(df -g $HOME | awk 'NR==2 {print $4}') GB free disk; budget ${B} / ${HD} s; target ${TG} code tokens; ${RW} rows per check"
( while sleep 10; do
    if [ $(( $(swap) - SW0 )) -gt 2048 ]; then echo "== swap grew > 2 GB ($(swap) MB): stopping"; pkill -f dlog_capture.py; exit; fi
  done ) & G=$!
BIGMOE_TIER_CL=$((112 * C / 192)) BIGMOE_MAXK=$((RW * 10)) DLOG_ROWS=$RW DLOG_SRC=mbpp DLOG_NIGHT=d1102mbpp DLOG_TARGET=$TG DLOG_MAXTOK=512 DLOG_BUDGET=$B DLOG_HARD=$HD DLOG_CAP_GB=7 \
  DLOG_MIN_FREE_GB=114 HF_HUB_OFFLINE=1 $PY sticky/dlog_capture.py $C 2>&1 | tee -a logs/e2_cap.out \
  | grep --line-buffered -v "Token indices\|moe-sorted-gather"
kill $G 2>/dev/null
echo "== E2 capture done $(date +%H:%M:%S), swap $(swap) MB (start $SW0)"
