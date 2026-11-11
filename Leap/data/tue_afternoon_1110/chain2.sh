#!/bin/bash
# Tue 14:xx (orchestrator 14:0x): (1) the 24 GB line's peaks at 10K / 30K tokens, default prefill and PF-A (run.py --config chat
# --mem 24, 64 tokens); (2) the PF-B kernel-level bit check (kcheck.py; a few GB, seconds of GPU). One job at a time, share
# lock, swap guard +2 GB, all by 16:45.
A=/private/tmp/claude-501/-Users-kamen-OjoX/f2971937-6c95-4e86-a049-05beff6a8e32/scratchpad/aft
SP=/private/tmp/claude-501/-Users-kamen-OjoX/f2971937-6c95-4e86-a049-05beff6a8e32/scratchpad
CL=/Users/kamen/Leap/release/cleanclone/final/clone; cd $CL; LOG=$A/chain2.log
END=$(date -j -f "%Y-%m-%d %H:%M:%S" "2026-11-10 16:45:00" +%s)
noswap() { vm_stat | awk '/page size of/ {ps=$8} /^Pages (free|speculative|purgeable):/ || /^File-backed pages:/ {v=$NF; sub("[.]","",v); s+=v} END {printf "%d", s*ps/1e9}'; }
swap() { sysctl -n vm.swapusage | sed -E 's/.*used = ([0-9.]+)M.*/\1/' | cut -d. -f1; }
busy() { peer ls 2>/dev/null | grep "^- " | grep -oE "· (shared run \(others may work\)|benchmarking|waiting for a quiet window)[^·]*" | grep -cE "^· (benchmarking|waiting for a quiet window)|MEM ([6-9]|[1-9][0-9]) GB"; }
say() { echo "$(date +%T) $*" | tee -a $LOG; }
n=0; until [ "$(busy)" = 0 ] && [ "$(noswap)" -ge 42 ]; do n=$((n + 1)); [ $n -gt 60 ] && { say "SKIP: not clear (busy $(busy), NO-SWAP $(noswap))"; exit 0; }; sleep 30; done
trap 'peer unlock >/dev/null 2>&1' EXIT
peer lock share 60m --cpu 300 --gpu 100 "MEM 34 GB: Leap 24 GB peaks at 10K / 30K (4 short runs) + a PF-B kernel bit check" > /dev/null 2>&1
S0=$(swap); say "GO: NO-SWAP $(noswap) GB, swap $S0 MB"
for L in 10k 30k; do
  for arm in default pfa; do
    [ $(( $(date +%s) + 420 )) -le $END ] || { say "SKIP $L $arm: time"; continue; }
    E=""; [ $arm = pfa ] && E="--env R57_PFLM=1"
    .venv/bin/python run.py --config chat --mem 24 --prompt-file $SP/prefillbench/prompt_$L.txt --max-tokens 64 $E > $A/p24_${L}_$arm.out 2>&1 &
    P=$!
    while kill -0 $P 2>/dev/null; do [ $(( $(swap) - S0 )) -gt 2048 ] && { say "STOP $L $arm: swap $(swap) MB"; for c in $(pgrep -P $P); do kill -TERM $c; done; kill -TERM $P; }; sleep 3; done
    wait $P; say "  24 GB $L $arm: $(grep -a -E '^whole decode' $A/p24_${L}_$arm.out | cut -c1-170) | swap $(swap) MB"
  done
done
say "kernel check:"; .venv/bin/python $A/kcheck.py engine/models/bare 1 2>&1 | tee -a $LOG
say "kernel check, layer 30:"; .venv/bin/python $A/kcheck.py engine/models/bare 30 2>&1 | tail -8 | tee -a $LOG
say "chain2 end: NO-SWAP $(noswap) GB, swap $(swap) MB (start $S0)"
