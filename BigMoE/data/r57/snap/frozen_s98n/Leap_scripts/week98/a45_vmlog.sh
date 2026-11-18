#!/bin/bash
# a45_vmlog.sh (Leap, Mon 16 Nov 2026 03:1x): memory beside tonight's a45, which may start at NO-SWAP 42 GB, its measured
# peak (the orchestrator 03:0x; normally its peak + 4 = 47). Every 10 s: swap used, NO-SWAP (cc3_runner's formula, one
# decimal), the compressor, and the cumulative swapouts / pageouts / compressions (vm_stat).
# Bias check, PRE-REGISTERED (03:1x, before a45 starts): if Swapouts grow inside a45's timed window (cc3.log "run a45" ->
# "a45 rc"), its misses / SSD-wait part is labelled "tight memory: may read high" (swap writes share the SSD with the
# expert reads); the dense part (d1 / d16 / d32, GPU-bound, no SSD reads) stands either way. No swapouts in the window:
# the split stands as measured.
# Ends when the marker appears (a45_last.sh writes it on ANY end) or at the end time.
# usage: a45_vmlog.sh <out file> <marker> "<END YYYY-MM-DD HH:MM:SS>"
O=$1; M=$2; END=$(date -j -f "%Y-%m-%d %H:%M:%S" "${3:?END}" +%s) || exit 2
echo "# a45_vmlog $(date '+%m-%d %H:%M:%S'): time, swap used MB, NO-SWAP GB, compressor GB, cumulative swapouts / pageouts / compressions" >> $O
while [ ! -e $M ] && [ $(date +%s) -lt $END ]; do
  sw=$(sysctl -n vm.swapusage | sed -E 's/.*used = ([0-9.]+)M.*/\1/')
  vm_stat | awk -v t="$(date +%T)" -v sw="$sw" '/page size of/ {ps=$8} {v=$NF; sub("[.]","",v)}
    /^Pages free:/ {f=v} /^Pages purgeable:/ {p=v} /^File-backed pages:/ {fb=v} /^Pages occupied by compressor:/ {c=v}
    /^Swapouts:/ {so=v} /^Pageouts:/ {po=v} /^Compressions:/ {cm=v}
    END {g=ps/1e9; printf "%s swap %s MB, NO-SWAP %.1f, compressor %.1f, swapouts %d, pageouts %d, compressions %d\n", t, sw, (f+p+fb)*g, c*g, so, po, cm}' >> $O
  sleep 10
done
echo "# end $(date '+%m-%d %H:%M:%S')$([ -e $M ] && echo ": the marker appeared")" >> $O
