#!/bin/sh
# Fit12 B4 (2026-10-25): expert read-path microbench under a short quiet lock (SSD timing; no GPU compute). Read-only on
# ~/BigMoE/models/bare/records3. Pre-registered in PLAN.md (B4). Launch (background task) after the quiet-lock grant:
# cd ~/Fit12 && sh ~/Fit12/scripts/b4.sh
F=$HOME/Fit12; OUT=$F/data/b4; LOG=$F/logs/summary.txt
mkdir -p "$OUT"
log() { echo "$(date +%H:%M:%S) $*" >> "$LOG"; }
log "B4 start (swap $(sysctl -n vm.swapusage | awk '{print $6}'), free $(df -g / | awk 'NR==2 {print $4}') GB)"
cd $F/b4 && xcrun swiftc -O -o b4bench b4bench.swift > "$OUT/build.txt" 2>&1 || { log "B4: build failed: $(head -3 $OUT/build.txt | tr '\n' ' ' | cut -c1-300)"; [ -n "$B4_KEEP" ] || peer unlock; exit 1; }
./b4bench 50 > "$OUT/b4.txt" 2>&1; rc=$?
log "B4 rc $rc: $(grep -A5 'fixed a' $OUT/b4.txt | tr '\n' ' ' | cut -c1-400)"
[ -n "$B4_KEEP" ] || peer unlock > /dev/null 2>&1   # B4_KEEP=1: the caller keeps the lock (b4tk3.sh)
