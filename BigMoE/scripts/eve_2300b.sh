#!/bin/bash
# Wed 21:3x: the 23:00 ticket, made memory-adaptive (RULES: NO-SWAP >= lifetime peak + 4 GB under a quiet lock).
# The early ticket (20:35-21:15) never queued: NO-SWAP stayed 41 GB < its WAITMEM 44 with Kamen's apps open. The old 23:00
# ticket (eve_2300.sh: C 224, PEAK 40) would not have started either, and PEAK 40 understated its biggest arm.
# Measured lifetime peaks (data/r57/*_runs.json, mem[1]): U3C on the 12 HumanEval 30.9 GB at C 160, 39.4 at C 224
# (0.134 GB per row = recsz x 48 layers); the QOM all-in +0.3 (qalq 31.2); forecast reads (FC_F 4) +2.6-3.3 (qaf / qfx
# 33.5-33.6 vs qs / qo1 30.2-30.9 on 2 prompts at C 160). So qallf peaks at ~34.8 + 0.134 (C - 160) GB (0.5 spare):
# C 224 needs NO-SWAP >= 47.4, 208 >= 45.2, 192 >= 43.1, 176 >= 40.9. Here every all-in arm runs at the largest of
# 224 / 208 / 192 / 176 that fits at the grant (one C for all arms, so the ratios stay clean); lever 2 stays at C 160
# (l2_wed.sh, ~35 GB with forecast reads on 14 prompts).
cd "$HOME/BigMoE" || exit 1
avail_gb() { vm_stat | awk '/page size of/ {ps=$8} /^Pages (free|speculative|purgeable):/ || /^File-backed pages:/ {v=$NF; sub("[.]","",v); s+=v} END {printf "%.1f", s*ps/1e9}'; }
A=${EVE_AVAIL:-$(avail_gb)}; C=176
for c in 224 208 192; do
  awk -v a="$A" -v c="$c" 'BEGIN {exit !(34.8 + 0.134 * (c - 160) + 4 <= a)}' && { C=$c; break; }
done
[ -n "$EVE_DRY" ] && { echo "C $C"; exit 0; }
echo "$(date '+%m-%d %H:%M:%S') eve_2300b: NO-SWAP $A GB at the start -> the all-in at C $C (the largest C whose peak + 4 fits)" >> data/allin/allin.log
ALLIN_C=$C ALLIN_DEADLINE=${ALLIN_DEADLINE:-2325} /bin/bash scripts/allin_wed.sh
L2_DEADLINE=${L2_DEADLINE:-2340} /bin/bash scripts/l2_wed.sh
