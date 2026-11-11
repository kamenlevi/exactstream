#!/bin/sh
# D-22 on the shipped 45 GB MTP line (MTP Head, Fri 10-30 22:2x; settings from BigMoE 21:56): H50 = scripts/mtp_depth.sh's
# stack (X + FULL + JOINT + MTP_PF_GROUP 48 + M01) + BIGMOE_EXT_BUILD=build_m01, BIGMOE_MTP_STREAM UNSET (the head install
# would bypass its streamed pool), C 256; stock head (installed, same path) vs the trained head, interleaved per prompt
# (sticky/d22_ship.py). SHARED GPU first (rough, FAST INTEGRATION); outputs must be identical across arms.
# QUIET lock (orchestrator 23:5x: a 45 GB-route run; one such ticket at a time, 15-min gaps). Guards: 30 s after the grant,
# vm_stat available >= 45 GB (peak ~41 + 4) and no other process over 20 GB resident, stop if swap grows > 2 GB.
cd $HOME/TrainLab
PY=$HOME/BigMoE/.venv/bin/python
RUN=${D22_SHIP_RUN:-b3}
trap 'pkill -P $$; exit 143' TERM
swap() { sysctl -n vm.swapusage | awk '{v = $6; sub("M", "", v); print int(v)}'; }
avail() { vm_stat | awk '/Pages free/{f=$3} /Pages inactive/{i=$3} /Pages speculative/{s=$3} /Pages purgeable/{p=$3} END{printf "%d", (f+i+s+p)*16384/1e9}'; }
noswap() { vm_stat | awk '/page size of/ {ps=$8} /^Pages (free|speculative|purgeable):/ || /^File-backed pages:/ {v=$NF; sub("[.]","",v); s+=v} END {printf "%d", s*ps/1e9}'; }
sleep 30                                                    # RULES.md (orchestrator 23:5x): settle 30 s after the grant
SW0=$(swap); F=$(noswap); BIG=$(ps -axo rss= | awk '$1 > 20000000' | wc -l | tr -d ' ')
echo "== D-22 shipped-line A/B ($RUN) $(date +%H:%M:%S): ${F} GB NO-SWAP (orchestrator 23:5x gate), ${BIG} big processes, swap ${SW0} MB"
CE=256; NEED=45                                             # largest cache that fits (orchestrator 01:4x): C 256 ~41 GB,
[ "$F" -lt 45 ] && { CE=224; NEED=41; }                    # C 224 ~37 GB, C 160 ~28 GB (peak + 4 GB)
[ "$F" -lt 41 ] && { CE=160; NEED=32; }
[ "$F" -ge $NEED ] && [ "$BIG" -eq 0 ] || { echo "== needs >= $NEED GB available (peak + 4) and no other big model: not starting (re-queue in 15 min)"; exit 1; }
( while sleep 10; do
    if [ $(( $(swap) - SW0 )) -gt 2048 ]; then echo "== swap grew > 2 GB ($(swap) MB): stopping"; pkill -f d22_ship.py; exit; fi
  done ) & G=$!
REF="BIGMOE_DUPROW=1 BIGMOE_COMPILE_HC=1 BIGMOE_HC_VREAD=1 BIGMOE_ASYNC_EVERY=1 BIGMOE_PF_JIT=1 BIGMOE_PF_QD=4 BIGMOE_PREFETCH=12"
X="$REF BIGMOE_DUPROW_CHUNK=5 BIGMOE_MAXK=80 BIGMOE_FRSPEC=1 BIGMOE_SCOUT_BUNDLE=1 MTPLX_QWEN4_VERIFY_GLUE=1"
V3="BIGMOE_TIER=4433333333 BIGMOE_TIER_GLU=3c BIGMOE_TIER_COMBINE=1 BIGMOE_OOO=1"
FULL="BIGMOE_GDN_EXACT=1 $V3 BIGMOE_TIER_PF_GROUP=64 BIGMOE_PF_NOLOGITS=1 BIGMOE_EVICT_LFU=0.99 BIGMOE_VERIFY_ROWS_SDPA=1"
env -u BIGMOE_MTP_STREAM D22_SHIP_C=$CE $X $FULL BIGMOE_JOINT=1 BIGMOE_MTP_PF_GROUP=48 BIGMOE_M01=1 BIGMOE_EXT_BUILD=build_m01 HF_HUB_OFFLINE=1 \
  $PY sticky/d22_ship.py $RUN 256 code,gsm,ifeval,multi 2>&1 | grep --line-buffered -v "Token indices\|moe-sorted-gather"
kill $G 2>/dev/null
echo "== done $(date +%H:%M:%S), swap $(swap) MB (start $SW0)"
