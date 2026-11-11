#!/bin/bash
# MTP Head, Thu 11-05 11:1x: PORT diagnosis 2 (diagnosis 1: the layer-0 gradients of MLX and the port MATCH at bf16 noise;
# REPORT 'PORT steps FAIL, diagnosis 1'). Its two gaps:
#  2a (port only): the full run's recompute / chunking (--ckpt-layers, --attn-chunk 4 of 8 anchors, --rows 128) vs none,
#     same batch, 1 layer: must agree to fp32 rounding (else the recompute path is wrong).
#  2b (MLX GPU vs port): the selector path. All 5 layers (fp32, frozen), only the final norm + candidate selector get
#     gradients, so the selector loss is active (the 5-layer drafter ranks the true token in its top 16).
# Both on A 8, the first train group (gsm_0, alone in a scratch folder: same batch, ~1 GB less per process), pmask 0.5.
# One process at a time, share lock MEM 5 GB; swap +2 GB or available < 12 GB -> stop. Out: data/port/gradpar2/.
cd ~/TrainLab || exit 1
log() { echo "[$(date +%H:%M:%S)] $*"; }
swap_mb() { sysctl -n vm.swapusage | awk '{print $6}' | tr -d M; }
avail_gb() { vm_stat | awk '/page size of/ {ps=$8} /Pages free/ {f=$3} /Pages inactive/ {i=$3} /Pages speculative/ {s=$3} /Pages purgeable/ {p=$3} END {printf "%d", (f+i+s+p)*ps/2^30}'; }
others() { peer ls 2>/dev/null | grep "^- "; }
locks() { grep -oE "· (shared run \(others may work\)|benchmarking|waiting for a quiet window)[^·]*"; }
nquiet() { others | locks | grep -cE "^· (benchmarking|waiting for a quiet window)"; }
q=$(nquiet); [ "$q" -eq 0 ] || { log "another session holds or queues a quiet lock: not now"; exit 3; }
out=$(peer lock share 20m --cpu 800 --gpu 20 "MEM 5 GB: MTP Head PORT gradient parity 2 (recompute check + selector path), CPU then GPU, ~3 min" 2>&1)
log "lock: $(printf '%s' "$out" | head -1)"
printf '%s' "$out" | grep -q "reserved" || { log "NO LOCK"; exit 4; }
D=data/port/gradpar2; mkdir -p $D
W=/private/tmp/claude-501/-Users-kamen-OjoX/4e355fd6-5387-4e1c-8f8a-f46ddfe41f52/scratchpad/l120_first
TP=~/TrainLab/.venv-torch/bin/python; MP=~/BigMoE/.venv/bin/python3
run() {   # $1 label, $2 pkill pattern, rest = command
  local label=$1 pat=$2; shift 2
  local s0 a0; s0=$(swap_mb); a0=$(avail_gb); log "$label: start (swap $s0 MB, $a0 GB available)"
  /usr/bin/time -l "$@" > $D/$label.log 2>&1 &
  local pid=$!
  while kill -0 $pid 2>/dev/null; do
    sleep 5
    local s a; s=$(swap_mb); a=$(avail_gb)
    if awk -v x="$s" -v y="$s0" 'BEGIN{exit !(x-y>2048)}' || [ "$a" -lt 12 ]; then
      log "$label: swap $s MB / $a GB available: stopping"; pkill -TERM -f "$pat"; break
    fi
  done
  wait $pid; local rc=$?
  log "$label: rc=$rc, footprint $(grep 'peak memory footprint' $D/$label.log | awk '{printf "%.2f GB", $1/2^30}'): $(grep -E '^\[(ref|port)\] grad' $D/$label.log | cut -c1-150)"
  return $rc
}
C="--A 8 --groups 1 --pmask 0.5 --data $W"
run port_plain "port/test_parit[y].py" $TP -I port/test_parity.py grad $D/port_plain.json --layers 1 $C --threads 6 \
  && run port_ckpt "port/test_parit[y].py" $TP -I port/test_parity.py grad $D/port_ckpt.json --layers 1 $C --threads 6 \
       --ckpt-layers --attn-chunk 4 --rows 128 \
  && run mlx_sel "port/ref_ml[x].py" $MP -I port/ref_mlx.py grad $D/mlx_sel.json --gpu --freeze-layers $C \
  && run port_sel "port/test_parit[y].py" $TP -I port/test_parity.py grad $D/port_sel.json --freeze-layers $C --threads 6
peer unlock 2>&1 | tail -1
for pair in "port_plain port_ckpt 2a" "mlx_sel port_sel 2b"; do
  set -- $pair
  if [ -f $D/$1.json ] && [ -f $D/$2.json ]; then
    $TP -I port/compare_grads.py $D/$1.json $D/$2.json > $D/compare_$3.txt 2>&1
    log "== $3: $1 vs $2"; head -12 $D/compare_$3.txt
  fi
done
log "DONE"
