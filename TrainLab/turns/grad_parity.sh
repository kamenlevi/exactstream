#!/bin/bash
# MTP Head, Thu 11-05 11:0x: the PORT steps check FAILED at steps 2-5 (step 1's loss matched to 1e-4). Diagnosis, step 1:
# the gradient on one batch, MLX (the trainer's own code, sticky/d09_train.py via port/ref_mlx.py, on the GPU) vs the port
# (port/test_parity.py, CPU): first layer only, A 8, 1 group, pmask 0.5 (the trainer's PM_RNG). Read-out: per-parameter
# norms and the estimated relative difference (port/grad_stats.py; port/compare_grads.py). Each side is expected at
# <= ~4 GB, one process at a time, under a share lock MEM 4 GB (not during anyone's quiet lock). Swap +2 GB or
# available < 12 GB -> stop. Logs and JSONs in data/port/gradpar/.
cd ~/TrainLab || exit 1
log() { echo "[$(date +%H:%M:%S)] $*"; }
swap_mb() { sysctl -n vm.swapusage | awk '{print $6}' | tr -d M; }
avail_gb() { vm_stat | awk '/page size of/ {ps=$8} /Pages free/ {f=$3} /Pages inactive/ {i=$3} /Pages speculative/ {s=$3} /Pages purgeable/ {p=$3} END {printf "%d", (f+i+s+p)*ps/2^30}'; }
others() { peer ls 2>/dev/null | grep "^- "; }
locks() { grep -oE "· (shared run \(others may work\)|benchmarking|waiting for a quiet window)[^·]*"; }
nquiet() { others | locks | grep -cE "^· (benchmarking|waiting for a quiet window)"; }
q=$(nquiet); [ "$q" -eq 0 ] || { log "another session holds or queues a quiet lock: not now"; exit 3; }
out=$(peer lock share 20m --cpu 800 --gpu 20 "MEM 4 GB: MTP Head PORT gradient parity, MLX (GPU) then port (CPU), 1 layer, ~5 min" 2>&1)
log "lock: $(printf '%s' "$out" | head -1)"
printf '%s' "$out" | grep -q "reserved" || { log "NO LOCK"; exit 4; }
D=data/port/gradpar; mkdir -p $D
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
run mlx "port/ref_ml[x].py" ~/BigMoE/.venv/bin/python3 -I port/ref_mlx.py grad $D/mlx.json --gpu --layers 1 --A 8 --groups 1 --pmask 0.5 \
  && run port "port/test_parit[y].py" ~/TrainLab/.venv-torch/bin/python -I port/test_parity.py grad $D/port.json --layers 1 --A 8 --groups 1 --pmask 0.5 --threads 6
peer unlock 2>&1 | tail -1
if [ -f $D/mlx.json ] && [ -f $D/port.json ]; then
  ~/TrainLab/.venv-torch/bin/python -I port/compare_grads.py $D/mlx.json $D/port.json > $D/compare.txt 2>&1
  head -30 $D/compare.txt
fi
log "DONE"
