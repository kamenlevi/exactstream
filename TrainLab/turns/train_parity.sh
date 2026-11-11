#!/bin/bash
# MTP Head, Thu 11-05 11:1x: PORT diagnosis 4 (data path identical over 5 steps, diagnosis 3; single-batch gradients match,
# diagnoses 1-2): the UPDATE over steps. 5 optimizer steps (the trainers own step code: summed grads, clip 1.0, MLX
# AdamW, the lr ramp), MLX (GPU) vs port, first layer + final norm (selector frozen: inactive at 1 layer), A 8, step k
# on train group k of gsm_0 (one-window folder), pmask 0.5. PASS = losses within ~1e-3 (the per-batch noise).
# One process at a time, share lock MEM 5 GB; swap +2 GB or available < 12 GB -> stop. Out: data/port/trainpar/.
cd ~/TrainLab || exit 1
log() { echo "[$(date +%H:%M:%S)] $*"; }
swap_mb() { sysctl -n vm.swapusage | awk '{print $6}' | tr -d M; }
avail_gb() { vm_stat | awk '/page size of/ {ps=$8} /Pages free/ {f=$3} /Pages inactive/ {i=$3} /Pages speculative/ {s=$3} /Pages purgeable/ {p=$3} END {printf "%d", (f+i+s+p)*ps/2^30}'; }
others() { peer ls 2>/dev/null | grep "^- "; }
locks() { grep -oE "· (shared run \(others may work\)|benchmarking|waiting for a quiet window)[^·]*"; }
nquiet() { others | locks | grep -cE "^· (benchmarking|waiting for a quiet window)"; }
q=$(nquiet); [ "$q" -eq 0 ] || { log "another session holds or queues a quiet lock: not now"; exit 3; }
out=$(peer lock share 20m --cpu 800 --gpu 20 "MEM 5 GB: MTP Head PORT update parity (5 steps, 1 layer), GPU then CPU, ~2 min" 2>&1)
log "lock: $(printf '%s' "$out" | head -1)"
printf '%s' "$out" | grep -q "reserved" || { log "NO LOCK"; exit 4; }
D=data/port/trainpar; mkdir -p $D
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
run mlx_train "port/ref_ml[x].py" $MP -I port/ref_mlx.py train $D/mlx_train.npz --gpu --layers 1 --freeze-selector --steps 5 $C \
  && run port_train "port/test_parit[y].py" $TP -I port/test_parity.py train $D/mlx_train.npz --layers 1 --freeze-selector --steps 5 $C --threads 6
peer unlock 2>&1 | tail -1
grep -E "^\[(ref|port)\] (step|RESULT)" $D/mlx_train.log $D/port_train.log | cut -c1-220
log "DONE"
