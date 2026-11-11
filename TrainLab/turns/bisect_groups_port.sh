#!/bin/bash
# MTP Head, Thu 11-05 11:1x: PORT diagnosis 8b: the same per-group test on the PORT forward, start weights + ONE groups
# step-1 change (from the swap test-s mlx_p1.npz), per group; zzz = no change (baseline 12.3438).
# AdamW, the lr ramp), MLX (GPU) vs port, first layer + final norm (selector frozen: inactive at 1 layer), A 8, step k
# on train group k of gsm_0 (one-window folder), pmask 0.5. PASS = losses within ~1e-3 (the per-batch noise).
# One process at a time, share lock MEM 5 GB; swap +2 GB or available < 12 GB -> stop. Out: data/port/bisect/.
cd ~/TrainLab || exit 1
log() { echo "[$(date +%H:%M:%S)] $*"; }
swap_mb() { sysctl -n vm.swapusage | awk '{print $6}' | tr -d M; }
avail_gb() { vm_stat | awk '/page size of/ {ps=$8} /Pages free/ {f=$3} /Pages inactive/ {i=$3} /Pages speculative/ {s=$3} /Pages purgeable/ {p=$3} END {printf "%d", (f+i+s+p)*ps/2^30}'; }
others() { peer ls 2>/dev/null | grep "^- "; }
locks() { grep -oE "· (shared run \(others may work\)|benchmarking|waiting for a quiet window)[^·]*"; }
nquiet() { others | locks | grep -cE "^· (benchmarking|waiting for a quiet window)"; }
q=$(nquiet); [ "$q" -eq 0 ] || { log "another session holds or queues a quiet lock: not now"; exit 3; }
out=$(peer lock share 20m --cpu 800 --gpu 20 "MEM 5 GB: MTP Head PORT group bisection (port evals, 1 layer), ~1 min" 2>&1)
log "lock: $(printf '%s' "$out" | head -1)"
printf '%s' "$out" | grep -q "reserved" || { log "NO LOCK"; exit 4; }
D=data/port/bisect; mkdir -p $D
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
SP=/private/tmp/claude-501/-Users-kamen-OjoX/4e355fd6-5387-4e1c-8f8a-f46ddfe41f52/scratchpad
for g in zzz layers.0.self_attn layers.0.mlp. layers.0.attention_conv layers.0.mlp_conv layers.0.input_layernorm layers.0.post_attention_layernorm norm.weight; do
  run "port_$g" "port/test_parit[y].py" $TP -I port/test_parity.py evalp $D/unused --threads 6 --layers 1 --freeze-selector --params $SP/mlx_p1.npz --only $g $C || break
done
peer unlock 2>&1 | tail -1
grep -hE "^.port. evalp" $D/port_*.log | cut -c1-200
log "DONE"
