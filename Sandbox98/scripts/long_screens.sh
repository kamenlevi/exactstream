#!/bin/bash
# Long screens (orchestrator 21:3x; lines in results/screens/LONG_LINES.md, written first). Polite guards as in
# standin_screens.sh: never during another session's quiet lock or queued ticket (checked before each step and every 20 s
# in it; on one, the step is killed and rerun from scratch later), PAUSE, NO-SWAP >= 12 GB, swap growth < 1.5 GB.
cd "$(dirname "$0")/.."
PY=$HOME/BigMoE/.venv/bin/python
quiet_others() { peer ls 2>/dev/null | grep "^- " | grep -oE "· (benchmarking|waiting for a quiet window)[^·]*" | grep -c . ; }
noswap() { vm_stat | awk '/page size of/ {ps=$8} /^Pages (free|purgeable):/ || /^File-backed pages:/ {v=$NF; sub("[.]","",v); s+=v} END {printf "%d", s*ps/1e9}'; }
swap_mb() { sysctl -n vm.swapusage | sed -E 's/.*used = ([0-9.]+)M.*/\1/' | cut -d. -f1; }
SWAP0=$(swap_mb)
ready() {
  [ -f $HOME/.local/share/quiet/PAUSE ] && return 1
  [ "$(quiet_others)" -eq 0 ] || return 1
  [ "$(noswap)" -ge 12 ] || return 1
  [ $(( $(swap_mb) - SWAP0 )) -lt 1500 ] || { echo "$(date +%H:%M:%S) swap grew > 1.5 GB since the start: stopping"; exit 0; }
  return 0
}
guarded() {   # NAME LOG MINUTES CMD...   run CMD in its own session under a share lock; yield to quiet tickets
  local name=$1 log=$2 mins=$3; shift 3
  local tries=0
  while true; do
    until ready; do sleep 20; done
    sleep 60; ready || continue
    echo "$(date +%H:%M:%S) start $name (swap $(swap_mb) MB, NO-SWAP $(noswap) GB)"
    peer lock share ${mins}m --cpu 200 --gpu 25 "Sandbox98: long screen step $name, MEM 3.6 GB measured peak" 2>&1 | head -1
    $PY -I -c 'import os, sys; os.setsid(); os.execvp(sys.argv[1], sys.argv[1:])' "$@" > $log 2>&1 &
    local SP=$! yielded=0
    while kill -0 $SP 2>/dev/null; do
      sleep 20
      if [ "$(quiet_others)" -gt 0 ]; then
        echo "$(date +%H:%M:%S) quiet ticket appeared: yield (kill $name)"; kill -TERM -- -$SP 2>/dev/null; sleep 5; kill -KILL -- -$SP 2>/dev/null
        yielded=1; break
      fi
    done
    wait $SP; local rc=$?
    peer unlock 2>&1 | tail -1
    if [ $yielded = 0 ]; then echo "$(date +%H:%M:%S) done $name (exit $rc, swap $(swap_mb) MB)"; return $rc; fi
    tries=$((tries + 1)); [ $tries -ge 3 ] && { echo "$(date +%H:%M:%S) $name: yielded 3 times, giving up"; exit 1; }
  done
}
# 1. the base again, with Adam's state
if [ ! -f ckpt/s30o_opt.safetensors ]; then
  rm -f ckpt/s30o.safetensors ckpt/s30o.json
  guarded base_s30o logs/s30o.log 30 $PY sb/train.py --run s30o --start native --files base --steps 100000 --rows 96 --lr 1e-5 --save-opt || exit 1
fi
$PY - <<'PYEOF' | tee -a results/screens/LONG_LINES.md
import mlx.core as mx
a, b = mx.load("ckpt/s30.safetensors"), mx.load("ckpt/s30o.safetensors")
d = max(float(mx.abs(a[k].astype(mx.float32) - b[k].astype(mx.float32)).max()) for k in a)
print(f"\nBase check: s30o vs s30, max |weight difference| over {len(a)} tensors = {d:.3g} -> " + ("EQUAL: the gate's numbers apply" if d == 0 else "DIFFERENT: read s30o out separately"))
PYEOF
export BASE=s30o OPT=s30o LR=1e-5 EPOCHS=2
# 2. the control (+ its line vs s30)
guarded control logs/long_ctl.log 20 ./screen.sh ctl - "control" 1000 50 30 || exit 1
$PY - <<'PYEOF' | tee -a results/screens/LONG_LINES.md
import json, time
g = json.load(open("results/gate_chain.json")); c = json.load(open("results/screens/_eval_ctl_s30o_1000_lr1e-5_opt_50_30.json"))
out = []
for st in ("mbpp", "he"):
    ps, pc = g["per"][f"s30|{st}|chain"], c["per"][f"ctl_s30o_1000_lr1e-5_opt|{st}|chain"]
    k = sorted(pc); r = lambda p, b: sum(p[x][b][0] for x in k) / sum(p[x][b][1] for x in k)
    out.append((st, 100 * r(ps, "23-30"), 100 * r(pc, "23-30"), r(ps, "kept"), r(pc, "kept"), 100 * r(ps, "0-0"), 100 * r(pc, "0-0")))
m = out[0]; d23, dk = m[2] - m[1], m[4] - m[3]
flags = []
flags.append("no drift (valid)" if abs(d23) <= 1.5 and abs(dk) <= 0.15 else "outside the no-drift band")
if d23 > 1.0: flags.append("MOVES UP: the budget moves the head (the rule is met)")
if d23 < -1.5: flags.append("DRIFT DOWN: reported beside the comparisons")
if -1.0 <= d23 <= 1.0: flags.append("does not measurably move: NO EFFECT below would still mean untested")
v = "; ".join(flags)
print(f"\nControl ({time.strftime('%H:%M')}): " + "; ".join(f"{st} 23-30 {a:.2f} -> {b:.2f} ({b - a:+.2f}), kept/32 {c_:.3f} -> {d:.3f}, depth 0 {e:.2f} -> {f:.2f}" for st, a, b, c_, d, e, f in out) + f" -> {v}")
PYEOF
# 3. the levers
LINE=$(cat results/screens/LONG_LINES.md | tr '\n' ' ' | cut -c1-1200)
guarded distill logs/long_distill.log 25 ./screen.sh distill distill "$LINE" 1000 50 30
guarded noisehead logs/long_noisehead.log 22 ./screen.sh noisehead noisehead "$LINE" 1000 50 30
echo "long screens done"
