#!/bin/sh
# HPD-1 (MTP Head, Mon 11-02; REPORT "HPD-1: pre-registration"): hard-position distillation screen on L1-20's own data.
# Runs inside a lock that is already held (run_locked.sh for a quiet ticket, or a share lock by day); each phase is skipped
# when its marker data/hpd/<phase>.done exists, the trainers resume from their checkpoints. Phases:
# compile -> conf (L1-20's cf on ~/Leap/data/l120) -> walk0 (L1-20 CF walk, self-checks) -> trainH (loss only at cf < 0.999)
# -> walkH (own + at L1-20's anchors) -> readH -> trainU (same, uniform: d13_train.py) -> walkU -> read -> data/HPD_DONE.
# Guards: NO-SWAP >= peak + HPD_MARGIN (5 in my quiet ticket, 8 when shared) before each phase; swap +2 GB stops a phase.
cd $HOME/TrainLab
LOG=logs/turns.log
PY=$HOME/BigMoE/.venv/bin/python
L120=$HOME/Leap/data/l120_ft.safetensors
M=${HPD_MARGIN:-5}
WSETS="code=$HOME/Leap/data/dfl2d_code/*.npz,math=$HOME/Leap/data/dfl2d_math/*.npz,multi=$HOME/TrainLab/data/d02a/tf_multi_*.npz,ifeval=$HOME/TrainLab/data/d02a/tf_ifeval_*.npz"
ARGS="--data $HOME/Leap/data/l120 --init $L120 --noval --bs 32 --micro 1 --S 32 --A 8 --cache-gb 0.5 --epochs 2 --minutes 120 --pmask 0.5 --div fkl"
mkdir -p data/hpd
say() { echo "$(date '+%m-%d %H:%M:%S') hpd1: $*" >> $LOG; echo "== $(date +%H:%M:%S) $*"; }
swap() { sysctl -n vm.swapusage | awk '{v = $6; sub("M", "", v); print int(v)}'; }
noswap() { vm_stat | awk '/page size of/ {ps=$8} /^Pages (free|speculative|purgeable):/ || /^File-backed pages:/ {v=$NF; sub("[.]","",v); s+=v} END {printf "%d", s*ps/1e9}'; }

# phase NAME PEAK CMD...: wait up to 10 min for the memory gate, run CMD with the swap guard; 0 = done (marker written)
phase() {
  name=$1; peak=$2; shift 2
  [ -e data/hpd/$name.done ] && return 0
  i=0
  while [ "$(noswap)" -lt $((peak + M)) ]; do
    i=$((i + 1)); [ $i -gt 20 ] && { say "$name: NO-SWAP $(noswap) GB < gate $((peak + M)) for 10 min: stopping"; return 1; }
    sleep 30
  done
  SW0=$(swap)
  say "$name: start ($(noswap) GB NO-SWAP, gate $((peak + M)); swap ${SW0} MB)"
  set -m
  "$@" >> logs/hpd1_$name.out 2>&1 &
  P=$!
  while kill -0 $P 2>/dev/null; do
    sleep 5
    if [ $(( $(swap) - SW0 )) -gt 2048 ]; then
      say "$name: swap over +2 GB: stopping (resumes from its checkpoint)"; kill -TERM -- -$P
      j=0; while kill -0 $P 2>/dev/null && [ $j -lt 25 ]; do sleep 1; j=$((j + 1)); done
      kill -9 -- -$P 2>/dev/null; return 1
    fi
  done
  wait $P; rc=$?
  set +m
  [ $rc = 0 ] || { say "$name: exited rc $rc (logs/hpd1_$name.out)"; return 1; }
  touch data/hpd/$name.done; say "$name: done"; return 0
}
compile() { $PY -m py_compile sticky/hpd_train.py sticky/d02b_walk.py sticky/hpd_read.py; }
conf() {
  $PY sticky/hpd_train.py $ARGS --conf-out $HOME/TrainLab/data/hpd/conf_l120 || return 1
  [ "$(ls data/hpd/conf_l120/*.npy | wc -l | tr -d ' ')" = 109 ]
}
walk0() {
  TL_WALK_CF=1 TL_WALK_TAG=_hpd0 TL_WALK_SETS="$WSETS" $PY sticky/d02b_walk.py || return 1
  # self-checks: the CF walk keeps L1-20's own numbers (code 9.57, math 7.00, multi 2.06, IFEval 2.45: D-02b, E15 read-out),
  # and L1-20 walked at its own anchors gives the same checks (the fixed-anchor mode)
  for x in "code: .*kept per check 9.57" "math: .*kept per check 7.00" "multi: .*kept per check 2.06" "ifeval: .*kept per check 2.45"; do
    grep -q "$x" data/d02b_walk_hpd0.txt || { echo "SELF-CHECK FAIL: $x"; return 1; }
  done
  TL_WALK_CF=1 TL_WALK_TAG=_hpd0x TL_WALK_ANCHORS=_hpd0 TL_WALK_SETS="$WSETS" $PY sticky/d02b_walk.py || return 1
  $PY -c "
import numpy as np, os
for st in ('code', 'math', 'multi', 'ifeval'):
    a = np.load(os.path.expanduser('~/TrainLab/data/d02b_walk_hpd0_%s.npz' % st)); b = np.load(os.path.expanduser('~/TrainLab/data/d02b_walk_hpd0x_%s.npz' % st))
    for s in range(len(a['names'])):
        for k in ('checks', 'dtok', 'ttok'):
            assert np.array_equal(a['%d_%s' % (s, k)], b['%d_%s' % (s, k)]), (st, s, k)
print('SELF-CHECK OK: fixed-anchor walk == own walk for L1-20 on every set')"
}
trainH() {
  R=""; grep -q '"status"' data/hpdH.state.json 2>/dev/null && R="--resume"
  $PY sticky/hpd_train.py $ARGS --hpd-conf $HOME/TrainLab/data/hpd/conf_l120 --hpd-thr 0.999 --out $HOME/TrainLab/data/hpdH.safetensors $R
  grep -q '"status": "done"' data/hpdH.state.json
}
trainU() {
  R=""; grep -q '"status"' data/hpdU.state.json 2>/dev/null && R="--resume"
  $PY sticky/d13_train.py $ARGS --out $HOME/TrainLab/data/hpdU.safetensors $R
  grep -q '"status": "done"' data/hpdU.state.json
}
walkarm() {   # $1 = H / U
  TL_DRAFT=$HOME/TrainLab/data/hpd$1.safetensors TL_WALK_CF=1 TL_WALK_TAG=_hpd$1 TL_WALK_SETS="$WSETS" $PY sticky/d02b_walk.py || return 1
  TL_DRAFT=$HOME/TrainLab/data/hpd$1.safetensors TL_WALK_CF=1 TL_WALK_TAG=_hpd$1_fx TL_WALK_ANCHORS=_hpd0 TL_WALK_SETS="$WSETS" $PY sticky/d02b_walk.py
}
walkH() { walkarm H; }
walkU() { walkarm U; }
readH() { HPD_ARMS=H:_hpdH $PY sticky/hpd_read.py > logs/hpd1_readH.txt; }
readall() { HPD_ARMS=H:_hpdH,U:_hpdU $PY sticky/hpd_read.py > logs/hpd1_read.txt; }

say "start (margin $M GB)"
phase compile 0 compile && phase conf 12 conf && phase walk0 6 walk0 && phase trainH 15 trainH && phase walkH 6 walkH \
  && phase readH 0 readH && phase trainU 15 trainU && phase walkU 6 walkU && phase read 0 readall || { say "stopped early"; exit 1; }
touch data/HPD_DONE
say "HPD-1 done: $(grep 'HPD line' logs/hpd1_read.txt | tail -1 | cut -c1-240)"
