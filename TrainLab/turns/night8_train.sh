#!/bin/sh
> (a quote from the project owner, kept private)
# Data: Fable's E1 set from tonight's DLOG files (sticky/e15_set.py: 20K answer tokens per language de / ru / ja), built once
# night7_chain's main ticket (D-23c + DLOG) is done. Phases in order, each resumable / skipped when done (data/e15_<phase>.done):
# E1 train (L1-20 recipe, full body) -> E1 walk -> E5 train (DFlash-X branch, body frozen, lr 1e-4) -> E5 walk -> read-out
# (sticky/e15_read.py, Fable's lines) -> data/E15_DONE.
# A. Beside EvalSuite's runner 36 (orchestrator: small-model training may share the GPU only with non-timed runs and only if the
#    memory gate passes; EvalSuite 21:3x, its conditions): start only inside its lock with why "evalsuite: decode ref2ct" held
#    >= 5 min, NO-SWAP >= peak + 9; gone within ~25 s of a lock change; stop at +1 GB swap; ~/TrainLab/NO_TRAIN pauses.
#    No peer lock of its own (night7_chain is the session's lock user then).
# B. The exclusive slot (orchestrator 21:4x, NewIdeas slots it first after Leap's runner 62, ~05:45): once runner 62 (pid 44084)
#    has exited, ONE quiet FIFO ticket of 3 h for whatever is left (gate peak + 5, swap +2 GB stop, keepalive as run_locked);
#    released at the end. night7_chain then only waits for Leap's capture (later in NewIdeas' order), so one lock user at a time.
# Peaks (--time, S 32 / A 8, cache 0.5 GB): E1 14.4 GB (10.3 s per 256 anchors), E5 11.0 GB (8.5 s); walks ~5 GB.
# Stops at Mon 12:00 whatever is unfinished (the rest by day, NewIdeas' order).
cd $HOME/TrainLab
set -m                                                   # each phase in its own process group: a pause reaches its python
LOG=logs/turns.log
PY=$HOME/BigMoE/.venv/bin/python
L120=$HOME/Leap/data/l120_ft.safetensors
SET=$HOME/TrainLab/data/dlog/e15set
E5LR=${E5_LR:-1e-4}       # E5a (21:37): 2e-5 and 2e-4 both null in 35 steps; 1e-4 for a zero-init branch over ~470 steps
MODE=r36
say() { echo "$(date '+%m-%d %H:%M:%S') night8_train: $*" >> $LOG; echo "== $(date +%H:%M:%S) $*"; }
swap() { sysctl -n vm.swapusage | awk '{v = $6; sub("M", "", v); print int(v)}'; }
noswap() { vm_stat | awk '/page size of/ {ps=$8} /^Pages (free|speculative|purgeable):/ || /^File-backed pages:/ {v=$NF; sub("[.]","",v); s+=v} END {printf "%d", s*ps/1e9}'; }
lk() { $PY -c "
import json, sys, time
try:
    d = json.load(open('$HOME/.claude-peers/locks/63835.json'))
except Exception:
    print(0); raise SystemExit
ok = d.get('mode') == 'quiet' and d.get('until', 0) > time.time() and (d.get('why') or '').startswith('evalsuite: decode ref2ct')
held = time.time() - float(d.get('taken') or time.time())
print(int(d.get('taken') or 0) if ok and held >= float(sys.argv[1]) else 0)" "$1" 2>/dev/null || echo 0; }
mine() { $PY -c "
import json, os, time
try:
    d = json.load(open(os.path.expanduser('~/.claude-peers/locks/%s.json' % os.environ['CLAUDE_PID'])))
except Exception:
    print(0); raise SystemExit
print(int(d.get('taken') or 1) if d.get('mode') == 'quiet' and d.get('until', 0) > time.time() else 0)" 2>/dev/null || echo 0; }
keepalive() { $PY -I -S -c '
import json, os, sys, time
p = os.path.expanduser("~/.claude-peers/locks/%s.json" % os.environ["CLAUDE_PID"])
try: d = json.load(open(p))
except Exception: sys.exit(1)
if d.get("mode") != "quiet" or d.get("until", 0) <= time.time(): sys.exit(1)
d["active_at"] = time.time(); t = "%s.tmp%d" % (p, os.getpid()); json.dump(d, open(t, "w")); os.replace(t, p)'; }
r62live() { ps -p 44084 -o command= 2>/dev/null | grep -q runner_r62; }
lockid() { if [ $MODE = r36 ]; then [ -e $HOME/TrainLab/NO_TRAIN ] && { echo 0; return; }; lk 300; else mine; fi; }
STOPAT=$(date -j -f "%Y-%m-%d %H:%M" "2026-11-02 12:00" +%s)
late() { [ $(date +%s) -ge $STOPAT ]; }

# phase NAME PEAK CMD...: run CMD while allowed (mode r36: inside r36's lock, NO-SWAP >= PEAK + 9; mode own: inside my quiet
# lock, NO-SWAP >= PEAK + 5); pause / resume around it. Returns 0 when CMD exits 0 (done), 1 at the stop time, 2 when mode
# r36 should hand over to the exclusive slot (runner 62 has exited).
phase() {
  name=$1; peak=$2; shift 2
  [ -e data/e15_$name.done ] && return 0
  while :; do
    late && { say "$name: Mon 12:00, not finished: left for the day"; return 1; }
    [ $MODE = r36 ] && ! r62live && return 2
    if [ $MODE = r36 ]; then gate=$((peak + 9)); else gate=$((peak + 5)); fi
    ID=$(lockid)
    if [ "$ID" = 0 ]; then
      [ $MODE = own ] && { say "$name: my quiet lock is gone"; return 1; }
      sleep 30; continue
    fi
    if [ "$(noswap)" -lt "$gate" ]; then sleep 30; continue; fi
    SW0=$(swap)
    if [ $MODE = r36 ]; then SWL=1024; else SWL=2048; fi
    say "$name: start ($MODE, lock $ID; $(noswap) GB NO-SWAP, gate $gate; swap ${SW0} MB)"
    "$@" >> logs/night8_$name.out 2>&1 &
    P=$!
    i=0
    while kill -0 $P 2>/dev/null; do
      sleep 5; i=$((i + 5))
      [ $MODE = own ] && [ $((i % 150)) -eq 0 ] && keepalive
      if [ "$(lockid)" != "$ID" ] || [ $(( $(swap) - SW0 )) -gt $SWL ]; then
        say "$name: pausing (lock $ID ended / NO_TRAIN / swap over +$SWL MB)"; kill -TERM -- -$P
        j=0; while kill -0 $P 2>/dev/null && [ $j -lt 22 ]; do sleep 1; j=$((j + 1)); done
        kill -0 $P 2>/dev/null && { kill -9 -- -$P; say "$name: killed after 22 s (resumes from its last checkpoint)"; }
        break
      fi
    done
    wait $P; rc=$?
    [ $rc = 0 ] && { touch data/e15_$name.done; say "$name: done"; return 0; }
    say "$name: exited rc $rc; will resume"
    sleep 30
  done
}
train_e1() {
  R=""; grep -q '"status"' data/e1_dlog.state.json 2>/dev/null && R="--resume"
  $PY sticky/d13_train.py --data $SET --init $L120 --noval --out $HOME/TrainLab/data/e1_dlog.safetensors \
    --bs 32 --micro 1 --S 32 --A 8 --cache-gb 0.5 --epochs 2 --minutes 400 --pmask 0.5 --div fkl $R
  grep -q '"status": "done"' data/e1_dlog.state.json
}
train_e5() {
  R=""; grep -q '"status"' data/e5_dlog.state.json 2>/dev/null && R="--resume"
  E5_BODY16=1 $PY sticky/e5_train.py --data $SET --init $L120 --noval --out $HOME/TrainLab/data/e5_dlog.safetensors \
    --bs 32 --micro 1 --S 32 --A 8 --cache-gb 0.5 --epochs 2 --minutes 400 --pmask 0.5 --lr $E5LR $R
  grep -q '"status": "done"' data/e5_dlog.state.json
}
WSETS="multi=$HOME/TrainLab/data/d02a/tf_multi_*.npz,ifeval=$HOME/TrainLab/data/d02a/tf_ifeval_*.npz,code=$HOME/Leap/data/dfl2d_code/*.npz,math=$HOME/Leap/data/dfl2d_math/*.npz"
walk_e1() { TL_DRAFT=$HOME/TrainLab/data/e1_dlog.safetensors TL_WALK_TAG=_e1 TL_WALK_SETS="$WSETS" $PY sticky/d02b_walk.py; }
walk_e5() { TL_E5=$HOME/TrainLab/data/e5_dlog.safetensors TL_WALK_TAG=_e5 TL_WALK_SETS="$WSETS" $PY sticky/d02b_walk.py; }
run_all() {
  phase e1train 15 train_e1 || return $?
  phase e1walk 5 walk_e1 || return $?
  phase e5train 12 train_e5 || return $?
  phase e5walk 6 walk_e5 || return $?
  return 0
}

# 23:2x: started by turns/night9_chain.sh once DLOG has E1's set; data/E15_SLOT_DONE marks its exit (BigMoE's C50 queues after it)
trap 'touch $HOME/TrainLab/data/E15_SLOT_DONE' EXIT
say "start (from night9_chain): E1 -> E5 beside r36 (gated), the rest in a 3-h slot after runner 62"
$PY sticky/e15_set.py >> logs/night8_set.out 2>&1 || { say "no DLOG data for a language: stopping ($(tail -1 logs/night8_set.out))"; exit 1; }
say "$(tail -1 logs/night8_set.out)"
if [ "$NIGHT8_INLOCK" = 1 ]; then                  # 01:0x (orchestrator: E1 then E5 right after D-23c): inside turn69's ticket
  MODE=own
  say "in-lock mode (turn69's ticket, after D-23c): E1 -> E5 now"
  run_all; rc=$?
else
run_all; rc=$?
fi
if [ $rc = 2 ]; then
  MODE=own
  say "runner 62 done: queueing the exclusive 3-h slot for what is left (NewIdeas orders it first after runner 62)"
  PEER_QUEUE_TIMEOUT=28800 peer lock quiet --queue 3h "MTP Head E1 + E5: drafter training (DFlash2 on tonight's DLOG data; expert-borrowing branch), small model, ~15 GB, 3 h slot" >> $LOG 2>&1 || { say "queue failed: stopping"; exit 1; }
  say "slot granted"
  run_all; rc=$?
  peer unlock > /dev/null 2>&1; say "slot released (rc $rc)"
fi
[ $rc = 0 ] || exit 0
E15_ARMS=e1,e5 $PY sticky/e15_read.py > logs/night8_read.out 2>&1
touch data/E15_DONE
say "E1 / E5 done: $(grep -E 'E1 line|E5 line' logs/night8_read.out | tr '\n' ' ' | cut -c1-300)"
