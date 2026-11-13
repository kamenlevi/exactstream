#!/bin/bash
# prof_fast4.sh (Leap 11-11 16:3x; from prof_pipe3.sh): the FAST QSA option = MTPLX's own sparse QSA prefill lane (NAX flash)
# engaged from 4K context instead of 32K (MTPLX_QSA_PREFILL_MIN_CONTEXT / _FLASH_MIN_CONTEXT / _DIRECT_MIN_CONTEXT = 4096; NOT
# exact: PrefillKernels max |d| 0.009-0.016), PF-B PIPE=1, 10K and 30K, after prof_pipe3.sh (pipe3.done). Timing + how many of
# the 32 decoded tokens equal the exact PIPE=1 run's (prof_pipe2.sh). EvalSuite's same-scores gate decides adoption.
# v2 (orchestrator 14:2x): the share lock is taken INSIDE the gate loop and retried until granted (a failed `peer lock`
# used to fall through and run beside another job); after the grant, 20 s later, no other lock / engine and NO-SWAP
# >= 46 again, else unlock and keep waiting. Start 00:33 so busy() sees a ticket queued at 00:30.
# prof_pipe3.sh (Leap 11-11 16:3x; from prof_qsa2.sh): after prof_qsa2.sh (qsa2.done), PF-B with R57_PFLM_PIPE=2 (background
# expert reads through bigmoe_ext_ov, the GIL-free copy) at 10K and 30K, bitwise vs prof_pipe2.sh's PIPE=0 runs (tokens + LOGHASH).
# (prof_qsa2.sh's header follows.) Prefill 1.5K, tonight after prof_pipe.sh (waits for <work>/pipe.done): PF-B pipelined (R57_PFLM_PIPE=1) at 32 GB on the 30K
# prompt with MTPLX's exact dense-band QSA (MTPLX_QSA_DENSE_BAND_SDPA=1; the engine runs MTPLX's stock dense-mask fallback
# today; PrefillKernels: band == stock, 0 differing) and a cProfile of every QSA self_attn call (R57_PFLM_QSA_PROFILE).
# Bitwise check: tokens + R57_LOGHASH verify-logit hashes vs prof_pipe.sh's 30K PIPE=1 run. Gates as prof_pipe.sh; last start
# LAST. Usage: prof_qsa.sh <work dir> "<LAST YYYY-MM-DD HH:MM:SS>"
W=$1; P2=/Users/kamen/Leap/scripts/prefill15/prof; SO=/private/tmp/claude-501/-Users-kamen-OjoX/f2971937-6c95-4e86-a049-05beff6a8e32/scratchpad/agent/ext_ov_build/bigmoe_ext_ov.cpython-312-darwin.so; CL=/Users/kamen/Leap/release/cleanclone/cache/clone; LOG=$W/fast4.log
LAST=$(date -j -f "%Y-%m-%d %H:%M:%S" "$2" +%s)
noswap() { vm_stat | awk '/page size of/ {ps=$8} /^Pages (free|speculative|purgeable):/ || /^File-backed pages:/ {v=$NF; sub("[.]","",v); s+=v} END {printf "%d", s*ps/1e9}'; }
swap() { sysctl -n vm.swapusage | sed -E 's/.*used = ([0-9.]+)M.*/\1/' | cut -d. -f1; }
idle() { ioreg -c IOHIDSystem | awk '/HIDIdleTime/ {print int($NF/1000000000); exit}'; }
procs() { ps -axo pid=,args= | awk '/[Pp]ython/ && /r57_screen_|[ \/](chat|serve|run)[.]py( |$)/ && !/ awk / {printf "%s ", $1}'; }
busy() { peer ls 2>/dev/null | grep "^- " | grep -oE "· (shared run \(others may work\)|benchmarking|waiting for a quiet window)[^·]*" | grep -cE "^· (benchmarking|waiting for a quiet window)|MEM ([6-9]|[1-9][0-9]) GB"; }
held() { [ -e ~/.local/share/quiet/PAUSE ] || [ -e ~/.local/share/quiet/HOLD ]; }
say() { echo "$(date '+%m-%d %H:%M:%S') $*" | tee -a $LOG; }
RP=""; GUARD=""; LOCKED=0
cleanup() { [ -n "$GUARD" ] && kill $GUARD 2>/dev/null; [ -n "$RP" ] && kill -0 $RP 2>/dev/null && kill -TERM $RP
            [ $LOCKED = 1 ] && peer unlock >/dev/null 2>&1; touch $W/fast4.done; say "exit (NO-SWAP $(noswap) GB, swap $(swap) MB)"; }
trap cleanup EXIT; trap 'say "stopped by a signal"; exit 1' INT TERM
say "armed (pid $$): FAST QSA (MTPLX sparse lane from 4K) + PIPE=1 at 10K / 30K, after $W/pipe3.done; last start $2"
NOTE=""; CLR=0
while :; do
  now=$(date +%s); [ $now -ge $LAST ] && { say "SKIP: no free window by $2 (last wait: $NOTE)"; exit 0; }
  why=""
  [ ! -e $W/pipe3.done ] && why="waits for prof_pipe3.sh (pipe3.done)"
> (a quote from the project owner, kept private)
  [ -z "$why" ] && [ -n "$(procs)" ] && why="an engine runs"
  [ -z "$why" ] && [ "$(busy)" -gt 0 ] && why="another lock"
  [ -z "$why" ] && held && why="PAUSE / HOLD"
  [ -z "$why" ] && [ "$(noswap)" -lt 46 ] && why="NO-SWAP $(noswap) < 46"
  if [ -z "$why" ]; then CLR=$((CLR + 1)); else CLR=0; fi
  [ -z "$why" ] && [ $CLR -lt 2 ] && why="clear once"
  if [ -z "$why" ]; then
    if peer lock share 30m --cpu 400 --gpu 100 "MEM 47 GB: Leap prefill FAST QSA lane (32 GB, 10K / 30K), ~5 min" > /dev/null 2>&1; then
      LOCKED=1; sleep 20
      if [ "$(busy)" = 0 ] && [ -z "$(procs)" ] && [ "$(noswap)" -ge 46 ]; then say "lock granted; clear after 20 s"; break; fi
      peer unlock >/dev/null 2>&1; LOCKED=0; CLR=0; why="after the lock: another lock / an engine / NO-SWAP $(noswap)"
    else
      CLR=0; why="share lock not granted"
    fi
  fi
  [ "$why" != "$NOTE" ] && say "wait: $why"; NOTE=$why
  sleep 60
done
S0=$(swap); ( while :; do [ $(( $(swap) - S0 )) -gt 2048 ] && { echo "$(date +%T) SWAP GUARD: $(swap) MB" >> $LOG; kill -TERM $$; exit; }; sleep 5; done ) &
GUARD=$!
for N in 10k 30k; do
  [ -n "$(procs)" ] && { say "SKIP $N: an engine runs"; continue; }
  [ "$(noswap)" -lt 44 ] && { say "SKIP $N: NO-SWAP $(noswap) < 44"; continue; }
  J=$W/fast_${N}.jsonl; rm -f $J
  say "run $N fast QSA, pipe=1 (NO-SWAP $(noswap) GB)"
  ( cd $CL && exec env PYTHONPATH=$P2 R57_PFLM_PROF=$J R57_PFLM_PROF_SYNC=0 R57_PFLM_PIPE=1 MTPLX_QSA_PREFILL_MIN_CONTEXT=4096 MTPLX_QSA_PREFILL_FLASH_MIN_CONTEXT=4096 MTPLX_QSA_PREFILL_DIRECT_MIN_CONTEXT=4096 .venv/bin/python run.py \
      --config chat --mem 32 --prompt-file $W/prompt_$N.txt --max-tokens 32 --env R57_PFLM=3 --env R57_LOGHASH=1 ) > $W/fast_${N}.out 2>&1 &
  RP=$!; wait $RP; rc=$?; RP=""
  tag=$(grep -o 'out/cli_[0-9_]*_tokens.json' $W/fast_${N}.out | head -1 | sed 's|out/||; s|_tokens.json||')
  [ -n "$tag" ] && cp $CL/out/${tag}_runs.json $W/fast_${N}_runs.json 2>/dev/null && cp $CL/out/${tag}_tokens.json $W/fast_${N}_tokens.json 2>/dev/null
  say "  rc $rc; $(grep -o '"total": \[[^]]*\]' $J 2>/dev/null); $(grep -o 'first token [0-9.]* s' $W/fast_${N}.out | head -1); $(grep -o 'lifetime peak [0-9.]* GB' $W/fast_${N}.out | tail -1); tag $tag"
done
/opt/homebrew/opt/python@3.12/bin/python3.12 -I - $W <<'PY' | tee -a $LOG
import json, sys
W = sys.argv[1]
def toks(f):
    try:
        d = json.load(open(f"{W}/{f}"))
    except Exception:
        return None
    out = []
    def walk(o):
        if isinstance(o, list) and o and all(isinstance(x, int) for x in o): out.append(o)
        elif isinstance(o, dict): [walk(v) for v in o.values()]
        elif isinstance(o, list): [walk(v) for v in o]
    walk(d)
    return out[0] if out else None
for n in ("10k", "30k"):
    a, b = toks(f"pipe_{n}_p1_tokens.json"), toks(f"fast_{n}_tokens.json")
    if not a or not b: print(n, "FAST vs exact: missing files"); continue
    same = sum(1 for x, y in zip(a, b) if x == y); first = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None)
    print(n, f"FAST vs exact PIPE=1: {same} of {min(len(a), len(b))} tokens equal; first difference at token {first}")
PY
