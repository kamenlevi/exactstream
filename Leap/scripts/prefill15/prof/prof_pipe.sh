#!/bin/bash
# Prefill 1.5K, step 1 (exact host work), tonight: PF-B (R57_PFLM=3) at 32 GB, today's loop (PIPE=0) vs the pipelined loop
# (R57_PFLM_PIPE=1: two full-E scratches, the next layer's read while the GPU runs this layer, no syncs in the MoE build),
# through the instrumented pflm copy (timestamps only, + the PLE-stage timer). Bitwise check: --max-tokens 32 with
# R57_LOGHASH=1 (every decode check's verify logits hashed) -> the tokens and the "lh" lists of PIPE=0 and PIPE=1 must be
# equal. Order alternates (10K: 0 then 1; 30K: 1 then 0). Gates before each run: no engine / chat / server, no PAUSE /
# HOLD, Kamen idle >= 600 s, NO-SWAP >= 46 GB; a share lock with MEM; swap +2 GB stops; from FROM, last start LAST.
# Usage: prof_pipe.sh <work dir> "<FROM YYYY-MM-DD HH:MM:SS>" "<LAST ...>"
W=$1; P=/Users/kamen/Leap/scripts/prefill15/prof; CL=/Users/kamen/Leap/release/cleanclone/cache/clone; LOG=$W/pipe.log
FROM=$(date -j -f "%Y-%m-%d %H:%M:%S" "$2" +%s); LAST=$(date -j -f "%Y-%m-%d %H:%M:%S" "$3" +%s)
noswap() { vm_stat | awk '/page size of/ {ps=$8} /^Pages (free|speculative|purgeable):/ || /^File-backed pages:/ {v=$NF; sub("[.]","",v); s+=v} END {printf "%d", s*ps/1e9}'; }
swap() { sysctl -n vm.swapusage | sed -E 's/.*used = ([0-9.]+)M.*/\1/' | cut -d. -f1; }
idle() { ioreg -c IOHIDSystem | awk '/HIDIdleTime/ {print int($NF/1000000000); exit}'; }
procs() { ps -axo pid=,args= | awk '/[Pp]ython/ && /r57_screen_|[ \/](chat|serve|run)[.]py( |$)/ && !/ awk / {printf "%s ", $1}'; }
busy() { peer ls 2>/dev/null | grep "^- " | grep -oE "· (shared run \(others may work\)|benchmarking|waiting for a quiet window)[^·]*" | grep -cE "^· (benchmarking|waiting for a quiet window)|MEM ([6-9]|[1-9][0-9]) GB"; }
held() { [ -e ~/.local/share/quiet/PAUSE ] || [ -e ~/.local/share/quiet/HOLD ]; }
say() { echo "$(date '+%m-%d %H:%M:%S') $*" | tee -a $LOG; }
RP=""; GUARD=""; LOCKED=0
cleanup() { [ -n "$GUARD" ] && kill $GUARD 2>/dev/null; [ -n "$RP" ] && kill -0 $RP 2>/dev/null && kill -TERM $RP
            [ $LOCKED = 1 ] && peer unlock >/dev/null 2>&1; touch $W/pipe.done; say "exit (NO-SWAP $(noswap) GB, swap $(swap) MB)"; }
trap cleanup EXIT; trap 'say "stopped by a signal"; exit 1' INT TERM
say "armed (pid $$): PF-B 32 GB PIPE 0 vs 1 at 10K / 30K; from $2, last start $3"
NOTE=""; CLR=0
while :; do
  now=$(date +%s); [ $now -ge $LAST ] && { say "SKIP: no free window by $3 (last wait: $NOTE)"; exit 0; }
  why=""
  [ $now -lt $FROM ] && why="before $2"
> (a quote from the project owner, kept private)
  [ -z "$why" ] && [ -n "$(procs)" ] && why="an engine runs"
  [ -z "$why" ] && [ "$(busy)" -gt 0 ] && why="another lock"
  [ -z "$why" ] && held && why="PAUSE / HOLD"
  [ -z "$why" ] && [ "$(noswap)" -lt 46 ] && why="NO-SWAP $(noswap) < 46"
  if [ -z "$why" ]; then CLR=$((CLR + 1)); else CLR=0; fi
  [ -z "$why" ] && [ $CLR -lt 2 ] && why="clear once"
  [ -z "$why" ] && break
  [ "$why" != "$NOTE" ] && say "wait: $why"; NOTE=$why
  sleep 60
done
peer lock share 60m --cpu 400 --gpu 100 "MEM 46 GB: Leap prefill PF-B pipelining A/B (32 GB, 10K / 30K), ~10 min" > /dev/null 2>&1 && LOCKED=1
S0=$(swap); ( while :; do [ $(( $(swap) - S0 )) -gt 2048 ] && { echo "$(date +%T) SWAP GUARD: $(swap) MB" >> $LOG; kill -TERM $$; exit; }; sleep 5; done ) &
GUARD=$!
for spec in "10k 0" "10k 1" "30k 1" "30k 0"; do
  set -- $spec; N=$1; PI=$2
  [ -n "$(procs)" ] && { say "SKIP $N pipe=$PI: an engine runs"; continue; }
  [ "$(noswap)" -lt 44 ] && { say "SKIP $N pipe=$PI: NO-SWAP $(noswap) < 44"; continue; }
  J=$W/pipe_${N}_p$PI.jsonl; rm -f $J
  say "run $N pipe=$PI (NO-SWAP $(noswap) GB)"
  ( cd $CL && exec env PYTHONPATH=$P R57_PFLM_PROF=$J R57_PFLM_PROF_SYNC=0 R57_PFLM_PIPE=$PI .venv/bin/python run.py --config chat \
      --mem 32 --prompt-file $W/prompt_$N.txt --max-tokens 32 --env R57_PFLM=3 --env R57_LOGHASH=1 ) > $W/pipe_${N}_p$PI.out 2>&1 &
  RP=$!; wait $RP; rc=$?; RP=""
  tag=$(grep -o 'out/cli_[0-9_]*_tokens.json' $W/pipe_${N}_p$PI.out | head -1 | sed 's|out/||; s|_tokens.json||')
  [ -n "$tag" ] && cp $CL/out/${tag}_runs.json $W/pipe_${N}_p${PI}_runs.json 2>/dev/null && cp $CL/out/${tag}_tokens.json $W/pipe_${N}_p${PI}_tokens.json 2>/dev/null
  say "  rc $rc; $(grep -o '"total": \[[^]]*\]' $J 2>/dev/null) tokens $(grep -o '"tokens": [0-9]*' $J 2>/dev/null | cut -d' ' -f2); $(grep -o 'first token [0-9.]* s' $W/pipe_${N}_p$PI.out | head -1); $(grep -o 'lifetime peak [0-9.]* GB' $W/pipe_${N}_p$PI.out | tail -1); tag $tag"
done
/opt/homebrew/opt/python@3.12/bin/python3.12 -I - $W <<'PY' | tee -a $LOG
import json, sys
W = sys.argv[1]
def get(n, p):
    try:
        tk = json.load(open(f"{W}/pipe_{n}_p{p}_tokens.json")); rr = json.load(open(f"{W}/pipe_{n}_p{p}_runs.json"))
    except Exception as e:
        return None, None
    toks = json.dumps(tk, sort_keys=True)
    lh = []
    def walk(o):
        if isinstance(o, dict):
            for k, v in o.items():
                if k == "lh": lh.append(v)
                else: walk(v)
        elif isinstance(o, list):
            for v in o: walk(v)
    walk(rr)
    return toks, json.dumps(lh)
for n in ("10k", "30k"):
    a, b = get(n, 0), get(n, 1)
    if None in a or None in b: print(n, "BITWISE: missing files"); continue
    print(n, "BITWISE: tokens", "EQUAL" if a[0] == b[0] else "DIFFER", "| verify-logit hashes", "EQUAL" if a[1] == b[1] else "DIFFER",
          f"({len(json.loads(a[1])[0]) if json.loads(a[1]) and isinstance(json.loads(a[1])[0], list) else '?'} checks)")
PY
