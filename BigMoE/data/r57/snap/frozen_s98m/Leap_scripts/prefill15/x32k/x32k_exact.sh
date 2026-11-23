#!/bin/bash
# The >32K exact check (the orchestrator 11-12 17:0x; from prof/prof_pipe2.sh's gates). Does `exact` stay bitwise past 32K
# tokens of context? MTPLX's sparse QSA prefill lane resolves AUTO (on, M5) past 32,768 of context, but its gate also needs
# MTPLX's "prefill" attention phase, which this engine's loop never sets. A CPU check of the gate gave False at 40K as shipped
# (data/prefill15_1112.md). This is the engine-level confirmation.
# - Two arms, one process each, run.py --config exact --mem 32 (the default prefill), ~40K-token prompt (40,016 tokens; cut
#   from long_ctx's papers prompt), --max-tokens 32, R57_LOGHASH=1:
#   auto  = as shipped;
#   dense = MTPLX_QSA_PREFILL=0 (the lane pinned off).
# - Expected: tokens and every verify-logit hash ("lh") EQUAL. If they differ, the AUTO lane engages and exact needs the pin.
# - No timings: heat pauses do not matter. The lifetime peaks are recorded for the gates.
# Gates before each run: no engine / chat / server, no other big or quiet lock, no PAUSE / HOLD, Kamen idle >= 600 s,
# NO-SWAP >= 46 GB (exact-32's peak at 30K is ~35-36 GB; 40K adds about 1 GB of KV and score plane). A share lock with MEM,
# retried until granted, re-checked 20 s after the grant. Swap +2 GB stops it. From FROM, last start LAST.
# Usage: x32k_exact.sh <work dir with prompt_40k.txt> "<FROM YYYY-MM-DD HH:MM:SS>" "<LAST ...>"
W=$1; CL=/Users/kamen/Leap/release/cleanclone/cache/clone; LOG=$W/x32k.log
FROM=$(date -j -f "%Y-%m-%d %H:%M:%S" "$2" +%s); LAST=$(date -j -f "%Y-%m-%d %H:%M:%S" "$3" +%s)
noswap() { vm_stat | awk '/page size of/ {ps=$8} /^Pages (free|purgeable):/ || /^File-backed pages:/ {v=$NF; sub("[.]","",v); s+=v} END {printf "%d", s*ps/1e9}'; }
swap() { sysctl -n vm.swapusage | sed -E 's/.*used = ([0-9.]+)M.*/\1/' | cut -d. -f1; }
idle() { ioreg -c IOHIDSystem | awk '/HIDIdleTime/ {print int($NF/1000000000); exit}'; }
procs() { ps -axo pid=,args= | awk '/[Pp]ython/ && /r57_screen_|[ \/](chat|serve|run)[.]py( |$)/ && !/ awk / {printf "%s ", $1}'; }
busy() { peer ls 2>/dev/null | grep "^- " | grep -oE "· (shared run \(others may work\)|benchmarking|waiting for a quiet window)[^·]*" | grep -cE "^· (benchmarking|waiting for a quiet window)|MEM ([6-9]|[1-9][0-9]) GB"; }
held() { [ -e ~/.local/share/quiet/PAUSE ] || [ -e ~/.local/share/quiet/HOLD ]; }
say() { echo "$(date '+%m-%d %H:%M:%S') $*" | tee -a $LOG; }
RP=""; GUARD=""; LOCKED=0
cleanup() { [ -n "$GUARD" ] && kill $GUARD 2>/dev/null; [ -n "$RP" ] && kill -0 $RP 2>/dev/null && kill -TERM $RP
            [ $LOCKED = 1 ] && peer unlock >/dev/null 2>&1; touch $W/x32k.done; say "exit (NO-SWAP $(noswap) GB, swap $(swap) MB)"; }
trap cleanup EXIT; trap 'say "stopped by a signal"; exit 1' INT TERM
[ -f $W/prompt_40k.txt ] || { say "no $W/prompt_40k.txt"; exit 2; }
say "armed (pid $$): exact-32 at 40K, auto vs MTPLX_QSA_PREFILL=0, tokens + LOGHASH; from $2, last start $3"
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
  if [ -z "$why" ]; then
    if peer lock share 40m --cpu 400 --gpu 100 "MEM 38 GB: Leap exact check past 32K (exact-32, 40K prompt, 2 runs), ~10 min" > /dev/null 2>&1; then
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
for arm in auto dense; do
  [ -n "$(procs)" ] && { say "SKIP $arm: an engine runs"; continue; }
  [ "$(noswap)" -lt 44 ] && { say "SKIP $arm: NO-SWAP $(noswap) < 44"; continue; }
  PIN=""; [ $arm = dense ] && PIN="MTPLX_QSA_PREFILL=0"
  say "run $arm (NO-SWAP $(noswap) GB)"
  ( cd $CL && exec env $PIN .venv/bin/python run.py --config exact --mem 32 --prompt-file $W/prompt_40k.txt --max-tokens 32 \
      --env R57_LOGHASH=1 ) > $W/x32k_$arm.out 2>&1 &
  RP=$!; wait $RP; rc=$?; RP=""
  tag=$(grep -o 'out/cli_[0-9_]*_tokens.json' $W/x32k_$arm.out | head -1 | sed 's|out/||; s|_tokens.json||')
  [ -n "$tag" ] && cp $CL/out/${tag}_runs.json $W/x32k_${arm}_runs.json 2>/dev/null && cp $CL/out/${tag}_tokens.json $W/x32k_${arm}_tokens.json 2>/dev/null
  say "  rc $rc; $(grep -o 'lifetime peak [0-9.]* GB' $W/x32k_$arm.out | head -1)"
done
/opt/homebrew/opt/python@3.12/bin/python3.12 -I - $W <<'PY' | tee -a $LOG
import json, sys
W = sys.argv[1]
def get(arm):
    try:
        tk = json.load(open(f"{W}/x32k_{arm}_tokens.json")); rr = json.load(open(f"{W}/x32k_{arm}_runs.json"))
    except Exception:
        return None, None
    lh = []
    def walk(o):
        if isinstance(o, dict):
            for k, v in o.items():
                if k == "lh": lh.append(v)
                else: walk(v)
        elif isinstance(o, list):
            for v in o: walk(v)
    walk(rr)
    # the token lists only: a prompt-file run's item key carries its own timestamp (cli_<time>), so whole files always differ
    return json.dumps([v for k in sorted(tk) for a in sorted(tk[k]) for v in tk[k][a]]), json.dumps(lh)
a, b = get("auto"), get("dense")
if None in a or None in b:
    print("x32k BITWISE: missing files")
else:
    print("x32k BITWISE (exact-32, 40K): tokens", "EQUAL" if a[0] == b[0] else "DIFFER", "| verify-logit hashes",
          "EQUAL" if a[1] == b[1] else "DIFFER", "| lh lists", len(json.loads(a[1])))
PY
