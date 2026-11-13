#!/bin/bash
# Prefill 1.5K, tonight after prof_pipe.sh (waits for <work>/pipe.done): PF-B pipelined (R57_PFLM_PIPE=1) at 32 GB on the 30K
# prompt with MTPLX's exact dense-band QSA (MTPLX_QSA_DENSE_BAND_SDPA=1; the engine runs MTPLX's stock dense-mask fallback
# today; PrefillKernels: band == stock, 0 differing) and a cProfile of every QSA self_attn call (R57_PFLM_QSA_PROFILE).
# Bitwise check: tokens + R57_LOGHASH verify-logit hashes vs prof_pipe.sh's 30K PIPE=1 run. Gates as prof_pipe.sh; last start
# LAST. Usage: prof_qsa.sh <work dir> "<LAST YYYY-MM-DD HH:MM:SS>"
W=$1; P2=/Users/kamen/Leap/scripts/prefill15/prof2; CL=/Users/kamen/Leap/release/cleanclone/cache/clone; LOG=$W/qsa.log
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
            [ $LOCKED = 1 ] && peer unlock >/dev/null 2>&1; touch $W/qsa.done; say "exit (NO-SWAP $(noswap) GB, swap $(swap) MB)"; }
trap cleanup EXIT; trap 'say "stopped by a signal"; exit 1' INT TERM
say "armed (pid $$): 30K PIPE=1 + dense band + QSA cProfile, after $W/pipe.done; last start $2"
NOTE=""; CLR=0
while :; do
  now=$(date +%s); [ $now -ge $LAST ] && { say "SKIP: no free window by $2 (last wait: $NOTE)"; exit 0; }
  why=""
  [ ! -e $W/pipe.done ] && why="waits for prof_pipe.sh (pipe.done)"
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
peer lock share 30m --cpu 400 --gpu 100 "MEM 46 GB: Leap prefill QSA dense band + profile (32 GB, 30K), ~3 min" > /dev/null 2>&1 && LOCKED=1
S0=$(swap); ( while :; do [ $(( $(swap) - S0 )) -gt 2048 ] && { echo "$(date +%T) SWAP GUARD: $(swap) MB" >> $LOG; kill -TERM $$; exit; }; sleep 5; done ) &
GUARD=$!
J=$W/qsa_30k_band.jsonl; QP=$W/qsa_30k_band_cprofile.txt; rm -f $J $QP
say "run 30k pipe=1 band=1 (NO-SWAP $(noswap) GB)"
( cd $CL && exec env PYTHONPATH=$P2 R57_PFLM_PROF=$J R57_PFLM_PROF_SYNC=0 R57_PFLM_PIPE=1 R57_PFLM_QSA_PROFILE=$QP MTPLX_QSA_DENSE_BAND_SDPA=1 \
    .venv/bin/python run.py --config chat --mem 32 --prompt-file $W/prompt_30k.txt --max-tokens 32 --env R57_PFLM=3 --env R57_LOGHASH=1 ) \
    > $W/qsa_30k_band.out 2>&1 &
RP=$!; wait $RP; rc=$?; RP=""
tag=$(grep -o 'out/cli_[0-9_]*_tokens.json' $W/qsa_30k_band.out | head -1 | sed 's|out/||; s|_tokens.json||')
[ -n "$tag" ] && cp $CL/out/${tag}_runs.json $W/qsa_30k_band_runs.json 2>/dev/null && cp $CL/out/${tag}_tokens.json $W/qsa_30k_band_tokens.json 2>/dev/null
say "  rc $rc; $(grep -o '"total": \[[^]]*\]' $J 2>/dev/null); $(grep -o 'first token [0-9.]* s' $W/qsa_30k_band.out | head -1); $(grep -o 'lifetime peak [0-9.]* GB' $W/qsa_30k_band.out | tail -1); tag $tag"
/opt/homebrew/opt/python@3.12/bin/python3.12 -I - $W <<'PY' | tee -a $LOG
import json, sys
W = sys.argv[1]
def get(t, r):
    try:
        tk = json.load(open(f"{W}/{t}")); rr = json.load(open(f"{W}/{r}"))
    except Exception:
        return None
    lh = []
    def walk(o):
        if isinstance(o, dict):
            for k, v in o.items():
                (lh.append(v) if k == "lh" else walk(v))
        elif isinstance(o, list):
            for v in o: walk(v)
    walk(rr)
    return json.dumps(tk, sort_keys=True), json.dumps(lh)
a = get("pipe_30k_p1_tokens.json", "pipe_30k_p1_runs.json"); b = get("qsa_30k_band_tokens.json", "qsa_30k_band_runs.json")
print("30k band vs stock (both PIPE=1) BITWISE:", "missing files" if None in (a, b) else
      f"tokens {'EQUAL' if a[0] == b[0] else 'DIFFER'} | verify-logit hashes {'EQUAL' if a[1] == b[1] else 'DIFFER'}")
PY
