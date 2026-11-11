#!/bin/bash
# BIGMOE_IDLE_MS=50 A/B for 1.1.0 + release/next (orchestrator 01:3x: "Yes to BIGMOE_IDLE_MS=50 ... idle CPU, plus decode
# speed on a few prompts in separate passes, so the arm order can't favour one side"). The v110 clean clone (= the 1.1.0
# candidate, unchanged; the variable is set in the process environment, which the ext reads at load).
#   1. Idle CPU with BIGMOE_IDLE_MS=50: serve.py --mem 24, then 32; one 16-token request, 10 s rest, CPU time over 60 s
#      (tonight's control without it, 01:27-01:30: 100 % of one core at both).
#   2. Decode: run.py --config chat --kind code, 3 HumanEval items x 256 tokens, at --mem 24 then 32; arms A = unset,
#      B = BIGMOE_IDLE_MS=50, in the order A B B A per setting (each run its own process). Tokens must be equal in all 4.
# Starts as soon as pid 14659 (T1b + PF-A check) has exited and the Mac is free (one clear poll, 10-s polls: before the
# agent test's runner, which needs two clear polls 60 s apart); skips if the agent test went first. Last start 07:00.
A=/private/tmp/claude-501/-Users-kamen-OjoX/f2971937-6c95-4e86-a049-05beff6a8e32/scratchpad/agent
V=/Users/kamen/Leap/release/cleanclone/v110/clone; OUT=$A/ab_idle; mkdir -p $OUT; LOG=$OUT/runner.log
PORT=8098; NEED=42; WAITPID=14659
KEYS=humaneval/HumanEval/68,humaneval/HumanEval/16,humaneval/HumanEval/91
AGL=$A/agentcc_runs/32_1111/runner.log; G0=$(grep -c 'GO:' $AGL 2>/dev/null)
LAST=$(date -j -f "%Y-%m-%d %H:%M:%S" "2026-11-11 07:00:00" +%s)
noswap() { vm_stat | awk '/page size of/ {ps=$8} /^Pages (free|speculative|purgeable):/ || /^File-backed pages:/ {v=$NF; sub("[.]","",v); s+=v} END {printf "%d", s*ps/1e9}'; }
swap() { sysctl -n vm.swapusage | sed -E 's/.*used = ([0-9.]+)M.*/\1/' | cut -d. -f1; }
idle() { ioreg -c IOHIDSystem | awk '/HIDIdleTime/ {print int($NF/1000000000); exit}'; }
busy() { peer ls 2>/dev/null | grep "^- " | grep -oE "· (shared run \(others may work\)|benchmarking|waiting for a quiet window)[^·]*" | grep -cE "^· (benchmarking|waiting for a quiet window)|MEM ([6-9]|[1-9][0-9]) GB"; }
procs() { ps -axo pid=,args= | awk '/[Pp]ython/ && /r57_screen_|run_map[.]py|fit_onecall[.]py|[ \/](chat|serve|run)[.]py( |$)/ && !/ awk / {printf "%s ", $1}'; }
held() { [ -e ~/.local/share/quiet/PAUSE ] || [ -e ~/.local/share/quiet/HOLD ]; }
desc() { local c; for c in $(pgrep -P "$1"); do desc "$c"; echo "$c"; done; }
say() { echo "$(date '+%m-%d %H:%M:%S') $*" | tee -a $LOG; }
cpus() { ps -o time= -p $1 2>/dev/null | awk -F: '{if (NF == 3) print $1 * 3600 + $2 * 60 + $3; else print $1 * 60 + $2}'; }
LOCKED=0; SP=""; GUARD=""; TP=""
stopsrv() { [ -z "$SP" ] && return; kill -TERM $SP 2>/dev/null
            for i in $(seq 1 60); do kill -0 $SP 2>/dev/null || break; sleep 1; done
            kill -0 $SP 2>/dev/null && { say "serve.py did not stop in 60 s: SIGKILL"; kill -KILL $SP; }
            lsof -nP -iTCP:$PORT -sTCP:LISTEN >/dev/null 2>&1 && say "WARNING: port $PORT still in use"; SP=""; }
stoptp() { [ -n "$TP" ] && kill -0 $TP 2>/dev/null && { for c in $(desc $TP); do kill -TERM $c 2>/dev/null; done; kill -TERM $TP; }; TP=""; }
waitcap() { local cap=$1 t0=$(date +%s) p=$TP rc; [ -z "$p" ] && return 1
            while kill -0 $p 2>/dev/null; do [ $(( $(date +%s) - t0 )) -gt $cap ] && { say "  cap $cap s: stopping"; stoptp; break; }; sleep 2; done
            wait $p 2>/dev/null; rc=$?; TP=""; return $rc; }
cleanup() { [ -n "$GUARD" ] && kill $GUARD 2>/dev/null; stoptp; stopsrv
            [ $LOCKED = 1 ] && peer unlock >/dev/null 2>&1; touch $OUT/runner.done; say "exit (NO-SWAP $(noswap) GB, swap $(swap) MB)"; }
trap cleanup EXIT
trap 'say "stopped by a signal"; exit 1' INT TERM
say "armed (pid $$): BIGMOE_IDLE_MS=50 A/B -> $OUT; after pid $WAITPID; agent-test GO lines at arming: $G0; last start 07:00"
NOTE=""
while :; do
  now=$(date +%s); [ $now -ge $LAST ] && { say "SKIP: no free window by 07:00 (last wait: $NOTE)"; exit 0; }
  [ "$(grep -c 'GO:' $AGL 2>/dev/null)" -gt "$G0" ] && { say "SKIP: the agent test went first (the A/B goes at the next free slot)"; exit 0; }
  why=""
  kill -0 $WAITPID 2>/dev/null && why="waits for pid $WAITPID"
> (a quote from the project owner, kept private)
  [ -z "$why" ] && [ -n "$(procs)" ] && why="an engine / chat / server runs"
  [ -z "$why" ] && [ "$(busy)" -gt 0 ] && why="another lock"
  [ -z "$why" ] && held && why="PAUSE / HOLD"
  [ -z "$why" ] && [ "$(noswap)" -lt $NEED ] && why="NO-SWAP $(noswap) < $NEED"
  if [ -z "$why" ]; then
    peer lock share 30m --cpu 400 --gpu 100 "MEM $NEED GB: Leap BIGMOE_IDLE_MS A/B for 1.1.0 (idle CPU + decode, ~12 min)" > /dev/null 2>&1 && LOCKED=1
    sleep 2
    if [ -z "$(procs)" ] && [ "$(grep -c 'GO:' $AGL 2>/dev/null)" -le "$G0" ]; then break; fi
    peer unlock >/dev/null 2>&1; LOCKED=0; why="after the lock"
  fi
  [ "$why" != "$NOTE" ] && say "wait: $why"; NOTE=$why
  sleep 10
done
S0=$(swap); say "GO: NO-SWAP $(noswap) GB, swap $S0 MB"
( while :; do [ $(( $(swap) - S0 )) -gt 2048 ] && { echo "$(date +%T) SWAP GUARD: $(swap) MB" >> $LOG; kill -TERM $$; exit; }; sleep 5; done ) &
GUARD=$!

# 1. idle CPU with the nap
for M in 24 32; do
  ( cd $V && BIGMOE_IDLE_MS=50 exec .venv/bin/python serve.py --mem $M --port $PORT ) > $OUT/idle50_serve_$M.log 2>&1 &
  SP=$!; t0=$(date +%s); UP=1
  until curl -s -m 2 http://127.0.0.1:$PORT/health 2>/dev/null | grep -q ready; do
    kill -0 $SP 2>/dev/null && [ $(( $(date +%s) - t0 )) -lt 300 ] || { say "$M GB: serve.py did not come up"; UP=0; break; }; sleep 2
  done
  if [ $UP = 1 ]; then
    q0=$(date +%s)
    r=$(curl -s -m 300 http://127.0.0.1:$PORT/v1/chat/completions -H 'content-type: application/json' \
        -d '{"model": "exactstream", "max_tokens": 16, "messages": [{"role": "user", "content": "Say hello in three words."}]}' | cut -c1-120)
    q1=$(date +%s)
    sleep 10; c0=$(cpus $SP); sleep 60; c1=$(cpus $SP)
    pct=$(awk -v a="$c0" -v b="$c1" 'BEGIN {printf "%.1f", (b - a) / 60 * 100}')
    say "IDLE50 $M GB: idle CPU over 60 s after the request: $pct % of one core (cpu time $c0 -> $c1 s); request $((q1 - q0)) s: $r"
    q0=$(date +%s)                                           # a second request after the long idle: still served
    r2=$(curl -s -m 300 http://127.0.0.1:$PORT/v1/chat/completions -H 'content-type: application/json' \
        -d '{"model": "exactstream", "max_tokens": 16, "messages": [{"role": "user", "content": "Say hello in three words."}]}' | cut -c1-120)
    say "IDLE50 $M GB: request after 70 s idle: $(( $(date +%s) - q0 )) s, $(echo "$r2" | grep -c chat.completion) answer"
    grep -E 'done:' $OUT/idle50_serve_$M.log | tail -2 | cut -c1-200 | while read l; do say "  $l"; done
  fi
  stopsrv
done

# 2. decode A B B A per setting (separate processes)
for M in 24 32; do
  k=0
  for arm in A B B A; do
    k=$((k + 1))
    [ "$(noswap)" -ge $NEED ] || { say "SKIP $M $arm$k: NO-SWAP $(noswap)"; continue; }
    if [ $arm = B ]; then
      ( cd $V && BIGMOE_IDLE_MS=50 exec .venv/bin/python run.py --config chat --mem $M --kind code --keys $KEYS --max-tokens 256 ) > $OUT/dec_${M}_${k}${arm}.out 2>&1 &
    else
      ( cd $V && exec .venv/bin/python run.py --config chat --mem $M --kind code --keys $KEYS --max-tokens 256 ) > $OUT/dec_${M}_${k}${arm}.out 2>&1 &
    fi
    TP=$!; waitcap 600
    say "  decode $M GB run $k arm $arm: $(grep -a -E '^whole decode' $OUT/dec_${M}_${k}${arm}.out | cut -c1-170)"
  done
done
/opt/homebrew/opt/python@3.12/bin/python3.12 -I - $OUT <<'EOF' | tee -a $LOG
import glob, json, os, re, statistics as st, sys
O = sys.argv[1]
for M in ("24", "32"):
    rows = {}
    for f in sorted(glob.glob(f"{O}/dec_{M}_*.out")):
        t = open(f, errors="replace").read()
        m = re.search(r"^whole decode ([\d.]+) tok/s .*?first token ([\d.]+) s", t, re.M)
        tp = re.search(r"^tokens: ([^;]+)", t, re.M)
        arm = os.path.basename(f)[-5]
        toks = json.load(open(tp.group(1).strip())) if tp else None
        sig = json.dumps({k: sorted(json.dumps(v2) for v2 in (v.values() if isinstance(v, dict) else [v])) for k, v in toks.items()},
                         sort_keys=True) if toks else None          # the token lists per item (tag names ignored)
        rows[os.path.basename(f)] = (arm, float(m.group(1)) if m else None, float(m.group(2)) if m else None, sig)
    a = [r[1] for r in rows.values() if r[0] == "A" and r[1]]; b = [r[1] for r in rows.values() if r[0] == "B" and r[1]]
    fa = [r[2] for r in rows.values() if r[0] == "A" and r[2]]; fb = [r[2] for r in rows.values() if r[0] == "B" and r[2]]
    tk = {r[3] for r in rows.values() if r[3]}
    if a and b:
        print(f"A/B {M} GB: decode A (unset) {', '.join(f'{x:.1f}' for x in a)} -> mean {st.mean(a):.1f} tok/s; "
              f"B (IDLE_MS 50) {', '.join(f'{x:.1f}' for x in b)} -> mean {st.mean(b):.1f}; B/A {st.mean(b) / st.mean(a):.3f}; "
              f"first token A {st.mean(fa):.2f} s, B {st.mean(fb):.2f} s; tokens {'EQUAL in all ' + str(len(rows)) if len(tk) == 1 else f'{len(tk)} distinct outputs'}")
    else:
        print(f"A/B {M} GB: incomplete ({len(a)} A, {len(b)} B)")
EOF
say "done: NO-SWAP $(noswap) GB, swap $(swap) MB (start $S0)"
