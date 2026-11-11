#!/bin/bash
# 1.1.0 candidate pkg_110g (7f61f48 = pkg_110f + the PF-A qualifier + BIGMOE_IDLE_MS=50 in chat.py / serve.py): the
# real-engine checks the orchestrator asked for (01:5x), right after EvalSuite's cold 32 GB agent test (pid 15962), before
# the 64K / 128K runs. In the prepped clean clone release/cleanclone/v110g (install.sh rc 0, dry runs OK at 02:00):
#   1. BIGMOE_IDLE_MS confirmation at 32 GB: run.py --config chat --kind code, 3 HumanEval x 256 tokens, 8 runs in the order
#      B A A B B A A B (A = BIGMOE_IDLE_MS=0, B = 50; run.py does not set it), each its own process; tokens must be equal.
#   2. The clean-clone check, as pkg_110c's: tools/clean_clone_test.sh run with CONFIGS="lossy-24 chat-24", the text prompt
#      on chat-24, chat.py and serve.py --mem 24 (HOME empty; tokens vs the shipped references; closure).
#   3. The packaged defaults at idle, no variable set: serve.py --mem 24 (one request, 10 s rest, CPU over 60 s) and chat.py
#      --mem 24 (one piped message, the input kept open, CPU over 60 s after the answer, then /exit).
# Starts once pid 15962 has exited and the Mac is free (one clear poll, 10-s polls: before long_ctx's two 60-s polls).
# Last start 09:00. Share lock; swap +2 GB stops everything; every server / run is stopped and the port checked.
A=/private/tmp/claude-501/-Users-kamen-OjoX/f2971937-6c95-4e86-a049-05beff6a8e32/scratchpad/agent
W=/Users/kamen/Leap/release/cleanclone/v110g; CL=$W/clone; OUT=$A/cc110g; mkdir -p $OUT; LOG=$OUT/runner.log
PORT=8099; NEED=42; WAITPID=15962
KEYS=humaneval/HumanEval/68,humaneval/HumanEval/16,humaneval/HumanEval/91
LAST=$(date -j -f "%Y-%m-%d %H:%M:%S" "2026-11-11 09:00:00" +%s)
noswap() { vm_stat | awk '/page size of/ {ps=$8} /^Pages (free|speculative|purgeable):/ || /^File-backed pages:/ {v=$NF; sub("[.]","",v); s+=v} END {printf "%d", s*ps/1e9}'; }
swap() { sysctl -n vm.swapusage | sed -E 's/.*used = ([0-9.]+)M.*/\1/' | cut -d. -f1; }
idle() { ioreg -c IOHIDSystem | awk '/HIDIdleTime/ {print int($NF/1000000000); exit}'; }
busy() { peer ls 2>/dev/null | grep "^- " | grep -oE "· (shared run \(others may work\)|benchmarking|waiting for a quiet window)[^·]*" | grep -cE "^· (benchmarking|waiting for a quiet window)|MEM ([6-9]|[1-9][0-9]) GB"; }
procs() { ps -axo pid=,args= | awk '/[Pp]ython/ && /r57_screen_|run_map[.]py|fit_onecall[.]py|[ \/](chat|serve|run)[.]py( |$)/ && !/ awk / {printf "%s ", $1}'; }
held() { [ -e ~/.local/share/quiet/PAUSE ] || [ -e ~/.local/share/quiet/HOLD ]; }
desc() { local c; for c in $(pgrep -P "$1"); do desc "$c"; echo "$c"; done; }
say() { echo "$(date '+%m-%d %H:%M:%S') $*" | tee -a $LOG; }
cpus() { ps -o time= -p $1 2>/dev/null | awk -F: '{if (NF == 3) print $1 * 3600 + $2 * 60 + $3; else print $1 * 60 + $2}'; }
LOCKED=0; SP=""; GUARD=""; TP=""; FW=""
stopsrv() { [ -z "$SP" ] && return; kill -TERM $SP 2>/dev/null
            for i in $(seq 1 60); do kill -0 $SP 2>/dev/null || break; sleep 1; done
            kill -0 $SP 2>/dev/null && { say "serve.py did not stop in 60 s: SIGKILL"; kill -KILL $SP; }
            lsof -nP -iTCP:$PORT -sTCP:LISTEN >/dev/null 2>&1 && say "WARNING: port $PORT still in use"; SP=""; }
stoptp() { [ -n "$TP" ] && kill -0 $TP 2>/dev/null && { for c in $(desc $TP); do kill -TERM $c 2>/dev/null; done; kill -TERM $TP; }; TP=""; }
waitcap() { local cap=$1 t0=$(date +%s) p=$TP rc; [ -z "$p" ] && return 1
            while kill -0 $p 2>/dev/null; do [ $(( $(date +%s) - t0 )) -gt $cap ] && { say "  cap $cap s: stopping"; stoptp; break; }; sleep 2; done
            wait $p 2>/dev/null; rc=$?; TP=""; return $rc; }
cleanup() { [ -n "$GUARD" ] && kill $GUARD 2>/dev/null; [ -n "$FW" ] && { for c in $(desc $FW); do kill $c 2>/dev/null; done; kill $FW 2>/dev/null; }
            stoptp; stopsrv; rm -f $OUT/chat_in.fifo
            [ $LOCKED = 1 ] && peer unlock >/dev/null 2>&1; touch $OUT/runner.done; say "exit (NO-SWAP $(noswap) GB, swap $(swap) MB)"; }
trap cleanup EXIT
trap 'say "stopped by a signal"; exit 1' INT TERM
say "armed (pid $$): pkg_110g real-engine checks -> $OUT; after pid $WAITPID; last start 09:00"
NOTE=""
while :; do
  now=$(date +%s); [ $now -ge $LAST ] && { say "SKIP: no free window by 09:00 (last wait: $NOTE)"; exit 0; }
  why=""
  kill -0 $WAITPID 2>/dev/null && why="waits for pid $WAITPID (the agent test)"
> (a quote from the project owner, kept private)
  [ -z "$why" ] && [ -n "$(procs)" ] && why="an engine / chat / server runs"
  [ -z "$why" ] && [ "$(busy)" -gt 0 ] && why="another lock"
  [ -z "$why" ] && held && why="PAUSE / HOLD"
  [ -z "$why" ] && [ "$(noswap)" -lt $NEED ] && why="NO-SWAP $(noswap) < $NEED"
  if [ -z "$why" ]; then
    peer lock share 30m --cpu 400 --gpu 100 "MEM $NEED GB: Leap 1.1.0 candidate pkg_110g checks (IDLE_MS A/B + clean clone + idle), ~15 min" > /dev/null 2>&1 && LOCKED=1
    sleep 2
    [ -z "$(procs)" ] && break
    peer unlock >/dev/null 2>&1; LOCKED=0; why="after the lock"
  fi
  [ "$why" != "$NOTE" ] && say "wait: $why"; NOTE=$why
  sleep 10
done
S0=$(swap); say "GO: NO-SWAP $(noswap) GB, swap $S0 MB"
( while :; do [ $(( $(swap) - S0 )) -gt 2048 ] && { echo "$(date +%T) SWAP GUARD: $(swap) MB" >> $LOG; kill -TERM $$; exit; }; sleep 5; done ) &
GUARD=$!

# 1. BIGMOE_IDLE_MS confirmation at 32 GB (B A A B B A A B)
k=0
for arm in B A A B B A A B; do
  k=$((k + 1)); v=50; [ $arm = A ] && v=0
  ( cd $CL && BIGMOE_IDLE_MS=$v exec .venv/bin/python run.py --config chat --mem 32 --kind code --keys $KEYS --max-tokens 256 ) > $OUT/dec_32_${k}${arm}.out 2>&1 &
  TP=$!; waitcap 600
  say "  IDLE A/B 32 GB run $k arm $arm: $(grep -a -E '^whole decode' $OUT/dec_32_${k}${arm}.out | cut -c1-150)"
done
/opt/homebrew/opt/python@3.12/bin/python3.12 -I - $OUT $A/ab_idle <<'EOF' | tee -a $LOG
import glob, json, os, re, statistics as st, sys
O, O1 = sys.argv[1], sys.argv[2]
def rows(d):
    out = []
    for f in sorted(glob.glob(f"{d}/dec_32_*.out")):
        t = open(f, errors="replace").read()
        m = re.search(r"^whole decode ([\d.]+) tok/s .*?first token ([\d.]+) s", t, re.M)
        tp = re.search(r"^tokens: ([^;]+)", t, re.M)
        toks = json.load(open(tp.group(1).strip())) if tp else None
        sig = json.dumps({k: sorted(json.dumps(v2) for v2 in (v.values() if isinstance(v, dict) else [v])) for k, v in toks.items()}, sort_keys=True) if toks else None
        out.append((os.path.basename(f)[-5], float(m.group(1)) if m else None, sig))
    return out
new, old = rows(O), rows(O1)
for name, rs in (("tonight 02:xx (8 runs)", new), ("with 01:5x (12 runs)", new + old)):
    a = [r[1] for r in rs if r[0] == "A" and r[1]]; b = [r[1] for r in rs if r[0] == "B" and r[1]]
    if len(a) > 1 and len(b) > 1:
        d = st.mean(b) - st.mean(a); se = (st.variance(a) / len(a) + st.variance(b) / len(b)) ** 0.5
        print(f"IDLE A/B 32 GB {name}: A {st.mean(a):.1f} tok/s (n {len(a)}, sd {st.stdev(a):.2f}), B {st.mean(b):.1f} (n {len(b)}, sd {st.stdev(b):.2f}); "
              f"B/A {st.mean(b) / st.mean(a):.4f}; diff {d:+.2f} tok/s = {d / se:+.1f} SE; tokens {'EQUAL in all ' + str(len(rs)) if len({r[2] for r in rs if r[2]}) == 1 else 'DIFFER'}")
EOF

# 2. the clean-clone check (as pkg_110c's)
say "clean-clone check: CONFIGS lossy-24 chat-24, text on chat-24, chat.py + serve.py --mem 24"
WHEELS=/Users/kamen/Leap/release/cleanclone/wheels CONFIGS="lossy-24 chat-24" TEXT_CONFIG=chat-24 CHAT_SERVE=1 CS_MEM=24 \
  bash $CL/tools/clean_clone_test.sh $CL $W run > $OUT/clean_clone_run.log 2>&1 &
TP=$!; waitcap 1200
say "clean-clone check: $(tail -1 $W/report.txt)"
grep -E 'prefix-equal|closure|tok/s|chat.py rc|serve.py|stopped|PASS|FAIL' $W/report.txt | tail -14 | cut -c1-220 | while read l; do say "  $l"; done

# 3. the packaged defaults at idle (no BIGMOE_IDLE_MS in the environment)
unset BIGMOE_IDLE_MS
( cd $CL && exec .venv/bin/python serve.py --mem 24 --port $PORT ) > $OUT/idle_serve_24.log 2>&1 &
SP=$!; t0=$(date +%s); UP=1
until curl -s -m 2 http://127.0.0.1:$PORT/health 2>/dev/null | grep -q ready; do
  kill -0 $SP 2>/dev/null && [ $(( $(date +%s) - t0 )) -lt 300 ] || { say "idle serve.py did not come up"; UP=0; break; }; sleep 2
done
if [ $UP = 1 ]; then
  curl -s -m 300 http://127.0.0.1:$PORT/v1/chat/completions -H 'content-type: application/json' \
    -d '{"model": "exactstream", "max_tokens": 16, "messages": [{"role": "user", "content": "Say hello in three words."}]}' > /dev/null
  sleep 10; c0=$(cpus $SP); sleep 60; c1=$(cpus $SP)
  say "packaged serve.py --mem 24 idle CPU over 60 s: $(awk -v a="$c0" -v b="$c1" 'BEGIN {printf "%.1f", (b - a) / 60 * 100}') % of one core (no variable set)"
fi
stopsrv
F=$OUT/chat_in.fifo; rm -f $F; mkfifo $F
( cd $CL && exec .venv/bin/python chat.py --mem 24 --max-tokens 64 < $F ) > $OUT/idle_chat_24.txt 2>&1 &
TP=$!
( printf '%s\n' "Say hello in three words."; sleep 400; printf '/exit\n' ) > $F &
FW=$!
t0=$(date +%s); ANS=0
while kill -0 $TP 2>/dev/null && [ $(( $(date +%s) - t0 )) -lt 300 ]; do
  grep -a -q -E 'tokens \| whole decode' $OUT/idle_chat_24.txt 2>/dev/null && { ANS=1; break; }; sleep 2
done
if [ $ANS = 1 ]; then                                     # TP is chat.py itself (the subshell exec'd it)
  sleep 10; c0=$(cpus $TP); sleep 60; c1=$(cpus $TP)
  say "packaged chat.py --mem 24 idle CPU over 60 s after its answer: $(awk -v a="$c0" -v b="$c1" 'BEGIN {printf "%.1f", (b - a) / 60 * 100}') % of one core (no variable set)"
else
  say "packaged chat.py: no answer within 300 s (see $OUT/idle_chat_24.txt)"
fi
for c in $(desc $FW); do kill $c 2>/dev/null; done; kill $FW 2>/dev/null; FW=""; stoptp; rm -f $F
say "done: NO-SWAP $(noswap) GB, swap $(swap) MB (start $S0)"
