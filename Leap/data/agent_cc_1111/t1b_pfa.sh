#!/bin/bash
# T1b + a PF-A short-tail check, ahead of the cold agent test (Leap 11-11 01:3x; BigMoE agreed 01:3x).
#   0. BigMoE's stand-in test (conv/test_mock.py, MOCK_PFLM=1) on my patched convcache.py: a crash or FAIL stops T1b.
#   1. T1b: T1 again on the cache clone with my patched convcache.py (7eb3401 + (a) the path chosen by the whole prompt,
#      N > chunk, as the loop does; (b) the check names each differing leaf): serve.py --mem 32, CONV_CACHE_CHECK=1, the 45
#      recorded requests of sessions_real_1110 in order (replay.py).
#   2. PF-A vs the default prefill on prompts whose last 1,024-token chunk is short: P = 5 x 1,024 + 116 and + 300 (exact-32,
#      the v110 clean clone = the 1.1.0 candidate, R57_LOGHASH=1, 64 tokens, R57_PFLM=1 vs 0).
# Starts as soon as pid 15929 (the /file check) has exited and the Mac is free (Kamen idle, no other lock / engine, no
# PAUSE / HOLD, NO-SWAP >= 42), polling every 10 s with ONE clear poll, so it starts before the agent test's runner (two
# clear polls 60 s apart). The stand-in runs as .../t1b_mock/run.py so the other runners' engine gate sees it. Skips if
# the agent test went first. Last start 07:00. Share lock; swap +2 GB stops everything; every server / run is stopped.
A=/private/tmp/claude-501/-Users-kamen-OjoX/f2971937-6c95-4e86-a049-05beff6a8e32/scratchpad/agent
C=/Users/kamen/Leap/release/cleanclone/cache/clone; V=/Users/kamen/Leap/release/cleanclone/v110/clone
RUN=$A/t1_runs/32_1111b; PC=$A/pfa_tail; MK=$A/t1b_mock; mkdir -p $RUN $PC $MK; LOG=$RUN/runner.log
PORT=8121; NEED=42; WAITPID=15929
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
LOCKED=0; SP=""; GUARD=""; TP=""
stopsrv() { [ -z "$SP" ] && return; kill -TERM $SP 2>/dev/null
            for i in $(seq 1 60); do kill -0 $SP 2>/dev/null || break; sleep 1; done
            kill -0 $SP 2>/dev/null && { say "serve.py did not stop in 60 s: SIGKILL"; kill -KILL $SP; }
            lsof -nP -iTCP:$PORT -sTCP:LISTEN >/dev/null 2>&1 && say "WARNING: port $PORT still in use"; SP=""; }
stoptp() { [ -n "$TP" ] && kill -0 $TP 2>/dev/null && { for c in $(desc $TP); do kill -TERM $c 2>/dev/null; done; kill -TERM $TP; }; TP=""; }
cleanup() { [ -n "$GUARD" ] && kill $GUARD 2>/dev/null; stoptp; stopsrv
            [ $LOCKED = 1 ] && peer unlock >/dev/null 2>&1; touch $RUN/runner.done; say "exit (NO-SWAP $(noswap) GB, swap $(swap) MB)"; }
trap cleanup EXIT
trap 'say "stopped by a signal"; exit 1' INT TERM
# waits for a child with a wall cap (s); the swap guard kills this script (and so the child) on +2 GB
waitcap() { local cap=$1 t0=$(date +%s) p=$TP rc; [ -z "$p" ] && return 1
            while kill -0 $p 2>/dev/null; do [ $(( $(date +%s) - t0 )) -gt $cap ] && { say "  cap $cap s: stopping"; stoptp; break; }; sleep 3; done
            wait $p 2>/dev/null; rc=$?; TP=""; return $rc; }
say "armed (pid $$): T1b + PF-A tail check -> $RUN, $PC; after pid $WAITPID; agent-test GO lines at arming: $G0; last start 07:00"
NOTE=""
while :; do
  now=$(date +%s); [ $now -ge $LAST ] && { say "SKIP: no free window by 07:00 (last wait: $NOTE)"; exit 0; }
  [ "$(grep -c 'GO:' $AGL 2>/dev/null)" -gt "$G0" ] && { say "SKIP: the agent test went first (T1b goes after the long runs instead)"; exit 0; }
  why=""
  kill -0 $WAITPID 2>/dev/null && why="waits for pid $WAITPID"
> (a quote from the project owner, kept private)
  [ -z "$why" ] && [ -n "$(procs)" ] && why="an engine / chat / server runs"
  [ -z "$why" ] && [ "$(busy)" -gt 0 ] && why="another lock"
  [ -z "$why" ] && held && why="PAUSE / HOLD"
  [ -z "$why" ] && [ "$(noswap)" -lt $NEED ] && why="NO-SWAP $(noswap) < $NEED"
  if [ -z "$why" ]; then
    peer lock share 45m --cpu 400 --gpu 100 "MEM $NEED GB: Leap T1b (conversation-cache fix check, BigMoE) + PF-A short-tail check, ~25 min" > /dev/null 2>&1 && LOCKED=1
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

# 0. the stand-in (BigMoE's test_mock.py with my module; out file in my run folder; no .pyc writes into pkg_110f)
cp -p ~/BigMoE/conv/test_mock.py $MK/run.py; cp -p $C/engine/loop/convcache.py $MK/convcache.py
( cd $MK && PYTHONDONTWRITEBYTECODE=1 CONV_CACHE_CHECK=1 MOCK_PFLM=1 exec ~/BigMoE/.venv/bin/python $MK/run.py $RUN/mock_pflm1.json ) > $RUN/mock_pflm1.out 2>&1 &
TP=$!; waitcap 900; MRC=$?
say "stand-in (MOCK_PFLM=1, patched module): rc $MRC; $(grep -c 'check PASS' $RUN/mock_pflm1.out) PASS / $(grep -c 'check FAIL' $RUN/mock_pflm1.out) FAIL lines; tail: $(tail -1 $RUN/mock_pflm1.out | cut -c1-160)"
MOK=1; { [ $MRC != 0 ] || [ "$(grep -c 'check FAIL' $RUN/mock_pflm1.out)" != 0 ] || grep -q Traceback $RUN/mock_pflm1.out \
         || ! tail -1 $RUN/mock_pflm1.out | grep -q '^PASS'; } && MOK=0

# 1. T1b
if [ $MOK = 1 ]; then
  ( cd $C && CONV_CACHE=1 CONV_CACHE_CHECK=1 CONV_CACHE_LOG=$RUN/convcache.log EXACTSTREAM_LOG_REQUESTS=$RUN/requests.jsonl exec .venv/bin/python serve.py --mem 32 --port $PORT ) > $RUN/serve.log 2>&1 &
  SP=$!; t0=$(date +%s); UP=1
  until curl -s -m 2 http://127.0.0.1:$PORT/health 2>/dev/null | grep -q ready; do
    kill -0 $SP 2>/dev/null && [ $(( $(date +%s) - t0 )) -lt 600 ] || { say "T1b: serve.py did not come up (tail: $(tail -2 $RUN/serve.log | tr '\n' ' '))"; UP=0; break; }; sleep 2
  done
  if [ $UP = 1 ]; then
    say "T1b: serve.py ready in $(( $(date +%s) - t0 )) s; module sha $(shasum -a 256 $C/engine/loop/convcache.py | cut -c1-16)"
    /opt/homebrew/opt/python@3.12/bin/python3.12 -I $A/replay.py $PORT $A/sessions_real_1110.jsonl $RUN/replay.jsonl > $RUN/replay.out 2>&1 &
    TP=$!; waitcap 3600
    say "T1b: $(tail -1 $RUN/replay.out); hits $(grep -c ' hit ' $RUN/convcache.log 2>/dev/null), PASS $(grep -c 'check PASS' $RUN/convcache.log 2>/dev/null), FAIL $(grep -c 'check FAIL' $RUN/convcache.log 2>/dev/null); server errors $(grep -c -E 'Traceback|Error' $RUN/serve.log)"
    grep 'check FAIL' $RUN/convcache.log | cut -c1-400 | while read l; do say "  $l"; done
  fi
  stopsrv; say "T1b: serve.py stopped (NO-SWAP $(noswap) GB)"
else
  say "T1b NOT RUN: the stand-in failed on the patched module (see $RUN/mock_pflm1.out)"
fi

# 2. PF-A vs default on short last chunks (exact-32, the 1.1.0 clean clone)
[ "$(noswap)" -ge $NEED ] || { say "PF-A check SKIP: NO-SWAP $(noswap)"; exit 0; }
( cd $V && PYTHONDONTWRITEBYTECODE=1 exec .venv/bin/python - /Users/kamen/Leap/data/prefill_1109/prompt_30k.txt $PC ) <<'EOF' >> $LOG 2>&1
import sys
from transformers import AutoTokenizer
src, out = sys.argv[1:3]; t = open(src).read(); ins = "Summarize the main findings of the document above in five bullet points."
doc = t[:t.rindex(ins)].rstrip(); tk = AutoTokenizer.from_pretrained("engine/models/bare"); ids = tk.encode(doc, add_special_tokens=False)
for name, tail in (("t116", 116), ("t300", 300)):
    target = 5 * 1024 + tail - 12                     # + 12 template tokens (Tue's P 5,412 prompt)
    for n in range(target - 30, target + 1):
        s = tk.decode(ids[:n]) + "\n\n" + ins + "\n"
        if len(tk.encode(s, add_special_tokens=False)) >= target:
            break
    open(f"{out}/prompt_{name}.txt", "w").write(s)
    print(f"prompt_{name}.txt: {len(tk.encode(s, add_special_tokens=False))} tokens before the template")
EOF
for P_ in t116 t300; do
  for arm in pfa default; do
    [ "$(noswap)" -ge $NEED ] || { say "SKIP $P_ $arm: NO-SWAP $(noswap)"; continue; }
    E="--env R57_PFLM=1"; [ $arm = default ] && E="--env R57_PFLM=0"
    ( cd $V && exec .venv/bin/python run.py --config exact --mem 32 --prompt-file $PC/prompt_$P_.txt --max-tokens 64 --env R57_LOGHASH=1 $E ) > $PC/${P_}_$arm.out 2>&1 &
    TP=$!; waitcap 900
    say "  PF-A check $P_ $arm: $(grep -a -E '^whole decode' $PC/${P_}_$arm.out | cut -c1-150)"
  done
done
/opt/homebrew/opt/python@3.12/bin/python3.12 -I - $PC <<'EOF' | tee -a $LOG
import json, re, sys
A = sys.argv[1]
def files(n):
    try:
        tp = re.search(r"^tokens: ([^;]+)", open(f"{A}/{n}.out").read(), re.M).group(1).strip()
        return json.load(open(tp)), json.load(open(tp.replace("_tokens.json", "_runs.json")))[0]
    except Exception as e:
        return None, None
def toks(x):
    if isinstance(x, list) and x and all(isinstance(t, int) for t in x): return x
    for v in (x.values() if isinstance(x, dict) else x if isinstance(x, list) else []):
        t = toks(v)
        if t: return t
for P_ in ("t116", "t300"):
    ta, ra = files(f"{P_}_default"); tb, rb = files(f"{P_}_pfa")
    if not ra or not rb:
        print(f"PF-A CHECK {P_}: missing result (default {bool(ra)}, PF-A {bool(rb)})"); continue
    a, b = toks(ta), toks(tb); ha, hb = ra.get("lh") or [], rb.get("lh") or []
    ft = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None); eq = sum(x == y for x, y in zip(ha, hb))
    fh = next((i for i, (x, y) in enumerate(zip(ha, hb)) if x != y), None)
    print(f"PF-A CHECK {P_}: P {ra.get('P')} (last chunk {ra.get('P') % 1024 if ra.get('P') else '?'}); tokens {'EQUAL' if a == b else f'differ from token {ft + 1}'}; "
          f"hashes equal on {eq} of {min(len(ha), len(hb))}{'' if fh is None else f' (first difference at check {fh + 1})'}; "
          f"first token {ra['pfs']:.2f} s default vs {rb['pfs']:.2f} s PF-A; peaks {ra['mem'][6]:.2f} / {rb['mem'][6]:.2f} GB")
EOF
say "done: NO-SWAP $(noswap) GB, swap $(swap) MB (start $S0)"
