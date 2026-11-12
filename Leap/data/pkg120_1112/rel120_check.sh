#!/bin/bash
# The 1.2.0 candidate's GPU checks (the orchestrator 18:2x: "the clean-clone test on the candidate, a P2 smoke on chat-24
# (tokens equal to the gated arm)"), on the prepared clean clone (release/cleanclone/v120, prep PASS), one stage at a time,
# each only if its memory floor holds when it starts (else logged as skipped for memory):
#   1. (NO-SWAP >= 28) the 24 GB configs, P2: clean_clone_test.sh run with CONFIGS="chat-24 lossy-24" (2 IFEval keys each vs
#      verify/*_w24p2.json, the module and file closure), a text prompt on chat-24, chat.py and serve.py at --mem 24 (P2 on
#      short prompts); then the P2 SMOKE: chat-24 on all 12 reference prompts at cap 256 vs verify/chat-24_w24p2.json (P2's
#      gated arm, BigMoE W24T w4p1_100; its own repeat matched 12 / 12); then, information only, serve.py --mem 24 with
#      CONV_CACHE=1 (not the shipped default) over four turns: two short ones (P2) and a long first turn + its follow-up.
#   2. (NO-SWAP >= 36) the 32 GB configs: exact-32 (with the MTPLX_QSA_PREFILL=0 pin; must match token for token),
#      lossy-32, chat-32.
#   3. (NO-SWAP >= 52) the 45 GB configs: exact-45 (pin), lossy-45, chat-45.
# Gates before stage 1: no engine / chat / server, no other big or quiet lock, no PAUSE / HOLD, Kamen idle >= 600 s, NO-SWAP
# >= 28, two clear polls, a share lock with MEM (re-checked 20 s after the grant); swap +2 GB stops it. From FROM, last
# start LAST. Writes <W>/report.txt (clean_clone_test.sh's) and <W>/rel120_check.log; <W>/rel120_check.done at the end.
# Usage: rel120_check.sh "<FROM YYYY-MM-DD HH:MM:SS>" "<LAST ...>"
W=${W:-/Users/kamen/Leap/release/cleanclone/v120b}; CL=$W/clone; PKG=${PKG:-/Users/kamen/Leap/release/out/pkg_120b}; LOG=$W/rel120_check.log   # pkg_120b: the Host / Origin fix
CCT="$CL/tools/clean_clone_test.sh"; WH=/Users/kamen/Leap/release/cleanclone/wheels; HFH=$HOME/.cache/huggingface
FROM=$(date -j -f "%Y-%m-%d %H:%M:%S" "$1" +%s); LAST=$(date -j -f "%Y-%m-%d %H:%M:%S" "$2" +%s)
noswap() { vm_stat | awk '/page size of/ {ps=$8} /^Pages (free|speculative|purgeable):/ || /^File-backed pages:/ {v=$NF; sub("[.]","",v); s+=v} END {printf "%d", s*ps/1e9}'; }
swap() { sysctl -n vm.swapusage | sed -E 's/.*used = ([0-9.]+)M.*/\1/' | cut -d. -f1; }
idle() { ioreg -c IOHIDSystem | awk '/HIDIdleTime/ {print int($NF/1000000000); exit}'; }
procs() { ps -axo pid=,args= | awk '/[Pp]ython/ && /r57_screen_|run_map[.]py|fit_onecall[.]py|[ \/](chat|serve|run)[.]py( |$)/ && !/ awk / {printf "%s ", $1}'; }
busy() { peer ls 2>/dev/null | grep "^- " | grep -oE "· (shared run \(others may work\)|benchmarking|waiting for a quiet window)[^·]*" | grep -cE "^· (benchmarking|waiting for a quiet window)|MEM ([6-9]|[1-9][0-9]) GB"; }
held() { [ -e ~/.local/share/quiet/PAUSE ] || [ -e ~/.local/share/quiet/HOLD ]; }
say() { echo "$(date '+%m-%d %H:%M:%S') $*" | tee -a $LOG; }
GUARD=""; LOCKED=0; SP=""
cleanup() { [ -n "$GUARD" ] && kill $GUARD 2>/dev/null; [ -n "$SP" ] && kill -0 $SP 2>/dev/null && kill -TERM $SP
            [ $LOCKED = 1 ] && peer unlock >/dev/null 2>&1; touch $W/rel120_check.done; say "exit (NO-SWAP $(noswap) GB, swap $(swap) MB)"; }
trap cleanup EXIT; trap 'say "stopped by a signal"; exit 1' INT TERM
[ "$(git -C $CL rev-parse --short HEAD)" = "$(git -C $PKG rev-parse --short HEAD)" ] || { say "STOP: the clone is not at the candidate's HEAD"; exit 2; }
say "armed (pid $$): 1.2.0 candidate $(git -C $CL rev-parse --short HEAD) GPU checks; from $1, last start $2"
NOTE=""; CLR=0
while :; do
  now=$(date +%s); [ $now -ge $LAST ] && { say "SKIP: no free window by $2 (last wait: $NOTE)"; exit 0; }
  why=""
  [ $now -lt $FROM ] && why="before $1"
> (a quote from the project owner, kept private)
  [ -z "$why" ] && [ -n "$(procs)" ] && why="an engine runs"
  [ -z "$why" ] && [ "$(busy)" -gt 0 ] && why="another lock"
  [ -z "$why" ] && held && why="PAUSE / HOLD"
  [ -z "$why" ] && [ "$(noswap)" -lt 28 ] && why="NO-SWAP $(noswap) < 28"
  if [ -z "$why" ]; then CLR=$((CLR + 1)); else CLR=0; fi
  [ -z "$why" ] && [ $CLR -lt 2 ] && why="clear once"
  if [ -z "$why" ]; then
    if peer lock share 60m --cpu 400 --gpu 100 "MEM 45 GB: Leap 1.2.0 candidate checks (P2 smoke at 24 GB; 32 / 45 GB configs if room), ~30 min" > /dev/null 2>&1; then
      LOCKED=1; sleep 20
      if [ "$(busy)" = 0 ] && [ -z "$(procs)" ] && [ "$(noswap)" -ge 28 ]; then say "lock granted; clear after 20 s (NO-SWAP $(noswap) GB)"; break; fi
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
# ---- stage 1: the 24 GB configs (P2) ----
say "stage 1 (NO-SWAP $(noswap) GB): clean_clone_test.sh run, CONFIGS=chat-24 lossy-24, text prompt on chat-24, chat.py + serve.py at 24"
CONFIGS="chat-24 lossy-24" TEXT_CONFIG=chat-24 CS_MEM=24 WHEELS=$WH bash "$CCT" "$PKG" "$W" run > $W/stage1.out 2>&1
say "  clean_clone_test run rc $?: $(tail -1 $W/report.txt)"
KEYS=$("$CL/.venv/bin/python" -c 'import json, sys; print(",".join(json.load(open(sys.argv[1]))))' "$CL/verify/chat-24_w24p2.json")
say "  P2 SMOKE: chat-24 on the 12 reference prompts, cap 256 ($KEYS)"
HOME="$W/home" HF_HOME="$HFH" "$CL/.venv/bin/python" "$CL/run.py" --config chat --mem 24 --keys "$KEYS" --max-tokens 256 \
  --out "$W/out/p2_12" > "$W/out/p2_12.txt" 2>&1; rc=$?
grep -E "^whole decode|^first token|^lifetime" "$W/out/p2_12.txt" | head -4 | sed 's/^/    /' | tee -a $LOG
tok=$(ls -t "$W/out/p2_12"/cli_*_tokens.json 2>/dev/null | head -1)
if [ $rc = 0 ] && [ -n "$tok" ]; then
  "$CL/.venv/bin/python" "$CL/tools/check_tokens.py" "$tok" "$CL/verify/chat-24_w24p2.json" --history > "$W/out/p2_12_check.txt" 2>&1; cr=$?
  say "  P2 SMOKE tokens vs the gated arm: $(tail -1 "$W/out/p2_12_check.txt") (rc $cr)"
else
  say "  P2 SMOKE: run.py rc $rc (see out/p2_12.txt)"
fi
say "  information only: serve.py --mem 24 with CONV_CACHE=1 (unshipped), four turns"
PORT=8095; CONV_CACHE=1 CONV_CACHE_LOG=$W/out/p2cache_convcache.log HOME="$W/home" HF_HOME="$HFH" \
  "$CL/.venv/bin/python" "$CL/serve.py" --mem 24 --port $PORT > "$W/out/p2cache_serve.txt" 2>&1 &
SP=$!; t0=$(date +%s); ok=1
until [ "$(curl -s -m 2 http://127.0.0.1:$PORT/health 2>/dev/null | grep -o ready)" = ready ]; do
  kill -0 $SP 2>/dev/null && [ $(( $(date +%s) - t0 )) -lt 300 ] || { ok=0; break; }; sleep 2
done
if [ $ok = 1 ]; then
  "$CL/.venv/bin/python" - $PORT "$W/out/p2cache_turns.jsonl" <<'PY' 2>&1 | tee -a $LOG
import json, sys, urllib.request
port, out = sys.argv[1], sys.argv[2]
def ask(msgs):
    body = json.dumps({"model": "exactstream", "messages": msgs, "max_tokens": 64}).encode()
    r = urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", body,
                                                      {"content-type": "application/json"}), timeout=600)
    d = json.load(r); return r.status, d["choices"][0]["message"]["content"], d.get("usage", {})
long_q = ("Here are notes from a meeting. " + " ".join(f"Item {i}: the team reviewed task {i}, agreed on an owner and set a date."
          for i in range(1, 31)) + " Summarise the notes in two sentences.")
turns, res = [], []
for q in ["Name three primary colours.", "Which of them is the warmest?"]:
    turns.append({"role": "user", "content": q}); s, a, u = ask(turns); turns.append({"role": "assistant", "content": a}); res.append((s, u))
turns2 = [{"role": "user", "content": long_q}]
s, a, u = ask(turns2); res.append((s, u)); turns2 += [{"role": "assistant", "content": a}, {"role": "user", "content": "Which item came first?"}]
s, a, u = ask(turns2); res.append((s, u))
open(out, "w").write("\n".join(json.dumps({"status": s, "usage": u}) for s, u in res) + "\n")
print("    four turns: HTTP", [s for s, _ in res], "prompt tokens", [u.get("prompt_tokens") for _, u in res])
PY
  grep -E "^[0-9:]+ done:" "$W/out/p2cache_serve.txt" | sed 's/^/    /' | cut -c1-260 | tee -a $LOG
  say "    errors in the server log: $(grep -c -E 'Traceback|Error' "$W/out/p2cache_serve.txt"); convcache lines: $(grep -c . $W/out/p2cache_convcache.log 2>/dev/null)"
else
  say "    serve.py did not come up within 300 s (see out/p2cache_serve.txt)"
fi
kill -TERM $SP 2>/dev/null; for i in $(seq 1 60); do kill -0 $SP 2>/dev/null || break; sleep 1; done; kill -0 $SP 2>/dev/null && kill -KILL $SP; SP=""
lsof -nP -iTCP:$PORT -sTCP:LISTEN >/dev/null 2>&1 && say "    WARNING: port $PORT still in use" || say "    serve.py stopped, port $PORT free"
# ---- stages 2 and 3: the 32 and 45 GB configs, if there is room ----
for st in "2 36 chat-32 exact-32 lossy-32 chat-32" "3 52 chat-45 exact-45 lossy-45 chat-45"; do
  set -- $st; n=$1; need=$2; tc=$3; shift 3
  [ -n "$(procs)" ] && { say "stage $n skipped: an engine runs"; continue; }
  [ "$(noswap)" -ge $need ] || { say "stage $n skipped for memory: NO-SWAP $(noswap) < $need"; continue; }
  say "stage $n (NO-SWAP $(noswap) GB): clean_clone_test.sh run, CONFIGS=$*"
  CONFIGS="$*" TEXT_CONFIG=$tc CHAT_SERVE=0 WHEELS=$WH bash "$CCT" "$PKG" "$W" run > $W/stage$n.out 2>&1
  say "  rc $?: $(tail -1 $W/report.txt)"
done
say "DONE: report $W/report.txt"
