#!/bin/bash
# rel121_check.sh (Leap 11-13 15:5x): the release checks of package 1.2.1 (candidate release/out/pkg_121 at b830caf, clean
# clone release/cleanclone/v121, prepared 15:42), at night under ONE quiet ticket of their own, queued only after onebuild's
# runner has exited (its EXIT trap releases its ticket first). The orchestrator 15:2x (the list), 15:3x (the two token
# checks: if either fails there is no release, and it is told), 15:4x (the start rule: after onebuild; nothing may END after
# Sat 10:00). Parts, the 45 GB ones first, each with its own NO-SWAP gate (its lifetime peak + 4, the quiet-lock margin;
# waits up to 2 min for it) and an end-time estimate that must fit before CUTOFF; a part that does not fit is skipped:
#  A  stage 3 (45 GB): clean_clone_test.sh run, exact-45 lossy-45 chat-45 (2 IFEval keys each vs verify/, the module and
#     file closure), a text prompt on chat-45, chat.py and serve.py at --mem 45 (P2 on their short prompts)   gate 48, est 20 min
#  B  P2 SMOKE 45: chat-45 on the 12 reference prompts at cap 256 vs verify/chat-45_p2.json (--history); first token median
#     (the A/B: 0.47 s), lifetime peak                                                                         48, 10
#  C  TOKEN CHECK 2: the first 5 keys of the 45 GB P2 gate's P2 arm (card_p2_45_fwd, process 1 = BigMoE
#     data/r57/es_cdp2_45_131341_fwd1), in its order, cap 2048 (the gate's), warm-up on (the gate's main processes): the
#     packaged chat-45's tokens must equal the gate's rows, 5 / 5. EvalSuite's data/items/dec_gpqad.jsonl (Nov 8, the file
#     the gate read) is copied into the clone's harness for this part only                                   48, 12
#  D  TOKEN CHECK 1: a 1,972-token prompt (P2 does not engage) on 1.2.1 chat-45 and on 1.2.0 chat-45 (the v120b clone),
#     cap 256, R57_LOGHASH=1: tokens bit-identical (logits hashes reported)                                   50, 12
#  E  serve.py --mem 45, cache on (the shipped default) + CONV_CACHE_CHECK=1: the 1,972-token turn, a short (P2) request,
#     the turn's follow-up, another short one, a second follow-up; every cache hit's cold-prefill check must PASS, and
#     the short requests must report no cache use (the 1.2.1 serve.py fix)                                    50, 12
#  F  long-prompt peaks, chat-45 at cap 256: 289 / 965 / 3,999 / 8,953 tokens (D gives 1,972)               50/50/51/52
#  G  stage 2 (32 GB): exact-32 lossy-32 chat-32, text on chat-32, chat.py / serve.py at 32                  40, 20
#  H  stage 1 (24 GB): lossy-24 chat-24, text on chat-24, chat.py / serve.py at 24                           28, 15
#  I  long-prompt peak at 28,966 tokens (chat-45, cap 256)                                                     53, 15
# Kamen-idle, lid, HOLD, PAUSE (waits up to 3 min) and HEALTH ACT are checked before every part; swap +2 GB stops all.
# Usage: rel121_check.sh <AFTER pid> "<FROM YYYY-MM-DD HH:MM:SS>" "<LAST queue ...>" "<CUTOFF ...>"
#   AFTER: onebuild's runner pid (wait until it has exited); FROM: not before (the GO file counts as reached).
V=/Users/kamen/Leap/release/cleanclone/v121; CL=$V/clone; PKG=/Users/kamen/Leap/release/out/pkg_121; CCT=$CL/tools/clean_clone_test.sh
V0=/Users/kamen/Leap/release/cleanclone/v120b/clone
WH=/Users/kamen/Leap/release/cleanclone/wheels; HFH=$HOME/.cache/huggingface; STORE=$HOME/BigMoE/models/bare
SCR=/private/tmp/claude-501/-Users-kamen-OjoX/f2971937-6c95-4e86-a049-05beff6a8e32/scratchpad/agent
P=$SCR/r121; OBW=$SCR/onebuild45; GOF=$SCR/GO_night
GATE1=/Users/kamen/BigMoE/data/r57/es_cdp2_45_131341_fwd1; GPQ=/Users/kamen/EvalSuite/data/items/dec_gpqad.jsonl
LOG=$V/rel121_check.log; Q=$HOME/.local/share/quiet; H2=$V/home2; O=$V/out
AFTER=${1:?AFTER pid}
FROM=$(date -j -f "%Y-%m-%d %H:%M:%S" "${2:?FROM}" +%s); LAST=$(date -j -f "%Y-%m-%d %H:%M:%S" "${3:?LAST}" +%s)
CUT=$(date -j -f "%Y-%m-%d %H:%M:%S" "${4:?CUTOFF}" +%s)
noswap() { vm_stat | awk '/page size of/ {ps=$8} /^Pages (free|speculative|purgeable):/ || /^File-backed pages:/ {v=$NF; sub("[.]","",v); s+=v} END {printf "%d", s*ps/1e9}'; }
swap() { sysctl -n vm.swapusage | sed -E 's/.*used = ([0-9.]+)M.*/\1/' | cut -d. -f1; }
idle() { ioreg -c IOHIDSystem | awk '/HIDIdleTime/ {print int($NF/1000000000); exit}'; }
procs() { ps -axo pid=,args= | awk '/[Pp]ython/ && /r57_screen_|run_map[.]py|fit_onecall[.]py|[ \/](chat|serve|run)[.]py( |$)/ && !/ awk / {printf "%s ", $1}'; }
lid_closed() { ioreg -r -k AppleClamshellState -d 1 | grep -q '"AppleClamshellState" = Yes'; }
say() { echo "$(date '+%m-%d %H:%M:%S') $*" | tee -a $LOG; }
res() { echo "RESULT $*" >> $V/rel121_results.txt; say "RESULT $*"; }
PY=/opt/homebrew/opt/python@3.12/bin/python3.12; ST="$PY /Users/kamen/Leap/scripts/night/rel121_stats.py"
CLIENT=/Users/kamen/Leap/scripts/night/rel121_serve_client.py
LOCKED=0; CP=""; SP=""; GUARD=""; COPIED=0; RC=""
killtree() { local c; for c in $(pgrep -P $1 2>/dev/null); do killtree $c; done; kill -TERM $1 2>/dev/null; }
cleanup() { [ -n "$GUARD" ] && kill $GUARD 2>/dev/null
            for p_ in $CP $SP; do kill -0 $p_ 2>/dev/null && killtree $p_; done
            [ $COPIED = 1 ] && rm -f $CL/harness/data/items/dec_gpqad.jsonl && say "removed the copied dec_gpqad.jsonl"
            [ $LOCKED = 1 ] && { peer unlock >/dev/null 2>&1; say "ticket released"; }
            say "exit (NO-SWAP $(noswap) GB, swap $(swap) MB)"; }
trap cleanup EXIT; trap 'say "stopped by a signal"; exit 1' INT TERM
for f in $CLIENT /Users/kamen/Leap/scripts/night/rel121_stats.py $CCT $V0/run.py $P/long_2k.txt $P/len_300.txt $P/len_1k.txt $P/len_4k.txt $OBW/prompt_10k.txt $OBW/prompt_30k.txt \
         ${GATE1}_tokens.json ${GATE1}_runs.json $GPQ $CL/verify/chat-45_p2.json; do
  [ -f $f ] || { say "STOP: no $f"; exit 2; }
done
[ "$(git -C $CL rev-parse --short HEAD)" = "$(git -C $PKG rev-parse --short HEAD)" ] || { say "STOP: the clone is not at the candidate's HEAD"; exit 2; }
mkdir -p $H2 $O
say "armed (pid $$): 1.2.1 release checks of $(git -C $CL rev-parse --short HEAD) under one quiet ticket after pid $AFTER exits; from $2 (or $GOF), queue until $3, every part ends by $4"

# ---- 1. wait: onebuild's runner gone, FROM reached (or GO) ----
NOTE=""
while :; do
  now=$(date +%s)
  [ $now -ge $LAST ] && { say "SKIP: past $3 before queueing"; exit 0; }
  if kill -0 $AFTER 2>/dev/null; then w="waiting for pid $AFTER (onebuild's runner) to exit"
  elif [ $now -lt $FROM ] && [ ! -e $GOF ]; then w="before $2"
  else break; fi
  [ "$w" != "$NOTE" ] && say "wait: $w"; NOTE=$w
  sleep 30
done

# ---- 2. one quiet ticket, attached ----
TO=$(( LAST - $(date +%s) )); [ $TO -lt 120 ] && { say "SKIP: under 2 min to $3"; exit 0; }
say "queue the quiet ticket (PEER_QUEUE_TIMEOUT $TO s; NO-SWAP $(noswap) GB)"
if PEER_QUEUE_TIMEOUT=$TO peer lock quiet --queue 150m "Leap: 45 GB release checks of package 1.2.1 (clean-clone 24 / 32 / 45 GB, token checks, long-prompt peaks), MEM 53 GB, ~100 min" >> $LOG 2>&1; then
  LOCKED=1
else
  say "the ticket left the queue without a grant"; exit 0
fi
say "granted (NO-SWAP $(noswap) GB); settle 30 s"
sleep 30
S0=$(swap); ( while :; do [ $(( $(swap) - S0 )) -gt 2048 ] && { echo "$(date +%T) SWAP GUARD: $(swap) MB" >> $LOG; kill -TERM $$; exit; }; sleep 5; done ) &
GUARD=$!
: > $V/rel121_results.txt

# part <id> <gate GB> <est s> "<label>": 0 = start it now
part() {
  local w=0
  while [ -e $Q/PAUSE ] && [ $w -lt 180 ]; do sleep 15; w=$((w + 15)); done
  [ -e $Q/HOLD ] && { say "STOP: HOLD"; exit 0; }
  lid_closed && { say "STOP: lid closed"; exit 0; }
  grep -q "Verdict: ACT" $Q/HEALTH.md 2>/dev/null && { say "STOP: HEALTH ACT"; exit 0; }
> (a quote from the project owner, kept private)
  w=0; while [ -n "$(procs)" ] && [ $w -lt 60 ]; do sleep 5; w=$((w + 5)); done
  [ -n "$(procs)" ] && { res "$1 SKIPPED: an engine runs ($(procs))"; return 1; }
  [ $(( $(date +%s) + $3 )) -gt $CUT ] && { res "$1 SKIPPED: it could end after the cutoff (est $3 s)"; return 1; }
  w=0; while [ "$(noswap)" -lt $2 ] && [ $w -lt 120 ]; do sleep 15; w=$((w + 15)); done
  [ "$(noswap)" -lt $2 ] && { res "$1 SKIPPED: NO-SWAP $(noswap) < $2 (its peak + 4)"; return 1; }
  say "== part $1: $4 (NO-SWAP $(noswap) GB, swap $(swap) MB)"
  return 0
}
runbg() { "$@" & CP=$!; wait $CP; local rc=$?; CP=""; return $rc; }
eng() {   # eng <clone> <out tag> <args...>: run.py with HOME empty, in this shell (CP stays visible to cleanup); rc in RC
  local c=$1 t=$2; shift 2
  runbg env HOME=$H2 HF_HOME=$HFH $c/.venv/bin/python $c/run.py "$@" --out $O/$t > $O/$t.txt 2>&1; RC=$?
}
newest() { ls -t $O/$1/cli_*_$2.json 2>/dev/null | head -1; }

# ---- A: stage 3 ----
if part A 48 1200 "stage 3 (45 GB configs, text on chat-45, chat.py + serve.py at 45)"; then
  runbg env CONFIGS="exact-45 lossy-45 chat-45" TEXT_CONFIG=chat-45 CHAT_SERVE=1 CS_MEM=45 CS_PORT=8097 WHEELS=$WH bash $CCT $PKG $V run > $V/stage3.out 2>&1
  rc=$?; res "A stage 3 (45 GB) rc $rc: $(grep -E 'CLEAN-CLONE (PASS|FAIL)' $V/stage3.out | tail -1 | sed 's/^[0-9: -]*//')"
  grep -E "tokens vs|closure:|chat.py rc|OpenAI stream|serve.py stopped|lifetime|^ *whole decode" $V/stage3.out | sed 's/^/    /' | tee -a $LOG
fi

# ---- B: P2 smoke 45 ----
if part B 48 600 "P2 SMOKE: chat-45 on the 12 reference prompts, cap 256"; then
  K12=$($PY -c 'import json, sys; print(",".join(json.load(open(sys.argv[1]))))' $CL/verify/chat-45_p2.json)
  t0=$(date +%s); eng $CL p2_12_45 --config chat --mem 45 --keys "$K12" --max-tokens 256; rc=$RC; t1=$(date +%s)
  tok=$(newest p2_12_45 tokens); runs=$(newest p2_12_45 runs)
  if [ -n "$tok" ]; then
    $CL/.venv/bin/python $CL/tools/check_tokens.py "$tok" $CL/verify/chat-45_p2.json --history > $O/p2_12_45_check.txt 2>&1; cr=$?
    st=$($ST p2smoke "$runs" 2>&1)
    res "B P2 SMOKE 45 rc $rc, check rc $cr: $(tail -1 $O/p2_12_45_check.txt); $st; ran $(date -r $t0 +%T)-$(date -r $t1 +%T)"
  else
    res "B P2 SMOKE 45 rc $rc: no tokens file (see $O/p2_12_45.txt)"
  fi
fi

# ---- C: token check 2, the gate's first 5 P2 keys ----
if part C 48 720 "TOKEN CHECK 2: the gate's process 1, first 5 keys, cap 2048"; then
  K5=$($PY -c 'import json, sys; print(",".join([r["key"] for r in json.load(open(sys.argv[1]))][:5]))' ${GATE1}_runs.json)
  if [ -e $CL/harness/data/items/dec_gpqad.jsonl ]; then say "  dec_gpqad.jsonl already in the clone: STOP"; exit 2; fi
  cp $GPQ $CL/harness/data/items/dec_gpqad.jsonl && COPIED=1
  say "  keys $K5; dec_gpqad.jsonl md5 $(md5 -q $GPQ) copied in"
  t0=$(date +%s); eng $CL gate5 --config chat --mem 45 --keys "$K5" --max-tokens 2048; rc=$RC; t1=$(date +%s)
  rm -f $CL/harness/data/items/dec_gpqad.jsonl; COPIED=0
  tok=$(newest gate5 tokens)
  v=$($PY - "$tok" ${GATE1}_tokens.json "$K5" <<'EOF'
import json, sys
a = json.load(open(sys.argv[1])) if sys.argv[1] else {}
b = json.load(open(sys.argv[2])); keys = sys.argv[3].split(",")
eq = []
for k in keys:
    x = [v for arm in sorted(a.get(k, {})) for v in a[k][arm]]; y = [v for arm in sorted(b[k]) for v in b[k][arm]]
    eq.append(x == y and bool(x))
    if x != y:
        xs, ys = (x[0] if x else []), (y[0] if y else [])
        d = next((i for i, (p, q) in enumerate(zip(xs, ys)) if p != q), min(len(xs), len(ys)))
        print(f"  {k}: DIFFER (lengths {len(xs)} vs {len(ys)}, first difference at token {d})", file=sys.stderr)
print(f"{'PASS' if all(eq) else 'FAIL'}: tokens equal the gate's rows on {sum(eq)} / {len(keys)} keys")
EOF
)
  res "C TOKEN CHECK 2 rc $rc: $v; ran $(date -r $t0 +%T)-$(date -r $t1 +%T)"
fi

# ---- D: token check 1, a long prompt, 1.2.1 vs 1.2.0 ----
if part D 50 720 "TOKEN CHECK 1: 1,972 tokens, 1.2.1 chat-45 vs 1.2.0 chat-45"; then
  eng $CL long2k_121 --config chat --mem 45 --prompt-file $P/long_2k.txt --max-tokens 256 --env R57_LOGHASH=1; rc1=$RC
  eng $V0 long2k_120 --config chat --mem 45 --prompt-file $P/long_2k.txt --max-tokens 256 --env R57_LOGHASH=1; rc0=$RC
  v=$($PY - "$(newest long2k_121 tokens)" "$(newest long2k_120 tokens)" "$(newest long2k_121 runs)" "$(newest long2k_120 runs)" <<'EOF'
import json, sys
def toks(p):
    tk = json.load(open(p)); return [v for k in sorted(tk) for a in sorted(tk[k]) for v in tk[k][a]]
def lh(p):
    out = []
    def walk(o):
        if isinstance(o, dict):
            for k, v in o.items():
                (out.append(v) if k == "lh" else walk(v))
        elif isinstance(o, list):
            for v in o: walk(v)
    walk(json.load(open(p))); return out
try:
    a, b = toks(sys.argv[1]), toks(sys.argv[2]); la, lb = lh(sys.argv[3]), lh(sys.argv[4])
    ra = json.load(open(sys.argv[3])); P = [r.get("P") for r in ra]; p2 = [r.get("p2") for r in ra]
    print(f"{'PASS' if a == b and a else 'FAIL'}: tokens {'bit-identical' if a == b else 'DIFFER'} ({len(a[0]) if a else 0} tokens; prompt {P}, p2 {p2}); "
          f"logits hashes {'equal' if la == lb else 'DIFFER'} ({len(json.dumps(la))} chars)")
except Exception as e:
    print(f"FAIL: {type(e).__name__}: {e}")
EOF
)
  res "D TOKEN CHECK 1 rc $rc1 / $rc0: $v; peaks 1.2.1 $(grep -o 'lifetime peak [0-9.]* GB' $O/long2k_121.txt | tail -1), 1.2.0 $(grep -o 'lifetime peak [0-9.]* GB' $O/long2k_120.txt | tail -1)"
fi

# ---- E: serve.py at 45 GB, P2 + cache ----
if part E 50 720 "serve.py --mem 45, cache on + CONV_CACHE_CHECK=1, five requests"; then
  PORT=8098; rm -f $O/p2cache45_convcache.log
  env CONV_CACHE_CHECK=1 CONV_CACHE_LOG=$O/p2cache45_convcache.log HOME=$H2 HF_HOME=$HFH $CL/.venv/bin/python $CL/serve.py --mem 45 --port $PORT > $O/p2cache45_serve.txt 2>&1 &
  SP=$!; t0=$(date +%s); ok=1
  until [ "$(curl -s -m 2 http://127.0.0.1:$PORT/health 2>/dev/null | grep -o ready)" = ready ]; do
    kill -0 $SP 2>/dev/null && [ $(( $(date +%s) - t0 )) -lt 300 ] || { ok=0; break; }; sleep 2
  done
  if [ $ok = 1 ]; then
    runbg $PY $CLIENT $PORT $P/long_2k.txt $O/p2cache45_turns.jsonl > $O/p2cache45_client.txt 2>&1
    sed 's/^/    /' $O/p2cache45_client.txt | tee -a $LOG
    nh=$(grep -c "check PASS" $O/p2cache45_convcache.log 2>/dev/null); nf=$(grep -c "check FAIL" $O/p2cache45_convcache.log 2>/dev/null)
    ne=$(grep -c -E "Traceback|Error" $O/p2cache45_serve.txt)
    grep -E "done:" $O/p2cache45_serve.txt | cut -c1-260 | sed 's/^/    /' | tee -a $LOG
    sc=$(grep -E "done: [0-9]+ prompt tokens" $O/p2cache45_serve.txt | awk '{for (i = 1; i <= NF; i++) if ($i == "done:") n = $(i + 1); if (n + 0 <= 256) { s++; if ($0 ~ /; cache /) c++ } } END {printf "%d short, %d with cache statistics", s, c}')
    res "E serve.py 45 + cache: $(tail -1 $O/p2cache45_client.txt); cache checks PASS $nh, FAIL $nf; P2 requests: $sc (want 0); errors in the server log $ne"
  else
    res "E serve.py 45 + cache: did not come up within 300 s (see $O/p2cache45_serve.txt)"
  fi
  kill -TERM $SP 2>/dev/null; for i in $(seq 1 60); do kill -0 $SP 2>/dev/null || break; sleep 1; done; kill -0 $SP 2>/dev/null && kill -KILL $SP; SP=""
  lsof -nP -iTCP:$PORT -sTCP:LISTEN >/dev/null 2>&1 && say "  WARNING: port $PORT still in use" || say "  serve.py stopped, port $PORT free"
fi

# ---- F: long-prompt peaks ----
for job in "len_300 50 300" "len_1k 50 300" "len_4k 51 420" "prompt_10k 52 600"; do
  set -- $job; nm=$1
  f=$P/$nm.txt; [ $nm = prompt_10k ] && f=$OBW/prompt_10k.txt
  if part "F-$nm" $2 $3 "chat-45 long-prompt peak, $nm"; then
    t0=$(date +%s); eng $CL peak_$nm --config chat --mem 45 --prompt-file $f --max-tokens 256; rc=$RC; t1=$(date +%s)
    res "F peak $nm rc $rc: $($ST one "$(newest peak_$nm runs)" 2>&1); ran $(date -r $t0 +%T)-$(date -r $t1 +%T)"
  fi
done

# ---- G, H: stages 2 and 1 ----
if part G 40 1200 "stage 2 (32 GB configs, text on chat-32, chat.py + serve.py at 32)"; then
  runbg env CONFIGS="exact-32 lossy-32 chat-32" TEXT_CONFIG=chat-32 CHAT_SERVE=1 CS_MEM=32 CS_PORT=8097 WHEELS=$WH bash $CCT $PKG $V run > $V/stage2.out 2>&1
  rc=$?; res "G stage 2 (32 GB) rc $rc: $(grep -E 'CLEAN-CLONE (PASS|FAIL)' $V/stage2.out | tail -1 | sed 's/^[0-9: -]*//')"
  grep -E "tokens vs|closure:|chat.py rc|OpenAI stream|serve.py stopped|^ *whole decode" $V/stage2.out | sed 's/^/    /' | tee -a $LOG
fi
if part H 28 900 "stage 1 (24 GB configs, text on chat-24, chat.py + serve.py at 24)"; then
  runbg env CONFIGS="lossy-24 chat-24" TEXT_CONFIG=chat-24 CHAT_SERVE=1 CS_MEM=24 CS_PORT=8097 WHEELS=$WH bash $CCT $PKG $V run > $V/stage1.out 2>&1
  rc=$?; res "H stage 1 (24 GB) rc $rc: $(grep -E 'CLEAN-CLONE (PASS|FAIL)' $V/stage1.out | tail -1 | sed 's/^[0-9: -]*//')"
  grep -E "tokens vs|closure:|chat.py rc|OpenAI stream|serve.py stopped|^ *whole decode" $V/stage1.out | sed 's/^/    /' | tee -a $LOG
fi

# ---- I: the 30K peak ----
if part I 53 900 "chat-45 long-prompt peak, 28,966 tokens"; then
  t0=$(date +%s); eng $CL peak_prompt_30k --config chat --mem 45 --prompt-file $OBW/prompt_30k.txt --max-tokens 256; rc=$RC; t1=$(date +%s)
  res "I peak prompt_30k rc $rc: $($ST one "$(newest peak_prompt_30k runs)" 2>&1); ran $(date -r $t0 +%T)-$(date -r $t1 +%T)"
fi
say "SUMMARY"; sed 's/^/  /' $V/rel121_results.txt | tee -a $LOG
say "DONE"
exit 0
