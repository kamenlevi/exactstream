#!/bin/bash
# tools/clean_clone_test.sh <package repo or url> <work dir> prep|run [keys] [cap]
# The package's clean-clone test on one Mac (RELEASE section 4).
#   prep (CPU only): git clone into <work dir>/clone; install.sh --offline (venv from the local uv cache, extension build),
#        links the store and the drafter weights given by STORE= and WEIGHTS= (defaults: the research machine's paths);
#        then `run.py --dry-run` for every config.
#   run  (GPU, ~45 GB, one process at a time): every config on the given benchmark keys (default: 2 IFEval prompts, cap 256)
#        plus one free-text prompt, each with HOME set to an EMPTY directory, so any path left pointing outside the clone
#        fails instead of reading the research trees; the Hugging Face cache (HF_HOME) is the one outside path besides the
#        store and the weights. Then: tokens vs the shipped references (verify/), every module the engine imported
#        must lie in the clone or its venv, and every file Python opened in the clone / venv / store / weights / HF cache.
# Writes <work dir>/report.txt (last line: CLEAN-CLONE PASS / FAIL).
set -uo pipefail
PKG=${1:?package repo}; W=${2:?work dir}; PHASE=${3:?prep|run}
KEYS_ARG=${4:-}; KEYS=${4:-ifeval/1220,ifeval/3280}; CAP=${5:-256}
# Without [keys], each config runs the first two prompts of its own shipped reference run, cap 256: the 45 / 32 GB
# references are 12 IFEval prompts (ifeval/1220, ifeval/3280 first; since 1.2.0 the 24 GB ones too, P2's run). The lossy
# and chat configs' tokens depend on what the process decoded before (the relaxed acceptance takes the drafter's proposals),
# so their check needs the reference's earlier prompts run to the same length: hence the same cap (check_tokens.py --history
# reports a prompt whose history differs as information only). The exact configs do not depend on history.
STORE=${STORE:-$HOME/BigMoE/models/bare}; WEIGHTS=${WEIGHTS:-}; HFH=${HF_HOME_REAL:-$HOME/.cache/huggingface}
WHEELS=${WHEELS:-}                                  # a local wheel dir for install.sh --offline --find-links (mtplx)
export HF_HUB_OFFLINE=1
CONFIGS=${CONFIGS:-exact-45 exact-32 lossy-45 lossy-32 chat-45 chat-32 lossy-24 chat-24}
mkdir -p "$W"; W=$(cd "$W" && pwd -P); R="$W/report.txt"; CL="$W/clone"
say() { echo "$(date '+%m-%d %H:%M:%S') $*" | tee -a "$R"; }
keys_for() {    # a config's keys: the given ones, else the first two of its shipped reference run, else the default
  local ref k=""
  [ -n "$KEYS_ARG" ] && { echo "$KEYS_ARG"; return; }
  ref=$(ls "$CL/verify/$1"*.json 2>/dev/null | head -1)
  [ -n "$ref" ] && k=$("$CL/.venv/bin/python" -c 'import json, sys; print(",".join(list(json.load(open(sys.argv[1])))[:2]))' "$ref" 2>/dev/null)
  echo "${k:-$KEYS}"
}
if [ "$PHASE" = prep ]; then
  rm -rf "$CL" "$W/home" "$W/out"; : > "$R"
  git clone -q --single-branch "$PKG" "$CL" || { say "clone failed"; exit 2; }
  say "cloned $(git -C "$CL" log -1 --format='%h %s' | cut -c1-100)"
  if [ -z "$WEIGHTS" ]; then                      # the research machine: the two weight files live in two trees
    WEIGHTS="$W/weights_src"; mkdir -p "$WEIGHTS"
    ln -sfn "$HOME/Leap/data/l120_ft.safetensors" "$WEIGHTS/l120_ft.safetensors"
    ln -sfn "$HOME/TrainLab/ckpt/d22_b5.safetensors" "$WEIGHTS/d22_b5.safetensors"
  fi
  t0=$(date +%s)
  bash "$CL/install.sh" --offline ${WHEELS:+--find-links "$WHEELS"} --store "$STORE" --weights "$WEIGHTS" --jobs 4 2>&1 | tee -a "$R" | tail -8
  rc=${PIPESTATUS[0]}; say "install.sh rc $rc in $(( $(date +%s) - t0 )) s"; [ $rc = 0 ] || exit 3
  left=$(grep -rlF "@ROOT@" "$CL" --include='*.py' --include='*.sh' 2>/dev/null | grep -v '/third_party/nanobind/' | grep -v '/install.sh$' | grep -v '/tools/' | head -3)
  [ -z "$left" ] || { say "@ROOT@ left in: $left"; exit 3; }
  for c in $CONFIGS; do
    "$CL/.venv/bin/python" "$CL/run.py" --config ${c%-*} --mem ${c#*-} --keys "$(keys_for $c)" --dry-run > "$W/dry_$c.txt" 2>&1 \
      && say "dry run $c OK: $(head -1 "$W/dry_$c.txt" | cut -c1-110)" || { say "dry run $c FAILED"; cat "$W/dry_$c.txt"; exit 3; }
  done
  say "PREP DONE"; exit 0
fi
[ "$PHASE" = run ] || { echo "phase: prep|run"; exit 2; }
[ -x "$CL/.venv/bin/python" ] || { say "no prepared clone (run prep first)"; exit 2; }
mkdir -p "$W/home" "$W/out"; [ -z "$(ls -A "$W/home")" ] || { rm -rf "$W/home"; mkdir -p "$W/home"; }
fails=0
for c in $CONFIGS; do
  K=$(keys_for $c)
  say "run $c: keys $K, cap $CAP"
  HOME="$W/home" HF_HOME="$HFH" "$CL/.venv/bin/python" "$CL/run.py" --config ${c%-*} --mem ${c#*-} --keys "$K" \
    --max-tokens "$CAP" --trace --out "$W/out/$c" > "$W/out/$c.txt" 2>&1
  rc=$?
  grep -E "^whole decode|^=== " "$W/out/$c.txt" | sed 's/^/    /' | tee -a "$R"
  [ $rc = 0 ] || { say "  $c rc $rc"; tail -15 "$W/out/$c.txt" | sed 's/^/    | /' | tee -a "$R"; fails=$((fails + 1)); continue; }
  ref=$(ls "$CL/verify/${c}"*.json 2>/dev/null | head -1)
  tok=$(ls -t "$W/out/$c"/cli_*_tokens.json 2>/dev/null | head -1)
  if [ -n "$ref" ] && [ -n "$tok" ]; then
    H=""; [ "${c%-*}" = exact ] || H="--history"
    "$CL/.venv/bin/python" "$CL/tools/check_tokens.py" "$tok" "$ref" $H > "$W/out/${c}_check.txt" 2>&1; cr=$?
    case "$ref" in                                # a reference named *.info.json is compared and reported, never failed on
      *.info.json) say "  tokens vs $(basename "$ref") (information only: one reference run of a config whose run-to-run identity is not shown): $(tail -1 "$W/out/${c}_check.txt") (rc $cr)" ;;
      *) say "  tokens vs $(basename "$ref"): $(tail -1 "$W/out/${c}_check.txt") (rc $cr)"
         [ $cr = 0 ] || fails=$((fails + 1)) ;;
    esac
  else
    say "  tokens: no reference shipped for $c (information only)"
  fi
  man=$(ls -t "$W/out/$c"/cli_*_modules.json 2>/dev/null | head -1)
  "$CL/.venv/bin/python" - "$man" "$CL" "$STORE" "$HFH" "$W" > "$W/out/${c}_closure.txt" 2>&1 <<'EOF'
import json, os, sys
man, cl, store, hfh, w = sys.argv[1:6]
d = json.load(open(man))
real = lambda p: os.path.realpath(p)
ok_mod = (real(cl) + "/", )
py_std = os.path.realpath(os.path.dirname(os.__file__))            # the base interpreter's stdlib
mods_out = [m for m in d["modules"] if not real(m[1]).startswith(ok_mod) and not real(m[1]).startswith(py_std + "/")]
ok_open = (real(cl) + "/", real(store) + "/", real(hfh) + "/", real(w) + "/", py_std + "/", "/dev/", "/private/var/", "/var/",
           "/System/", "/usr/", "/private/tmp/", "/tmp/", "/Library/", "/opt/homebrew/")
wd = os.path.join(cl, "weights")                                   # the drafter files install.sh linked (--weights): allowed
ok_w = {real(os.path.join(wd, f)) for f in os.listdir(wd)} if os.path.isdir(wd) else set()   # where the links resolve
opens_out = sorted({p for p in d["opened"] if not real(p).startswith(ok_open) and real(p) not in ok_w})
print(f"modules {len(d['modules'])}, outside the clone + stdlib: {len(mods_out)}")
for m in mods_out[:30]:
    print("  MODULE", m[0], m[1])
print(f"files opened {len(d['opened'])}, outside the allowed roots: {len(opens_out)}")
for p in opens_out[:30]:
    print("  OPEN", p)
sys.exit(1 if mods_out or opens_out else 0)
EOF
  cr=$?; say "  closure: $(head -1 "$W/out/${c}_closure.txt"); $(sed -n '/^files opened/p' "$W/out/${c}_closure.txt") (rc $cr)"
  [ $cr = 0 ] || { grep -E "MODULE|OPEN" "$W/out/${c}_closure.txt" | head -8 | sed 's/^/    /' | tee -a "$R"; fails=$((fails + 1)); }
done
if [ -n "${TEXT_PROMPT:-Explain in three sentences how a refrigerator keeps food cold.}" ]; then
  c=${TEXT_CONFIG:-chat-45}
  say "run $c with a text prompt"
  HOME="$W/home" HF_HOME="$HFH" "$CL/.venv/bin/python" "$CL/run.py" --config ${c%-*} --mem ${c#*-} --max-tokens "$CAP" \
    --prompt "${TEXT_PROMPT:-Explain in three sentences how a refrigerator keeps food cold.}" --out "$W/out/text" > "$W/out/text.txt" 2>&1
  rc=$?; sed -n '/^=== /,$p' "$W/out/text.txt" | head -12 | sed 's/^/    /' | tee -a "$R"
  [ $rc = 0 ] || { say "  text prompt rc $rc"; fails=$((fails + 1)); }
fi
if [ "${CHAT_SERVE:-1}" = 1 ]; then              # chat.py and serve.py on the real engine, one short run each (CS_MEM GB)
  m=${CS_MEM:-24}
  say "chat.py --mem $m: one message"
  printf '%s\n/exit\n' "Explain in two sentences what a mixture-of-experts language model is." \
    | HOME="$W/home" HF_HOME="$HFH" "$CL/.venv/bin/python" "$CL/chat.py" --mem $m --max-tokens 128 > "$W/out/chat.txt" 2>&1
  rc=$?; s=$(grep -a -o -E "\[[0-9]+ tokens \| whole decode [0-9.]+ tok/s[^]]*\]" "$W/out/chat.txt" | tail -1)
  say "  chat.py rc $rc: ${s:-no answer line (see out/chat.txt)}"
  { [ $rc = 0 ] && [ -n "$s" ]; } || fails=$((fails + 1))
  say "serve.py --mem $m: one streamed OpenAI and one streamed Anthropic request, then SIGTERM"
  PORT=${CS_PORT:-8093}
  HOME="$W/home" HF_HOME="$HFH" "$CL/.venv/bin/python" "$CL/serve.py" --mem $m --port $PORT > "$W/out/serve.txt" 2>&1 &
  SPID=$!; t0=$(date +%s); ok=1
  until [ "$(curl -s -m 2 http://127.0.0.1:$PORT/health 2>/dev/null | grep -o ready)" = ready ]; do
    kill -0 $SPID 2>/dev/null && [ $(( $(date +%s) - t0 )) -lt 300 ] || { ok=0; break; }
    sleep 2
  done
  if [ $ok = 1 ]; then
    curl -sN -m 120 http://127.0.0.1:$PORT/v1/chat/completions -H 'content-type: application/json' \
      -d '{"model":"exactstream","messages":[{"role":"user","content":"Say hello in five words."}],"stream":true,"max_tokens":64}' > "$W/out/serve_openai.sse"
    curl -sN -m 120 http://127.0.0.1:$PORT/v1/messages -H 'content-type: application/json' -H 'anthropic-version: 2023-06-01' \
      -d '{"model":"exactstream","max_tokens":64,"stream":true,"messages":[{"role":"user","content":"Say hello in five words."}]}' > "$W/out/serve_anthropic.sse"
    o=$(grep -c -E '^data: \[DONE\]' "$W/out/serve_openai.sse"); f=$(grep -c '"finish_reason": *"[a-z]' "$W/out/serve_openai.sse")
    a=$(grep -c -E '^event: message_stop' "$W/out/serve_anthropic.sse"); d=$(grep -c -E '^event: content_block_delta' "$W/out/serve_anthropic.sse")
    say "  OpenAI stream: [DONE] $o, finish_reason $f; Anthropic stream: message_stop $a, text deltas $d"
    { [ "$o" = 1 ] && [ "$f" -ge 1 ] && [ "$a" = 1 ] && [ "$d" -ge 1 ]; } || fails=$((fails + 1))
  else
    say "  serve.py did not come up within 300 s (see out/serve.txt)"; fails=$((fails + 1))
  fi
  kill -TERM $SPID 2>/dev/null
  for i in $(seq 1 30); do kill -0 $SPID 2>/dev/null || break; sleep 1; done
  if kill -0 $SPID 2>/dev/null; then say "  serve.py ignored SIGTERM for 30 s"; kill -KILL $SPID; fails=$((fails + 1))
  else say "  serve.py stopped on SIGTERM: $(grep -a -c 'stopped; log:' "$W/out/serve.txt") 'stopped' line(s)"; fi
fi
[ -z "$(ls -A "$W/home" 2>/dev/null)" ] || say "note: the engine wrote into the empty HOME: $(cd "$W/home" && find . -maxdepth 3 | head -5 | tr '\n' ' ')"
[ $fails = 0 ] && say "CLEAN-CLONE PASS" || say "CLEAN-CLONE FAIL ($fails)"
exit $fails
