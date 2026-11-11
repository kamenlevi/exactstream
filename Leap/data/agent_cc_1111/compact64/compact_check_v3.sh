#!/bin/bash
# Does Claude Code 2.1.321 compact before 64K when it is told a 64K window? (orchestrator 07:2x: option (b), tell Claude
# Code 64K at --mem 32 and 45; "first check on your stand-in that Claude Code really compacts before 64K with that
# setting; if it clamps to 100K or more, tell me before relying on it".)
# Stand-in engine (mock_tools.py; no GPU, < 1 GB). serve.py keeps its default --max-context 131072, so a clamped window
# would show as a conversation growing past 64K with no compact_boundary. Claude Code runs through the clone's
# tools/claude_local.sh with EXACTSTREAM_CONTEXT=$CTX (= CLAUDE_CODE_MAX_CONTEXT_TOKENS and CLAUDE_CODE_AUTO_COMPACT_WINDOW),
# in the night runs' sandbox (regex read-deny, writes only in the run folder / temp, network only to localhost), with an
# isolated config folder and the neutral HOME. Each turn: claude -p (--continue after the first) with about 6K tokens of
# new text; the mock answers with one Write call, then "Done". Per turn: compact_boundary events (trigger, pre / post
# tokens), the result's contextWindow / maxOutputTokens, and the server's largest prompt in the turn.
# Stops after the first auto compaction (plus one more turn, to see the session continue), once a prompt passes 100K
# tokens (only when CTX < 100000), or after MAXT turns. v2: WORDS per turn (env, default 4600 = ~6K tokens), MAXT (env, 24).
# v3: LAUNCHER (env) = the claude_local.sh to run (default: the clone's). Usage: compact_check_v3.sh [CTX (65536)] [TAG]
A=/private/tmp/claude-501/-Users-kamen-OjoX/f2971937-6c95-4e86-a049-05beff6a8e32/scratchpad/agent
CL=/Users/kamen/Leap/release/cleanclone/agent/clone; PORT=8132; CTX=${1:-65536}; TAG=${2:-w$CTX}; H=/Users/kamen
B=$A/compact64/v3_$TAG; rm -rf $B; mkdir -p $B/work $B/cctmp $B/home; LOG=$B/check.log
say() { echo "$(date '+%H:%M:%S') $*" | tee -a $LOG; }
SP=""
stopsrv() { [ -z "$SP" ] && return; kill -TERM $SP 2>/dev/null
            for i in $(seq 1 60); do kill -0 $SP 2>/dev/null || break; sleep 1; done
            kill -0 $SP 2>/dev/null && { say "serve.py did not stop in 60 s: SIGKILL"; kill -KILL $SP; }
            lsof -nP -iTCP:$PORT -sTCP:LISTEN >/dev/null 2>&1 && say "WARNING: port $PORT still in use"; SP=""; }
trap stopsrv EXIT
cp $A/agent_home_tpl/.zshrc $A/agent_home_tpl/.gitconfig $B/home/
rsync -a --exclude projects --exclude sessions --exclude backups --exclude shell-snapshots --exclude todos $A/cc_home/ $B/cc_home/
cat > $B/sandbox.sb <<SB
(version 1)
(allow default)
(deny network-outbound)
(allow network-outbound (remote ip "localhost:*"))
(allow network-outbound (remote unix-socket))
(deny file-write*)
(allow file-write* (subpath "$B") (subpath "/private/var/folders") (regex #"^/dev/"))
(deny file-read* (subpath "$H/.ssh") (subpath "$H/.aws") (subpath "$H/.gnupg") (regex #"^/Users/[^/]+/\.claude[^/]*(/|$)")
  (subpath "$H/.codex") (subpath "$H/.config/gh") (literal "$H/.netrc") (literal "$H/.git-credentials")
  (subpath "$H/Library/Keychains") (subpath "$H/Library/Mail") (subpath "$H/Library/Messages") (subpath "$H/Library/Cookies")
  (subpath "$H/Library/Safari") (subpath "$H/Library/Application Support/Google") (subpath "$H/Library/Application Support/Claude")
  (subpath "$H/Documents") (subpath "$H/Desktop") (subpath "$H/Downloads") (subpath "$H/Papers") (subpath "$H/.local/share/exactstream-sessions"))
SB
# 24 turn texts of about 6K tokens each (plain words; fixed seed)
/opt/homebrew/opt/python@3.12/bin/python3.12 -I - $B <<'EOF'
import random, sys, os
B = sys.argv[1]; r = random.Random(1111)
W = ("river stone light paper garden window morning silver market winter forest letter yellow bridge orange harbor "
     "candle meadow pocket marble thunder lantern velvet copper island shadow ribbon summer engine button violet cabin "
     "falcon tunnel planet mirror basket feather canyon pepper saddle anchor kettle quarry blossom compass lighthouse").split()
for t in range(1, int(os.environ.get("MAXT", "24")) + 1):
    words = [r.choice(W) for _ in range(int(os.environ.get("WORDS", "4600")))]
    lines = [" ".join(words[i:i + 20]) for i in range(0, len(words), 20)]
    txt = (f"Turn {t}: please keep these notes in mind; no action is needed beyond saving a small file.\n\n"
           + "\n".join(f"note {t}.{k}: {l}" for k, l in enumerate(lines)))
    open(f"{B}/turn{t:02d}.txt", "w").write(txt)
EOF
( cd $CL && MOCK_STORE=$CL/engine/models/bare MOCK_WORKDIR=$B/work exec .venv/bin/python serve.py --mem 32 --port $PORT --_screen $A/mock_tools.py ) > $B/serve.log 2>&1 &
SP=$!; t0=$(date +%s)
until curl -s -m 2 http://127.0.0.1:$PORT/health 2>/dev/null | grep -q ready; do
  kill -0 $SP 2>/dev/null && [ $(( $(date +%s) - t0 )) -lt 120 ] || { say "mock serve.py did not come up: $(tail -3 $B/serve.log)"; exit 2; }; sleep 1
done
say "mock serve.py ready in $(( $(date +%s) - t0 )) s (pid $SP); EXACTSTREAM_CONTEXT=$CTX; claude $(CLAUDE_CONFIG_DIR=$B/cc_version /Users/kamen/.local/bin/claude --version 2>&1 | head -1)"
SEEN=0; AFTER=0
for i in $(seq 1 ${MAXT:-24}); do
  n=$(printf "%02d" $i); CONT=""; [ $i -gt 1 ] && CONT="--continue"
  l0=$(grep -c "done:" $B/serve.log)
  ( cd $B/work && exec env HOME=$B/home CLAUDE_CONFIG_DIR=$B/cc_home EXACTSTREAM_PORT=$PORT EXACTSTREAM_CONTEXT=$CTX \
      CLAUDE_BIN=/Users/kamen/.local/bin/claude CLAUDE_CODE_TMPDIR=$B/cctmp \
      sandbox-exec -f $B/sandbox.sb ${LAUNCHER:-$CL/tools/claude_local.sh} -p $CONT "$(cat $B/turn$n.txt)" --output-format stream-json \
      --verbose --max-turns 3 --allowedTools Write ) < /dev/null > $B/t$n.jsonl 2> $B/t$n.err &
  CP=$!; c0=$(date +%s)
  while kill -0 $CP 2>/dev/null; do [ $(( $(date +%s) - c0 )) -gt 300 ] && { say "turn $i: claude over 300 s: stopping"; kill -TERM $CP; }; sleep 1; done
  wait $CP; rc=$?
  maxp=$(tail -n +$((l0 + 1)) <(grep "done:" $B/serve.log) | sed -E 's/.*done: ([0-9]+) prompt tokens.*/\1/' | sort -n | tail -1)
  line=$(/opt/homebrew/opt/python@3.12/bin/python3.12 -I - $B/t$n.jsonl <<'EOF'
import json, sys
cb, win, res = [], None, None
for l in open(sys.argv[1]):
    try: d = json.loads(l)
    except Exception: continue
    if d.get("type") == "system" and d.get("subtype") == "compact_boundary":
        m = d.get("compact_metadata", {}); cb.append(f"{m.get('trigger')} {m.get('pre_tokens')}->{m.get('post_tokens')}")
    if d.get("type") == "result":
        res = d.get("subtype")
        for k, v in (d.get("modelUsage") or {}).items():
            if v.get("contextWindow"): win = f"{k} window {v.get('contextWindow')} maxOut {v.get('maxOutputTokens')}"
print(f"result {res}; {win}; compactions [{'; '.join(cb)}]")
EOF
)
  say "turn $i: rc $rc; server's largest prompt ${maxp:-?} tokens; $line"
  echo "$line" | grep -q "compactions \[auto" && SEEN=1
  [ $SEEN = 1 ] && AFTER=$((AFTER + 1))
  [ $AFTER -ge 2 ] && { say "stop: an auto compaction seen, and one more turn after it"; break; }
  [ $CTX -lt 100000 ] && [ -n "$maxp" ] && [ "$maxp" -gt 100000 ] && { say "stop: a prompt passed 100K tokens with no auto compaction (CLAMPED?)"; break; }
done
stopsrv; say "mock serve.py stopped"
