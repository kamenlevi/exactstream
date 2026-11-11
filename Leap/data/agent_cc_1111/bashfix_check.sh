#!/bin/bash
# Sandboxed Claude Code Bash on the stand-in engine (mock_bash.py = mock_tools.py + one Bash call; no GPU, < 1 GB).
# EvalSuite 03:5x: in agentcc_runs/32_1111 all 79 Bash calls failed with EPERM: Claude Code mkdirs a per-folder temp dir
# under /private/tmp/claude-501/ before it runs a command, and the sandbox allows writes only in the run folder,
# /private/var/folders and /dev. Variants (each its own run folder, the same profile generator as agentcc_night.sh):
#   V0: as in 32_1111 (expect the EPERM);
#   V1: CLAUDE_CODE_TMPDIR=<run>/cctmp (Claude Code's temp root inside the run folder; the profile unchanged).
# 04:0x: the read-deny rule for ~/.claude* is now one regex (no account names); the command also counts the
# ~/.claude* entries it can read (must be 0) and tries ~/.ssh (must be denied).
# The command prints the shell's TMPDIR, writes a file in the task folder, writes a tempfile (TMPDIR), and tries a write
# outside (must be denied). Usage: bashfix_check.sh [V0 V1 ...]
A=/private/tmp/claude-501/-Users-kamen-OjoX/f2971937-6c95-4e86-a049-05beff6a8e32/scratchpad/agent
CL=/Users/kamen/Leap/release/cleanclone/agent/clone; PORT=8131; B=$A/bashfix; mkdir -p $B
VARS=${*:-V0 V1}; H=/Users/kamen
SP=""
stopsrv() { [ -z "$SP" ] && return; kill -TERM $SP 2>/dev/null
            for i in $(seq 1 60); do kill -0 $SP 2>/dev/null || break; sleep 1; done
            kill -0 $SP 2>/dev/null && { echo "serve.py did not stop in 60 s: SIGKILL"; kill -KILL $SP; }
            lsof -nP -iTCP:$PORT -sTCP:LISTEN >/dev/null 2>&1 && echo "WARNING: port $PORT still in use"; SP=""; }
trap stopsrv EXIT
export MOCK_BASH_CMD='echo "TMPDIR=$TMPDIR"; echo "PWD=$(pwd)"; echo hi > made_by_bash.txt && echo WROTE_OK; /usr/bin/python3 -c "import tempfile; f = tempfile.NamedTemporaryFile(delete=False); f.write(b\"x\"); f.close(); print(\"TMPFILE_OK\", f.name)"; P=$HOME/.bashfix_probe_$$; (echo x > $P) 2>/dev/null && echo "OUTSIDE_WRITE_ALLOWED $P" || echo OUTSIDE_DENIED_OK; n=0; r=0; for d in $HOME/.claude*; do n=$((n+1)); { ls "$d" >/dev/null 2>&1 || head -c1 "$d" >/dev/null 2>&1; } && r=$((r+1)); done; echo "DOTCLAUDE_ENTRIES $n READABLE $r"; head -c1 $HOME/.ssh/known_hosts >/dev/null 2>&1 && echo SSH_READABLE || echo SSH_DENIED_OK'
( cd $CL && MOCK_STORE=$CL/engine/models/bare MOCK_WORKDIR=$B exec .venv/bin/python serve.py --mem 32 --port $PORT --_screen $A/mock_bash.py ) > $B/serve.log 2>&1 &
SP=$!; t0=$(date +%s)
until curl -s -m 2 http://127.0.0.1:$PORT/health 2>/dev/null | grep -q ready; do
  kill -0 $SP 2>/dev/null && [ $(( $(date +%s) - t0 )) -lt 120 ] || { echo "mock serve.py did not come up: $(tail -3 $B/serve.log)"; exit 2; }; sleep 1
done
echo "mock serve.py ready in $(( $(date +%s) - t0 )) s (pid $SP)"
for V in $VARS; do
  RUN=$B/$V; rm -rf $RUN; mkdir -p $RUN/tasks/t00 $RUN/cctmp
  cat > $RUN/sandbox.sb <<SB
(version 1)
(allow default)
(deny network-outbound)
(allow network-outbound (remote ip "localhost:*"))
(allow network-outbound (remote unix-socket))
(deny file-write*)
(allow file-write* (subpath "$RUN") (subpath "/private/var/folders") (regex #"^/dev/"))
(deny file-read* (subpath "$H/.ssh") (subpath "$H/.aws") (subpath "$H/.gnupg") (regex #"^/Users/[^/]+/\.claude[^/]*(/|$)")
  (subpath "$H/.codex") (subpath "$H/.config/gh") (literal "$H/.netrc") (literal "$H/.git-credentials")
  (subpath "$H/Library/Keychains") (subpath "$H/Library/Mail") (subpath "$H/Library/Messages") (subpath "$H/Library/Cookies")
  (subpath "$H/Library/Safari") (subpath "$H/Library/Application Support/Google") (subpath "$H/Library/Application Support/Claude")
  (subpath "$H/Documents") (subpath "$H/Desktop") (subpath "$H/Downloads") (subpath "$H/Papers") (subpath "$H/.local/share/exactstream-sessions"))
SB
  rsync -a --exclude projects --exclude sessions --exclude backups --exclude shell-snapshots --exclude todos $A/cc_home/ $RUN/cc_home/
  EXTRA=""; [ $V = V1 ] && EXTRA="CLAUDE_CODE_TMPDIR=$RUN/cctmp"
  ( cd $RUN/tasks/t00 && exec env CLAUDE_CONFIG_DIR=$RUN/cc_home EXACTSTREAM_PORT=$PORT CLAUDE_BIN=/Users/kamen/.local/bin/claude $EXTRA \
      sandbox-exec -f $RUN/sandbox.sb $CL/tools/claude_local.sh -p "BASHFIX: run the shell check." --output-format stream-json \
      --verbose --max-turns 3 --allowedTools Bash Edit Glob Grep Read Write ) < /dev/null > $RUN/out.jsonl 2> $RUN/stderr.txt &
  CP=$!; c0=$(date +%s)
  while kill -0 $CP 2>/dev/null; do [ $(( $(date +%s) - c0 )) -gt 180 ] && { echo "$V: claude over 180 s: stopping"; kill -TERM $CP; }; sleep 1; done
  wait $CP; rc=$?
  /opt/homebrew/opt/python@3.12/bin/python3.12 -I - $RUN/out.jsonl $V $rc <<'EOF'
import json, sys
f, V, rc = sys.argv[1:4]
res = []
for line in open(f):
    try:
        d = json.loads(line)
    except Exception:
        continue
    if d.get("type") == "user":
        for c in d.get("message", {}).get("content", []) if isinstance(d.get("message", {}).get("content"), list) else []:
            if c.get("type") == "tool_result":
                t = c.get("content"); t = t if isinstance(t, str) else json.dumps(t)
                res.append((bool(c.get("is_error")), t))
print(f"{V}: claude rc {rc}; tool results {len(res)}")
for e, t in res:
    print(f"  is_error {e}: " + t[:600].replace("\n", " | "))
EOF
  ls -la $RUN/tasks/t00 | grep -v "^total" | sed 's/^/  folder: /'
  echo "  cctmp: $(find $RUN/cctmp -maxdepth 3 | head -5 | tr '\n' ' ')"
done
ls -d $H/.bashfix_probe_* 2>/dev/null && echo "OUTSIDE FILE PRESENT (sandbox leak)" || echo "no probe file outside: OK"
stopsrv; echo "mock serve.py stopped"
