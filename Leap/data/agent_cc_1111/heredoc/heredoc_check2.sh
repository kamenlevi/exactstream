#!/bin/bash
# The zsh heredoc limit seen in 32c_1111 (t09: "can't create temp file for here document: operation not permitted"):
# zsh writes heredoc / here-string temp files under TMPPREFIX (default /tmp/zsh), and the sandbox allows writes only in
# the run folder, /private/var/folders and /dev. Orchestrator 07:3x: TMPPREFIX in the neutral .zshrc, pointing at the
# run's own temp folder, for runs AFTER the armed 45 GB one (EvalSuite amendment first; agent_home_tpl stays as it is,
# because the armed 45 GB runner copies it). Stand-in check (mock_bash.py; no GPU, < 1 GB) in the night runs' setup
# (regex read-deny sandbox, CLAUDE_CODE_TMPDIR=<run>/cctmp, HOME=<run>/home from agent_home_tpl):
#   H: the neutral .zshrc as registered (the exact PATH only): expect the heredoc to fail;
#   E: the same .zshrc + TMPPREFIX=<run>/cctmp/zsh in the claude process environment (heredoc_check.sh showed a .zshrc export does not reach the Bash tool)
A=/private/tmp/claude-501/-Users-kamen-OjoX/f2971937-6c95-4e86-a049-05beff6a8e32/scratchpad/agent
CL=/Users/kamen/Leap/release/cleanclone/agent/clone; PORT=8131; B=$A/heredoc_check2; mkdir -p $B; H=/Users/kamen
SP=""
stopsrv() { [ -z "$SP" ] && return; kill -TERM $SP 2>/dev/null
            for i in $(seq 1 60); do kill -0 $SP 2>/dev/null || break; sleep 1; done
            kill -0 $SP 2>/dev/null && { echo "serve.py did not stop in 60 s: SIGKILL"; kill -KILL $SP; }
            lsof -nP -iTCP:$PORT -sTCP:LISTEN >/dev/null 2>&1 && echo "WARNING: port $PORT still in use"; SP=""; }
trap stopsrv EXIT
export MOCK_BASH_CMD=$'cat <<EOT\nHEREDOC_OK\nEOT\ncat <<< "HERESTRING_OK"; echo "TMPPREFIX=${TMPPREFIX:-unset}"; echo "PATH=$PATH"; echo "GITID $(git config --global user.name) <$(git config --global user.email)>"; echo "ALIASES $(alias | wc -l | tr -d " ")"'
( cd $CL && MOCK_STORE=$CL/engine/models/bare MOCK_WORKDIR=$B exec .venv/bin/python serve.py --mem 32 --port $PORT --_screen $A/mock_bash.py ) > $B/serve.log 2>&1 &
SP=$!; t0=$(date +%s)
until curl -s -m 2 http://127.0.0.1:$PORT/health 2>/dev/null | grep -q ready; do
  kill -0 $SP 2>/dev/null && [ $(( $(date +%s) - t0 )) -lt 120 ] || { echo "mock serve.py did not come up: $(tail -3 $B/serve.log)"; exit 2; }; sleep 1
done
echo "mock serve.py ready in $(( $(date +%s) - t0 )) s (pid $SP)"
for V in H E; do
  RUN=$B/$V; rm -rf $RUN; mkdir -p $RUN/tasks/t00 $RUN/cctmp $RUN/home
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
  cp $A/agent_home_tpl/.zshrc $A/agent_home_tpl/.gitconfig $RUN/home/
  EXTRA=""; [ $V = E ] && EXTRA="TMPPREFIX=$RUN/cctmp/zsh"
  echo "$V: .zshrc sha256 $(shasum -a 256 $RUN/home/.zshrc | cut -c1-12), lines $(grep -c . $RUN/home/.zshrc)"
  ( cd $RUN/tasks/t00 && exec env HOME=$RUN/home CLAUDE_CONFIG_DIR=$RUN/cc_home EXACTSTREAM_PORT=$PORT \
      CLAUDE_BIN=/Users/kamen/.local/bin/claude CLAUDE_CODE_TMPDIR=$RUN/cctmp $EXTRA \
      sandbox-exec -f $RUN/sandbox.sb $CL/tools/claude_local.sh -p "BASHFIX: run the shell check." --output-format stream-json \
      --verbose --max-turns 3 --allowedTools Bash ) < /dev/null > $RUN/out.jsonl 2> $RUN/stderr.txt &
  CP=$!; c0=$(date +%s)
  while kill -0 $CP 2>/dev/null; do [ $(( $(date +%s) - c0 )) -gt 180 ] && { echo "$V: claude over 180 s: stopping"; kill -TERM $CP; }; sleep 1; done
  wait $CP; rc=$?
  /opt/homebrew/opt/python@3.12/bin/python3.12 -I - $RUN/out.jsonl $V $rc <<'EOF'
import json, sys
f, V, rc = sys.argv[1:4]
for line in open(f):
    try:
        d = json.loads(line)
    except Exception:
        continue
    if d.get("type") == "user" and isinstance(d.get("message", {}).get("content"), list):
        for c in d["message"]["content"]:
            if c.get("type") == "tool_result":
                t = c.get("content"); t = t if isinstance(t, str) else json.dumps(t)
                print(f"{V}: claude rc {rc}; is_error {bool(c.get('is_error'))}")
                for l in t.splitlines():
                    print("   ", l[:400])
EOF
  echo "  $V cctmp: $(find $RUN/cctmp -maxdepth 1 -name 'zsh*' | wc -l | tr -d ' ') zsh temp files left"
done
stopsrv; echo "mock serve.py stopped"
