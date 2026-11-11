#!/bin/bash
# Stand-in check of agentcc_tasks2.py (for the runs after the cached 45 GB one; EvalSuite amendment first): one profile per
# task = the run's sandbox.sb + read denies (file contents and listings) of ~/EvalSuite/agentcc and of the run folder
# outside the task's own folder, its config folder, cctmp and home; TMPPREFIX=<run>/cctmp/zsh in claude's environment.
# Two scratch tasks (t00a, t00b) on mock_bash.py (no GPU, < 1 GB), the night runs' base profile, HOME=<run>/home from
# agent_home_tpl, CLAUDE_CODE_TMPDIR. Each task's Bash call probes (labels only, no contents printed):
#   expected READ_OK: its own folder, git status, a write + read there, a heredoc;
#   expected DENIED: <run>/tasks (siblings), the other task's folder, <run>/runner.log, its own transcript <run>/<id>.jsonl,
#   ~/EvalSuite/agentcc (listing) and its manifest.json.
# Then the runner's check (sandboxed with the same per-task profile) must pass in the task folder.
A=/private/tmp/claude-501/-Users-kamen-OjoX/f2971937-6c95-4e86-a049-05beff6a8e32/scratchpad/agent
CL=/Users/kamen/Leap/release/cleanclone/agent/clone; PORT=8131; B=$A/profile_check; H=/Users/kamen
rm -rf $B; mkdir -p $B/man/tpl $B/run/home $B/run/cctmp; RUN=$B/run
SP=""
stopsrv() { [ -z "$SP" ] && return; kill -TERM $SP 2>/dev/null
            for i in $(seq 1 60); do kill -0 $SP 2>/dev/null || break; sleep 1; done
            kill -0 $SP 2>/dev/null && { echo "serve.py did not stop in 60 s: SIGKILL"; kill -KILL $SP; }
            lsof -nP -iTCP:$PORT -sTCP:LISTEN >/dev/null 2>&1 && echo "WARNING: port $PORT still in use"; SP=""; }
trap stopsrv EXIT
echo "template file" > $B/man/tpl/README.txt
echo "check" > $B/man/check.py
/opt/homebrew/opt/python@3.12/bin/python3.12 -I - $B <<'EOF'
import json, sys
B = sys.argv[1]
t = lambda i: dict(id=i, template=f"{B}/man/tpl", prompt="BASHFIX: run the shell check.", timeout_s={"32": 240},
                   check_file=f"{B}/man/check.py", check_copy_as=".agentcc_check.py",
                   check="test -f made_by_bash.txt && test -f .agentcc_check.py && echo CHECK_OK")
json.dump({"tasks": [t("t00a"), t("t00b")]}, open(f"{B}/man/manifest.json", "w"))
EOF
echo "runner log line" > $RUN/runner.log
cp $A/agent_home_tpl/.zshrc $A/agent_home_tpl/.gitconfig $RUN/home/
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
export MOCK_BASH_CMD=$'R=$(dirname "$(dirname "$PWD")"); ME=$(basename "$PWD"); OTHER=t00a; [ "$ME" = t00a ] && OTHER=t00b\nP() { if eval "$2" >/dev/null 2>&1; then echo "$1 READ_OK"; else echo "$1 DENIED"; fi; }\necho hi > made_by_bash.txt && echo "WRITE_TASK ok"\nP OWN_FOLDER "ls . && cat made_by_bash.txt && cat README.txt"\nP GIT "git status --porcelain"\nP RUN_TASKS "ls $R/tasks"\nP OTHER_TASK "ls $R/tasks/$OTHER"\nP RUNNER_LOG "cat $R/runner.log"\nP OWN_TRANSCRIPT "cat $R/$ME.jsonl"\nP EVAL_LS "ls /Users/kamen/EvalSuite/agentcc"\nP EVAL_MANIFEST "cat /Users/kamen/EvalSuite/agentcc/manifest.json"\ncat <<EOT\nHEREDOC_OK\nEOT'
( cd $CL && MOCK_STORE=$CL/engine/models/bare MOCK_WORKDIR=$B exec .venv/bin/python serve.py --mem 32 --port $PORT --_screen $A/mock_bash.py ) > $B/serve.log 2>&1 &
SP=$!; t0=$(date +%s)
until curl -s -m 2 http://127.0.0.1:$PORT/health 2>/dev/null | grep -q ready; do
  kill -0 $SP 2>/dev/null && [ $(( $(date +%s) - t0 )) -lt 120 ] || { echo "mock serve.py did not come up: $(tail -3 $B/serve.log)"; exit 2; }; sleep 1
done
echo "mock serve.py ready in $(( $(date +%s) - t0 )) s (pid $SP)"
HOME=$RUN/home CLAUDE_CODE_TMPDIR=$RUN/cctmp /opt/homebrew/opt/python@3.12/bin/python3.12 -I $A/agentcc_tasks2.py $RUN 32 $B/man/manifest.json $PORT \
  $RUN/sandbox.sb $A/cc_home $CL/tools/claude_local.sh > $B/tasks.out 2>&1
echo "tasks.out:"; sed 's/^/  /' $B/tasks.out
for t in t00a t00b; do
  echo "== $t profile tail:"; tail -2 $RUN/sandbox_$t.sb | sed 's|'"$RUN"'|<run>|g; s/^/  /' | cut -c1-200
  /opt/homebrew/opt/python@3.12/bin/python3.12 -I - $RUN/$t.jsonl <<'EOF'
import json, sys
for line in open(sys.argv[1]):
    try: d = json.loads(line)
    except Exception: continue
    if d.get("type") == "user" and isinstance(d.get("message", {}).get("content"), list):
        for c in d["message"]["content"]:
            if c.get("type") == "tool_result":
                t = c.get("content"); t = t if isinstance(t, str) else json.dumps(t)
                print(f"  tool result (is_error {bool(c.get('is_error'))}):")
                for l in t.splitlines(): print("    " + l[:200])
    if d.get("type") == "result": print(f"  result {d.get('subtype')}, turns {d.get('num_turns')}")
EOF
  echo "  check: $(cat $RUN/$t.check.txt | tr '\n' ' ')"; echo "  stderr: $(grep -v unrecognized_model $RUN/$t.stderr | head -3 | tr '\n' ' ')"
done
echo "results:"; cat $RUN/results.jsonl
stopsrv; echo "mock serve.py stopped"
