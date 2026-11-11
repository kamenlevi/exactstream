#!/bin/bash
# Stand-in check of agentcc_tasks4.py (the per-task WRITE scope + no link-following by the runner; a security review of
# Leap 68ec944): two scratch tasks (t00a, t00b) on mock_bash.py (no GPU, < 1 GB), HOME=<run>/home from agent_home_tpl,
# Claude Code = $CLAUDE_BIN (Leap's 2.1.321 copy). Each task's Bash call probes and prints labels only:
#   reads (as profile_check.sh): its own folder and git OK; <run>/tasks, the other task, runner.log, its own transcript,
#     ~/EvalSuite/agentcc and its manifest DENIED;
#   writes: its own folder, a git commit there and its own temp folder ($TMPPREFIX's folder) OK; a file in <run>, a link at
#     the next task's transcript name (<run>/t00b.jsonl from t00a), HOME and the other task's temp folder DENIED;
#   the attack: a link at <task>/.agentcc_check.py (the check's name) to a canary file outside the run folder; the runner
#     then copies the check in, and the canary must stay unchanged;
#   a heredoc OK.
# Then: the canary, each init event's claude_code_version, the checks, the kept per-task profiles.
# MODE=new (default): agentcc_tasks4.py with a base profile that does not name <run> (agentcc_cache3.sh's);
# MODE=old (the control): agentcc_tasks3.py with 32c2's base profile (writes allowed in all of <run>).
A=/private/tmp/claude-501/-Users-kamen-OjoX/f2971937-6c95-4e86-a049-05beff6a8e32/scratchpad/agent
CL=/Users/kamen/Leap/release/cleanclone/agent/clone; PORT=8131; MODE=${MODE:-new}; B=$A/profile_check3_$MODE; H=/Users/kamen
TASKS_PY=$A/agentcc_tasks4.py; [ $MODE = old ] && TASKS_PY=$A/agentcc_tasks3.py
rm -rf $B; mkdir -p $B/man/tpl $B/run/home $B/run/cctmp; RUN=$B/run; CANARY=$B/canary.txt
echo "CANARY ORIGINAL CONTENT" > $CANARY
SP=""
stopsrv() { [ -z "$SP" ] && return; kill -TERM $SP 2>/dev/null
            for i in $(seq 1 60); do kill -0 $SP 2>/dev/null || break; sleep 1; done
            kill -0 $SP 2>/dev/null && { echo "serve.py did not stop in 60 s: SIGKILL"; kill -KILL $SP; }
            lsof -nP -iTCP:$PORT -sTCP:LISTEN >/dev/null 2>&1 && echo "WARNING: port $PORT still in use"; SP=""; }
trap stopsrv EXIT
echo "template file" > $B/man/tpl/README.txt
echo "print('check ran')" > $B/man/check.py
/opt/homebrew/opt/python@3.12/bin/python3.12 -I - $B <<'EOF'
import json, sys
B = sys.argv[1]
t = lambda i: dict(id=i, template=f"{B}/man/tpl", prompt="BASHFIX: run the shell check.", timeout_s={"32": 240},
                   check_file=f"{B}/man/check.py", check_copy_as=".agentcc_check.py",
                   check="test -f made_by_bash.txt && test -f .agentcc_check.py && test ! -L .agentcc_check.py && /usr/bin/python3 .agentcc_check.py && echo CHECK_OK")
json.dump({"tasks": [t("t00a"), t("t00b")]}, open(f"{B}/man/manifest.json", "w"))
EOF
echo "runner log line" > $RUN/runner.log
cp $A/agent_home_tpl/.zshrc $A/agent_home_tpl/.gitconfig $RUN/home/
WRITES='(allow file-write* (subpath "/private/var/folders") (regex #"^/dev/"))'
[ $MODE = old ] && WRITES="(allow file-write* (subpath \"$RUN\") (subpath \"/private/var/folders\") (regex #\"^/dev/\"))"
cat > $RUN/sandbox.sb <<SB
(version 1)
(allow default)
(deny network-outbound)
(allow network-outbound (remote ip "localhost:*"))
(allow network-outbound (remote unix-socket))
(deny file-write*)
$WRITES
(deny file-read* (subpath "$H/.ssh") (subpath "$H/.aws") (subpath "$H/.gnupg") (regex #"^/Users/[^/]+/\.claude[^/]*(/|$)")
  (subpath "$H/.codex") (subpath "$H/.config/gh") (literal "$H/.netrc") (literal "$H/.git-credentials")
  (subpath "$H/Library/Keychains") (subpath "$H/Library/Mail") (subpath "$H/Library/Messages") (subpath "$H/Library/Cookies")
  (subpath "$H/Library/Safari") (subpath "$H/Library/Application Support/Google") (subpath "$H/Library/Application Support/Claude")
  (subpath "$H/Documents") (subpath "$H/Desktop") (subpath "$H/Downloads") (subpath "$H/Papers") (subpath "$H/.local/share/exactstream-sessions"))
SB
export CANARY
export MOCK_BASH_CMD=$'R=$(dirname "$(dirname "$PWD")"); ME=$(basename "$PWD"); OTHER=t00a; NEXT=t00b; [ "$ME" = t00a ] && OTHER=t00b; [ "$ME" = t00b ] && NEXT=t00c\nP() { if eval "$2" >/dev/null 2>&1; then echo "$1 OK"; else echo "$1 DENIED"; fi; }\necho hi > made_by_bash.txt && echo "WRITE_TASK OK"\nP OWN_FOLDER "ls . && cat made_by_bash.txt && cat README.txt"\nP GIT "git status --porcelain"\nP GIT_COMMIT "git add -A && git commit -q -m probe"\nP OWN_TMP "echo x > $(dirname "$TMPPREFIX")/own_tmp_probe"\nP READ_RUN_TASKS "ls $R/tasks"\nP READ_OTHER_TASK "ls $R/tasks/$OTHER"\nP READ_RUNNER_LOG "cat $R/runner.log"\nP READ_OWN_TRANSCRIPT "cat $R/$ME.jsonl"\nP READ_EVAL_LS "ls /Users/kamen/EvalSuite/agentcc"\nP READ_EVAL_MANIFEST "cat /Users/kamen/EvalSuite/agentcc/manifest.json"\nP WRITE_RUN "echo x > $R/planted.txt"\nP LINK_NEXT_TRANSCRIPT "ln -s $CANARY $R/$NEXT.jsonl"\nP WRITE_HOME "echo x > $HOME/home_probe"\nP WRITE_OTHER_TMP "echo x > $R/cctmp/$OTHER/other_tmp_probe"\nP LINK_CHECK_NAME "ln -s $CANARY .agentcc_check.py"\ncat <<EOT\nHEREDOC_OK\nEOT\necho "TMPPREFIX under the task temp: $(case "$TMPPREFIX" in "$R/cctmp/$ME/"*) echo yes;; *) echo "no ($TMPPREFIX)";; esac)"'
( cd $CL && MOCK_STORE=$CL/engine/models/bare MOCK_WORKDIR=$B exec .venv/bin/python serve.py --mem 32 --port $PORT --_screen $A/mock_bash.py ) > $B/serve.log 2>&1 &
SP=$!; t0=$(date +%s)
until curl -s -m 2 http://127.0.0.1:$PORT/health 2>/dev/null | grep -q ready; do
  kill -0 $SP 2>/dev/null && [ $(( $(date +%s) - t0 )) -lt 120 ] || { echo "mock serve.py did not come up: $(tail -3 $B/serve.log)"; exit 2; }; sleep 1
done
echo "MODE=$MODE ($(basename $TASKS_PY)); mock serve.py ready in $(( $(date +%s) - t0 )) s (pid $SP); CLAUDE_BIN=${CLAUDE_BIN:-unset}"
EX=""; [ $MODE = old ] && EX="CLAUDE_CODE_TMPDIR=$RUN/cctmp"
env HOME=$RUN/home $EX /opt/homebrew/opt/python@3.12/bin/python3.12 -I $TASKS_PY $RUN 32 $B/man/manifest.json $PORT \
  $RUN/sandbox.sb $A/cc_home $CL/tools/claude_local.sh > $B/tasks.out 2>&1
echo "tasks.out:"; sed 's/^/  /' $B/tasks.out
for t in t00a t00b; do
  echo "== $t (init: $(head -1 $RUN/$t.jsonl 2>/dev/null | grep -o '"claude_code_version":"[^"]*"'); transcript is a link: $([ -L $RUN/$t.jsonl ] && echo YES || echo no))"
  /opt/homebrew/opt/python@3.12/bin/python3.12 -I - $RUN/$t.jsonl <<'EOF'
import json, sys
try: lines = open(sys.argv[1]).read().splitlines()
except Exception as e: lines = []; print("  (transcript unreadable:", e, ")")
for line in lines:
    try: d = json.loads(line)
    except Exception: continue
    if d.get("type") == "user" and isinstance(d.get("message", {}).get("content"), list):
        for c in d["message"]["content"]:
            if c.get("type") == "tool_result":
                t = c.get("content"); t = t if isinstance(t, str) else json.dumps(t)
                for l in t.splitlines(): print("    " + l[:160])
    if d.get("type") == "result": print(f"  result {d.get('subtype')}, turns {d.get('num_turns')}")
EOF
  echo "  check: $(cat $RUN/$t.check.txt 2>/dev/null | tr '\n' ' ')"
done
echo "results:"; cat $RUN/results.jsonl 2>/dev/null
echo "CANARY now: $(head -c 60 $CANARY | tr '\n' ' ') -> $([ "$(cat $CANARY)" = "CANARY ORIGINAL CONTENT" ] && echo UNCHANGED || echo OVERWRITTEN)"
echo "per-task profiles kept: $(ls $RUN/sandbox_t*.sb 2>/dev/null | wc -l | tr -d ' '); links anywhere in the run folder: $(find $RUN -type l | sed "s|$RUN|<run>|" | tr '\n' ' ')"
stopsrv; echo "mock serve.py stopped"
