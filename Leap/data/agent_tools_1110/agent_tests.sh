#!/bin/bash
# Real-engine tool tests (orchestrator 20:3x), after tonight's 1.1.0 clean-clone check (pid $1), when the Mac is free.
# serve.py (agent working copy, --mem 24, PF-A) on port 8097; then probe.py (30 calls); then 3 Claude Code tasks (claude binary
# called directly, isolated CLAUDE_CONFIG_DIR, macOS sandbox profile: writes only in the test folders, network only to
# localhost; tools limited to Read / Write / Edit / Glob / Grep) and 3 Codex tasks (its own workspace-write sandbox: the test
# folder only, no /tmp, no network). Share lock 'MEM 27 GB'; swap guard +2 GB; 8-min cap per task. Usage: agent_tests.sh <pid>
A=/private/tmp/claude-501/-Users-kamen-OjoX/f2971937-6c95-4e86-a049-05beff6a8e32/scratchpad/agent
CL=/Users/kamen/Leap/release/cleanclone/agent/clone; LOG=$A/agent_tests.log; WAITPID=${1:-0}
LAST=$(date -j -f "%Y-%m-%d %H:%M:%S" "2026-11-11 06:00:00" +%s)
noswap() { vm_stat | awk '/page size of/ {ps=$8} /^Pages (free|speculative|purgeable):/ || /^File-backed pages:/ {v=$NF; sub("[.]","",v); s+=v} END {printf "%d", s*ps/1e9}'; }
swap() { sysctl -n vm.swapusage | sed -E 's/.*used = ([0-9.]+)M.*/\1/' | cut -d. -f1; }
idle() { ioreg -c IOHIDSystem | awk '/HIDIdleTime/ {print int($NF/1000000000); exit}'; }
busy() { peer ls 2>/dev/null | grep "^- " | grep -oE "· (shared run \(others may work\)|benchmarking|waiting for a quiet window)[^·]*" | grep -cE "^· (benchmarking|waiting for a quiet window)|MEM ([6-9]|[1-9][0-9]) GB"; }
procs() { ps -axo pid=,args= | awk '/[Pp]ython/ && /r57_screen_|run_map[.]py|fit_onecall[.]py|[ \/](chat|serve|run)[.]py( |$)/ && !/ awk / {printf "%s ", $1}'; }
held() { [ -e ~/.local/share/quiet/PAUSE ] || [ -e ~/.local/share/quiet/HOLD ]; }
desc() { local c; for c in $(pgrep -P "$1"); do desc "$c"; echo "$c"; done; }
say() { echo "$(date '+%m-%d %H:%M:%S') $*" | tee -a $LOG; }
LOCKED=0; SP=""; GUARD=""
stopsrv() { [ -z "$SP" ] && return; kill -TERM $SP 2>/dev/null                # SP = serve.py itself (exec below)
            for i in $(seq 1 60); do kill -0 $SP 2>/dev/null || break; sleep 1; done
            kill -0 $SP 2>/dev/null && { say "serve.py did not stop in 60 s: SIGKILL"; kill -KILL $SP; }
            lsof -nP -iTCP:8097 -sTCP:LISTEN >/dev/null 2>&1 && say "WARNING: port 8097 still in use"; SP=""; }
cleanup() { [ -n "$GUARD" ] && kill $GUARD 2>/dev/null; stopsrv
            [ $LOCKED = 1 ] && peer unlock >/dev/null 2>&1; touch $A/agent_tests.done; say "exit (NO-SWAP $(noswap) GB, swap $(swap) MB)"; }
trap cleanup EXIT
trap 'say "stopped by a signal"; exit 1' INT TERM
say "armed (pid $$): after pid $WAITPID; free Mac; last start 11-11 06:00"
CLR=0; NOTE=""
while :; do
  [ "$(date +%s)" -ge $LAST ] && { say "SKIP: no free window by 06:00 (last wait: $NOTE)"; exit 0; }
  why=""
  [ "$WAITPID" -gt 1 ] && kill -0 $WAITPID 2>/dev/null && why="the 1.1.0 check runs first (pid $WAITPID)"
> (a quote from the project owner, kept private)
  [ -z "$why" ] && [ -n "$(procs)" ] && why="an engine / chat / server process runs"
  [ -z "$why" ] && [ "$(busy)" -gt 0 ] && why="another lock"
  [ -z "$why" ] && held && why="PAUSE / HOLD"
  [ -z "$why" ] && [ "$(noswap)" -lt 30 ] && why="NO-SWAP $(noswap) < 30"
  if [ -z "$why" ]; then CLR=$((CLR + 1)); else CLR=0; fi
  [ -z "$why" ] && [ $CLR -lt 2 ] && why="clear once"
  if [ -z "$why" ]; then
    peer lock share 60m --cpu 300 --gpu 100 "MEM 27 GB: Leap agent tests (serve.py --mem 24 + Claude Code / Codex, sandboxed)" > /dev/null 2>&1 && LOCKED=1
    sleep 20; [ "$(busy)" = 0 ] && [ -z "$(procs)" ] && break
    peer unlock >/dev/null 2>&1; LOCKED=0; CLR=0; why="after the lock"
  fi
  [ "$why" != "$NOTE" ] && say "wait: $why"; NOTE=$why
  sleep 60
done
S0=$(swap); say "GO: NO-SWAP $(noswap) GB, swap $S0 MB"
( while :; do [ $(( $(swap) - S0 )) -gt 2048 ] && { echo "$(date +%T) SWAP GUARD: $(swap) MB" >> $LOG; kill -TERM $$; exit; }; sleep 5; done ) &
GUARD=$!
( cd $CL && EXACTSTREAM_LOG_REQUESTS=$A/real_req.jsonl exec .venv/bin/python serve.py --mem 24 --port 8097 ) > $A/real_serve.log 2>&1 &
SP=$!; t0=$(date +%s)
until curl -s -m 2 http://127.0.0.1:8097/health 2>/dev/null | grep -q ready; do
  kill -0 $SP 2>/dev/null && [ $(( $(date +%s) - t0 )) -lt 300 ] || { say "serve.py did not come up"; exit 2; }; sleep 2
done
say "serve.py ready in $(( $(date +%s) - t0 )) s"
capped() { local s=$1; shift; "$@" & local P=$!; for i in $(seq 1 $s); do kill -0 $P 2>/dev/null || { wait $P; return $?; }; sleep 1; done
           for c in $(desc $P); do kill -TERM $c 2>/dev/null; done; kill -TERM $P; echo "(killed after $s s)"; return 124; }
say "probe:"; capped 1500 /opt/homebrew/opt/python@3.12/bin/python3.12 -I $A/probe.py 8097 $A/probe.jsonl > $A/probe.out 2>&1
say "  $(tail -1 $A/probe.out)"
W=$A/work_cc; rm -rf $W; mkdir -p $W; cd $W
cc() { capped 480 sandbox-exec -f $A/sandbox.sb env CLAUDE_CONFIG_DIR=$A/cc_home EXACTSTREAM_PORT=8097 CLAUDE_BIN=/Users/kamen/.local/bin/claude \
         $CL/tools/claude_local.sh -p "$1" --allowedTools "Read" "Write" "Edit" "Glob" "Grep" < /dev/null > $A/cc_$2.out 2>&1; }
cc "Create a file named notes.txt in this folder that contains exactly the word hello." t1
say "  claude t1 (create): rc $?; notes.txt = '$(cat $W/notes.txt 2>/dev/null | tr '\n' ' ')'; $(tail -c 120 $A/cc_t1.out | tr '\n' ' ')"
cc "Read notes.txt in this folder and tell me the word it contains." t2
say "  claude t2 (read): rc $?; answer: $(tail -c 160 $A/cc_t2.out | tr '\n' ' ')"
cc "In notes.txt in this folder, replace the word hello with goodbye." t3
say "  claude t3 (edit): rc $?; notes.txt = '$(cat $W/notes.txt 2>/dev/null | tr '\n' ' ')'"
W=$A/work_codex; rm -rf $W; mkdir -p $W; cd $W
cx() { capped 480 env EXACTSTREAM_CODEX_HOME=$A/codex_home EXACTSTREAM_PORT=8097 CODEX_BIN=/Users/kamen/.local/bin/codex $CL/tools/codex_local.sh \
         exec --sandbox workspace-write --skip-git-repo-check -c sandbox_workspace_write.exclude_tmpdir_env_var=true \
         -c sandbox_workspace_write.exclude_slash_tmp=true -c sandbox_workspace_write.network_access=false -C $W "$1" < /dev/null > $A/cx_$2.out 2>&1; }
cx "Create a file named notes.txt in this folder that contains exactly the word hello." t1
say "  codex t1 (create): rc $?; notes.txt = '$(cat $W/notes.txt 2>/dev/null | tr '\n' ' ')'"
cx "How many files are in this folder? Answer with just the number." t2
say "  codex t2 (count): rc $?; answer: $(tail -c 120 $A/cx_t2.out | tr '\n' ' ')"
cx "Append a second line containing the word world to notes.txt in this folder." t3
say "  codex t3 (append): rc $?; notes.txt = '$(cat $W/notes.txt 2>/dev/null | tr '\n' '|')'"
say "side-lane probe (model exactstream-side): $(curl -s -m 300 http://127.0.0.1:8097/v1/messages -H 'content-type: application/json' \
  -d '{"model": "exactstream-side", "max_tokens": 8, "messages": [{"role": "user", "content": "Reply with one word: ok"}]}' | cut -c1-200)"
say "server lines: $(grep -c 'done:' $A/real_serve.log) requests; errors: $(grep -c -E 'Traceback|Error' $A/real_serve.log)"
stopsrv; say "serve.py stopped (NO-SWAP $(noswap) GB)"
cp $A/real_req.jsonl $A/sessions_real_1110.jsonl 2>/dev/null
$CL/.venv/bin/python -I $A/prefix_check.py $A/real_req.jsonl $CL/engine/models/bare > $A/prefix_real.out 2>&1
say "prefix check: $(tail -1 $A/prefix_real.out)"
