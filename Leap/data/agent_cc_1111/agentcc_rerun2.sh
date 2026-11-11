#!/bin/bash
# EvalSuite's 12 Claude Code tasks again at 32 GB (PREREG_agentcc amendment, EvalSuite 03:5x): in agentcc_runs/32_1111 all
# 79 Bash calls failed with EPERM, because Claude Code mkdirs a per-folder temp dir under /private/tmp/claude-501/ and the
# sandbox allows writes only in the run folder, /private/var/folders and /dev. The fix: CLAUDE_CODE_TMPDIR=<run>/cctmp
# (Claude Code's temp root inside the run folder; the sandbox profile is unchanged). Mock check 03:54 (bashfix_check.sh):
# V0 = the EPERM, V1 = the command ran (TMPDIR /var/folders/..., a file written in the task folder, a tempfile written,
# a write outside denied). Everything else as 32_1111: the same server files (the clone = release/next at pin 39947b1),
# serve.py --mem 32 with its request log, agentcc_tasks.py with the same manifest, base config and launcher. No probe
# (EvalSuite: the sandbox cannot affect it). Guard: if t01's transcript still holds the EPERM, stop at once.
# 04:0x (agentcc_rerun2.sh, replaces agentcc_rerun.sh before it started): the read-deny rule for ~/.claude and the two
# account config folders is one regex covering every ~/.claude* entry (also ~/.claude.json); no account names in
# files (orchestrator 04:0x). Mock check 04:05 (bashfix_check2.out): Bash runs, ~/.claude* 7 entries 0 readable.
# Starts once pid WAITPID (T1d's runner) has exited and the Mac is free: one clear poll, 10-s polls (before long_ctx's two
# 60-s polls). Last start 08:00. Share lock; swap +2 GB stops everything; the server is stopped and the port checked.
GB=32; RUN=${1:-/private/tmp/claude-501/-Users-kamen-OjoX/f2971937-6c95-4e86-a049-05beff6a8e32/scratchpad/agent/agentcc_runs/32_1111b}
WAITPID=${2:-75700}
A=/private/tmp/claude-501/-Users-kamen-OjoX/f2971937-6c95-4e86-a049-05beff6a8e32/scratchpad/agent
CL=/Users/kamen/Leap/release/cleanclone/agent/clone; PIN=39947b1; MAN=/Users/kamen/EvalSuite/agentcc/manifest.json
PORT=8124; NEED=42; mkdir -p $RUN $RUN/cctmp; LOG=$RUN/runner.log
LAST=$(date -j -f "%Y-%m-%d %H:%M:%S" "2026-11-11 08:00:00" +%s)
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
say "armed (pid $$): agentcc rerun2 $GB GB (CLAUDE_CODE_TMPDIR fix, regex read-deny) -> $RUN; after pid $WAITPID; last start 08:00; NO-SWAP >= $NEED"
NOTE=""
while :; do
  now=$(date +%s); [ $now -ge $LAST ] && { say "SKIP: no free window by 08:00 (last wait: $NOTE)"; exit 0; }
  why=""
  kill -0 $WAITPID 2>/dev/null && why="waits for pid $WAITPID (T1d)"
> (a quote from the project owner, kept private)
  [ -z "$why" ] && [ -n "$(procs)" ] && why="an engine / chat / server runs"
  [ -z "$why" ] && [ "$(busy)" -gt 0 ] && why="another lock"
  [ -z "$why" ] && held && why="PAUSE / HOLD"
  [ -z "$why" ] && [ "$(noswap)" -lt $NEED ] && why="NO-SWAP $(noswap) < $NEED"
  if [ -z "$why" ]; then
    peer lock share 180m --cpu 400 --gpu 100 "MEM $NEED GB: Leap Claude Code agent test rerun at 32 GB (EvalSuite PREREG_agentcc, Bash fix), ~95 min" > /dev/null 2>&1 && LOCKED=1
    sleep 2
    [ -z "$(procs)" ] && break
    peer unlock >/dev/null 2>&1; LOCKED=0; why="after the lock"
  fi
  [ "$why" != "$NOTE" ] && say "wait: $why"; NOTE=$why
  sleep 10
done
S0=$(swap); say "GO: NO-SWAP $(noswap) GB, swap $S0 MB"
( while :; do [ $(( $(swap) - S0 )) -gt 2048 ] && { echo "$(date +%T) SWAP GUARD: $(swap) MB" >> $LOG; touch $RUN/SWAP_GUARD; kill -TERM $$; exit; }; sleep 5; done ) &
GUARD=$!
# provenance: the clone's server files = release/next at the pin (the cold run's), Claude Code's version, the fix
git -C /Users/kamen/Leap show $PIN:release/next/SHA256SUMS > $RUN/SHA256SUMS.pin
for f in serve.py agentio.py tools/claude_local.sh; do
  a=$(shasum -a 256 $CL/$f | cut -d' ' -f1); b=$(grep " $f$" $RUN/SHA256SUMS.pin | cut -d' ' -f1)
  [ -n "$b" ] && [ "$a" = "$b" ] && echo "clone $f = release/next at $PIN ($a)" >> $RUN/provenance.txt || { say "STOP: clone $f is not release/next at $PIN"; exit 3; }
done
echo "claude --version: $(CLAUDE_CONFIG_DIR=$RUN/cc_version /Users/kamen/.local/bin/claude --version 2>&1 | head -1)" >> $RUN/provenance.txt
echo "manifest sha256: $(shasum -a 256 $MAN | cut -c1-12); setting $GB GB; port $PORT; date $(date '+%F %T')" >> $RUN/provenance.txt
echo "fix: CLAUDE_CODE_TMPDIR=$RUN/cctmp; sandbox as 32_1111 except the ~/.claude* read-deny = one regex (covers ~/.claude.json too)" >> $RUN/provenance.txt
H=/Users/kamen
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
( cd $CL && EXACTSTREAM_LOG_REQUESTS=$RUN/server_requests.jsonl exec .venv/bin/python serve.py --mem $GB --port $PORT ) > $RUN/serve.log 2>&1 &
SP=$!; t0=$(date +%s)
until curl -s -m 2 http://127.0.0.1:$PORT/health 2>/dev/null | grep -q ready; do
  kill -0 $SP 2>/dev/null && [ $(( $(date +%s) - t0 )) -lt 600 ] || { say "serve.py did not come up (tail: $(tail -2 $RUN/serve.log | tr '\n' ' '))"; exit 2; }; sleep 2
done
say "serve.py --mem $GB ready in $(( $(date +%s) - t0 )) s (pid $SP)"
export CLAUDE_CODE_TMPDIR=$RUN/cctmp
/opt/homebrew/opt/python@3.12/bin/python3.12 -I $A/agentcc_tasks.py $RUN $GB $MAN $PORT $RUN/sandbox.sb $A/cc_home $CL/tools/claude_local.sh > $RUN/tasks.out 2>&1 &
TP=$!; G1=0
while kill -0 $TP 2>/dev/null; do
  if [ $G1 = 0 ] && [ -s $RUN/results.jsonl ]; then
    G1=1; f=$(ls $RUN/t01_*.jsonl 2>/dev/null | head -1)
    if [ -n "$f" ] && grep -q "EPERM: operation not permitted, mkdir" "$f"; then
      say "STOP: t01's transcript still holds the EPERM (the fix did not take): stopping the tasks"; stoptp; break
    fi
    say "  t01 done: $(head -1 $RUN/results.jsonl); Bash results with EPERM 0"
  fi
  sleep 5
done
wait $TP 2>/dev/null; TP=""
say "tasks: $(grep -c . $RUN/results.jsonl 2>/dev/null) results; checks passed $(grep -c '"check_rc": 0' $RUN/results.jsonl 2>/dev/null); $(tail -1 $RUN/tasks.out)"
say "Bash: $(cat $RUN/t*_*.jsonl 2>/dev/null | grep -c '"name":"Bash"\|"name": "Bash"') call lines; EPERM lines $(cat $RUN/t*_*.jsonl 2>/dev/null | grep -c 'EPERM: operation not permitted, mkdir')"
say "server: $(grep -c 'done:' $RUN/serve.log) requests; errors $(grep -c -E 'Traceback|Error' $RUN/serve.log)"
stopsrv; say "serve.py stopped (NO-SWAP $(noswap) GB)"
