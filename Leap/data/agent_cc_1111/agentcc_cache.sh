#!/bin/bash
# CACHED variant (conversation cache on; the cache clone; its files incl. engine/loop/convcache.py must match release/next's
# SHA256SUMS; the server logs every cache line to <run>/convcache.log). Otherwise as agentcc_night.sh:
# EvalSuite's Claude Code agent test (PREREG_agentcc_2026-11-11, manifest ~/EvalSuite/agentcc/manifest.json) at GB GB, at
# night (Kamen's go 23:1x: apps closed 00:29-07:30; one job over 5 GB at a time; the Claude Code push first). Waits for
# the given pids, FROM, a free Mac (Kamen idle, no other lock / engine, no PAUSE / HOLD, NO-SWAP >= the setting's need,
# 2 clear polls); last start LAST. Then, under a share lock: the pinned agent server (release/next; the clone's files must
# match its SHA256SUMS) as serve.py --mem GB on port 8120 with its request log; probe_cc.py (EvalSuite's 30 calls); the
# 12 tasks (agentcc_tasks.py); the server stopped and checked gone. Swap +2 GB stops everything.
# Wed 04:0x: + the Bash fix (CLAUDE_CODE_TMPDIR=<run>/cctmp; 32_1111's 79 Bash calls all hit EPERM on Claude Code's temp dir
# under /private/tmp/claude-501) and BRANCH=0 in the environment -> CONV_CACHE_BRANCH=0 (the orchestrator's 09:00 fallback:
# a would-be copy becomes a miss). 05:2x: LONGPID / LONGLOG = long_ctx's pid and log: wait while long_ctx still has arms
# it can run (its log's last line is not "no arm fits"); once it only waits for room that never comes, stop it at GO
# (TERM; it holds no lock then) so it cannot start an arm beside this run (it would skip those arms at 07:30 anyway).
# 04:2x (orchestrator): the claude processes run with HOME=<run>/home, a neutral folder
# (agent_home_tpl: .zshrc = only the exact PATH of the 11-11 runs; .gitconfig = agentcc@localhost), so Claude Code's shell
# snapshot holds no user aliases (it sources $HOME/.zshrc; ZDOTDIR has no effect). Usage: [BRANCH=0] agentcc_cache.sh <GB> <run dir> "<FROM YYYY-MM-DD HH:MM:SS>" "<LAST ...>" [pid ...]
GB=$1; RUN=$2; FROMS=$3; LASTS=$4; shift 4; AFTER="$*"
A=/private/tmp/claude-501/-Users-kamen-OjoX/f2971937-6c95-4e86-a049-05beff6a8e32/scratchpad/agent
CL=/Users/kamen/Leap/release/cleanclone/cache/clone; NEXT=/Users/kamen/Leap/release/next; MAN=/Users/kamen/EvalSuite/agentcc/manifest.json
PORT=8122; mkdir -p $RUN $RUN/cctmp; LOG=$RUN/runner.log; BR=${BRANCH:-1}
case $GB in 32) NEED=45;; 45) NEED=${NEED45:-58};; 24) NEED=35;; *) echo "GB must be 24 / 32 / 45"; exit 2;; esac
ep() { date -j -f "%Y-%m-%d %H:%M:%S" "$1" +%s; }
FROM=$(ep "$FROMS"); LAST=$(ep "$LASTS")
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
cleanup() { [ -n "$GUARD" ] && kill $GUARD 2>/dev/null
            [ -n "$TP" ] && kill -0 $TP 2>/dev/null && { for c in $(desc $TP); do kill -TERM $c 2>/dev/null; done; kill -TERM $TP; }
            stopsrv; [ $LOCKED = 1 ] && peer unlock >/dev/null 2>&1; touch $RUN/runner.done; say "exit (NO-SWAP $(noswap) GB, swap $(swap) MB)"; }
trap cleanup EXIT
trap 'say "stopped by a signal"; exit 1' INT TERM
say "armed (pid $$): agentcc $GB GB -> $RUN; after [$AFTER], from $FROMS, last start $LASTS, NO-SWAP >= $NEED"
CLR=0; NOTE=""
while :; do
  now=$(date +%s); [ $now -ge $LAST ] && { say "SKIP: no free window by $LASTS (last wait: $NOTE)"; exit 0; }
  why=""
  [ $now -lt $FROM ] && why="before $FROMS"
  [ -z "$why" ] && for w in $AFTER; do kill -0 $w 2>/dev/null && why="waits for pid $w"; done
  [ -z "$why" ] && [ -n "${LONGPID:-}" ] && kill -0 $LONGPID 2>/dev/null && ! tail -1 "$LONGLOG" 2>/dev/null | grep -q "no arm fits" \
    && why="long_ctx (pid $LONGPID) still has arms it can run"
> (a quote from the project owner, kept private)
  [ -z "$why" ] && [ -n "$(procs)" ] && why="an engine / chat / server runs"
  [ -z "$why" ] && [ "$(busy)" -gt 0 ] && why="another lock"
  [ -z "$why" ] && held && why="PAUSE / HOLD"
  [ -z "$why" ] && [ "$(noswap)" -lt $NEED ] && why="NO-SWAP $(noswap) < $NEED"
  if [ -z "$why" ]; then CLR=$((CLR + 1)); else CLR=0; fi
  [ -z "$why" ] && [ $CLR -lt 2 ] && why="clear once"
  if [ -z "$why" ]; then
    peer lock share 300m --cpu 400 --gpu 100 "MEM $NEED GB: Leap Claude Code agent test at $GB GB, conversation cache on (EvalSuite PREREG_agentcc amendment)" > /dev/null 2>&1 && LOCKED=1
    sleep 20
    if [ "$(busy)" = 0 ] && [ -z "$(procs)" ]; then
      if [ -n "${LONGPID:-}" ] && kill -0 $LONGPID 2>/dev/null; then
        if tail -1 "$LONGLOG" 2>/dev/null | grep -q "no arm fits"; then
          say "long_ctx (pid $LONGPID) only waits for room (\"$(tail -1 "$LONGLOG" | cut -c19-140)\"): stopping it"
          kill -TERM $LONGPID; for i in $(seq 1 75); do kill -0 $LONGPID 2>/dev/null || break; sleep 1; done
          kill -0 $LONGPID 2>/dev/null && say "WARNING: long_ctx still alive after 75 s"
        fi
      fi
      [ -z "$(procs)" ] && { [ -z "${LONGPID:-}" ] || ! kill -0 $LONGPID 2>/dev/null; } && break
    fi
    peer unlock >/dev/null 2>&1; LOCKED=0; CLR=0; why="after the lock"
  fi
  [ "$why" != "$NOTE" ] && say "wait: $why"; NOTE=$why
  sleep 60
done
S0=$(swap); say "GO: NO-SWAP $(noswap) GB, swap $S0 MB"
( while :; do [ $(( $(swap) - S0 )) -gt 2048 ] && { echo "$(date +%T) SWAP GUARD: $(swap) MB" >> $LOG; touch $RUN/SWAP_GUARD; kill -TERM $$; exit; }; sleep 5; done ) &
GUARD=$!
# provenance: the pinned server, Claude Code's version
(cd $NEXT && shasum -a 256 -c SHA256SUMS 2>/dev/null | sed 's/^/release\/next: /') > $RUN/provenance.txt
for f in serve.py agentio.py tools/claude_local.sh tools/codex_local.sh engine/loop/convcache.py; do
  a=$(shasum -a 256 $CL/$f | cut -d' ' -f1); b=$(grep " $f$" $NEXT/SHA256SUMS | cut -d' ' -f1)
  [ "$a" = "$b" ] && echo "clone $f = release/next ($a)" >> $RUN/provenance.txt || { say "STOP: clone $f differs from release/next"; exit 3; }
done
echo "release/next commit: $(git -C /Users/kamen/Leap log -1 --format='%h %s' -- release/next | cut -c1-120)" >> $RUN/provenance.txt
echo "claude --version: $(CLAUDE_CONFIG_DIR=$RUN/cc_version /Users/kamen/.local/bin/claude --version 2>&1 | head -1)" >> $RUN/provenance.txt
echo "manifest sha256: $(shasum -a 256 $MAN | cut -c1-12); setting $GB GB; port $PORT; date $(date '+%F %T')" >> $RUN/provenance.txt
echo "cache: CONV_CACHE=1 CONV_CACHE_BRANCH=$BR; Bash fix: CLAUDE_CODE_TMPDIR=$RUN/cctmp" >> $RUN/provenance.txt
mkdir -p $RUN/home && cp -p $A/agent_home_tpl/.zshrc $A/agent_home_tpl/.gitconfig $RUN/home/
echo "claude HOME: $RUN/home (.zshrc sha256 $(shasum -a 256 $RUN/home/.zshrc | cut -c1-12): only the exact PATH; .gitconfig: agentcc@localhost)" >> $RUN/provenance.txt
grep "^export PATH=" $RUN/home/.zshrc | sed "s/^/  /" >> $RUN/provenance.txt
echo "git identity under that HOME: $(HOME=$RUN/home git config --global user.name) <$(HOME=$RUN/home git config --global user.email)>" >> $RUN/provenance.txt
# the sandbox: writes only in the run folder and temp, network only to localhost, private home folders unreadable
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
( cd $CL && CONV_CACHE=1 CONV_CACHE_BRANCH=$BR CONV_CACHE_LOG=$RUN/convcache.log EXACTSTREAM_LOG_REQUESTS=$RUN/server_requests.jsonl exec .venv/bin/python serve.py --mem $GB --port $PORT ) > $RUN/serve.log 2>&1 &
SP=$!; t0=$(date +%s)
until curl -s -m 2 http://127.0.0.1:$PORT/health 2>/dev/null | grep -q ready; do
  kill -0 $SP 2>/dev/null && [ $(( $(date +%s) - t0 )) -lt 600 ] || { say "serve.py did not come up (tail: $(tail -2 $RUN/serve.log | tr '\n' ' '))"; exit 2; }; sleep 2
done
say "serve.py --mem $GB ready in $(( $(date +%s) - t0 )) s (pid $SP)"
P0=$(date +%s); python3 /Users/kamen/EvalSuite/tools/probe_cc.py $PORT $RUN/probe_cc.jsonl > $RUN/probe_cc.out 2>&1 &
TP=$!; while kill -0 $TP 2>/dev/null; do [ $(( $(date +%s) - P0 )) -gt 7200 ] && { say "probe: 2-h cap"; for c in $(desc $TP); do kill -TERM $c; done; kill -TERM $TP; }; sleep 5; done
wait $TP; say "probe rc $? in $(( $(date +%s) - P0 )) s: $(tail -1 $RUN/probe_cc.out | cut -c1-160)"; TP=""
export CLAUDE_CODE_TMPDIR=$RUN/cctmp
HOME=$RUN/home /opt/homebrew/opt/python@3.12/bin/python3.12 -I $A/agentcc_tasks.py $RUN $GB $MAN $PORT $RUN/sandbox.sb $A/cc_home $CL/tools/claude_local.sh > $RUN/tasks.out 2>&1 &
TP=$!; G1=0
while kill -0 $TP 2>/dev/null; do
  if [ $G1 = 0 ] && [ -s $RUN/results.jsonl ]; then
    G1=1; f=$(ls $RUN/t01_*.jsonl 2>/dev/null | head -1)
    if [ -n "$f" ] && grep -q "EPERM: operation not permitted, mkdir" "$f"; then
      say "STOP: t01's transcript still holds the EPERM (the Bash fix did not take): stopping the tasks"
      for c in $(desc $TP); do kill -TERM $c 2>/dev/null; done; kill -TERM $TP; break
    fi
  fi
  sleep 5
done
wait $TP 2>/dev/null; say "tasks: $(grep -c . $RUN/results.jsonl 2>/dev/null) results; checks passed $(grep -c '"check_rc": 0' $RUN/results.jsonl 2>/dev/null); $(tail -1 $RUN/tasks.out)"; TP=""
say "server: $(grep -c 'done:' $RUN/serve.log) requests; errors $(grep -c -E 'Traceback|Error' $RUN/serve.log); cache hits $(grep -c ' hit ' $RUN/convcache.log 2>/dev/null) of $(grep -cE ' (hit|miss) ' $RUN/convcache.log 2>/dev/null)"
stopsrv; say "serve.py stopped (NO-SWAP $(noswap) GB)"
