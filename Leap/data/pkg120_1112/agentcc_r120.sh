#!/bin/bash
# r120 (agentcc_r120.sh = agentcc_cache4.sh, the 32c2 runner, with the changes EvalSuite's PREREG_agentcc amendment 5 lists):
#  - the server and the launcher are the package candidate's, AS SHIPPED: the clean clone CL of pkg_120 at PIN (its files
#    checked against PIN's MANIFEST.json), serve.py with its own defaults at --mem GB (no CONV_CACHE / CONV_CACHE_GB /
#    CONV_CACHE_BRANCH in the environment; the runner unsets them), and tools/claude_local.sh as shipped;
#  - the context window tools/claude_code.sh sets at --mem GB (EXACTSTREAM_CONTEXT: 65536 at 45 / 32, 131072 at 24),
#    exported for the launcher, since the runner starts serve.py itself instead of through claude_code.sh;
#  - Claude Code pinned to 2.1.322 (Leap's copy logs/claude_versions/2.1.322, sha256 a9d23794..., 323,702,464 bytes, equal
#    to the installed versions/2.1.322); the t01 guard and the end-of-run count check 2.1.322;
#  - GB / NEED / ARM from the environment (default 32 GB, NO-SWAP >= 40 = 32c2's peak 36.17 GB rounded up + 3, arm r120);
#    the 24 GB arm (r120s, information only) runs the same file with GB=24 and its own NEED.
# Everything else is 32c2's: tasks5, the hardened per-task sandbox, the neutral HOME, CLAUDE_CODE_TMPDIR and TMPPREFIX per
# task, probe_cc.py unchanged, the manifest unchanged, a fresh folder, the GO file, the gates, the swap guard.
# The 32c2 runner's own description (agentcc_cache4.sh):
#  | 32c2 (v4, agentcc_cache4.sh): the cached 32 GB confirmation rerun of EvalSuite's Claude Code agent test (PREREG_agentcc amendments 4, 4a and
#  | 4c; agentcc_cache2.sh / 4b is void: a security review of Leap 68ec944 found that the sandbox let the agent write anywhere
#  | in <run> while the unsandboxed runner followed links there and at the check's name). As 32c (agentcc_cache.sh) with:
#  |  - agentcc_tasks5.py (frozen, sha-checked copy in the run folder; = tasks4 + v5: after claude ends, the sandboxed
#  |    processes still in the task's folders are killed (lingering.jsonl), and the check runs on the runner's copy of the
#  |    task folder, <run>/checks/<id>, under <run>/sandbox_check_<id>.sb whose only writable folder is that copy):
#  |    one profile per task (<run>/sandbox_<id>.sb, kept):
#  |    WRITES only in the task's own folder, its config folder and its temp folder <run>/cctmp/<id> (+ /private/var/folders
#  |    and /dev from the base), READS denied in ~/EvalSuite/agentcc and in <run> outside those and the read-only HOME;
#  |    CLAUDE_CODE_TMPDIR=<run>/cctmp/<id> and TMPPREFIX=<run>/cctmp/<id>/zsh per task; runner outputs opened with
#  |    O_NOFOLLOW (O_EXCL when new); the check copy removes any entry at its name first; CLAUDE_BIN from the environment;
#  |  - the base sandbox.sb names no part of <run> (agentcc_tasks5.py refuses one that does);
#  |  - Claude Code pinned to 2.1.321: CLAUDE_BIN = Leap's kept copy (logs/claude_versions/2.1.321, sha256 abc9feee...,
#  |    323,686,048 bytes; the installed claude updated itself to 2.1.322 on 11-11 08:15); a guard stops the tasks if t01's
#  |    init event is not claude_code_version 2.1.321; every transcript's version is counted at the end;
#  |  - the server files and launcher of the 80c8041 pin: the cache clone's serve.py, agentio.py, engine/loop/convcache.py and
#  |    tools/codex_local.sh checked against 80c8041's release/next/SHA256SUMS, and the launcher run from
#  |    tools/claude_local.sh.80c8041 (e5b900fa; the clone's tools/claude_local.sh is now release/next's newer one);
#  |  - CONV_CACHE=1, CONV_CACHE_GB=2.5 (set explicitly; 32c had it as serve.py's default), CONV_CACHE_BRANCH=1;
#  |  - NO-SWAP >= 45 GB, the swap guard (+2 GB stops everything), the neutral HOME, CLAUDE_CODE_TMPDIR, the same base
#  |    sandbox.sb, probe_cc.py unchanged.
# Starts only when <run>/GO exists (created once EvalSuite's dated line is in), from FROM (the orchestrator's window), on
# a free Mac (Kamen idle >= 600 s, no other lock / engine, no PAUSE / HOLD, 2 clear polls); last start LAST.
# DRYRUN=1: write <run>/provenance.txt with every pin check, then exit (no gates, no server).
# Usage: [DRYRUN=1] agentcc_cache4.sh <fresh run dir> "<FROM YYYY-MM-DD HH:MM:SS>" "<LAST ...>" [pid ...]
RUN=$1; FROMS=$2; LASTS=$3; shift 3; AFTER="$*"; GB=${GB:-32}; NEED=${NEED:-40}; ARM=${ARM:-r120}
[ -e "$RUN" ] && { echo "agentcc_r120.sh: $RUN exists; use a fresh folder"; exit 2; }
case $GB in 24) CTX=131072;; 32|45) CTX=65536;; *) echo "agentcc_r120.sh: GB 45, 32 or 24"; exit 2;; esac   # = tools/claude_code.sh
unset CONV_CACHE CONV_CACHE_GB CONV_CACHE_BRANCH CONV_CACHE_CHECK CONV_CACHE_GRID R57_PFLM BIGMOE_IDLE_MS EXACTSTREAM_CONTEXT   # as shipped
A=/private/tmp/claude-501/-Users-kamen-OjoX/f2971937-6c95-4e86-a049-05beff6a8e32/scratchpad/agent
L=/Users/kamen/Leap; CL=$L/release/cleanclone/v120b/clone; MAN=/Users/kamen/EvalSuite/agentcc/manifest.json
# 5d: the candidate pkg_120b, with serve.py's Host / Origin check
PKG=$L/release/out/pkg_120b; PIN=21fb59b6bfd980e9c7aec0bfb8e3094c553b92b7; TASKS_SRC=$L/data/agent_cc_1111/profile/agentcc_tasks5.py
TASKS_SHA=28caec4c6f92468cdf00884483a0c12417c7667a7a61b5bc0cf4ec96dc7989ef
CLAUDE_PIN=$L/logs/claude_versions/2.1.322; CLAUDE_SHA=a9d237949db816a0b7afd88a9a8e1df36b3a597a30f5a41d1c0539eb180a3450; CLAUDE_SIZE=323702464; CCV=2.1.322
LAUNCH=$CL/tools/claude_local.sh; PROBE=/Users/kamen/EvalSuite/tools/probe_cc.py
PORT=8122; mkdir -p $RUN $RUN/cctmp; LOG=$RUN/runner.log
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
sha() { shasum -a 256 "$1" | cut -d' ' -f1; }
initver() { head -1 "$1" 2>/dev/null | grep -o '"claude_code_version":"[^"]*"' | cut -d'"' -f4; }
LOCKED=0; SP=""; GUARD=""; TP=""
stopsrv() { [ -z "$SP" ] && return; kill -TERM $SP 2>/dev/null
            for i in $(seq 1 60); do kill -0 $SP 2>/dev/null || break; sleep 1; done
            kill -0 $SP 2>/dev/null && { say "serve.py did not stop in 60 s: SIGKILL"; kill -KILL $SP; }
            lsof -nP -iTCP:$PORT -sTCP:LISTEN >/dev/null 2>&1 && say "WARNING: port $PORT still in use"; SP=""; }
cleanup() { [ -n "$GUARD" ] && kill $GUARD 2>/dev/null
            [ -n "$TP" ] && kill -0 $TP 2>/dev/null && { for c in $(desc $TP); do kill -TERM $c 2>/dev/null; done; kill -TERM $TP; }
            stopsrv; [ $LOCKED = 1 ] && peer unlock >/dev/null 2>&1; touch $RUN/runner.done; say "exit (NO-SWAP $(noswap) GB, swap $(swap) MB)"; }
# provenance + every pin check; any mismatch stops the run before the server starts
provenance() {
  local P=$RUN/provenance.txt f a b
  git -C $PKG show $PIN:MANIFEST.json > $RUN/MANIFEST.$PIN.json || { say "STOP: no MANIFEST.json at $PIN"; return 3; }
  [ "$(git -C $CL rev-parse HEAD)" = "$PIN" ] || { say "STOP: the clone $CL is not at $PIN"; return 3; }
  RC=$(/opt/homebrew/opt/python@3.12/bin/python3.12 -I $L/data/pkg120_1112/clone_rootcheck.py $CL) || { say "STOP: the clone has other changes than install.sh's @ROOT@ rewrite: $RC"; return 3; }
  echo "pin: package candidate $PIN ($(git -C $PKG log -1 --format='%s' $PIN | cut -c1-100)); clone $CL at HEAD $PIN; $RC" > $P
  for f in serve.py agentio.py run.py engine/loop/convcache.py tools/claude_local.sh tools/claude_code.sh tools/statusline.sh tools/codex_local.sh configs/chat-$GB.env; do
    a=$(sha $CL/$f); b=$(/opt/homebrew/opt/python@3.12/bin/python3.12 -I -c 'import json, sys; print(json.load(open(sys.argv[1]))["files"][sys.argv[2]]["sha256"])' $RUN/MANIFEST.$PIN.json $f 2>/dev/null)
    [ -n "$b" ] && [ "$a" = "$b" ] && echo "clone $f = $PIN's ($a)" >> $P || { say "STOP: clone $f is not $PIN's"; return 3; }
  done
  [ "$LAUNCH" = "$CL/tools/claude_local.sh" ] && echo "launcher: $LAUNCH (as shipped, checked above)" >> $P || { say "STOP: the launcher is not the clone's"; return 3; }
  cp -p $TASKS_SRC $RUN/agentcc_tasks5.py; a=$(sha $RUN/agentcc_tasks5.py)
  [ "$a" = "$TASKS_SHA" ] && echo "tasks script: $RUN/agentcc_tasks5.py (frozen copy) sha256 $a" >> $P || { say "STOP: agentcc_tasks5.py sha256 $a is not the registered one"; return 3; }
  a=$(sha $CLAUDE_PIN); b=$(stat -f %z $CLAUDE_PIN)
  [ "$a" = "$CLAUDE_SHA" ] && [ "$b" = "$CLAUDE_SIZE" ] || { say "STOP: the pinned claude binary does not match (sha256 $a, $b bytes)"; return 3; }
  echo "CLAUDE_BIN=$CLAUDE_PIN sha256 $a ($b bytes)" >> $P
  echo "\$CLAUDE_BIN --version: $(CLAUDE_CONFIG_DIR=$RUN/cc_version $CLAUDE_PIN --version 2>&1 | head -1)" >> $P
  echo "probe: $PROBE sha256 $(sha $PROBE); manifest sha256 $(sha $MAN | cut -c1-12); setting $GB GB; port $PORT; date $(date '+%F %T')" >> $P
  echo "server: serve.py --mem $GB --port $PORT with the package's own defaults: conversation cache on at 45 / 32 GB and off at 24 (CONV_CACHE unset), CONV_CACHE_GB 2.5 at 45 / 32 (unset), CONV_CACHE_BRANCH 1 (unset), PF-A R57_PFLM=1 at 32 / 24 (unset), BIGMOE_IDLE_MS 50 (unset); set by the runner for its records only: CONV_CACHE_LOG=$RUN/convcache.log, EXACTSTREAM_LOG_REQUESTS=$RUN/server_requests.jsonl" >> $P
  echo "claude: EXACTSTREAM_CONTEXT=$CTX (tools/claude_code.sh's value at --mem $GB), read by the shipped claude_local.sh: CLAUDE_CODE_MAX_CONTEXT_TOKENS=$CTX$([ $CTX -ge 100000 ] && echo ", CLAUDE_CODE_AUTO_COMPACT_WINDOW=$CTX" || echo ", no AUTO_COMPACT_WINDOW")" >> $P
  echo "runner environment, cache / engine / window variables: [$(env | grep -E '^(CONV_CACHE|R57_|BIGMOE_|EXACTSTREAM_)' | tr '\n' ' ')]" >> $P
  echo "per task (set by agentcc_tasks5.py in claude's environment): CLAUDE_CODE_TMPDIR=$RUN/cctmp/<id>, TMPPREFIX=$RUN/cctmp/<id>/zsh" >> $P
  echo "per-task WRITE set for claude (the profile <run>/sandbox_<id>.sb, kept): $RUN/tasks/<id>, $RUN/cc_home/<id>, $RUN/cctmp/<id>, /private/var/folders, /dev/* (regex ^/dev/); nothing else (HOME $RUN/home read-only)" >> $P
  echo "per-task READ denies (file contents and listings): ~/EvalSuite/agentcc, the manifest's folder, and $RUN except $RUN/tasks/<id>, $RUN/cc_home/<id>, $RUN/cctmp/<id>, $RUN/home" >> $P
  echo "after claude ends: sandboxed processes with a cwd or open file in the task's folders are killed (logged to $RUN/lingering.jsonl)" >> $P
  echo "the check: on the runner's copy $RUN/checks/<id> (links copied as links, other file types skipped), placed through a directory fd, run with $RUN/sandbox_check_<id>.sb (kept): WRITE set $RUN/checks/<id>, /private/var/folders, /dev/* only; READ denies as above except that copy and HOME" >> $P
  mkdir -p $RUN/home && cp -p $A/agent_home_tpl/.zshrc $A/agent_home_tpl/.gitconfig $RUN/home/
  echo "claude HOME: $RUN/home (.zshrc sha256 $(sha $RUN/home/.zshrc | cut -c1-12): only the exact PATH; .gitconfig: agentcc@localhost)" >> $P
  grep "^export PATH=" $RUN/home/.zshrc | sed "s/^/  /" >> $P
  echo "git identity under that HOME: $(HOME=$RUN/home git config --global user.name) <$(HOME=$RUN/home git config --global user.email)>" >> $P
}
if [ "${DRYRUN:-0}" = 1 ]; then provenance; rc=$?; say "DRYRUN: provenance rc $rc"; cat $RUN/provenance.txt; exit $rc; fi
trap cleanup EXIT
trap 'say "stopped by a signal"; exit 1' INT TERM
say "armed (pid $$): $ARM (agentcc $GB GB, package candidate ${PIN:0:7} as shipped, per-task read + write profile, Claude Code $CCV) -> $RUN; after [$AFTER], needs $RUN/GO; from $FROMS, last start $LASTS, NO-SWAP >= $NEED"
CLR=0; NOTE=""
while :; do
  now=$(date +%s); [ $now -ge $LAST ] && { say "SKIP: no free window by $LASTS (last wait: $NOTE)"; exit 0; }
  why=""
  [ ! -e $RUN/GO ] && why="no GO file (EvalSuite's dated line)"
  [ -z "$why" ] && [ $now -lt $FROM ] && why="before $FROMS"
  [ -z "$why" ] && for w in $AFTER; do kill -0 $w 2>/dev/null && why="waits for pid $w"; done
> (a quote from the project owner, kept private)
  [ -z "$why" ] && [ -n "$(procs)" ] && why="an engine / chat / server runs"
  [ -z "$why" ] && [ "$(busy)" -gt 0 ] && why="another lock"
  [ -z "$why" ] && held && why="PAUSE / HOLD"
  [ -z "$why" ] && [ "$(noswap)" -lt $NEED ] && why="NO-SWAP $(noswap) < $NEED"
  if [ -z "$why" ]; then CLR=$((CLR + 1)); else CLR=0; fi
  [ -z "$why" ] && [ $CLR -lt 2 ] && why="clear once"
  if [ -z "$why" ]; then
    peer lock share 300m --cpu 400 --gpu 100 "MEM $NEED GB: Leap Claude Code agent test $ARM (1.2 release gate, $GB GB, EvalSuite PREREG_agentcc amendment 5)" > /dev/null 2>&1 && LOCKED=1
    sleep 20
    [ "$(busy)" = 0 ] && [ -z "$(procs)" ] && break
    peer unlock >/dev/null 2>&1; LOCKED=0; CLR=0; why="after the lock"
  fi
  [ "$why" != "$NOTE" ] && say "wait: $why"; NOTE=$why
  sleep 60
done
S0=$(swap); say "GO: NO-SWAP $(noswap) GB, swap $S0 MB"
( while :; do [ $(( $(swap) - S0 )) -gt 2048 ] && { echo "$(date +%T) SWAP GUARD: $(swap) MB" >> $LOG; touch $RUN/SWAP_GUARD; kill -TERM $$; exit; }; sleep 5; done ) &
GUARD=$!
provenance || exit 3
H=/Users/kamen
cat > $RUN/sandbox.sb <<SB
(version 1)
(allow default)
(deny network-outbound)
(allow network-outbound (remote ip "localhost:*"))
(allow network-outbound (remote unix-socket))
(deny file-write*)
(allow file-write* (subpath "/private/var/folders") (regex #"^/dev/"))
(deny file-read* (subpath "$H/.ssh") (subpath "$H/.aws") (subpath "$H/.gnupg") (regex #"^/Users/[^/]+/\.claude[^/]*(/|$)")
  (subpath "$H/.codex") (subpath "$H/.config/gh") (literal "$H/.netrc") (literal "$H/.git-credentials")
  (subpath "$H/Library/Keychains") (subpath "$H/Library/Mail") (subpath "$H/Library/Messages") (subpath "$H/Library/Cookies")
  (subpath "$H/Library/Safari") (subpath "$H/Library/Application Support/Google") (subpath "$H/Library/Application Support/Claude")
  (subpath "$H/Documents") (subpath "$H/Desktop") (subpath "$H/Downloads") (subpath "$H/Papers") (subpath "$H/.local/share/exactstream-sessions"))
SB
( cd $CL && CONV_CACHE_LOG=$RUN/convcache.log EXACTSTREAM_LOG_REQUESTS=$RUN/server_requests.jsonl exec .venv/bin/python serve.py --mem $GB --port $PORT ) > $RUN/serve.log 2>&1 &
SP=$!; t0=$(date +%s)
until curl -s -m 2 http://127.0.0.1:$PORT/health 2>/dev/null | grep -q ready; do
  kill -0 $SP 2>/dev/null && [ $(( $(date +%s) - t0 )) -lt 600 ] || { say "serve.py did not come up (tail: $(tail -2 $RUN/serve.log | tr '\n' ' '))"; exit 2; }; sleep 2
done
say "serve.py --mem $GB ready in $(( $(date +%s) - t0 )) s (pid $SP)"
P0=$(date +%s); python3 $PROBE $PORT $RUN/probe_cc.jsonl > $RUN/probe_cc.out 2>&1 &
TP=$!; while kill -0 $TP 2>/dev/null; do [ $(( $(date +%s) - P0 )) -gt 7200 ] && { say "probe: 2-h cap"; for c in $(desc $TP); do kill -TERM $c; done; kill -TERM $TP; }; sleep 5; done
wait $TP; say "probe rc $? in $(( $(date +%s) - P0 )) s: $(tail -1 $RUN/probe_cc.out | cut -c1-160)"; TP=""
export CLAUDE_BIN=$CLAUDE_PIN EXACTSTREAM_CONTEXT=$CTX
HOME=$RUN/home /opt/homebrew/opt/python@3.12/bin/python3.12 -I $RUN/agentcc_tasks5.py $RUN $GB $MAN $PORT $RUN/sandbox.sb $A/cc_home $LAUNCH > $RUN/tasks.out 2>&1 &
TP=$!; G1=0
while kill -0 $TP 2>/dev/null; do
  if [ $G1 = 0 ] && [ -s $RUN/results.jsonl ]; then
    G1=1; f=$(ls $RUN/t01_*.jsonl 2>/dev/null | head -1); v=$(initver "$f")
    if [ -n "$f" ] && grep -q "EPERM: operation not permitted, mkdir" "$f"; then
      say "STOP: t01's transcript holds the EPERM (the Bash fix did not take): stopping the tasks"
      for c in $(desc $TP); do kill -TERM $c 2>/dev/null; done; kill -TERM $TP; break
    fi
    if [ "$v" != "$CCV" ]; then
      say "STOP: t01's init event reads claude_code_version '$v', not $CCV: stopping the tasks"
      for c in $(desc $TP); do kill -TERM $c 2>/dev/null; done; kill -TERM $TP; break
    fi
    say "  t01 done: claude_code_version $v; $(head -1 $RUN/results.jsonl)"
  fi
  sleep 5
done
wait $TP 2>/dev/null; say "tasks: $(grep -c . $RUN/results.jsonl 2>/dev/null) results; checks passed $(grep -c '"check_rc": 0' $RUN/results.jsonl 2>/dev/null); $(tail -1 $RUN/tasks.out)"; TP=""
n=0; ok=0; for f in $RUN/t[0-9][0-9]_*.jsonl; do [ -e "$f" ] || continue; n=$((n + 1)); [ "$(initver "$f")" = "$CCV" ] && ok=$((ok + 1)); done
say "transcripts with claude_code_version $CCV: $ok of $n; per-task profiles kept: $(ls $RUN/sandbox_t*.sb 2>/dev/null | wc -l | tr -d ' '); symlinks in the run folder: $(find $RUN -type l 2>/dev/null | wc -l | tr -d ' ')"
find $RUN -type l 2>/dev/null | sed 's/^/  link: /' >> $LOG
say "lingering processes killed: $(grep -c . $RUN/lingering.jsonl 2>/dev/null || echo 0) task lines"
say "server: $(grep -c 'done:' $RUN/serve.log) requests; errors $(grep -c -E 'Traceback|Error' $RUN/serve.log); cache hits $(grep -c ' hit ' $RUN/convcache.log 2>/dev/null) of $(grep -cE ' (hit|miss) ' $RUN/convcache.log 2>/dev/null)"
stopsrv; say "serve.py stopped (NO-SWAP $(noswap) GB)"
