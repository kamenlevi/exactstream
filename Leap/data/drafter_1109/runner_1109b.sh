#!/bin/bash
# Leap, night of Mon 9 -> Tue 10 Nov (orchestrator 23:2x on Kamen's yes; NewIdeas 23:12 keeps the order: W24c, then
# Breakthrough's q1u_0_live and Fit12's S1p, then Leap; everything done and freed by 02:05, the gate at 02:10):
#   1. the 32 GB drafter pair (dfl, mtp; one 2,048-token code request through chat.py) -> ~/Leap/data/drafter_1109.done
#   2. the 32 GB prefill test (10K / 30K tokens x default / PF-A / PF-B, run.py)        -> ~/Leap/data/prefill_1109.done
# Start gate, polled every 30 s, all at once:
#   - order: ~/Breakthrough/data/m1/q1u_0_live.done touched after 22:30, or from 01:35 the fallback; both need no python
#     run_map.py / dm_oea_masks.py / fit_onecall.py process and no other session's MEM > 5 GB lock or >= 50 % GPU share;
#   - no quiet lock held or queued by others; no PAUSE / HOLD in ~/.local/share/quiet; lid open;
#   - Kamen idle: no chat.py / serve.py of his, HIDIdleTime >= 600 s;
#   - Chrome closed (orchestrator: "NO-SWAP needs 39 GB, so they need Chrome closed; skip and report if it isn't"):
#     still open 10 min after everything else passed = SKIP;
#   - NO-SWAP (free + speculative + purgeable + file-backed) >= 39 GB on 3 reads 10 s apart;
#   - start by 01:58 (the pair ends by 02:05), else SKIP.
# Then a share lock ('MEM 38 GB' first in the reason) to 02:05, a 20 s settle and re-check, the two benches (each run has
# its own last start, memory check and cap inside), peer unlock. The done-files are touched on every exit (EXIT trap).
# DRY=1: print one gate evaluation and exit (no lock, no run, no done-files).
# v2 (23:5x): at 23:50 v1 took its lock on Breakthrough's done-file, written by mistake and withdrawn at 23:53. W24d's quiet
# lock sent it back after the 20 s settle, but its 'MEM 38 GB' lock met S1p's lock in the same second and sent S1p back too.
# So v2 also waits for Fit12's S1p (~/Fit12/data/scr/s1p/DONE, written at its end or at its 01:30 'not started'; or from
# 01:31 with no fit_onecall.py), needs the gate clear on 2 polls in a row, and re-checks at +20 s and +40 s after the lock,
# GPU share included.
SP=/private/tmp/claude-501/-Users-kamen-OjoX/f2971937-6c95-4e86-a049-05beff6a8e32/scratchpad
N=$SP/night; LOG=$N/runner.log; mkdir -p $N
ep() { date -j -f "%Y-%m-%d %H:%M:%S" "$1" +%s; }
T2300=$(ep "2026-11-09 23:00:00"); T2230=$(ep "2026-11-09 22:30:00"); FALLB=$(ep "2026-11-10 01:35:00")
LAST=$(ep "2026-11-10 01:58:00"); END=$(ep "2026-11-10 02:05:00"); S1PDL=$(ep "2026-11-10 01:31:00")
DQ=/Users/kamen/Breakthrough/data/m1/q1u_0_live.done; DS=/Users/kamen/Fit12/data/scr/s1p/DONE
say() { echo "$(date +%T) $*" | tee -a $LOG; }
noswap() { vm_stat | awk '/page size of/ {ps=$8} /^Pages (free|speculative|purgeable):/ || /^File-backed pages:/ {v=$NF; sub("[.]","",v); s+=v} END {printf "%d", s*ps/1e9}'; }
swap() { sysctl -n vm.swapusage | sed -E 's/.*used = ([0-9.]+)M.*/\1/' | cut -d. -f1; }
idle() { ioreg -c IOHIDSystem | awk '/HIDIdleTime/ {print int($NF/1000000000); exit}'; }
hischat() { ps -axo pid=,comm=,args= | awk '$2 ~ /[Pp]ython/ && /[ \/](chat|serve)[.]py( |$)/ && !/--max-tokens 2048/ {printf "%s ", $1}'; }
pyjobs() { ps -axo pid=,comm=,args= | awk '$2 ~ /[Pp]ython/ && /run_map[.]py|dm_oea_masks[.]py|fit_onecall[.]py/ {printf "%s ", $1}'; }
others() { peer ls 2>/dev/null | grep "^- "; }
locks() { grep -oE "· (shared run \(others may work\)|benchmarking|waiting for a quiet window)[^·]*"; }
quiet_n() { others | locks | grep -cE "^· (benchmarking|waiting for a quiet window)"; }
bigmem_n() { others | locks | grep -oE "MEM [0-9]+ GB" | awk '$2 > 5 {n++} END {print n + 0}'; }
gpu_n() { others | sed -n 's/.*shared run[^:]*: about [0-9]*% CPU, \([0-9]*\)% GPU.*/\1/p' | awk '$1 >= 50 {n++} END {print n + 0}'; }
held() { [ -e ~/.local/share/quiet/PAUSE ] || [ -e ~/.local/share/quiet/HOLD ]; }
lid_closed() { ioreg -r -k AppleClamshellState -d 1 | grep -q '"AppleClamshellState" = Yes'; }
chrome() { pgrep -x "Google Chrome" >/dev/null; }
qdone() { [ -f $DQ ] && [ "$(stat -f %m $DQ)" -ge "$T2230" ]; }
sdone() { [ -f $DS ] && [ "$(stat -f %m $DS)" -ge "$T2300" ]; }
recheck() {   # after the lock: prints what is wrong now; empty = still clear
  local w=""
  [ "$(noswap)" -lt 39 ] && w="NO-SWAP $(noswap) GB"
  [ -n "$(pyjobs)" ] && w="$w; running: python $(pyjobs)"
  [ "$(quiet_n)" -gt 0 ] && w="$w; a quiet lock"
  [ "$(bigmem_n)" -gt 0 ] && w="$w; a MEM > 5 GB lock"
  [ "$(gpu_n)" -gt 0 ] && w="$w; a >= 50 % GPU shared run"
  echo "$w"
}
killtree() { for c in $(pgrep -P $1); do killtree $c $2; done; kill -$2 $1 2>/dev/null; }
gate() {   # prints why it can't start now; empty = clear (the Chrome and NO-SWAP checks come after this)
  local now=$(date +%s)
  [ $now -lt $T2300 ] && { echo "before 23:00"; return; }
  qdone || [ $now -ge $FALLB ] || { echo "waiting for q1u_0_live.done (fallback 01:35)"; return; }
  sdone || [ $now -ge $S1PDL ] || { echo "waiting for Fit12's S1p DONE (fallback 01:31)"; return; }
  [ -n "$(pyjobs)" ] && { echo "running: python $(pyjobs)"; return; }
  [ "$(quiet_n)" -gt 0 ] && { echo "a quiet lock is held or queued"; return; }
  [ "$(bigmem_n)" -gt 0 ] && { echo "another session holds a MEM > 5 GB lock"; return; }
  [ "$(gpu_n)" -gt 0 ] && { echo "another shared run uses >= 50 % GPU"; return; }
  held && { echo "PAUSE / HOLD"; return; }
  lid_closed && { echo "lid closed"; return; }
> (a quote from the project owner, kept private)
> (a quote from the project owner, kept private)
}
if [ "${DRY:-0}" = 1 ]; then
  g=$(gate); echo "gate: ${g:-clear}; q1u_0_live.done: $(qdone && echo yes || echo no); S1p DONE: $(sdone && echo yes || echo no); chrome: $(chrome && echo open || echo closed);" \
    "NO-SWAP $(noswap) GB; swap $(swap) MB; idle $(idle) s; quiet $(quiet_n), MEM>5 $(bigmem_n), GPU>=50 $(gpu_n); pyjobs '$(pyjobs)'; hischat '$(hischat)'"
  exit 0
fi
LOCKED=0; BP=""
finish() {
  [ -n "$BP" ] && kill -0 $BP 2>/dev/null && killtree $BP TERM
  [ $LOCKED = 1 ] && { peer unlock >/dev/null 2>&1; say "unlocked"; }
  touch /Users/kamen/Leap/data/drafter_1109.done /Users/kamen/Leap/data/prefill_1109.done $N/runner.end
  say "exit: done-files touched; NO-SWAP $(noswap) GB, swap $(swap) MB"
}
trap finish EXIT
trap 'say "runner stopped by a signal"; exit 1' INT TERM
> (a quote from the project owner, kept private)
CT=0; NOTE=""; CLR=0
while :; do
  now=$(date +%s)
  if [ $now -ge $LAST ]; then say "SKIP: no start by 01:58 (last wait: ${NOTE:-none})"; echo "SKIP: no start by 01:58 (${NOTE})" > $N/result; exit 0; fi
  why=$(gate)
  if [ -z "$why" ]; then CLR=$((CLR + 1)); else CLR=0; fi
  [ -z "$why" ] && [ $CLR -lt 2 ] && why="clear once: one more poll"
  if [ -z "$why" ] && chrome; then
    [ $CT = 0 ] && { CT=$now; say "everything else passed; Chrome is open (NO-SWAP $(noswap) GB): SKIP if still open at $(date -r $((now + 600)) +%H:%M)"; }
    if [ $((now - CT)) -ge 600 ]; then say "SKIP: Chrome still open (NO-SWAP $(noswap) GB)"; echo "SKIP: Chrome open (NO-SWAP $(noswap) GB)" > $N/result; exit 0; fi
    why="Chrome open"
  fi
  if [ -z "$why" ]; then
    a=$(noswap); sleep 10; b=$(noswap); sleep 10; c=$(noswap); m=$(printf "%s\n%s\n%s\n" $a $b $c | sort -n | head -1)
    [ "$m" -lt 39 ] && why="NO-SWAP $a / $b / $c GB < 39"
  fi
  if [ -z "$why" ]; then
    M=$(( (END - $(date +%s)) / 60 )); [ $M -lt 5 ] && M=5
    out=$(peer lock share ${M}m --cpu 300 --gpu 100 "MEM 38 GB: Leap 32 GB drafter pair + prefill 10K/30K (chat-32); ends by 02:05" 2>&1); rc=$?
    say "share lock (${M}m) rc $rc: $(echo "$out" | tr '\n' ' ' | cut -c1-400)"
    if [ $rc = 0 ]; then
      LOCKED=1; sleep 20; w2=$(recheck)
      [ -z "$w2" ] && { sleep 20; w2=$(recheck); }
      [ -z "$w2" ] && break
      say "after the lock: $w2: unlock and wait"; peer unlock >/dev/null 2>&1; LOCKED=0; CLR=0; why="after the lock: $w2"
    else
      why="share lock refused"
    fi
  fi
  [ "$why" != "$NOTE" ] && say "wait: $why"; NOTE=$why
  sleep 30
done
say "GO: NO-SWAP $(noswap) GB, swap $(swap) MB, idle $(idle) s, q1u_0_live.done $(qdone && echo yes || echo 'no (fallback)')"
bash $SP/codebench/bench_code.sh $SP/codebench >> $LOG 2>&1 &
BP=$!; wait $BP; BP=""
touch /Users/kamen/Leap/data/drafter_1109.done $N/drafter.end; say "drafter pair ended"
bash $SP/prefillbench/bench_prefill.sh $SP/prefillbench >> $LOG 2>&1 &
BP=$!; wait $BP; BP=""
touch /Users/kamen/Leap/data/prefill_1109.done; say "prefill test ended"
peer unlock >/dev/null 2>&1; LOCKED=0; say "unlocked"
echo "DONE" > $N/result
