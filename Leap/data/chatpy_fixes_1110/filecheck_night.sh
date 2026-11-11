#!/bin/bash
# Kamen's clone gets chat.py (built-in no-tools system line; /file = the file, then the request) when his chat (pid $1) has
# exited and nothing runs from the clone (2 polls, 5 s apart); then the real-engine /file check there: chat.py --kind code
# --mem 32 --max-tokens 600 with his 22:27 request on tetris_fix3.html, piped; PASS = code in the answer and no made-up
# tool calls. NO-SWAP >= 38, share lock, 300-s cap. Usage: chatfix_now2.sh <pid of the running chat>
A=/private/tmp/claude-501/-Users-kamen-OjoX/f2971937-6c95-4e86-a049-05beff6a8e32/scratchpad/agent
F=/Users/kamen/Leap/release/cleanclone/v110d/clone; LOG=$A/filecheck_night.log; OLD=0
say() { echo "$(date '+%m-%d %H:%M:%S') $*" | tee -a $LOG; }
noswap() { vm_stat | awk '/page size of/ {ps=$8} /^Pages (free|speculative|purgeable):/ || /^File-backed pages:/ {v=$NF; sub("[.]","",v); s+=v} END {printf "%d", s*ps/1e9}'; }
users() { for p in $(ps -axo pid=,args= | awk '/[Pp]ython/ && /[ \/](chat|serve|run)[.]py( |$)/ && !/ awk / {print $1}'); do
            lsof -a -p $p -d cwd -Fn 2>/dev/null | sed -n 's/^n//p' | grep -q "^$F" && printf "%s " $p; done; }
desc() { local c; for c in $(pgrep -P "$1"); do desc "$c"; echo "$c"; done; }
say "armed (pid $$): the real /file check in the v110d clone (release/src chat.py) when the Mac is free; last start 11-11 07:00"
idle() { ioreg -c IOHIDSystem | awk '/HIDIdleTime/ {print int($NF/1000000000); exit}'; }
anyeng() { ps -axo pid=,args= | awk '/[Pp]ython/ && /r57_screen_|run_map[.]py|fit_onecall[.]py|[ \/](chat|serve|run)[.]py( |$)/ && !/ awk / {printf "%s ", $1}'; }
busy() { peer ls 2>/dev/null | grep "^- " | grep -oE "· (shared run \(others may work\)|benchmarking|waiting for a quiet window)[^·]*" | grep -cE "^· (benchmarking|waiting for a quiet window)|MEM ([6-9]|[1-9][0-9]) GB"; }
LAST=$(date -j -f "%Y-%m-%d %H:%M:%S" "2026-11-11 12:00:00" +%s); CLR=0
while :; do
  [ "$(date +%s)" -ge $LAST ] && { say "SKIP: no free window by 07:00"; exit 0; }
  if [ "$(idle)" -ge 600 ] && [ -z "$(anyeng)" ] && [ "$(busy)" = 0 ] && [ ! -e ~/.local/share/quiet/PAUSE ] && [ ! -e ~/.local/share/quiet/HOLD ] && [ "$(noswap)" -ge 40 ]; then CLR=$((CLR + 1)); else CLR=0; fi
  [ $CLR -ge 2 ] && break
  sleep 60
done
NS=$(noswap)
if [ -n "$(users)" ]; then say "real-engine /file check SKIPPED: a chat started from the clone already ($(users))"; exit 0; fi
if [ "$NS" -lt 38 ]; then say "real-engine /file check SKIPPED: NO-SWAP $NS GB < 38"; exit 0; fi
> (a quote from the project owner, kept private)
say "real-engine /file check: NO-SWAP $NS GB; chat.py --kind code --mem 32 --max-tokens 600"
( cd $F && exec .venv/bin/python chat.py --kind code --mem 32 --max-tokens 600 < $A/filecheck_in.txt ) > $A/filecheck_real.out 2>&1 &
P=$!; T0=$(date +%s); SW0=$(sysctl -n vm.swapusage | sed -E 's/.*used = ([0-9.]+)M.*/\1/' | cut -d. -f1)
while kill -0 $P 2>/dev/null; do
  WHY=""
  [ $(( $(date +%s) - T0 )) -gt 300 ] && WHY="cap 300 s"
> (a quote from the project owner, kept private)
  [ $(( $(sysctl -n vm.swapusage | sed -E 's/.*used = ([0-9.]+)M.*/\1/' | cut -d. -f1) - SW0 )) -gt 1024 ] && WHY="swap +1 GB"
  [ -n "$WHY" ] && { say "  STOPPED: $WHY"; for c in $(desc $P); do kill -TERM $c 2>/dev/null; done; kill -TERM $P; break; }
  sleep 2
done
wait $P; RC=$?
peer unlock > /dev/null 2>&1
O=$A/filecheck_real.out
say "  rc $RC in $(( $(date +%s) - T0 )) s; $(grep -a '(/file:' $O | head -1 | cut -c1-80); $(grep -a 'tokens | whole decode' $O | tail -1 | tr -d '[]' | cut -c1-150)"
TOOLS=$(grep -a -c -E '<parameter|<tool_call>|<function=|exploring the project|ls -la' $O); CODE=$(grep -a -c -E '```html|<!DOCTYPE|<html|function |const |ctx\.' $O)
say "  made-up tool-call lines: $TOOLS; code lines: $CODE -> $([ $TOOLS = 0 ] && [ $CODE -gt 0 ] && echo PASS || echo FAIL)"
say "  answer starts: $(awk '/tokens\)/{f=1; next} f' $O | grep -a -v '^> *$\|^$' | head -3 | tr -d '\r' | tr '\n' ' ' | cut -c1-220)"
say "done (NO-SWAP $(noswap) GB)"
