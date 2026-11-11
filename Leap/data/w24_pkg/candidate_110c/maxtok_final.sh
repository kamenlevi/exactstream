#!/bin/bash
# Apply the 65,536 default to the local clone Kamen chats from (release/cleanclone/final/clone) once he is done: no chat.py /
# serve.py / run.py running from that clone on 2 polls in a row, 60 s apart. Then maxtok_apply.py (exact lines, atomic
# renames) and a --help check. Gives up at 11-11 12:00.
A=/private/tmp/claude-501/-Users-kamen-OjoX/f2971937-6c95-4e86-a049-05beff6a8e32/scratchpad/agent
F=/Users/kamen/Leap/release/cleanclone/final/clone; LOG=$A/maxtok_final.log
END=$(date -j -f "%Y-%m-%d %H:%M:%S" "2026-11-11 12:00:00" +%s)
say() { echo "$(date '+%m-%d %H:%M:%S') $*" | tee -a $LOG; }
users() { for p in $(ps -axo pid=,args= | awk '/[Pp]ython/ && /[ \/](chat|serve|run)[.]py( |$)/ && !/ awk / {print $1}'); do
            lsof -a -p $p -d cwd -Fn 2>/dev/null | sed -n 's/^n//p' | grep -q "^$F" && printf "%s " $p; done; }
say "armed (pid $$): waits until nothing runs from $F"
CLR=0; NOTE=""
while :; do
  [ "$(date +%s)" -ge $END ] && { say "GAVE UP at 12:00: still in use ($(users))"; exit 1; }
  u=$(users)
  if [ -z "$u" ]; then CLR=$((CLR + 1)); else CLR=0; fi
  [ $CLR -ge 2 ] && break
  w="in use: ${u:-clear once}"; [ "$w" != "$NOTE" ] && say "wait: $w"; NOTE=$w
  sleep 60
done
say "free: applying"
/opt/homebrew/opt/python@3.12/bin/python3.12 -I $A/maxtok_apply.py $F 2>&1 | while read l; do say "  $l"; done
say "check: chat.py --help: $(cd $F && .venv/bin/python chat.py --help 2>/dev/null | tr -s ' \n' ' ' | grep -o 'default 65536[^)]*' | head -1)"
say "check: serve.py --help: $(cd $F && .venv/bin/python serve.py --help 2>/dev/null | tr -s ' \n' ' ' | grep -o 'default 65536[^)]*' | head -1)"
say "done"
