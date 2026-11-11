#!/bin/bash
# Kamen's clone gets the fixed chat.py / serve.py the moment his old chat is gone (pid $1, plus nothing else running from
# the clone, 2 polls 5 s apart); then one short real-engine start of chat.py there (piped: one message, 16 tokens, /exit),
# if NO-SWAP >= 36, under a share lock. Usage: chatfix_now.sh <pid of the old chat>
A=/private/tmp/claude-501/-Users-kamen-OjoX/f2971937-6c95-4e86-a049-05beff6a8e32/scratchpad/agent
F=/Users/kamen/Leap/release/cleanclone/final/clone; S=/Users/kamen/Leap/release/src; LOG=$A/chatfix_now.log; OLD=$1
say() { echo "$(date '+%m-%d %H:%M:%S') $*" | tee -a $LOG; }
noswap() { vm_stat | awk '/page size of/ {ps=$8} /^Pages (free|speculative|purgeable):/ || /^File-backed pages:/ {v=$NF; sub("[.]","",v); s+=v} END {printf "%d", s*ps/1e9}'; }
users() { for p in $(ps -axo pid=,args= | awk '/[Pp]ython/ && /[ \/](chat|serve|run)[.]py( |$)/ && !/ awk / {print $1}'); do
            lsof -a -p $p -d cwd -Fn 2>/dev/null | sed -n 's/^n//p' | grep -q "^$F" && printf "%s " $p; done; }
desc() { local c; for c in $(pgrep -P "$1"); do desc "$c"; echo "$c"; done; }
say "armed (pid $$): waits for pid $OLD and a free clone"
CLR=0
while :; do
  if ! kill -0 $OLD 2>/dev/null && [ -z "$(users)" ]; then CLR=$((CLR + 1)); else CLR=0; fi
  [ $CLR -ge 2 ] && break
  sleep 5
done
say "clone free: replacing chat.py and serve.py"
for f in chat.py serve.py; do
  cp -p $F/$f $F/$f.bak-1110b && cp $S/$f $F/$f.tmp_fix && chmod $(stat -f %Lp $F/$f) $F/$f.tmp_fix
  /opt/homebrew/opt/python@3.12/bin/python3.12 -I -c "import sys; compile(open(sys.argv[1]).read(), sys.argv[1], 'exec')" $F/$f.tmp_fix \
    && mv -f $F/$f.tmp_fix $F/$f && say "  $f replaced (backup $f.bak-1110b)" || { say "  $f: parse failed, kept the old one"; rm -f $F/$f.tmp_fix; }
done
say "  markers in chat.py: Keys $(grep -c '^class Keys' $F/chat.py), title $(grep -c '033\]0;' $F/chat.py), /file $(grep -c 'def file_message' $F/chat.py), no-pause $(grep -c '_PAUSE = \"\"' $F/chat.py), 65536 $(grep -c 'default=65536' $F/chat.py)"
NS=$(noswap)
if [ -n "$(users)" ]; then say "real-engine check SKIPPED: a chat started from the clone already ($(users))"; exit 0; fi
if [ "$NS" -lt 36 ]; then say "real-engine check SKIPPED: NO-SWAP $NS GB < 36"; exit 0; fi
> (a quote from the project owner, kept private)
say "real-engine check: NO-SWAP $NS GB; chat.py --mem 32, one piped message"
( cd $F && printf 'Say hello in three words.\n/exit\n' | exec .venv/bin/python chat.py --mem 32 --max-tokens 16 ) > $A/chatfix_real.out 2>&1 &
P=$!; T0=$(date +%s)
while kill -0 $P 2>/dev/null; do
  [ $(( $(date +%s) - T0 )) -gt 240 ] && { say "  cap 240 s: stopping"; for c in $(desc $P); do kill -TERM $c 2>/dev/null; done; kill -TERM $P; break; }
  sleep 2
done
wait $P; RC=$?
peer unlock > /dev/null 2>&1
say "  rc $RC in $(( $(date +%s) - T0 )) s; $(grep -a 'tokens | whole decode' $A/chatfix_real.out | tail -1 | tr -d '[]' | cut -c1-160)"
say "  answer: $(grep -a -v 'Powered by\|github.com\|ExactStream chat\|tokens | whole\|^log:\|PyTorch\|^$\|^> *$' $A/chatfix_real.out | tr -d '\r' | head -3 | tr '\n' ' ' | cut -c1-160)"
say "done (NO-SWAP $(noswap) GB)"
