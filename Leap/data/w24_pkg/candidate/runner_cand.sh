#!/bin/bash
# Public-candidate checks with the 24 GB line (orchestrator Tue 10:1x), one job over 5 GB at a time, by 14:45:
#   1. tools/clean_clone_test.sh run on the prepared clone in <work dir>: every config on the first two prompts of its own
#      reference (45 / 32 GB: IFEval; 24 GB: HumanEval), cap 256, HOME empty, tokens vs verify/, module + file closure;
#      all 8 configs when NO-SWAP >= 51 GB (the 45 GB configs peak ~42-43 GB), else the 32 + 24 GB ones (reported).
#   2. serve.py's first real-engine test (--mem 24): servetest/serve_test.sh (one streamed OpenAI, one streamed Anthropic).
# Gate, every 30 s: no quiet lock held or queued; no other MEM > 5 GB lock or >= 50 % GPU shared run; no foreign engine
# process; no chat.py / serve.py running; no PAUSE / HOLD; NO-SWAP >= 42 GB on 3 reads; clear on 2 polls. Share lock,
# re-checked after 20 s. Guards: swap +2 GB, a 25 min cap for step 1. Usage: runner_cand.sh <work dir> <END epoch>
W=$1; END=$2; T=/private/tmp/claude-501/-Users-kamen-OjoX/f2971937-6c95-4e86-a049-05beff6a8e32/scratchpad/servetest
LOG=$W/cand.log
noswap() { vm_stat | awk '/page size of/ {ps=$8} /^Pages (free|speculative|purgeable):/ || /^File-backed pages:/ {v=$NF; sub("[.]","",v); s+=v} END {printf "%d", s*ps/1e9}'; }
swap() { sysctl -n vm.swapusage | sed -E 's/.*used = ([0-9.]+)M.*/\1/' | cut -d. -f1; }
chats() { ps -axo pid=,comm=,args= | awk '$2 ~ /[Pp]ython/ && /[ \/](chat|serve)[.]py( |$)/ {printf "%s ", $1}'; }
engines() { ps -axo pid=,comm=,args= | awk '$2 ~ /[Pp]ython/ && /r57_screen_|run_map[.]py|fit_onecall[.]py|dm_oea_masks[.]py/ {printf "%s ", $1}'; }
others() { peer ls 2>/dev/null | grep "^- "; }
locks() { grep -oE "· (shared run \(others may work\)|benchmarking|waiting for a quiet window)[^·]*"; }
quiet_n() { others | locks | grep -cE "^· (benchmarking|waiting for a quiet window)"; }
bigmem_n() { others | locks | grep -oE "MEM [0-9]+ GB" | awk '$2 > 5 {n++} END {print n + 0}'; }
gpu_n() { others | sed -n 's/.*shared run[^:]*: about [0-9]*% CPU, \([0-9]*\)% GPU.*/\1/p' | awk '$1 >= 50 {n++} END {print n + 0}'; }
held() { [ -e ~/.local/share/quiet/PAUSE ] || [ -e ~/.local/share/quiet/HOLD ]; }
say() { echo "$(date +%T) $*" | tee -a $LOG; }
desc() { local c; for c in $(pgrep -P "$1"); do desc "$c"; echo "$c"; done; }
gate() {
  [ "$(quiet_n)" -gt 0 ] && { echo "a quiet lock is held or queued"; return; }
  [ "$(bigmem_n)" -gt 0 ] && { echo "another session holds a MEM > 5 GB lock"; return; }
  [ "$(gpu_n)" -gt 0 ] && { echo "another shared run uses >= 50 % GPU"; return; }
  [ -n "$(engines)" ] && { echo "another engine run ($(engines))"; return; }
  [ -n "$(chats)" ] && { echo "a chat.py / serve.py is running"; return; }
  held && { echo "PAUSE / HOLD"; return; }
  local a b c; a=$(noswap); sleep 5; b=$(noswap); sleep 5; c=$(noswap)
  [ $a -lt 42 ] || [ $b -lt 42 ] || [ $c -lt 42 ] && { echo "NO-SWAP $a / $b / $c GB < 42"; return; }
}
LOCKED=0; BP=""
finish() { [ -n "$BP" ] && kill -0 $BP 2>/dev/null && { for c in $(desc $BP); do kill -TERM $c 2>/dev/null; done; kill -TERM $BP; }
           [ $LOCKED = 1 ] && { peer unlock >/dev/null 2>&1; say "unlocked"; }; touch $W/cand.done; say "exit (NO-SWAP $(noswap) GB, swap $(swap) MB)"; }
trap finish EXIT
trap 'say "stopped by a signal"; exit 1' INT TERM
say "armed (pid $$): clean-clone run (8 configs if NO-SWAP >= 51, else 32 + 24 GB) + serve.py --mem 24 test; end $(date -r $END +%H:%M)"
CLR=0; NOTE=""
while :; do
  [ $(( $(date +%s) + 900 )) -gt $END ] && { say "SKIP: no start that ends by $(date -r $END +%H:%M) (last wait: ${NOTE:-none})"; exit 0; }
  why=$(gate)
  if [ -z "$why" ]; then CLR=$((CLR + 1)); else CLR=0; fi
  [ -z "$why" ] && [ $CLR -lt 2 ] && why="clear once: one more poll"
  if [ -z "$why" ]; then
    A=$(noswap); MEM=$([ $A -ge 51 ] && echo 43 || echo 36)
    out=$(peer lock share 40m --cpu 400 --gpu 100 "MEM $MEM GB: Leap public candidate check (clean clone, configs one at a time) + serve.py --mem 24 test" 2>&1); rc=$?
    say "share lock (MEM $MEM) rc $rc: $(echo "$out" | tr '\n' ' ' | cut -c1-200)"
    if [ $rc = 0 ]; then
      LOCKED=1; sleep 20
      w2=""; [ "$(quiet_n)" -gt 0 ] && w2="quiet"; [ "$(bigmem_n)" -gt 0 ] && w2="$w2 MEM lock"; [ -n "$(engines)" ] && w2="$w2 engine"; [ "$(noswap)" -lt 40 ] && w2="$w2 NO-SWAP $(noswap)"
      [ -z "$w2" ] && break
      say "after the lock: $w2: unlock and wait"; peer unlock >/dev/null 2>&1; LOCKED=0; CLR=0; why="after the lock: $w2"
    else
      why="share lock refused"
    fi
  fi
  [ "$why" != "$NOTE" ] && say "wait: $why"; NOTE=$why
  sleep 30
done
A=$(noswap); S0=$(swap)
if [ $A -ge 51 ]; then CF="exact-45 exact-32 lossy-45 lossy-32 chat-45 chat-32 lossy-24 chat-24"; TC=chat-45
else CF="exact-32 lossy-32 chat-32 lossy-24 chat-24"; TC=chat-24; say "NO-SWAP $A GB < 51: the 45 GB configs are NOT run"; fi
say "GO: NO-SWAP $A GB, swap $S0 MB; configs: $CF; text prompt on $TC"
CONFIGS="$CF" TEXT_CONFIG=$TC bash $W/clone/tools/clean_clone_test.sh $W/clone $W run >> $LOG 2>&1 &
BP=$!; T0=$(date +%s)
while kill -0 $BP 2>/dev/null; do
  s=$(swap)
  if [ $((s - S0)) -gt 2048 ] || [ $(( $(date +%s) - T0 )) -gt 1500 ]; then
    say "STOP the clean-clone run: swap $s MB (start $S0) or 25 min"; for c in $(desc $BP); do kill -TERM $c 2>/dev/null; done; kill -TERM $BP; break
  fi
  sleep 5
done
wait $BP 2>/dev/null; BP=""
say "clean-clone run ended ($(( $(date +%s) - T0 )) s): $(tail -1 $W/report.txt)"
if [ $(( $(date +%s) + 420 )) -le $END ]; then
  cp /Users/kamen/Leap/release/cleanclone/final/clone/serve.py /Users/kamen/Leap/release/cleanclone/final/clone/chat.py $W/clone/
  bash $T/serve_test.sh $W/clone $W/serve 8093 >> $LOG 2>&1 &
  BP=$!; wait $BP; BP=""
  say "serve test ended: $(grep -E '^SERVE TEST' $LOG | tail -1)"
else
  say "SKIP the serve test: cannot end by $(date -r $END +%H:%M)"
fi
peer unlock >/dev/null 2>&1; LOCKED=0; say "unlocked"
