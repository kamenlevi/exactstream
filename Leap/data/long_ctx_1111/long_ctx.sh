#!/bin/bash
# Long-context chat runs (orchestrator 20:3x; night rule: from 00:30): the chat configs at 24 / 32 / 45 GB on a 64,000- and
# a 128,000-token prompt (the papers MAIN, KERNELS, RELEASE, EVALUATION, DRAFTER, cut; 128K repeats them from the start of
# MAIN; then "Summarize the main findings of the document above in five bullet points."), through the agent clone's run.py,
# 64 new tokens. PF-A (R57_PFLM=1, chat.py's and serve.py's prefill at 24 / 32 GB) and the default prefill (run.py's; the
# leaner path); 45 GB as published (no PF-A there). Per run: first token, whole decode, lifetime peak (run.py's line),
# NO-SWAP before, swap growth. Order: by predicted lifetime peak (linear from the measured 10K / 30K peaks; a measured 64K
# peak replaces the guess for the same arm's 128K). A run starts when NO-SWAP >= its predicted peak + 2 GB, Kamen idle
# >= 600 s, no other lock / engine / chat.py / serve.py, no PAUSE / HOLD, 2 clear polls; last start 07:30. During a run:
# swap +2 GB (then larger arms are skipped), Kamen back (input within 5 s on 5 polls in a row) or the 50-min cap stop it.
# Starts after the 1.1.0 check (pid $1) and the agent tests (pid $2) end. Usage: long_ctx.sh <v110 pid> <agent tests pid>
L=/private/tmp/claude-501/-Users-kamen-OjoX/f2971937-6c95-4e86-a049-05beff6a8e32/scratchpad/agent/long
CL=/Users/kamen/Leap/release/cleanclone/agent/clone; LOG=$L/long_ctx.log; SUM=$L/summary.txt
ep() { date -j -f "%Y-%m-%d %H:%M:%S" "$1" +%s; }
FROM=$(ep "2026-11-11 01:00:00"); LAST=$(ep "2026-11-11 11:00:00")
noswap() { vm_stat | awk '/page size of/ {ps=$8} /^Pages (free|speculative|purgeable):/ || /^File-backed pages:/ {v=$NF; sub("[.]","",v); s+=v} END {printf "%d", s*ps/1e9}'; }
swap() { sysctl -n vm.swapusage | sed -E 's/.*used = ([0-9.]+)M.*/\1/' | cut -d. -f1; }
idle() { ioreg -c IOHIDSystem | awk '/HIDIdleTime/ {print int($NF/1000000000); exit}'; }
busy() { peer ls 2>/dev/null | grep "^- " | grep -oE "· (shared run \(others may work\)|benchmarking|waiting for a quiet window)[^·]*" | grep -cE "^· (benchmarking|waiting for a quiet window)|MEM ([6-9]|[1-9][0-9]) GB"; }
procs() { ps -axo pid=,args= | awk '/[Pp]ython/ && /r57_screen_|run_map[.]py|fit_onecall[.]py|[ \/](chat|serve|run)[.]py( |$)/ && !/ awk / {printf "%s ", $1}'; }
held() { [ -e ~/.local/share/quiet/PAUSE ] || [ -e ~/.local/share/quiet/HOLD ]; }
desc() { local c; for c in $(pgrep -P "$1"); do desc "$c"; echo "$c"; done; }
say() { echo "$(date '+%m-%d %H:%M:%S') $*" | tee -a $LOG; }
fx() { awk "BEGIN {printf \"%.1f\", $1}"; }
LOCKED=0; RP=""
trap '[ -n "$RP" ] && kill -0 $RP 2>/dev/null && { for c in $(desc $RP); do kill -TERM $c 2>/dev/null; done; kill -TERM $RP; }; [ $LOCKED = 1 ] && peer unlock >/dev/null 2>&1; touch $L/long_ctx.done; say "exit"' EXIT
# arms.tsv: id mem path size predicted-peak state (pending / done / skip / stopped); p30: the measured 30K peaks (45: a guess)
ARMS=$L/arms.tsv
printf "%s\n" "24_default_64k 24 default 64k 32.2 pending" "24_pfa_64k 24 pfa 64k 37.5 pending" \
  "24_default_128k 24 default 128k 39.7 pending" "32_default_128k 32 default 128k 42.7 pending" "32_pfa_64k 32 pfa 64k 44.7 pending" \
  "24_pfa_128k 24 pfa 128k 50.5 pending" "45_default_64k 45 default 64k 51.7 pending" "32_pfa_128k 32 pfa 128k 56.4 pending" \
  "45_default_128k 45 default 128k 56.5 pending" > $ARMS
p30() { case "$1 $2" in "24 default") echo 28.17;; "24 pfa") echo 30.60;; "32 default") echo 35.48;; "32 pfa") echo 38.49;; *) echo 49.2;; esac; }
col() { awk -v id="$1" -v c="$2" '$1 == id {print $c}' $ARMS; }
setcol() { awk -v id="$1" -v c="$2" -v v="$3" '$1 == id {$c = v} {print}' $ARMS > $ARMS.tmp && mv $ARMS.tmp $ARMS; }
smallest() { awk '$6 == "pending" {print $5, $1}' $ARMS | sort -n | head -1 | cut -d" " -f2; }
fitting() { awk -v ns="$1" '$6 == "pending" && $5 + 2 <= ns {print $5, $1}' $ARMS | sort -n | head -1 | cut -d" " -f2; }
say "armed (pid $$): after pids $1 and $2, from 00:30, last start 07:30; NO-SWAP now $(noswap) GB"
echo "| arm | prompt | NO-SWAP before | first token | prefill rate | whole decode | lifetime peak (predicted) | swap growth | wall |" > $SUM
echo "|---|---|---|---|---|---|---|---|---|" >> $SUM
NOTE=""; CLR=0; MAXNS=0
while :; do
  now=$(date +%s); ns=$(noswap); [ "$ns" -gt "$MAXNS" ] && [ "$now" -ge "$FROM" ] && MAXNS=$ns
  next=$(smallest)
  [ -z "$next" ] && { say "all arms done or skipped"; break; }
  if [ "$now" -ge "$LAST" ]; then
    for k in $(awk '$6 == "pending" {print $1}' $ARMS); do
      setcol $k 6 skip; nd=$(fx "$(col $k 5) + 2")
      say "SKIP $k: no room by 07:30 (needs $nd GB, NO-SWAP max $MAXNS GB)"; echo "| $k | - | max $MAXNS GB | skipped: needs $nd GB NO-SWAP | | | | | |" >> $SUM
    done
    break
  fi
  pick=$(fitting $ns)
  why=""
  [ "$now" -lt "$FROM" ] && why="before 00:30"
  [ -z "$why" ] && for w in ${1:-0} ${2:-0}; do [ "$w" -gt 1 ] && kill -0 $w 2>/dev/null && why="the 1.1.0 check / agent tests run first"; done
> (a quote from the project owner, kept private)
  [ -z "$why" ] && [ -n "$(procs)" ] && why="an engine / chat / server / run.py process runs"
  [ -z "$why" ] && [ "$(busy)" -gt 0 ] && why="another lock"
  [ -z "$why" ] && held && why="PAUSE / HOLD"
  [ -z "$why" ] && [ -z "$pick" ] && why="NO-SWAP $ns GB: no arm fits (smallest pending: $next, needs $(fx "$(col $next 5) + 2"))"
  if [ -z "$why" ]; then CLR=$((CLR + 1)); else CLR=0; fi
  [ -z "$why" ] && [ $CLR -lt 2 ] && why="clear once"
  if [ -z "$why" ]; then
    NEED=$(fx "$(col $pick 5) + 2")
    peer lock share 60m --cpu 300 --gpu 100 "MEM ${NEED%.*} GB: Leap long-context chat run ($pick)" > /dev/null 2>&1 && LOCKED=1
    sleep 20
    if [ "$(busy)" = 0 ] && [ -z "$(procs)" ]; then
      M=$(col $pick 2); PT=$(col $pick 3); SZ=$(col $pick 4); PR=$(col $pick 5); ENV=""; [ $PT = pfa ] && ENV="--env R57_PFLM=1"
      ns0=$(noswap); S0=$(swap); T0=$(date +%s); OUT=$L/run_${M}_${PT}_${SZ}.out
      say "run $pick: NO-SWAP $ns0 GB (need $NEED), swap $S0 MB"
      cd $CL && .venv/bin/python run.py --config chat --mem $M --prompt-file $L/prompt_$SZ.txt --max-tokens 64 $ENV > $OUT 2>&1 &
      RP=$!; ACT=0; STOP=""
      while kill -0 $RP 2>/dev/null; do
        [ $(( $(swap) - S0 )) -gt 2048 ] && STOP="swap guard ($(swap) MB, start $S0)"
        [ $(( $(date +%s) - T0 )) -gt 3000 ] && STOP="50-min cap"
        if [ "$(idle)" -lt 5 ]; then ACT=$((ACT + 1)); else ACT=0; fi
> (a quote from the project owner, kept private)
        if [ -n "$STOP" ]; then for c in $(desc $RP); do kill -TERM $c 2>/dev/null; done; kill -TERM $RP 2>/dev/null; sleep 10; break; fi
        sleep 5
      done
      wait $RP 2>/dev/null; RP=""; W=$(( $(date +%s) - T0 )); SG=$(( $(swap) - S0 ))
      line=$(grep "whole decode" $OUT | tail -1)
      ft=$(echo "$line" | sed -nE 's/.*first token ([0-9.]+) s.*/\1/p'); dec=$(echo "$line" | sed -nE 's/^whole decode ([0-9.]+) tok\/s.*/\1/p')
      pk=$(echo "$line" | sed -nE 's/.*lifetime peak ([0-9.]+) GB.*/\1/p'); n=${SZ%k}000
      if [ -n "$STOP" ] || [ -z "$pk" ]; then
        setcol $pick 6 stopped; say "STOPPED $pick after $W s: ${STOP:-no result line} (tail: $(tail -2 $OUT | tr '\n' ' ' | cut -c1-200))"
        echo "| $pick | ${SZ} | $ns0 GB | stopped: ${STOP:-no result} after $W s | | | ($PR) | $SG MB | $W s |" >> $SUM
        if [ "${STOP%% *}" = swap ]; then for k in $(awk -v p="$PR" '$6 == "pending" && $5 >= p {print $1}' $ARMS); do setcol $k 6 skip; say "SKIP $k: a smaller arm hit the swap guard"; done; fi
> (a quote from the project owner, kept private)
      else
        setcol $pick 6 done; rate=$(fx "$n / $ft")
        say "DONE $pick: first token $ft s ($rate tok/s), whole decode $dec tok/s, lifetime peak $pk GB (predicted $PR), swap +$SG MB, wall $W s"
        echo "| $M GB, $PT | $n | $ns0 GB | $ft s | $rate tok/s | $dec tok/s | $pk GB ($PR) | $SG MB | $W s |" >> $SUM
        k2=${M}_${PT}_128k
        if [ $SZ = 64k ] && [ "$(col $k2 6)" = pending ]; then
          old=$(col $k2 5); setcol $k2 5 $(fx "$pk + ($pk - $(p30 $M $PT)) * 64 / 34")
          say "  128K prediction for $M $PT: $old -> $(col $k2 5) GB"
        fi
      fi
      peer unlock >/dev/null 2>&1; LOCKED=0; CLR=0; NOTE=""; sleep 30; continue
    fi
    peer unlock >/dev/null 2>&1; LOCKED=0; CLR=0; why="after the lock"
  fi
  [ "$why" != "$NOTE" ] && say "wait: $why"; NOTE=$why
  sleep 60
done
say "summary: $SUM"
