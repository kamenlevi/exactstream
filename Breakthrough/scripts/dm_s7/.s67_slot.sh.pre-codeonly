#!/bin/bash
# Map screens 6 + 7 + Fable C's 1(b) / 1(c) in one share-lock batch (Breakthrough, Wed 2026-11-04 15:5x; the orchestrator
# 14:3x + 15:3x; NewIdeas 14:5x for s7_ref3). Pre-registered in SCOREBOARD "MAP SCREENS 6 + 7" (614e16c) and its 15:5x
# addendum. Run ATTACHED (Bash run_in_background), so each share lock sees this session's processes.
#   Order (the orchestrator 15:0x): qom_0_live_ball_c147_t3 (6), s7_ref, s7_ref3, s7_cal, s7_cal_all, q1u_0.9_live_bonus9g
#   (1b), qom_0.5_live_bonus5g (1c), then s7_trl. Start 20:00 (after BigMoE's quiet lever 2 arms, ~19:00-19:35, if they
#   slip). One map process per arm (the model loads in ~1 s), so the clock rules hold per arm:
#   - an arm starts only if the clock is before LASTGO (23:10) and its expected minutes (est, from the 14:48 smoke and
#     earlier full runs, with margin) end by DEADLINE (23:30); the first arm in order that fits goes next;
#   - DEADLINE is a hard stop (RELAX's quiet arms, MEM 34-42 GB, take the GPU at 23:30); arms that don't fit are left for
#     Thu morning beside T0 (the orchestrator's night order).
#   s7_trl goes only when Fit12's TRL store is whole (dm_t3.trl_gate); until then the next arm that fits goes, and s7_trl
#   is re-checked each time (every 2 min once it is the only arm left that fits).
#   Last, optional (NewIdeas 16:5x, the orchestrator's OK; Wed 17:0x): qom_0_live_ball_c147_t3+dense3 = screen 6 + Leap's
#   DENSE-3 (install_dense3.sh; smoke 17:00), est 10 min. The orchestrator 18:4x: DENSE-3 goes LAST, only once s7_trl is
#   done and only if it can start before 23:10 (predicted FAIL, the lowest value tonight); after 1(c) the GPU goes to
#   MTP Head's P-CAP (~32 GB) or s7_trl, whichever is ready first (P-CAP waits for 3 clear minutes with no MEM >= 5 GB
#   lock; this slot waits while another session's shared run declares MEM > 5 GB).
#   The orchestrator 19:5x: MTP Head's P-OFF goes first (extraction ~20:40-21:15, training to ~22:15 if alive), then s7_trl
#   once the store is whole (~22:15-22:30), all before BigMoE's 23:00 ticket. DENSE-3 is OFF tonight (a Thu leftover with
#   BIGMOE_EXT_BUILD=build_nat after the conversion, or dropped if NewIdeas agrees). The interim read-out is not repeated
#   when the log already has one.
#   The orchestrator 20:1x: BigMoE's ticket takes the GPU at 23:00 and RELAX ~23:05, so armed with LASTGO 22:53 and
#   DEADLINE 23:00. One possible extra arm (NewIdeas' chat route) after s7_trl, if handed over by ~22:30 and it can start
#   by 22:50: listed in $EXTRA (see below), so it joins without a re-arm. The final read-out runs as soon as the scheduled
#   arms are done; the extra arms get a map_kl line at the end.
#   The orchestrator 17:3x: start NOW (the GPU is idle by day; armed with S67_START = the arming time); s7_trl joins when the
#   store is whole (~21:10). The first time only s7_trl is left, an interim read-out runs (the same read-outs as the end).
#   Wed 18:1x, by-day heat (fanwatch: a heat pause every ~5 min, 3-6 min each; s7_ref was stopped at 18:05 for "PAUSE over
#   5 min" after ~7 min of work): each run's working folder is ~/Breakthrough, so fanwatch freezes and thaws it with the
#   other research jobs; fanwatch's own PAUSE (PAUSE.fanwatch) no longer counts toward the 5-min stop (any other PAUSE
#   still does); share locks last 3 x est + 10 min (screen 6 took 20.5 min, est 6, so its lock lapsed before the end).
# Clear = no map process, no other session's quiet window held or waited for, no other session's shared run over 5 GB (by
# day one > 5 GB job at a time), lid open, no PAUSE, NO-SWAP >= 25 GB: 3 clear
# checks 10 s apart before the first arm and after any stop, 1 between arms. Share lock per arm, MEM 18 GB. Guards every
# 15 s: swap +2 GB, others' quiet window (3 strikes), lid closed (3 strikes), PAUSE over 5 min, DEADLINE: each stops the
# arm (a stopped arm writes nothing; it is retried from its start if it still fits). At most 14 runs.
# Read-outs: NewIdeas' map_kl.py (read-only use; prints) for screens 6 + 7 vs C; dm_screen_read.py for 6 vs C, and --null
# (+0.1 points vs the ungated twin, BYTES10.md:255-260) for 1(b) vs q1u_0.9_live and 1(c) vs qom_0.5_live.
# Override the clock at arming: S67_START / S67_LASTGO / S67_DEADLINE = "YYYY-mm-dd HH:MM:SS".
DM=~/FableGod/decision_map; PY=~/BigMoE/.venv/bin/python; LOG=~/Breakthrough/data/s67_slot.log; Q=$HOME/.local/share/quiet
ARMS="qom_0_live_ball_c147_t3 s7_ref s7_ref3 s7_cal s7_cal_all q1u_0.9_live_bonus9g qom_0.5_live_bonus5g s7_trl"   # DENSE-3 off tonight (the orchestrator 19:5x)
D3=qom_0_live_ball_c147_t3+dense3; D3SHA=ec71be4478bc606f   # Leap's dense3.py as smoked (17:00)
EXTRA=~/Breakthrough/data/s67_extra.txt   # the orchestrator 20:1x: optional extra arms, re-read every loop. Per line
# "<arm> <est min> <start by HH:MM>": an extra arm runs after s7_trl is done, once it is in dm_ablations.py, and only if it
# starts by its time. "PENDING 0 <HH:MM>" keeps the slot waiting for a hand-off until then.
xlines() { [ -f $EXTRA ] && awk 'NF >= 3 && $1 !~ /^#/' $EXTRA; }
xarms() { xlines | awk '$1 != "PENDING" && $1 != "ALONE" && $1 != "HOLD" {print $1}'; }
xby() { xlines | awk -v a="$1" '$1 == a {print $3; exit}'; }
by_ts() { date -j -f "%Y-%m-%d %H:%M" "$(date +%Y-%m-%d) $1" +%s; }
est() { local x; x=$(xlines | awk -v a="$1" '$1 == a {print $2; exit}'); [ -n "$x" ] && { echo $x; return; }
  case $1 in s7_cal_all) echo 35;; s7_ref3|$D3) echo 10;; s7_trl|s7_cal) echo 7;; q1u_0.9_live_bonus9g|qom_0.5_live_bonus5g) echo 5;; *) echo 6;; esac; }
ts() { date -j -f "%Y-%m-%d %H:%M:%S" "$1" +%s; }
START=$(ts "${S67_START:-2026-11-04 20:00:00}"); LASTGO=$(ts "${S67_LASTGO:-2026-11-04 23:10:00}")
DEADLINE=$(ts "${S67_DEADLINE:-2026-11-04 23:30:00}")
XALONE=$(ts "${S67_XALONE:-2026-11-04 22:40:00}")   # NewIdeas 20:2x: the chat arm needs no TRL store, so from 22:40 an extra arm
# no longer waits for s7_trl (the store may be late or never whole tonight); s7_trl still goes after it if it still fits
say() { echo "[s67] $* $(date '+%a %H:%M:%S')" >> $LOG; }
swap_mb() { sysctl -n vm.swapusage | sed -E 's/.*used = ([0-9.]+)M.*/\1/' | cut -d. -f1; }
noswap() { vm_stat | awk '/page size of/ {ps=$8} /^Pages (free|speculative|purgeable):/ || /^File-backed pages:/ {v=$NF; sub("[.]","",v); s+=v} END {printf "%d", s*ps/1e9}'; }
quiet_others() { peer ls 2>/dev/null | grep -v "^You:" | grep -qE "quiet lock until|benchmarking: quiet lock|waiting for a quiet window"; }
big_others() { peer ls 2>/dev/null | grep -v "^You:" | grep "shared run" | grep -oE "MEM [0-9]+ GB" | awk '$2 > 5 {f = 1} END {exit !f}'; }
lid_closed() { ioreg -r -k AppleClamshellState -d 1 | grep -q '"AppleClamshellState" = Yes'; }
kill_tree() { local p=$1 c; for c in $(pgrep -P $p); do kill_tree $c; done; kill -TERM $p 2>/dev/null; }
stop_run() { kill_tree $1; sleep 5; ps -o stat= -p $1 2>/dev/null | grep -q T && kill -KILL $1 2>/dev/null; }   # a frozen run holds SIGTERM until thawed; never SIGCONT
why_not() {
  if pgrep -f "run_ma[p].py" > /dev/null; then echo "a map process is running"
  elif quiet_others; then echo "a quiet window (others)"; elif big_others; then echo "another session's shared run > 5 GB"
  elif lid_closed; then echo "lid closed"
  elif [ -e "$Q/PAUSE" ]; then echo "PAUSE"; elif [ "$(noswap)" -lt 25 ]; then echo "NO-SWAP < 25 GB"; fi
}
trl_whole() { (cd $DM && DM_TRL_FALLBACK= $PY -c "import dm_t3; dm_t3.trl_gate([7, 18, 30, 41], 'mix')" > /dev/null 2>&1); }
nwin() { [ -f $DM/out/$1.json ] && $PY -c "import json; d = json.load(open('$DM/out/$1.json')); print(sum(1 for v in d.values() if isinstance(v, dict) and 'tk' in v))" 2>/dev/null || echo 0; }
installed() { local p; for p in $(echo "$1" | tr '+' ' '); do grep -qF "\"$p\"" $DM/dm_ablations.py || return 1; done; }   # a "+" stack: every part (Thu 07:1x)
btodo() { local a; for a in $ARMS; do [ "$(nwin $a)" = 36 ] || echo -n "$a "; done; }
todo() { local a; for a in $ARMS $(xarms); do [ "$(nwin $a)" = 36 ] || echo -n "$a "; done; }
pick_next() {   # the first arm in order that is not done, fits the clock, and (s7_trl) whose store is whole; DENSE-3 and the
  local a t by xa now=$(date +%s); PICK=; TRLWAIT=0; XWAIT=0; t=" $(todo)"   # extra arms only once s7_trl is done; sets
  xa=$(xby ALONE); if [ -n "$xa" ]; then xa=$(by_ts $xa); else xa=$XALONE; fi   # PICK, TRLWAIT, XWAIT. From the alone time
  [ $now -ge $xa ] && t=" $(for a in $(xarms); do [ "$(nwin $a)" = 36 ] || echo -n "$a "; done)$(btodo)"   # ($EXTRA's
  by=$(xby HOLD); if [ -n "$by" ] && [ $now -lt $(by_ts $by) ]; then XWAIT=1; return; fi   # "HOLD 0 HH:MM": nothing
  for a in $t; do                       # starts before then (P-CAP first, the orchestrator 21:18); "ALONE 0 HH:MM" line,
                                        # else XALONE: from then on the extra arms go FIRST
    [ $((now + $(est $a) * 60)) -le $DEADLINE ] || continue
    if [ $a = s7_trl ] && ! trl_whole; then TRLWAIT=1; continue; fi
    if [ $a = $D3 ]; then case "$t" in *" s7_trl "*) continue;; esac; fi
    by=$(xby $a)
    if [ -n "$by" ]; then
      [ $now -le $(by_ts $by) ] || continue
      if [ $now -lt $xa ]; then case "$t" in *" s7_trl "*) XWAIT=1; continue;; esac; fi   # from the alone time on, alone
      installed $a || { XWAIT=1; continue; }
    fi
    PICK=$a; return
  done
  by=$(xby PENDING); [ -n "$by" ] && [ $now -lt $(by_ts $by) ] && XWAIT=1
}
for n in s7_trl s7_ref3 q1u_0.9_live_bonus9g qom_0.5_live_bonus5g qom_0_live_ball_c147_t3 dense3; do
  grep -qF "\"$n\"" $DM/dm_ablations.py || { say "$n is not in dm_ablations.py (scripts/dm_s7/install_s7.sh, install_late.sh, install_dense3.sh); not run"; exit 1; }
done
say "armed (the orchestrator 14:3x / 15:3x, NewIdeas 14:5x / 16:5x): $ARMS; start >= $(date -r $START '+%a %H:%M'), no new arm after $(date -r $LASTGO '+%a %H:%M'), hard stop $(date -r $DEADLINE '+%a %H:%M')"
rd() { [ "$(nwin $1)" = 36 ] || return; $PY $DM/dm_screen_read.py "$@" >> $LOG 2>&1; say "read-out: $(grep '^VERDICT' $DM/out/SCREEN_read_$1.txt 2>/dev/null || echo "$1 none")"; }
readouts() {   # $1 = interim (the first wait for the TRL store; the orchestrator 17:3x) or final; CPU, the out/ JSONs only
  echo "[s67] read-outs ($1): map_kl.py (NewIdeas, read-only use): C, screen 6, screen 7, DENSE-3 $(date '+%a %H:%M:%S')" >> $LOG
  KL=$($PY ~/NewIdeas/scripts/map_kl.py qom_0_live_bq_c147_t3 qom_0_live_ball_c147_t3 s7_ref s7_ref3 s7_trl s7_cal s7_cal_all $D3 2>&1); echo "$KL" >> $LOG
  if [ "$(nwin $D3)" = 36 ] && [ "$(nwin qom_0_live_ball_c147_t3)" = 36 ]; then   # NewIdeas QUEUE 16:5x: INVALID rule, then code flips
    D3C=$($PY ~/Breakthrough/scripts/dm_s7/dense3_check.py $D3 qom_0_live_ball_c147_t3 2>&1); echo "$D3C" >> $LOG
    say "read-out DENSE-3: $(echo "$D3C" | grep '^VERDICT' || echo 'INVALID check failed'); $(echo "$KL" | grep -F "| $D3 |" | awk -F'|' '{split($3, a, "/"); if (a[1] !~ /[0-9]/) {print "no map_kl number"; exit} f = a[1] + 0; print (f <= 1.34 ? "PASS" : f <= 2.01 ? "BETWEEN" : "FAIL") " on code flips " f " % (PASS <= 1.34, BETWEEN <= 2.01)"}')"
  fi
  rd qom_0_live_ball_c147_t3 qom_0_live_bq_c147_t3
  rd q1u_0.9_live_bonus9g q1u_0.9_live 0.1 --null
  rd qom_0.5_live_bonus5g qom_0.5_live 0.1 --null
  say "read-outs ($1) done"
}
while [ $(date +%s) -lt $START ]; do sleep 30; done
cd $DM || exit 1
need=3; runs=0; last=; interim=0; final=0; grep -q "^\[s67\] read-outs (interim) done" $LOG && interim=1
while :; do
  [ $final = 0 ] && [ -z "$(btodo)" ] && { final=1; say "the scheduled arms are done"; readouts final; }
  [ $(date +%s) -ge $LASTGO ] && { say "LASTGO $(date -r $LASTGO +%H:%M): no new arm; left: $(todo)"; break; }
  pick_next
  if [ -z "$PICK" ]; then
    if [ $TRLWAIT = 1 ]; then
      if [ "$last" != trl ]; then
        say "waiting for Fit12's TRL store (s7_trl); no lock held, so MTP Head's jobs may take the GPU meanwhile"
        [ $interim = 0 ] && { interim=1; readouts interim; }
      fi
      last=trl; sleep 120; continue
    fi
    if [ $XWAIT = 1 ]; then [ "$last" = x ] || say "waiting for an extra arm (the orchestrator 20:1x; $EXTRA): $(xlines | tr '\n' ';')"; last=x; sleep 60; continue; fi
    [ -n "$(todo)" ] && say "nothing left fits the clock; left: $(todo)"; break
  fi
  w=$(why_not)
  if [ -n "$w" ]; then need=3; [ "$w" != "$last" ] && say "waiting: $w"; last=$w; sleep 10; continue; fi
  need=$((need - 1)); [ $need -gt 0 ] && { sleep 10; continue; }
  last=; runs=$((runs + 1)); [ $runs -gt 14 ] && { say "14 runs used; left: $(todo)"; break; }
  peer lock share $(( 3 * $(est $PICK) + 10 ))m --cpu 200 --gpu 100 "Breakthrough map screens 6 + 7 + 1(b)/(c) (the orchestrator 14:3x / 15:3x): MEM 18 GB (flips / KL, no timing): $PICK" >> $LOG 2>&1 \
    || { say "share lock not granted"; need=3; sleep 30; continue; }
  S0=$(swap_mb); say "go $PICK (est $(est $PICK) min; NO-SWAP $(noswap) GB, swap $S0 MB)"
  [ "$PICK" = "$D3" ] && say "Leap's dense3.py sha256 $(shasum -a 256 ~/Leap/scripts/dense3.py | cut -c1-16) (smoked: $D3SHA)"
  # cwd = the session folder (Wed 18:1x): fanwatch and quiet_window.sh freeze research jobs by working folder, and
  # ~/FableGod is not one, so a heat pause froze everyone's jobs but this map run (100 % CPU under PAUSE at 18:10)
  (cd ~/Breakthrough && exec $PY $DM/run_map.py "$PICK" --max 12 --skip-done) >> $LOG 2>&1 & P=$!
  bad=0; pz=0; stopped=
  while kill -0 $P 2>/dev/null; do
    sleep 15; w=
    [ $(date +%s) -ge $DEADLINE ] && { stopped="the hard stop ($(date -r $DEADLINE +%H:%M))"; stop_run $P; break; }
    [ $(( $(swap_mb) - S0 )) -gt 2048 ] && { stopped="swap grew > 2 GB"; stop_run $P; break; }
    # fanwatch's heat pause (its PAUSE.fanwatch mark) freezes the run and thaws it when cool: not a reason to stop it
    if [ -e "$Q/PAUSE" ] && [ ! -e "$Q/PAUSE.fanwatch" ]; then pz=$((pz + 1)); else pz=0; fi
    if quiet_others; then w="a quiet window (others)"; elif lid_closed; then w="lid closed"; elif [ $pz -ge 20 ]; then w="PAUSE over 5 min"; fi
    if [ -n "$w" ]; then bad=$((bad + 1)); else bad=0; fi
    [ $bad -ge 3 ] && { stopped="$w"; stop_run $P; break; }
  done
  wait $P 2>/dev/null; rc=$?; peer unlock >> $LOG 2>&1
  if [ -n "$stopped" ]; then need=3; say "stopped $PICK for: $stopped"; else need=1; fi
  say "$PICK ended (rc $rc, $([ "$(nwin $PICK)" = 36 ] && echo 'done, 36 windows' || echo 'NOT done')); left: $(todo); unlocked"
  [ $(date +%s) -ge $DEADLINE ] && break
done
say "batch over; left: $(todo)"
[ $final = 0 ] && readouts final
xs=$(for a in $(xarms); do [ "$(nwin $a)" = 36 ] && echo -n "$a "; done)
if [ -n "$xs" ]; then   # an extra line's fields 4+ are its twins, the primary first: dm_screen_read.py vs each (the primary
  tw=$(for a in $xs; do xlines | awk -v a="$a" '$1 == a {for (i = 4; i <= NF; i++) print $i}'; done | sort -u | tr '\n' ' ')
  echo "[s67] map_kl.py (NewIdeas, read-only use), extra arms + twins: $xs$tw $(date '+%a %H:%M:%S')" >> $LOG
  $PY ~/NewIdeas/scripts/map_kl.py $xs $tw >> $LOG 2>&1                   # last, so its SCREEN_read file holds the verdict)
  for a in $xs; do for t in $(xlines | awk -v a="$a" '$1 == a {for (i = NF; i >= 4; i--) print $i}'); do rd $a $t; done; done
  say "read-out (extra arms) done"
fi
say "done"
