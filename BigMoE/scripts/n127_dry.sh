#!/bin/bash
# The 127-node tree dump DRY RUN (BigMoE; MTP Head 22:2x, the orchestrator, NewIdeas; EvalSuite's start-gate marker 08:5x).
# Four head-only runs of scripts/dd_tree.py --nodes 127 (t30b; the bank's HE+ 24 / MBPP 100 windows), in this order:
#   heplus24 k 8, mbppbank k 8 (each with the nest check: the first 32 pops == the 32-node dump, bitwise),
#   heplus24 k 16, mbppbank k 16 (time / peak; the TOPK check on the first 8 of 16 children)
# -> data/deepdraft/ceiling/tree_t30b_<set>_n127k<k>.{npz,json} + dumps_sha256.json; log data/deepdraft/ceiling/n127_dry.log.
# Machine rules: Kamen away (away_secs >= 900) unless NIGHT=1; HOLD / the lid closed stop it; a PAUSE is waited out up to
# 30 min; NO-SWAP >= 18 GB on two reads 15 s apart before each part (heplus24 in 2, mbppbank in 5 window ranges, merged); ONE share lock ("MEM 13 GB" first in the reason);
# swap guard +2 GB (stops the part); one retry per part; HARD (epoch or "YYYY-mm-dd HH:MM") ends it; from YIELD_AT on, a
# waiting t30c (MTP Head, the drafter line) is let in first at a part boundary (the orchestrator 08:5x).
# EVERY end writes data/deepdraft/ceiling/N127_DRY_1117_DONE: "N127 DRY END <mm-dd HH:MM:SS>: rc <n> (<why>)" after the
# runs exited and the lock was released (EvalSuite's 1c-tiers runner gates on it), unless data/deepdraft/ceiling/N127.NOMARK.
# usage: HARD="2026-11-16 14:30" bash scripts/n127_dry.sh     DRY=1: argument / file checks + dd_tree --selftest, no runs
set -u
cd ~/BigMoE || exit 1
C=data/deepdraft/ceiling; W=${SIMD:-$C}   # SIMD: a simulation's log / marker dir (never the real marker)
L=$W/n127_dry.log; MK=$W/N127_DRY_1117_DONE; NOMARK=$W/N127.NOMARK; Q=$HOME/.local/share/quiet
PY=.venv/bin/python; RUNS="heplus24:8 mbppbank:8 heplus24:16 mbppbank:16"
YIELD_AT=${YIELD_AT:-}; HARD=${HARD:-}; NIGHT=${NIGHT:-0}; DRY=${DRY:-0}; STUB=${STUB:-}; NOLOCK=${NOLOCK:-0}; GATE_S=${GATE_S:-15}   # NOLOCK / GATE_S / STUB: simulation only
[ -n "$HARD" ] && { case "$HARD" in *-*) HARD=$(date -j -f "%Y-%m-%d %H:%M:%S" "$HARD:00" +%s) || exit 2;; esac; }
[ -n "$YIELD_AT" ] && { case "$YIELD_AT" in *-*) YIELD_AT=$(date -j -f "%Y-%m-%d %H:%M:%S" "$YIELD_AT:00" +%s) || exit 2;; esac; }
say() { echo "$(date '+%m-%d %H:%M:%S') $*" >> $L; }
noswap() { vm_stat | awk '/page size of/ {ps=$8} /^Pages (free|purgeable):/ || /^File-backed pages:/ {v=$NF; sub("[.]","",v); s+=v} END {printf "%d", s*ps/1e9}'; }
swapmb() { sysctl -n vm.swapusage | sed -E 's/.*used = ([0-9.]+)M.*/\1/' | cut -d. -f1; }
lid_closed() { ioreg -r -k AppleClamshellState -d 1 | grep -q '"AppleClamshellState" = Yes'; }
away() { bash $Q/away_secs.sh 2>/dev/null | tail -1; }
if [ "$DRY" = 1 ]; then
  for f in $C/topk_t30b_heplus24.npz $C/topk_t30b_mbppbank.npz $C/tree_t30b_heplus24.npz $C/tree_t30b_mbppbank.npz; do
    [ -s $f ] && echo "ok $f" || { echo "MISSING $f"; exit 3; }
  done
  $PY scripts/dd_tree.py --selftest 2>&1 | tail -1
  for r in $RUNS; do echo "would run: $PY scripts/dd_tree.py --set ${r%:*} --nodes 127 --k ${r#*:}"; done
  exit 0
fi
STATUS="not started"; RC=0; LOCKED=0; CP=""; GUARD=""
cleanup() { [ -n "$GUARD" ] && kill $GUARD 2>/dev/null
            if [ -n "$CP" ] && kill -0 $CP 2>/dev/null; then
              kill -TERM $CP 2>/dev/null; for i in $(seq 1 60); do kill -0 $CP 2>/dev/null || break; sleep 1; done
              kill -0 $CP 2>/dev/null && { kill -KILL $CP 2>/dev/null; sleep 2; }
            fi
            [ $LOCKED = 1 ] && { peer unlock > /dev/null 2>&1; say "lock released"; }
            if [ -e $NOMARK ]; then say "END without the marker (NOMARK): rc $RC ($STATUS)"
            else echo "N127 DRY END $(date '+%m-%d %H:%M:%S'): rc $RC ($STATUS)" > $MK
                 say "END: rc $RC ($STATUS); marker $MK written after the runs exited and the lock was released"; fi; }
trap cleanup EXIT; trap 'STATUS="$STATUS; stopped by a signal"; RC=9; exit 1' INT TERM
stop_checks() {   # HOLD / the lid closed / a 30-min PAUSE / HARD / Kamen at the Mac (by day) end the runner
  [ -e $Q/HOLD ] && { STATUS="$STATUS; STOP: HOLD"; RC=5; exit 0; }
  lid_closed && { STATUS="$STATUS; STOP: lid closed"; RC=5; exit 0; }
  local t0=$(date +%s)
  while [ -e $Q/PAUSE ]; do
    [ $(( $(date +%s) - t0 )) -gt 1800 ] && { STATUS="$STATUS; STOP: PAUSE for 30 min"; RC=5; exit 0; }
    [ -e $Q/HOLD ] && { STATUS="$STATUS; STOP: HOLD during a PAUSE"; RC=5; exit 0; }
    sleep 20
  done
  [ -n "$HARD" ] && [ "$(date +%s)" -ge "$HARD" ] && { STATUS="$STATUS; STOP: hard end"; RC=6; exit 0; }
> (a quote from the project owner, kept private)
}
HARDS=none; [ -n "$HARD" ] && HARDS=$(date -r $HARD '+%m-%d %H:%M')
say "start: $RUNS (--nodes 127, t30b); NO-SWAP gate 18 GB; swap guard +2 GB; hard end $HARDS; NIGHT=$NIGHT${STUB:+; STUB $STUB}"
stop_checks
if [ "$NOLOCK" = 1 ]; then LOCKED=2; else
  peer lock share 120m --cpu 200 --gpu 60 "BigMoE: MEM 13 GB, 127-node tree dump dry run (head only, 4 runs, ~1 h)" > /dev/null 2>&1 && LOCKED=1; fi
say "share lock $([ $LOCKED = 1 ] && echo taken || { [ $LOCKED = 2 ] && echo "skipped (NOLOCK)" || echo FAILED; })"
[ $LOCKED = 0 ] && { STATUS="no share lock"; RC=4; exit 0; }
DONE_RUNS=""
t30c_started() { ps -axo args | grep -qE "^[^ ]*python[^ ]* (-B )?scripts/dd_ttt\.py (train|eval) .*t30c"; }   # MTP Head's
t30c_waiting() { ps -axo args | grep -qE "^(/bin/)?bash [^ ]*sb98/t30c_run\.sh" && ! t30c_started; }   # sb98/t30c_run.sh, by shape
yield_t30c() {    # the orchestrator 08:5x: t30c (the drafter line, MEM 16 GB, ~09:40) beats this run. From YIELD_AT on, at a
                  # window (part) boundary: while t30c's runner waits and has not started, hold here (no GPU, lock kept,
                  # nothing loaded), up to 90 min; resume once it has started (or its waiter is gone)
  [ -n "$YIELD_AT" ] && [ "$(date +%s)" -ge "$YIELD_AT" ] || return 0
  t30c_waiting || return 0
  say "  yield: t30c is waiting at $(date '+%H:%M'); holding at a window boundary until it starts"
  local t0=$(date +%s)
  while t30c_waiting; do
    [ $(( $(date +%s) - t0 )) -gt 5400 ] && { say "  yield: t30c still waiting after 90 min: resuming"; break; }
    [ -e $Q/HOLD ] && { STATUS="$STATUS; STOP: HOLD while yielding"; RC=5; exit 0; }
    sleep 30
  done
  say "  yield over at $(date '+%H:%M') (t30c $(t30c_started && echo started || echo 'waiter gone'))"
}
for r in $RUNS; do
  s=${r%:*}; k=${r#*:}; [ $s = heplus24 ] && NP=2 || NP=5    # parts = window ranges (dd_tree --part i/n), merged after
  for i in $(seq 0 $((NP - 1))); do
    try=0
    while :; do
      yield_t30c; stop_checks
      ok=0; for g in $(seq 1 20); do a=$(noswap); sleep $GATE_S; b=$(noswap); [ $a -ge 18 ] && [ $b -ge 18 ] && { ok=1; break; }
        say "  $s k $k part $i/$NP: NO-SWAP $a / $b GB < 18"; stop_checks; done
      [ $ok = 1 ] || { STATUS="$STATUS; $s k $k part $i/$NP: NO-SWAP below 18 GB for 10 min"; RC=3; exit 0; }
      sw0=$(swapmb); t0=$(date +%s)
      say "$s k $k part $i/$NP start (try $((try + 1)); NO-SWAP $b GB, swap $sw0 MB)"
      if [ -n "$STUB" ]; then ( exec $PY $STUB $s $k $i/$NP ) >> $L 2>&1 &
      else ( exec $PY scripts/dd_tree.py --set $s --nodes 127 --k $k --part $i/$NP ) >> $L 2>&1 & fi
      CP=$!
      ( while kill -0 $CP 2>/dev/null; do sleep 10; [ $(( $(swapmb) - sw0 )) -gt 2000 ] && { echo "$(date '+%m-%d %H:%M:%S') swap guard: +2 GB, stopping $s k $k part $i" >> $L; kill -TERM $CP; break; }
          [ -e $Q/HOLD ] && { kill -TERM $CP; break; }; done ) &
      GUARD=$!
      wait $CP; rc=$?; CP=""; kill $GUARD 2>/dev/null; GUARD=""
      say "$s k $k part $i/$NP rc $rc ($(( $(date +%s) - t0 )) s, swap $(( $(swapmb) - sw0 )) MB)"
      [ $rc = 0 ] && break
      [ -e $Q/HOLD ] && { STATUS="$STATUS; $s k $k part $i stopped by HOLD"; RC=5; exit 0; }
      try=$((try + 1)); [ $try -ge 2 ] && { STATUS="$STATUS; $s k $k part $i/$NP failed twice (rc $rc)"; RC=$rc; exit 0; }
      say "  retry $s k $k part $i/$NP in 60 s"; sleep 60
    done
  done
  if [ -n "$STUB" ]; then mrc=0; else
    ( exec $PY scripts/dd_tree.py --set $s --nodes 127 --k $k --merge $NP ) >> $L 2>&1; mrc=$?    # CPU: concat + report
  fi
  say "$s k $k merge rc $mrc"
  [ $mrc = 0 ] || { STATUS="$STATUS; $s k $k merge failed (rc $mrc)"; RC=$mrc; exit 0; }
  j=$C/tree_t30b_${s}_n127k${k}.json
  [ -n "$STUB" ] || say "  $(python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print('anchors', d['anchors'], 'wall', d['wall_s'], 's, MLX peak', d['mlx_peak_gb'], 'GB; nest', (d.get('nest_vs_32node_dump') or {}).get('result', '-'), (d.get('nest_vs_32node_dump') or {}).get('windows', ''), 'windows; tree_32/64/127 kept (walk starts)', [d['preview_kept_per_check']['walk_starts'].get(f'tree_{n}') for n in (32, 64, 127)])" $j 2>&1 | cut -c1-300)"
  DONE_RUNS="$DONE_RUNS $s:k$k"
done
STATUS="all 4 runs done:$DONE_RUNS"; RC=0
