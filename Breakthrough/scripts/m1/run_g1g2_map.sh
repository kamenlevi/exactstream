#!/bin/bash
# run_g1g2_map.sh (Breakthrough, Mon 16 Nov 2026 13:xx): the G1 / G2 map pre-screens, pre-registered in
# docs/CHECK_BYTES_900.md section 2 (9ac2850). NewIdeas' GO 12:4x and order 12:5x: BigMoE's G1 key test first, then
# this screen, all before 22:30 (BigMoE's M4 at 23:05). EvalSuite's real end marker is MON3 (12:5x); BigMoE's 12:42.
# Waits for BOTH:
#   EvalSuite's resumed 1c-tiers chain end: ~/EvalSuite/results/T1C_MON3_END newer than this runner, with no
#   t1c_chain2.sh process left;
#   BigMoE's G1 key test end: ~/BigMoE/data/r57/G1A_KEYTEST_DONE with mtime >= Mon 13:00.
# Both are read twice, 30 s apart. A manual $D/GO starts it instead.
# Latest start 22:00 (else SKIP: the screen moves to Tue ~07:00); hard stop 22:28 (TERM, KILL 20 s later; the finished
# arms are kept: one json per arm).
# Then: HOLD / lid / HEALTH ACT skip; a PAUSE is waited out; NO-SWAP >= 26 GB (peak ~18 + 8), then a share lock
# (MEM 18 GB); the run (g1g2_run_map.py, cwd ~/Breakthrough); swap guard +2 GB; unlock; the read-out (g1g2_map.py read);
# an end marker $D/G1G2_MAP_END on every end.
# Simulation overrides: G1G2_D, G1G2_E, G1G2_BM, G1G2_Q, G1G2_ESPAT, G1G2_BM_MIN / G1G2_LATEST / G1G2_HARD (epoch),
# G1G2_NS_MIN, G1G2_NOLOCK=1, G1G2_JOB (a stub command; the read-out is skipped).
cd ~/Breakthrough || exit 9
D=${G1G2_D:-$HOME/Breakthrough/data/m1/g1g2_map}
E=${G1G2_E:-$HOME/EvalSuite/results}
BM=${G1G2_BM:-$HOME/BigMoE/data/r57/G1A_KEYTEST_DONE}
Q=${G1G2_Q:-$HOME/.local/share/quiet}
ESPAT=${G1G2_ESPAT:-tools/t1c_chain[2][.]sh}
NSMIN=${G1G2_NS_MIN:-26}
mkdir -p "$D"
L=$D/run.log
END=$D/G1G2_MAP_END
say() { echo "$(date '+%a %H:%M:%S') $*" >> "$L"; }
ns() { vm_stat | awk '/page size of/ {ps=$8} /^Pages (free|purgeable):/ || /^File-backed pages:/ {v=$NF; sub("[.]","",v); s+=v} END {printf "%d", s*ps/1e9}'; }
swmb() { sysctl -n vm.swapusage | awk '{v=$6; sub("M","",v); printf "%d", v}'; }
T0=$(date +%s)
LATEST=${G1G2_LATEST:-$(date -j -f "%Y-%m-%d %H:%M:%S" "2026-11-16 22:00:00" +%s)}
HARD=${G1G2_HARD:-$(date -j -f "%Y-%m-%d %H:%M:%S" "2026-11-16 22:28:00" +%s)}
BM_MIN=${G1G2_BM_MIN:-$(date -j -f "%Y-%m-%d %H:%M:%S" "2026-11-16 13:00:00" +%s)}
LOCKED=0
fin() { echo "G1G2 MAP END $(date '+%m-%d %H:%M:%S'): $1" > "$END"; say "END: $1"; }
quit() { [ $LOCKED = 1 ] && { peer unlock >/dev/null 2>&1; LOCKED=0; }; say "$1"; fin "$1"; exit 7; }
lid_closed() { ioreg -r -k AppleClamshellState -d 1 | grep -q '"AppleClamshellState" = Yes'; }
stop_checks() {
  [ -e "$Q/HOLD" ] && quit "SKIP ($1): HOLD present"
  lid_closed && quit "SKIP ($1): lid closed"
  grep -q "Verdict: ACT" "$Q/HEALTH.md" 2>/dev/null && quit "SKIP ($1): HEALTH ACT"
  if [ -e "$Q/PAUSE" ]; then
    say "PAUSE present ($1): waiting it out until $(date -r "$LATEST" '+%a %H:%M') at the latest"
    while [ -e "$Q/PAUSE" ] && [ "$(date +%s)" -lt "$LATEST" ]; do
      sleep 10
      [ -e "$Q/HOLD" ] && quit "SKIP ($1): HOLD (during a PAUSE)"
      lid_closed && quit "SKIP ($1): lid closed (during a PAUSE)"
    done
    [ -e "$Q/PAUSE" ] && quit "SKIP ($1): PAUSE still present at the latest start"
    say "PAUSE cleared ($1)"
  fi
}
# EvalSuite's own test (13:0x): T1C_MON3_END with mtime >= 12:58:15 AND its follower (pid 23473,
# tools/t1c_resume_follower.sh; checked by command line, so a reused pid can't fool it) gone; plus no chain process.
ESMIN=${G1G2_ES_MIN:-$(date -j -f "%Y-%m-%d %H:%M:%S" "2026-11-16 12:58:15" +%s)}
ESPID=${G1G2_ESPID:-23473}
ESPIDPAT=${G1G2_ESPIDPAT:-t1c_resume_follower}
es_follower() { ps -p "$ESPID" -o args= 2>/dev/null | grep -q "$ESPIDPAT"; }
es_done() { [ -f "$E/T1C_MON3_END" ] && [ "$(stat -f %m "$E/T1C_MON3_END")" -ge "$ESMIN" ] && ! es_follower && ! pgrep -f "$ESPAT" >/dev/null; }
bm_done() { [ -f "$BM" ] && [ "$(stat -f %m "$BM")" -ge "$BM_MIN" ]; }
yn() { "$@" && echo y || echo n; }
say "armed (pid $$): waits for EvalSuite's T1C_MON3_END (>= $(date -r "$ESMIN" '+%H:%M:%S'), follower pid $ESPID '$ESPIDPAT' gone (alive now: $(yn es_follower)), no '$ESPAT' process) AND BigMoE's $(basename "$BM") (>= $(date -r "$BM_MIN" '+%H:%M')), or $D/GO; latest start $(date -r "$LATEST" '+%a %H:%M'), hard stop $(date -r "$HARD" '+%a %H:%M')"
n=0
while true; do
  if [ -f "$D/GO" ]; then say "manual GO file: $(head -1 "$D/GO" | cut -c1-160)"; break; fi
  if es_done && bm_done; then
    sleep 30
    if es_done && bm_done; then
      say "triggers (read twice): EvalSuite: $(head -1 "$E/T1C_MON3_END" | cut -c1-160) | BigMoE: $(head -1 "$BM" | cut -c1-160)"
      break
    fi
  fi
  if [ "$(date +%s)" -ge "$LATEST" ]; then
    quit "SKIP: no trigger by the latest start (EvalSuite done $(yn es_done), BigMoE done $(yn bm_done)); the screen moves to Tue ~07:00"
  fi
  n=$((n + 1)); [ $((n % 60)) = 0 ] && say "waiting (EvalSuite done $(yn es_done), BigMoE done $(yn bm_done))"
  sleep 60
done
stop_checks "after the trigger"
a=$(ns)
say "NO-SWAP $a GB (need $NSMIN); power: $(pmset -g | grep -iE 'powermode|lowpowermode' | tr -s ' ' | paste -sd ';' -)"
while [ "$a" -lt "$NSMIN" ]; do
  [ "$(date +%s)" -ge "$LATEST" ] && quit "SKIP: NO-SWAP $a < $NSMIN GB at the latest start"
  stop_checks "NO-SWAP wait"
  sleep 20; a=$(ns)
done
[ "$(date +%s)" -ge "$LATEST" ] && quit "SKIP: past the latest start before the run"
if [ -z "${G1G2_NOLOCK:-}" ]; then
  peer lock share 45m --cpu 300 --gpu 60 "Breakthrough: G1/G2 map pre-screens (6 arms x 24 code windows, ~30 min), MEM 18 GB" >> "$L" 2>&1 \
    || quit "SKIP: the share lock was refused"
  LOCKED=1
fi
sw0=$(swmb)
say "running (NO-SWAP $(ns) GB, swap $sw0 MB)"
if [ -n "${G1G2_JOB:-}" ]; then JOB=($G1G2_JOB); else JOB=("$HOME/BigMoE/.venv/bin/python" -I -B scripts/m1/g1g2_run_map.py); fi
"${JOB[@]}" > >(grep --line-buffered -vE "^\[(tokenizer|mtplx|qwen4_exp|moe-sorted)" >> "$L") 2>&1 &
py=$!
stopped=""
while kill -0 $py 2>/dev/null; do
  sleep 10
  s=$(swmb)
  [ "$s" -gt $((sw0 + 2048)) ] && stopped="swap grew > 2 GB ($sw0 -> $s MB)"
  [ "$(date +%s)" -ge "$HARD" ] && stopped="hard stop $(date -r "$HARD" '+%H:%M')"
  { [ -e "$Q/HOLD" ] || lid_closed; } && stopped="HOLD or the lid closed"
  if [ -n "$stopped" ]; then say "stop: $stopped"; kill -TERM $py; sleep 20; kill -9 $py 2>/dev/null; break; fi
done
wait $py; rc=$?
[ $LOCKED = 1 ] && { peer unlock >/dev/null 2>&1; LOCKED=0; }
say "run ended rc $rc${stopped:+ ($stopped)}; swap $(swmb) MB (start $sw0)"
rd="skipped (simulation)"
if [ -z "${G1G2_JOB:-}" ]; then
  if "$HOME/BigMoE/.venv/bin/python" -I -B scripts/m1/g1g2_map.py read >> "$L" 2>&1; then rd="written (data/m1/g1g2_map.out)"; else rd="FAILED (log)"; fi
fi
fin "run rc $rc${stopped:+ ($stopped)}; read-out $rd"
