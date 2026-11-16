#!/bin/bash
# run_lband_1117.frozen.sh (Breakthrough, Mon 16 Nov 2026 21:0x): run_lband_1116.frozen.sh + the AWAY RULE + a second
# marker, for NewIdeas 20:5x's slot (Tue morning after Fit12's kill test). Changes only: no trigger while away_secs.sh
# < 900 (read twice); a run yields (TERM; marker written) when two reads 30 s apart say Kamen is back; LB_AFTER2 (my
# COMQOM_END) must also be newer than arming; defaults: latest start Tue 08:45, MEM 22 GB (measured 21.7), NO-SWAP 30.
# Sim hook: LB_AWAY_CMD. Gates, checks, job, read-out and the end marker are otherwise unchanged.
# run_lband_1116.frozen.sh (Breakthrough, Mon 16 Nov 2026 19:5x): run_lband.frozen.sh with ONLY the share lock changed, for
# NewIdeas 19:3x (band screen TONIGHT): the reason leads with 'MEM ${LB_MEM:-24} GB' (today's measured map peak on these 24
# windows: 23.7 GB lifetime), and the lock runs to the hard stop + 5 min (no lapse mid-run). Sim hooks LB_LOCKCMD /
# LB_UNLOCKCMD (default 'peer lock share' / 'peer unlock'). Gates, checks, job and read-out are unchanged.
# run_lband.sh (Breakthrough, Mon 16 Nov 2026 14:0x): the per-layer lambda band screen, docs/CHECK_BYTES_900.md section 6
# (NewIdeas 13:5x: Tue morning, AFTER Fit12's 4-key RUNAWAY kill test, latest start ~09:00; Leap's slot #1 at 09:30).
# Starts once Fit12's kill-test end marker (LB_AFTER, newer than this runner's arming) exists AND no map process runs
# (pgrep 'run_ma[p].py', read twice 30 s apart), not before Tue 07:00. Latest start Tue 09:00, hard stop 09:25 (TERM,
# KILL 20 s later). HOLD / lid / HEALTH ACT skip; a PAUSE is waited out; NO-SWAP >= 24 GB; a share lock (MEM 16 GB);
# swap guard +2 GB; the run (lband_run_map.py, cwd ~/Breakthrough), then the read-out (lband_read.py); an end marker
# on every end. Usage: run_lband.sh <Fit12's kill-test end marker path>
# Simulation overrides: LB_D, LB_Q, LB_START / LB_LATEST / LB_HARD (epoch), LB_PAT, LB_NS_MIN, LB_NOLOCK=1, LB_JOB.
cd ~/Breakthrough || exit 9
D=${LB_D:-$HOME/Breakthrough/data/m1/lband_map}
Q=${LB_Q:-$HOME/.local/share/quiet}
PAT=${LB_PAT:-run_ma[p][.]py}
NSMIN=${LB_NS_MIN:-30}
mkdir -p "$D"
L=$D/run.log
END=$D/LBAND_END
AFTER=${1:?usage: run_lband.sh MARKER_PATH}
AFTER2=${LB_AFTER2:-}
T0=$(date +%s)
say() { echo "$(date '+%a %H:%M:%S') $*" >> "$L"; }
ns() { vm_stat | awk '/page size of/ {ps=$8} /^Pages (free|purgeable):/ || /^File-backed pages:/ {v=$NF; sub("[.]","",v); s+=v} END {printf "%d", s*ps/1e9}'; }
swmb() { sysctl -n vm.swapusage | awk '{v=$6; sub("M","",v); printf "%d", v}'; }
START=${LB_START:-$(date -j -f "%Y-%m-%d %H:%M:%S" "2026-11-17 07:00:00" +%s)}
LATEST=${LB_LATEST:-$(date -j -f "%Y-%m-%d %H:%M:%S" "2026-11-17 08:45:00" +%s)}
HARD=${LB_HARD:-$(date -j -f "%Y-%m-%d %H:%M:%S" "2026-11-17 09:25:00" +%s)}
LOCKED=0
LOCKCMD=${LB_LOCKCMD:-peer lock share}; UNLOCKCMD=${LB_UNLOCKCMD:-peer unlock}; MEMGB=${LB_MEM:-22}
fin() { echo "LBAND END $(date '+%m-%d %H:%M:%S'): $1" > "$END"; say "END: $1"; }
quit() { [ $LOCKED = 1 ] && { $UNLOCKCMD >/dev/null 2>&1; LOCKED=0; }; say "$1"; fin "$1"; exit 7; }
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
mapfree() { ! pgrep -f "$PAT" >/dev/null; }
AWAYCMD=${LB_AWAY_CMD:-bash $HOME/.local/share/quiet/away_secs.sh}
away() { local a; a=$($AWAYCMD 2>/dev/null); case "$a" in ''|*[!0-9]*) echo 900;; *) echo "$a";; esac; }   # no reading = away (as EvalSuite)
away_ok() { [ "$(away)" -ge 900 ]; }
back() { [ "$(away)" -lt 900 ] || return 1; sleep 30; [ "$(away)" -lt 900 ]; }   # Kamen at the Mac: two reads 30 s apart
after_done() { [ -f "$AFTER" ] && [ "$(stat -f %m "$AFTER")" -ge "$T0" ] && { [ -z "$AFTER2" ] || { [ -f "$AFTER2" ] && [ "$(stat -f %m "$AFTER2")" -ge "$T0" ]; }; }; }
> (a quote from the project owner, kept private)
while true; do
  now=$(date +%s)
  if [ "$now" -ge "$START" ] && after_done && mapfree && away_ok; then
    sleep 30
> (a quote from the project owner, kept private)
  fi
> (a quote from the project owner, kept private)
  sleep 60
done
stop_checks "after the trigger"
a=$(ns)
say "NO-SWAP $a GB (need $NSMIN)"
while [ "$a" -lt "$NSMIN" ]; do
  [ "$(date +%s)" -ge "$LATEST" ] && quit "SKIP: NO-SWAP $a < $NSMIN GB at the latest start"
  stop_checks "NO-SWAP wait"
  sleep 20; a=$(ns)
done
if [ -z "${LB_NOLOCK:-}" ]; then
  lm=$(( ( HARD - $(date +%s) ) / 60 + 5 ))
  $LOCKCMD ${lm}m --cpu 300 --gpu 60 "MEM $MEMGB GB: Breakthrough per-layer lambda band screen (4 arms x 24 code windows), to $(date -r "$HARD" '+%H:%M') at the latest" >> "$L" 2>&1 \
    || quit "SKIP: the share lock was refused"
  LOCKED=1
fi
sw0=$(swmb)
say "running (NO-SWAP $(ns) GB, swap $sw0 MB)"
if [ -n "${LB_JOB:-}" ]; then JOB=($LB_JOB); else JOB=("$HOME/BigMoE/.venv/bin/python" -I -B scripts/m1/lband_run_map.py); fi
"${JOB[@]}" > >(grep --line-buffered -vE "^\[(tokenizer|mtplx|qwen4_exp|moe-sorted)" >> "$L") 2>&1 &
py=$!
stopped=""
while kill -0 $py 2>/dev/null; do
  sleep 10
  s=$(swmb)
  [ "$s" -gt $((sw0 + 2048)) ] && stopped="swap grew > 2 GB ($sw0 -> $s MB)"
  [ "$(date +%s)" -ge "$HARD" ] && stopped="hard stop $(date -r "$HARD" '+%H:%M')"
  { [ -e "$Q/HOLD" ] || lid_closed; } && stopped="HOLD or the lid closed"
> (a quote from the project owner, kept private)
  if [ -n "$stopped" ]; then say "stop: $stopped"; kill -TERM $py; sleep 20; kill -9 $py 2>/dev/null; break; fi
done
wait $py; rc=$?
[ $LOCKED = 1 ] && { $UNLOCKCMD >/dev/null 2>&1; LOCKED=0; }
say "run ended rc $rc${stopped:+ ($stopped)}; swap $(swmb) MB (start $sw0)"
rd="skipped (simulation)"
if [ -z "${LB_JOB:-}" ]; then
  if "$HOME/BigMoE/.venv/bin/python" -I -B scripts/m1/lband_read.py >> "$L" 2>&1; then rd="written (data/m1/lband_map.out)"; else rd="FAILED (log)"; fi
fi
fin "run rc $rc${stopped:+ ($stopped)}; read-out $rd"
