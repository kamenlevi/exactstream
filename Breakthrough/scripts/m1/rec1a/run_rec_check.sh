#!/bin/bash
# run_rec_check.sh (Breakthrough, Wed 18 Nov 2026 04:3x; the follow-up pre-registered in docs/MODULE1A.md's Wed 04:31
# entry, de499b6): ONE FUNCTIONAL run of rec_layer.py (now with an end-of-repeats bitwise check per arm, the outputs
# zeroed before each arm's repeats) in the official bench's env, under a share lock. Timings are NOT quotable.
# Gates: no HOLD / PAUSE / BREAK98_HOLD / HEALTH ACT, lid open, Kamen away (>= 900 s), NO-SWAP >= 12 GB (two reads
# 10 s apart). Guards: swap +2 GB, a 300 s cap, Kamen back -> TERM (never -9). Unlock on any end.
# Writes data/m1/rec1a_rset_layer_fast_check.{out,err,runner.log}.
cd "$HOME/Breakthrough" || exit 9
Q=$HOME/.local/share/quiet; o=data/m1/rec1a_rset_layer_fast_check; L=$o.runner.log
say() { echo "$(date '+%a %H:%M:%S') $*" | tee -a "$L"; }
ns() { vm_stat | awk '/page size of/ {ps=$8} /^Pages (free|purgeable):/ || /^File-backed pages:/ {v=$NF; sub("[.]","",v); s+=v} END {printf "%d", s*ps/1e9}'; }
swmb() { sysctl -n vm.swapusage | awk '{v=$6; sub("M","",v); printf "%d", v}'; }
away() { local a; a=$(bash "$Q/away_secs.sh" 2>/dev/null); case "$a" in ''|*[!0-9]*) echo 0;; *) echo "$a";; esac; }
for f in "$Q/HOLD" "$Q/PAUSE" "$HOME/NewIdeas/research/week98/BREAK98_HOLD"; do
  [ -e "$f" ] && { say "SKIP: $f exists"; exit 3; }
done
ioreg -r -k AppleClamshellState -d 1 | grep -q '"AppleClamshellState" = Yes' && { say "SKIP: lid closed"; exit 3; }
grep -q "Verdict: ACT" "$Q/HEALTH.md" 2>/dev/null && { say "SKIP: HEALTH ACT"; exit 3; }
> (a quote from the project owner, kept private)
a1=$(ns); sleep 10; a2=$(ns); a=$(( a1 < a2 ? a1 : a2 ))
[ "$a" -ge 12 ] || { say "SKIP: NO-SWAP $a GB < 12"; exit 3; }
peer lock share 5m --cpu 100 --gpu 30 "MEM 4 GB: Breakthrough module 1a follow-up check (residency-set repeats bitwise; ~10 s)" >> "$L" 2>&1 \
  || { say "SKIP: the share lock was refused"; exit 3; }
trap 'peer unlock >/dev/null 2>&1' EXIT
sw0=$(swmb); say "run (NO-SWAP $a GB, swap $sw0 MB, away $(away) s)"
REC_SCOPE=layer REC_MODES=9 REC_WINDOWS=4 REC_REPS=200 REC_RSET=1 "$HOME/BigMoE/.venv/bin/python" -I -B \
  scripts/m1/rec1a/rec_layer.py --fast > "$o.out" 2> "$o.err" &
py=$!; t0=$(date +%s); why=""
while kill -0 $py 2>/dev/null; do
  sleep 1
  [ "$(swmb)" -gt $((sw0 + 2048)) ] && why="swap grew > 2 GB"
  [ $(( $(date +%s) - t0 )) -ge 300 ] && why="the 300 s cap"
> (a quote from the project owner, kept private)
  [ -n "$why" ] && { say "stop: $why"; kill -TERM $py 2>/dev/null; break; }
done
wait $py; rc=$?
peer unlock >/dev/null 2>&1; trap - EXIT
say "ended rc $rc${why:+ ($why)}; swap $(swmb) MB (start $sw0); unlocked"
grep -E '^\[rec\] (ICB|residency|mode|RESULT)' "$o.out" | cut -c1-300 | tee -a "$L"
