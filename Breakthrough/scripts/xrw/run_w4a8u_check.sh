#!/bin/bash
# run_w4a8u_check.sh (Breakthrough, Wed 18 Nov 2026 07:4x; ALL_LEVERS_PLAN item 7, pre-registered in docs/EXPERT_REWRITE.md
# 'Wed 07:4x'): w4a8u through the FROZEN knob_v2 before BigMoE wires blob 2a. Three short runs, one process at a time,
# under ONE share lock: (1)+(2) w4a8u_check.py at nolsb share 0.3, then 0; (3) Leap's m32 scenarios (xrw_m32.py w4a8u).
# Gates: no HOLD / PAUSE / BREAK98_HOLD / HEALTH ACT, lid open, Kamen away (>= 900 s), NO-SWAP >= 12 GB (two reads 10 s
# apart). Guards per run: swap +2 GB, a 600 s cap, Kamen back -> TERM (never -9), then stop. Unlock on any end.
# Writes data/xrw/w4a8u_check_v2_p0.3.txt, w4a8u_check_v2_p0.txt, m32_w4a8u_knob_v2.txt (+ its stdout) and the log.
cd "$HOME/Breakthrough" || exit 9
Q=$HOME/.local/share/quiet; D=data/xrw; KD=$D/knob_v2; L=$D/w4a8u_check_runner.log
PY="$HOME/BigMoE/.venv/bin/python"
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
diff -q <(grep -E '(tgl16|tgl16v|w4a8|xgrp|xknob)\.py$' $D/knob_v2.md5 | awk '{print $1" "$2}' | sort) \
  <(cd "$KD" && md5 -r tgl16.py tgl16v.py w4a8.py xgrp.py xknob.py | sort) >/dev/null \
  || { say "SKIP: knob_v2 does not match knob_v2.md5"; exit 3; }
a1=$(ns); sleep 10; a2=$(ns); a=$(( a1 < a2 ? a1 : a2 ))
[ "$a" -ge 12 ] || { say "SKIP: NO-SWAP $a GB < 12"; exit 3; }
peer lock share 20m --cpu 100 --gpu 30 "MEM 4 GB: Breakthrough w4a8u release check through knob_v2 (ALL_LEVERS item 7; 3 short runs)" >> "$L" 2>&1 \
  || { say "SKIP: the share lock was refused"; exit 3; }
trap 'peer unlock >/dev/null 2>&1' EXIT
sw0=$(swmb); say "locked (NO-SWAP $a GB, swap $sw0 MB, away $(away) s)"
run() {   # $1 = label, $2 = output file, rest = the command (env words first)
  local lab=$1 out=$2; shift 2
  say "run $lab -> $out"
  env "$@" > "$out" 2>&1 &
  local py=$! t0; t0=$(date +%s); why=""; pk=0
  while kill -0 $py 2>/dev/null; do
    sleep 2
    local c r; c=$(pgrep -P $py | head -1); r=$(ps -o rss= -p "${c:-$py}" 2>/dev/null | tr -d ' ')
    [ -n "$r" ] && [ "$r" -gt "$pk" ] && pk=$r
    [ "$(swmb)" -gt $((sw0 + 2048)) ] && why="swap grew > 2 GB"
    [ $(( $(date +%s) - t0 )) -ge 600 ] && why="the 600 s cap"
> (a quote from the project owner, kept private)
    [ -n "$why" ] && { say "stop $lab: $why"; kill -TERM $py 2>/dev/null; break; }
  done
  wait $py; local rc=$?
  say "ended $lab rc $rc in $(( $(date +%s) - t0 )) s; peak RSS $((pk / 1024)) MB${why:+ ($why)}; swap $(swmb) MB"
  [ -z "$why" ]
}
run p0.3 $D/w4a8u_check_v2_p0.3.txt XRW_NOLSB_P=0.3 "$PY" -I -B scripts/xrw/w4a8u_check.py "$KD" \
  && run p0 $D/w4a8u_check_v2_p0.txt XRW_NOLSB_P=0 "$PY" -I -B scripts/xrw/w4a8u_check.py "$KD" \
  && run m32 $D/m32_w4a8u_knob_v2.stdout.txt "$PY" -I -B scripts/xrw/xrw_m32.py w4a8u "$KD"
peer unlock >/dev/null 2>&1; trap - EXIT
say "unlocked"
grep -h -E '^W4A8U CHECK|FAIL|False|Error|error' $D/w4a8u_check_v2_p0.3.txt $D/w4a8u_check_v2_p0.txt 2>/dev/null | cut -c1-300 | tee -a "$L"
m=$D/m32_w4a8u_knob_v2.txt
[ -e "$m" ] && say "m32: scenario lines $(grep -c 'base posts cover' $m); posts cover True $(grep -c -E 'base posts cover the [0-9]+ used experts: True' $m); 32 == 16+16 True $(grep -c 'bit for bit: True' $m); $(grep -h 'OOO1C M32 on' $m | cut -c1-200)"
