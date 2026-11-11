#!/bin/sh
# Fit12 (Tue 3 Nov 19:3x): dry test of fit_t3.py's T3R2 harness (F12_FAKE=1: no model, no GPU, CPU only, ~1 min). A scratch
# PAUSE file stands in for ~/.local/share/quiet/PAUSE (never touched). (1) a 1.5 s pause during the main pass: the prompts it
# touched are VOID, their files kept as void1_*, re-run once at the end (ok); (2) a pause that never clears: the touched
# prompts are re-run, paused again, excluded (void2_*); (Tue 22:3x, the 13:40 hard stop) (3) a pause, then the process cut
# mid main pass; (4) a pause that never clears, the process cut mid re-runs: no paused prompt may count. Then the T3R2
# read-out on all four outputs.
D=${1:-/private/tmp/claude-501/-Users-kamen-OjoX/7dd61b7e-2c45-43ec-b373-7768d2c96a18/scratchpad/t3r2dry}
PY=/Users/kamen/BigMoE/.venv/bin/python
rm -rf "$D"; mkdir -p "$D/o1" "$D/o2" "$D/o3" "$D/o4"
K=humaneval/HumanEval
KEYS=$K/156,$K/42,$K/68,$K/91,$K/16,$K/6,$K/79,$K/19,$K/161,$K/17,$K/56,$K/141,$K/101,$K/63,$K/114,$K/77
run() {  # run OUT PAUSE_AT PAUSE_FOR [KILL_AT] (s from the process start; FOR 0 = never cleared; KILL_AT: SIGTERM then)
  env F12_FAKE=1 F12_OUT="$D/$1" F12_CAP=64 F12_FAKE_DT=0.05 F12_PHLOG=1 F12_ALT=1 F12_RERUN=1 F12_PAUSEF="$D/$1.PAUSE" \
    F12_COOL_C=99 F12_COOL_S=3 F12_KEYS=$KEYS F12_T3ARMS=t80ck2,t80ck1um3,t80ck1u $PY -B /Users/kamen/Fit12/scripts/fit_t3.py r47 \
    > "$D/$1/r47.txt" 2>&1 &
  P=$!
  [ -n "$4" ] && ( sleep "$4"; kill -TERM $P 2>/dev/null ) &
  sleep "$2"; touch "$D/$1.PAUSE"
  [ "$3" -gt 0 ] && { sleep "$3"; rm -f "$D/$1.PAUSE"; }
  wait $P; echo "$1 rc $?"; rm -f "$D/$1.PAUSE"
}
run o1 6 2 &
run o2 6 0 &
run o3 6 2 14 &
run o4 6 0 30 &
wait
for o in o1 o2 o3 o4; do
  echo "== $o"; grep -E 'PAIR|PAUSE summary|TOTAL|done|Error|Traceback' "$D/$o/r47.txt" | sed 's/^\[r47\] //' | cut -c1-200
  echo "void1 files: $(ls "$D/$o" | grep -c '^void1_'), void2 files: $(ls "$D/$o" | grep -c '^void2_'), live ph files: $(ls "$D/$o" | grep -c '_ph.json$' ) (minus void: $(ls "$D/$o" | grep '_ph.json$' | grep -vc '^void'))"
done
for o in o1 o2 o3 o4; do echo "== read-out $o"; $PY -B /Users/kamen/Fit12/scripts/t3r2_verdict.py "$D/$o" r47 | grep -v -E " PAIR |PAUSE summary| TOTAL " | cut -c1-260; done
