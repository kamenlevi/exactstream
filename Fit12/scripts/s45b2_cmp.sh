#!/bin/sh
# Fit12 S45b-2 (Fri 30 Oct 2026): compare the points of the C sweep (data/s45b2/s45_speed_c<C>.txt, written by the copy of
# Leap's r40s3k_screen.py at the end of a clean run) against PLAN.md 'S45b-2': per prompt tok/s, misses and SSD ms per check,
# REF2-row agreement and the tokens sha per C (equal across C = the exactness check), then each point's summary line and
# the arm's PAUSE flag from the runner log (PAUSE seen = that point's tok/s is void).
# Usage: sh /Users/kamen/Fit12/scripts/s45b2_cmp.sh [C ...]   (default 224 256 288; missing points are listed and skipped)
D=${S45D:-/Users/kamen/Fit12/data/s45b2}; LOG=/Users/kamen/Fit12/logs/summary.txt
CS=${*:-224 256 288}; FILES=
for c in $CS; do
  if [ -f "$D/s45_speed_c$c.txt" ]; then FILES="$FILES $D/s45_speed_c$c.txt"; else echo "C $c: no $D/s45_speed_c$c.txt"; fi
done
[ -n "$FILES" ] || exit 1
echo "== per prompt (tok/s | misses per check | SSD ms per check | REF2 agree | tokens sha)"
awk '
FNR == 1 { n = split(FILENAME, a, "_c"); c = a[n]; sub(/\.txt$/, "", c); C[++nc] = c }
/ \[SDPA classes / {
  k = $1; K[k] = 1
  match($0, /, [0-9.]+ tok\/s/); t = substr($0, RSTART + 2, RLENGTH - 2); sub(/ tok\/s/, "", t); TS[k, c] = t
  match($0, /REF2 row [0-9]+ \/ [0-9]+/); AG[k, c] = substr($0, RSTART + 9, RLENGTH - 9)
  last = k
}
/ S45: per cycle/ {
  match($0, /per check [0-9.]+ misses/); m = substr($0, RSTART + 10, RLENGTH - 10); sub(/ misses/, "", m); MS[last, c] = m
  match($0, /SSD reads [0-9.]+ ms/); s = substr($0, RSTART + 10, RLENGTH - 10); sub(/ ms/, "", s); IO[last, c] = s
  match($0, /tokens sha [0-9a-f]+ \([0-9]+\)/); SH[last, c] = substr($0, RSTART + 11, RLENGTH - 11)
}
END {
  for (k in K) {
    printf "%s\n", k
    eq = 1; first = ""; nosha = 0
    for (i = 1; i <= nc; i++) {
      c = C[i]
      printf "   C %s: %6s tok/s | %6s misses | %6s ms SSD | agree %-9s | sha %s\n", c, TS[k, c], MS[k, c], IO[k, c], AG[k, c], SH[k, c]
      if (SH[k, c] == "") nosha = 1
      if (first == "") first = SH[k, c]; else if (SH[k, c] != first) eq = 0
    }
    printf "   tokens across C: %s\n", (nosha ? "NO TOKEN SHA in some point" : (nc < 2 ? "one point only" : (eq ? "EQUAL" : "DIFFER")))
  }
}' $FILES
echo "== summaries"
for f in $FILES; do grep "^S45 (REF2" "$f" | cut -c1-400; done
echo "== runner log (arms, PAUSE flags, swap trips)"
grep "s45b2/c" "$LOG" | grep "done (rc\|swap trip\|SSD\|skipped\|TIMEOUT\|lock lost" | cut -c1-200 | tail -12
echo "== errors"
grep -l "Traceback\|Error" $FILES /Users/kamen/Fit12/data/scr/s45b2/*.txt 2>/dev/null || echo "none"
