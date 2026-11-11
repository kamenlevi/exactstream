#!/bin/bash
# release/convert_full.sh <hard-stop epoch>
# Full expert-store conversion test (orchestrator, Mon 12:44): the TESTED package clone's convert/convert_store.sh on every
# layer of ~/BigMoE/models/bare into a temporary folder, then a byte compare of every layer file and of index.json with the
# measured store (records3), then the temporary store is deleted. Never writes to the model folder or its records3.
# Watchdog: the hard stop, swap growth > 2 GB, free disk < 40 GB -> stop convert_store.sh (by pid, children first).
# Run attached (run_in_background) inside a share lock taken beforehand. Last log line: CONVERT_FULL DONE / STOPPED / NOT STARTED.
set -uo pipefail
STOP=${1:?hard-stop epoch}
CLONE=/Users/kamen/Leap/release/cleanclone/final/clone
MD=/Users/kamen/BigMoE/models/bare
REF=$MD/records3
T=/Users/kamen/Leap/release/cleanclone/convert_full
OUT=$T/records3
LOG=$T/convert_full.log
PY=$CLONE/.venv/bin/python
DONE=/Users/kamen/Leap/data/release/convert_full.done   # Fit12's chw19 gate waits for this file (NewIdeas 12:51): always written
mkdir -p "$T"
say() { echo "$(date '+%m-%d %H:%M:%S') $*" | tee -a "$LOG"; }
CPID=""
stop_child() { if [ -n "$CPID" ] && kill -0 "$CPID" 2>/dev/null; then for c in $(pgrep -P "$CPID"); do kill -TERM "$c" 2>/dev/null; done; kill -TERM "$CPID" 2>/dev/null; fi; }
trap 'say "runner stopped by a signal (convert_store stopped; the temporary store may remain: delete $OUT and $OUT.work)"; stop_child; exit 143' TERM INT
trap 'echo "$(tail -1 "$LOG")" > "$DONE"' EXIT
free_gb() { df -g "$T" | tail -1 | awk '{print $4}'; }
swap_mb() { sysctl -n vm.swapusage | sed -E 's/.*used = ([0-9.]+)M.*/\1/' | cut -d. -f1; }
[ -e "$OUT" ] && { say "NOT STARTED: $OUT exists (remove it first)"; exit 3; }
case "$OUT" in /Users/kamen/Leap/release/cleanclone/convert_full/*) ;; *) say "NOT STARTED: unexpected OUT $OUT"; exit 3;; esac
F0=$(free_gb); S0=$(swap_mb)
say "start: clone $(git -C "$CLONE" log --oneline -1 | cut -c1-60); model $MD -> $OUT; free $F0 GB; swap $S0 MB; hard stop $(date -r "$STOP" '+%H:%M:%S')"
[ "$F0" -ge 100 ] || { say "NOT STARTED: free disk $F0 GB < 100 GB"; exit 3; }
[ "$(date +%s)" -lt "$((STOP - 600))" ] || { say "NOT STARTED: less than 10 min before the hard stop"; exit 3; }
export VECLIB_MAXIMUM_THREADS=4 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
T0=$(date +%s)
bash "$CLONE/convert/convert_store.sh" "$MD" "$OUT" > "$T/convert_store.stdout" 2>&1 &
CPID=$!
say "convert_store.sh pid $CPID"
WHY=""; FMIN=$F0; SMAX=$S0
while kill -0 "$CPID" 2>/dev/null; do
  sleep 10
  f=$(free_gb); s=$(swap_mb); [ "$f" -lt "$FMIN" ] && FMIN=$f; [ "$s" -gt "$SMAX" ] && SMAX=$s
  if [ "$(date +%s)" -ge "$STOP" ]; then WHY="hard stop"
  elif [ $((s - S0)) -gt 2048 ]; then WHY="swap grew $((s - S0)) MB"
  elif [ "$f" -lt 40 ]; then WHY="free disk $f GB < 40"; fi
  if [ -n "$WHY" ]; then
    say "STOPPING convert_store.sh ($WHY): pid $CPID and its children"
    for c in $(pgrep -P "$CPID"); do kill -TERM "$c" 2>/dev/null; done; kill -TERM "$CPID" 2>/dev/null
    break
  fi
done
wait "$CPID"; RC=$?
say "convert_store.sh rc $RC after $(( $(date +%s) - T0 )) s; free min $FMIN GB, swap max $SMAX MB (start $S0)"
cp "$OUT.work/convert_store_report.txt" "$T/convert_store_report.txt" 2>/dev/null
grep -E "^[0-9-]+ [0-9:]+ (L[0-9]+:|STORE DONE|convert_store:)" "$T/convert_store_report.txt" 2>/dev/null | tail -3 | sed 's/^/  /' | tee -a "$LOG"
# byte compare: every reference layer file + index.json
eq=0; ne=0; miss=0; NE=""; MISS=""
for f in "$REF"/L*.bin; do
  b=$(basename "$f")
  if [ ! -f "$OUT/$b" ]; then miss=$((miss + 1)); MISS="$MISS $b"
  elif cmp -s "$f" "$OUT/$b"; then eq=$((eq + 1))
  else ne=$((ne + 1)); NE="$NE $b"; fi
done
extra=$(ls "$OUT" 2>/dev/null | grep -c -v -E '^(L[0-9]+\.bin|index\.json)$')
say "layer files: $eq byte-identical, $ne differ${NE:+ ($NE)}, $miss missing${MISS:+ ($MISS)} (reference: $(ls "$REF"/L*.bin | wc -l | tr -d ' ') files); other files in the new store: $extra"
if cmp -s "$REF/index.json" "$OUT/index.json"; then say "index.json: byte-identical"
else
  say "index.json: bytes differ (or missing); entries:"
  "$PY" -I -c '
import json, sys
a = json.load(open(sys.argv[1])); b = json.load(open(sys.argv[2]))
la, lb = a.get("layers", {}), b.get("layers", {})
top = sorted(k for k in set(a) | set(b) if k != "layers" and a.get(k) != b.get(k))
same = sum(1 for k in la if lb.get(k) == la[k]); diff = sorted((k for k in la if k in lb and lb[k] != la[k]), key=int)
print(f"  layers in reference {len(la)}, in new {len(lb)}; entries equal {same}; differ {diff}; missing {sorted(set(la) - set(lb), key=int)}; top-level keys differing {top}")
' "$REF/index.json" "$OUT/index.json" 2>&1 | tee -a "$LOG"
fi
du -sg "$OUT" 2>/dev/null | awk '{print "  new store size " $1 " GB (du)"}' | tee -a "$LOG"
# delete the temporary store (only under $T)
rm -rf "$OUT" "$OUT.work"
say "temporary store deleted; free $(free_gb) GB"
if [ -n "$WHY" ]; then say "CONVERT_FULL STOPPED ($WHY)"
elif [ "$RC" -eq 0 ] && [ "$ne" -eq 0 ] && [ "$miss" -eq 0 ]; then say "CONVERT_FULL DONE: PASS"
else say "CONVERT_FULL DONE: FAIL (rc $RC, differ $ne, missing $miss)"; fi
