#!/bin/bash
# CERTIFY98's freeze (docs/PREREG_certify98_2026-11-15.md amendment 10, section 3). Waits for the capture's
# files_sha256.json, checks it (tools/certify98_freeze_check.py: the v2 manifest, the capture list's 647 windows, every
# file re-hashed), commits a copy as results/certify98/files_sha256_2026-11-18.json, then writes FILES_FROZEN (with the
# commit), which BigMoE's engine count check (iii) waits for. No GPU; one process.
#   bash tools/certify98_freeze.sh [deadline epoch]     # exit 0 frozen, 1 refused, 2 deadline
set -u
ES="$HOME/EvalSuite"                                          # the tools; the CF_* overrides are for the sim test only
REPO="${CF_REPO:-$ES}"
FS="${CF_FS:-$HOME/datasets/deepdraft/certify98/files_sha256.json}"
OUT="$REPO/results/certify98/files_sha256_2026-11-18.json"
MARK="$REPO/results/certify98/FILES_FROZEN"
LOG="$REPO/results/certify98/freeze_2026-11-18.log"
DEADLINE=${1:-$(date -j -f "%Y-%m-%d %H:%M:%S" "2026-11-19 12:00:00" +%s)}
say() { echo "$(date '+%Y-%m-%d %H:%M:%S') $*" | tee -a "$LOG"; }
late() { [ "$(date +%s)" -gt "$DEADLINE" ]; }
[ -e "$MARK" ] && { say "FILES_FROZEN already exists: no second freeze"; exit 1; }
say "armed (pid $$): waiting for $FS; deadline $(date -r "$DEADLINE" '+%Y-%m-%d %H:%M')"
while :; do
  if [ -s "$FS" ]; then
    sleep 20                                  # the writer finishes (one write, at the capture's PASS end)
    m=$(stat -f %m "$FS")
    if "$ES/.venv/bin/python" "$ES/tools/certify98_freeze_check.py" "$FS" >> "$LOG" 2>&1; then break; fi
    say "files_sha256.json present but not valid (above); waiting for a new one"
    while [ -s "$FS" ] && [ "$(stat -f %m "$FS")" = "$m" ]; do late && { say "deadline"; exit 2; }; sleep 30; done
    continue
  fi
  late && { say "deadline: no files_sha256.json"; exit 2; }
  sleep 30
done
cp "$FS" "$OUT" && cd "$REPO" || { say "copy failed"; exit 1; }
for i in 1 2 3 4 5 6 7 8 9 10; do git add -f "$OUT" 2>>"$LOG" && break; say "git add retry $i"; sleep 15; done
PAT="$ES/data/leak_patterns.txt"                               # the leak patterns live in a gitignored file
[ -s "$PAT" ] || { say "no leak pattern file $PAT; no commit"; exit 1; }
H=$(git grep --cached -l -f "$PAT" | wc -l | tr -d ' ')
[ "$H" = "0" ] || { say "leak gate: $H tracked files hit; no commit"; exit 1; }
MSG="CERTIFY98 freeze (amendment 10, section 3): the capture's files_sha256.json, checked (v2 manifest, the 647 listed windows, every file re-hashed), committed before any pass

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
if git ls-files --error-unmatch "$OUT" >/dev/null 2>&1 && git diff --quiet HEAD -- "$OUT"; then
  C=$(git log -1 --format=%H -- "$OUT"); say "already committed, unchanged, in $C"     # e.g. a retry after a stop
else
  ok=0
  for i in 1 2 3 4 5 6 7 8 9 10; do git commit -q -m "$MSG" -- "$OUT" 2>>"$LOG" && { ok=1; break; }; say "git commit retry $i"; sleep 15; done
  [ "$ok" = 1 ] || { say "commit failed"; exit 1; }
  C=$(git rev-parse HEAD)
fi
git push -q origin main 2>>"$LOG" && say "pushed $C" || say "push failed (the commit $C stands locally)"
echo "$C $(date '+%Y-%m-%d %H:%M:%S') $(shasum -a 256 "$OUT" | cut -d' ' -f1)" > "$MARK.tmp" && mv "$MARK.tmp" "$MARK"
say "FROZEN: $C; marker $MARK"
exit 0
