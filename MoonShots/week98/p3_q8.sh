#!/bin/sh
# WEEK98 Q8 + SCREENS (MoonShots; NewIdeas Mon 16 Nov 21:1x and 22:3x; amendment 2 Tue 17 Nov 09:5x, the orchestrator's
# r57 format decision: t30b runs at Q8 + 32K if Q8 holds its line, else bf16 + 32K): on the 125B, M3's harness
# (p3_dd_tree.py, gate PASSed Mon 08:4x), in the amended order:
#   Q8     Q4 with every 4 -> 8 (t30b's trained Linears + the head's HC mixes at 8 bits g64, router 8; full head, top-10)
#   Q8H32  Q8's weights with the 32K screen (the decided config)
#   H32    the screen: FR-Spec top-32K + the session extras (screen_masks, as H8); fp32 weights
#   B16H32 t30b's trained tensors rounded to bf16 + the 32K screen (the fallback config)
#   H16    the screen at top-16K
# Each cut on both sets (heplus24 first, then mbppbank); a cut whose dump exists is skipped (so a later run resumes).
# Then scores vs the full head (BigMoE's tree_t30b dumps) to p3/q8_scores.{txt,json}.
# Slots: today in share windows (Q8_CUTOFF=2359), else Wed 18 Nov 05:24 / 06:30 (no dump starts after Q8_CUTOFF, HHMM,
# default 0710, before noon). A dump doesn't start while another session waits for or holds a quiet lock. Share lock
# MEM 12 GB taken by the session before launch; this script drops it at the end. One dump process at a time under the
# memory guard (14 GB cap, swap +1 GB kill), vm_stat gate first.
cd "$HOME/MoonShots/week98" || exit 1
PY="$HOME/BigMoE/.venv/bin/python"
mkdir -p p3
LOG=p3/q8_run.log
CUTS_ORDER="Q8 Q8H32 H32 B16H32 H16"
avail_gb() { vm_stat | awk '/Pages free/ {f=$3} /Pages inactive/ {i=$3} /Pages speculative/ {s=$3} END {gsub("[.]","",f); gsub("[.]","",i); gsub("[.]","",s); printf "%d\n", (f+i+s)*16384/1e9}'; }
memgate() {
  n=0
  while [ "$(avail_gb)" -lt 20 ]; do
    [ $n -eq 0 ] && echo "[q8] waiting for memory: $(avail_gb) GB available < 20 at $(date '+%H:%M:%S')" >> $LOG
    n=$((n + 1)); [ $n -gt 60 ] && { echo "[q8] STOP: memory gate not met in 10 min" >> $LOG; return 4; }
    sleep 10
  done
}
quiet_wanted() {   # another session queued for, or holding, a quiet lock (read-only look at claude-peers' state)
  python3 - <<'EOF'
import glob, json, os, sys, time
now = time.time()
q = glob.glob(os.path.expanduser("~/.claude-peers/queue/*.json"))
held = []
for f in glob.glob(os.path.expanduser("~/.claude-peers/locks/*.json")):
    try:
        d = json.load(open(f))
    except Exception:
        continue
    if d.get("mode") == "quiet" and float(d.get("until", 0)) > now:
        held.append(os.path.basename(f))
sys.exit(0 if (q or held) else 1)
EOF
}
late() { [ "$(date '+%H%M')" -gt "${Q8_CUTOFF:-0710}" ] && [ "$(date '+%H%M')" -lt 1200 ]; }
run() {   # $1 set, $2 cut
  if [ -f "p3/tree_t30b_$1_$2.npz" ]; then echo "[q8] $2 $1 already dumped, skipped" >> $LOG; return 0; fi
  memgate || return 4
  if late; then echo "[q8] cutoff ${Q8_CUTOFF:-0710}: $2 $1 not started ($(date '+%H:%M:%S'))" >> $LOG; return 8; fi
  if quiet_wanted; then echo "[q8] another session waits for / holds a quiet lock: $2 $1 deferred ($(date '+%H:%M:%S'))" >> $LOG; return 8; fi
  echo "[q8] $2 $1 start $(date '+%H:%M:%S'), available $(avail_gb) GB" >> $LOG
  MAXGB=14 SWAPGB=1 LOG=p3/dump_$2_$1.log sh jcc_guard.sh "$PY" -I -B p3_dd_tree.py --set "$1" --cut "$2"
  rc=$?; echo "[q8] $2 $1 rc $rc at $(date '+%H:%M:%S')" >> $LOG; return $rc
}
echo "[q8] start $(date '+%a %d %b %H:%M:%S'), away $(sh $HOME/.local/share/quiet/away_secs.sh) s, cutoff ${Q8_CUTOFF:-0710}" >> $LOG
stop=0
for c in $CUTS_ORDER; do
  for s in heplus24 mbppbank; do
    run $s $c; rc=$?
    [ $rc -eq 8 ] && { stop=1; break; }
    [ $rc -ne 0 ] && echo "[q8] $c $s FAILED rc $rc (continuing with the next cut)" >> $LOG && break
  done
  [ $stop -eq 1 ] && break
done
peer unlock
echo "[q8] dumps done, lock released $(date '+%H:%M:%S')" >> $LOG
"$PY" -I -B p3_score.py --cuts none,Q4,H8,Q8,Q8H32,H32,B16H32,H16 --tag q8 >> $LOG 2>&1
echo "[q8] scored $(date '+%H:%M:%S') rc $?" >> $LOG
all=1
for c in $CUTS_ORDER; do for s in heplus24 mbppbank; do [ -f "p3/tree_t30b_${s}_${c}.npz" ] || all=0; done; done
[ $all -eq 1 ] && touch p3/Q8_DONE
