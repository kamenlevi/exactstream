#!/bin/sh
# WEEK98 C2 = MTP Head's path A for a tree-16 head (MoonShots; NewIdeas Tue 17 Nov 09:2x; pre-registered LEVERS_A "C2"):
# M3's harness (p3_dd_tree.py, gate PASS Mon 08:4x) with M3's ladder cut C2 = C1 (4-bit dense + the HC mixes at 4, router
# 8, FR-Spec top-8K + the session extras) with the routed experts at top-4: 76.7 MB per level (p3_m012_stock.log).
# heplus24 then mbppbank; then scores of none / C1 / C2 / C3 vs the full head to p3/c2_scores.{txt,json}.
# Share lock "MEM 12 GB" taken by the session before launch; this script drops it after the dumps.
# A dump is not started after C2_CUTOFF (HHMM, default 1005), while another session waits for or holds a quiet lock,
# or when its npz already exists (a deferred set re-runs with the same script later).
cd "$HOME/MoonShots/week98" || exit 1
PY="$HOME/BigMoE/.venv/bin/python"
mkdir -p p3
LOG=p3/c2_run.log
avail_gb() { vm_stat | awk '/Pages free/ {f=$3} /Pages inactive/ {i=$3} /Pages speculative/ {s=$3} END {gsub("[.]","",f); gsub("[.]","",i); gsub("[.]","",s); printf "%d\n", (f+i+s)*16384/1e9}'; }
memgate() {
  n=0
  while [ "$(avail_gb)" -lt 20 ]; do
    [ $n -eq 0 ] && echo "[c2] waiting for memory: $(avail_gb) GB available < 20 at $(date '+%H:%M:%S')" >> $LOG
    n=$((n + 1)); [ $n -gt 60 ] && { echo "[c2] STOP: memory gate not met in 10 min" >> $LOG; return 4; }
    sleep 10
  done
}
quiet_wanted() {   # another session queued for, or holding, a quiet lock (read-only look at claude-peers' state)
  python3 - "$$" <<'EOF'
import glob, json, os, sys, time
me, now = int(sys.argv[1]), time.time()
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
late() { [ "$(date '+%H%M')" -gt "${C2_CUTOFF:-1005}" ]; }
run() {   # $1 set, $2 cut
  if [ -f "p3/tree_t30b_$1_$2.npz" ]; then echo "[c2] $2 $1 already dumped, skipped" >> $LOG; return 0; fi
  memgate || return 4
  if late; then echo "[c2] cutoff ${C2_CUTOFF:-1005}: $2 $1 deferred ($(date '+%H:%M:%S'))" >> $LOG; return 8; fi
  if quiet_wanted; then echo "[c2] another session waits for / holds a quiet lock: $2 $1 deferred ($(date '+%H:%M:%S'))" >> $LOG; return 8; fi
  echo "[c2] $2 $1 start $(date '+%H:%M:%S'), available $(avail_gb) GB" >> $LOG
  MAXGB=14 SWAPGB=1 LOG=p3/dump_$2_$1.log sh jcc_guard.sh "$PY" -I -B p3_dd_tree.py --set "$1" --cut "$2"
  rc=$?; echo "[c2] $2 $1 rc $rc at $(date '+%H:%M:%S')" >> $LOG; return $rc
}
echo "[c2] start $(date '+%a %d %b %H:%M:%S'), away $(sh $HOME/.local/share/quiet/away_secs.sh) s" >> $LOG
for s in heplus24 mbppbank; do
  run $s C2; rc=$?
  [ $rc -eq 8 ] && break
  [ $rc -ne 0 ] && { echo "[c2] C2 $s FAILED rc $rc" >> $LOG; break; }
done
peer unlock
echo "[c2] dumps done, lock released $(date '+%H:%M:%S')" >> $LOG
"$PY" -I -B p3_score.py --cuts none,C1,C2,C3 --tag c2 >> $LOG 2>&1
echo "[c2] scored $(date '+%H:%M:%S') rc $?" >> $LOG
[ -f p3/tree_t30b_heplus24_C2.npz ] && [ -f p3/tree_t30b_mbppbank_C2.npz ] && touch p3/C2_DONE
