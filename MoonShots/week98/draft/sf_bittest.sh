#!/bin/sh
# DRAFT LINE piece 1: the SYNC-FREE bit test on the model (MoonShots, Tue 17 Nov; pre-registered LEVERS_A "SYNC-FREE").
# sf_tree.py's HS / SF / REF arms at B = 1 in four runs, one process at a time under the memory guard (14 GB cap, swap
# +1 GB kill), vm_stat gate first. A run doesn't start while another session waits for or holds a quiet lock.
# Share lock "MEM 12 GB" taken by the session before launch; this script drops it at the end.
# Outputs: draft/sf_bit_<tag>.json + draft/sf_bittest.log.
cd "$HOME/MoonShots/week98/draft" || exit 1
PY="$HOME/BigMoE/.venv/bin/python"
LOG=sf_bittest.log
avail_gb() { vm_stat | awk '/Pages free/ {f=$3} /Pages inactive/ {i=$3} /Pages speculative/ {s=$3} END {gsub("[.]","",f); gsub("[.]","",i); gsub("[.]","",s); printf "%d\n", (f+i+s)*16384/1e9}'; }
memgate() {
  n=0
  while [ "$(avail_gb)" -lt 20 ]; do
    [ $n -eq 0 ] && echo "[sf] waiting for memory: $(avail_gb) GB available < 20 at $(date '+%H:%M:%S')" >> $LOG
    n=$((n + 1)); [ $n -gt 60 ] && { echo "[sf] STOP: memory gate not met in 10 min" >> $LOG; return 4; }
    sleep 10
  done
}
quiet_wanted() {
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
run() {   # $1 tag, then sf_tree.py's arguments
  tag=$1; shift
  if [ -f "sf_bit_$tag.json" ]; then echo "[sf] $tag done before, skipped" >> $LOG; return 0; fi
  memgate || return 4
  if quiet_wanted; then echo "[sf] another session waits for / holds a quiet lock: $tag deferred ($(date '+%H:%M:%S'))" >> $LOG; return 8; fi
  echo "[sf] $tag start $(date '+%H:%M:%S'), available $(avail_gb) GB: $*" >> $LOG
  MAXGB=14 SWAPGB=1 LOG=$LOG sh ../jcc_guard.sh "$PY" -I -B sf_tree.py "$@" --out "sf_bit_$tag.json"
  rc=$?; echo "[sf] $tag rc $rc at $(date '+%H:%M:%S')" >> $LOG; return $rc
}
echo "[sf] start $(date '+%a %d %b %H:%M:%S'), away $(sh $HOME/.local/share/quiet/away_secs.sh) s" >> $LOG
for spec in "he16 --set heplus24 --windows 0,1 --nodes 16 --cut none --max-anchors 48" \
            "mb16 --set mbppbank --windows 0,1,2,3 --nodes 16 --cut none --max-anchors 48" \
            "he16q8h32 --set heplus24 --windows 0,1 --nodes 16 --cut Q8H32 --max-anchors 48" \
            "he32 --set heplus24 --windows 0,1 --nodes 32 --cut none --max-anchors 32"; do
  run $spec; rc=$?
  [ $rc -eq 8 ] && break
  [ $rc -ne 0 ] && echo "[sf] FAILED rc $rc: $spec" >> $LOG
done
peer unlock
echo "[sf] done, lock released $(date '+%H:%M:%S')" >> $LOG
"$PY" -I -B - >> $LOG 2>&1 <<'EOF'
import glob, json
for f in sorted(glob.glob("sf_bit_*.json")):
    r = json.load(open(f))
    mp = r["ms_per_pop"]
    print(f"{f}: PASS {r['PASS']} | trees {r['trees']}, SF == HS {r['sf_eq_hs']}, bad {r['sf_bad']} | REF tree diff "
          f"{r['ref_tree_diff']}, tp cells off {r['ref_tp_cells_diff']} / {r['ref_tp_cells']} (max {r['ref_tp_max_ulp']} ulp) | "
          f"ms per pop: " + ", ".join(f"{k} {v['median']:.3f} [{v['p10']:.3f}-{v['p90']:.3f}]" for k, v in mp.items() if v)
          + f" | peak {r['peak_gb']} GB" + (f" | first fail {r['first_fail']}" if r.get("first_fail") else ""))
EOF
