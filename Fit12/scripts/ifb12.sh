#!/bin/sh
# Fit12 ifb12.sh (Mon 9 Nov 23:4x; v2 Tue 02:0x with EvalSuite's 3 changes, 23:4x message). NewIdeas 23:30, route (i): Fit12 runs
# the 12 GB IFBench arm (P2', the 12 GB chat mode) as the w24 gate's neighbour; EvalSuite scores it afterwards (its amendment 13,
# pre-registered before the first row; tools/set_d4.py ifbench <run> ref2tnaxn.r6). A QUALITY arm: its speeds are rough.
# - harness: the FROZEN /Users/kamen/Fit12/snap/chw1108/fit_t3.py (7a9ffd8375ed), env exactly
#   /Users/kamen/Fit12/data/gate/card12_chw_env.txt's (git blob da1eb63570ee): F12_T3ARMS=ex F12_CAP=item F12_CMCL=47,32
#   F12_W1=cf:0.9:2 F12_W1SEED=1 F12_W1TAIL=1 F12_PR1=1; F12_IDS_OUT per chunk ({key: {"ex": [[ids]]}} + _meta.json);
# - KEYS: the 150 gate keys = the ifbench/ lines of /Users/kamen/EvalSuite/results/card/card_keys.txt, in that order (only these
#   have a NAX6-N reference row); prompts = the items' own prompt_ids, CAP = max_tokens 1536; chunks of CH keys (default 20),
#   one process each; keys already in an ifb12cNN ids file are skipped (resume); a key whose chunk was stopped is not in any
#   file and runs again;
# - REVERSE CHECK (EvalSuite 23:4x): after the 150, the first NREV (10) keys again in reverse order, in their own process
#   (ifb12rev); any token difference makes the arm VOID (ifb12_merge.py reports it);
# - ifb12_merge.py writes ifb12_tokens.json + ifb12_runs.json ([{key, arm "ex", cut false}]) + ifb12_meta.json.
# HOLD: while $O/WAIT exists (released by hand once EvalSuite's amendment 13 is in).
# START of a chunk (polled every 30 s): while the gate runs, its ETA file fresh (<= 30 min) with 'neighbour may start: yes' and
#   not 'passes 13:30: yes'; the FIRST chunk also needs Breakthrough's beside_gate.done; after the gate (a later chunk, the
#   gate's lock gone): alone, with no other MEM > 5 GB lock and no quiet ticket waiting; always NO-SWAP >= 20 GB, no PAUSE /
#   HOLD, before the hard end (arg 1, default Tue 23:00).
# LOCK per chunk: peer lock share, reason starting "MEM 12 GB"; released after every chunk.
# STOP (every 10 s while a chunk runs; ends the whole run, done keys kept): the process's phys_footprint_peak (footprint) >
#   12.0 GB; swap > the run's start + 0.5 GB (the gate's own guard is +2 GB); a fresh ETA file says 'passes 13:30: yes'.
# Dry test: IFB_PY (a fake harness), IFB_O, IFB_ETA, IFB_BGD, IFB_KEYS (a key subset), IFB_CH, IFB_NREV, IFB_PEAKGB and
#   IFB_NOLOCK=1 (no peer lock: a dry test must not show a MEM 12 GB lock to other sessions' gates) override the real ones.
F=/Users/kamen/Fit12; LOG=$F/logs/summary.txt
PYU=/Users/kamen/BigMoE/.venv/bin/python; PY=${IFB_PY:-$PYU}; H=$F/snap/chw1108/fit_t3.py   # PY = the harness's python
O=${IFB_O:-$F/data/scr/ifb12}; ETA=${IFB_ETA:-/Users/kamen/EvalSuite/results/w24_gate_eta.txt}
BGD=${IFB_BGD:-/Users/kamen/Breakthrough/data/m1/beside_gate.done}; CH=${IFB_CH:-20}; NREV=${IFB_NREV:-10}
Q=/Users/kamen/.local/share/quiet; PAUSEF=$Q/PAUSE; HOLDF=$Q/HOLD
END=$(date -j -f "%Y-%m-%d %H:%M" "${1:-2026-11-10 23:00}" +%s) || exit 9
log() { echo "[$(date '+%a %H:%M:%S')] $*"; }
swapmb() { sysctl -n vm.swapusage | sed -E 's/.*used = ([0-9.]+)M.*/\1/' | cut -d. -f1; }
nosw() { vm_stat | awk '/page size of/ {ps=$8} /^Pages (free|speculative|purgeable):/ || /^File-backed pages:/ {v=$NF; sub("[.]","",v); s+=v} END {printf "%d", s*ps/1e9}'; }
others() { peer ls 2>/dev/null | grep "^- "; }
locks() { grep -oE "· (shared run \(others may work\)|benchmarking|waiting for a quiet window)[^·]*"; }
nquiet() { others | locks | grep -cE "^· (benchmarking|waiting for a quiet window)"; }
nbig() { others | locks | grep -oE "MEM [~]?[0-9]+ GB" | tr -d '~' | awk '$2 > 5' | wc -l | tr -d " "; }
eta_fresh() { u=$(sed -n 's/^updated: //p' "$ETA" 2>/dev/null | head -1); [ -n "$u" ] || return 1
  e=$(date -j -f "%Y-%m-%d %H:%M:%S" "$u" +%s 2>/dev/null) || return 1; [ $(( $(date +%s) - e )) -le 1800 ]; }
eta_may() { eta_fresh && grep -q '^neighbour may start: yes' "$ETA" && ! grep -q '^passes 13:30: yes' "$ETA"; }
eta_stop() { eta_fresh && grep -q '^passes 13:30: yes' "$ETA"; }
fpeak() { footprint -p "$1" 2>/dev/null | awk '/phys_footprint_peak:/ {v = $2; u = $3; m = (u == "KB") ? 1024 : (u == "MB") ? 1048576 : (u == "GB") ? 1073741824 : 1; printf "%.3f", v * m / 1e9; exit}'; }
cd $F || exit 9
mkdir -p $O
rm -f $O/DONE
[ -e $O/GO ] || touch $O/WAIT
KEYS=${IFB_KEYS:-$(grep '^ifbench/' /Users/kamen/EvalSuite/results/card/card_keys.txt | paste -sd, -)}
REVK=$(echo "$KEYS" | tr ',' '\n' | head -$NREV | tail -r | paste -sd, -)
done_keys() { $PYU -I -c 'import glob, json, sys
s = set()
for f in glob.glob(sys.argv[1] + "/ifb12c*_ids.json"):
    s |= set(json.load(open(f)))
print(",".join(sorted(s)))' $O; }
left() { d=",$(done_keys),"; echo "$KEYS" | tr ',' '\n' | while read -r k; do case "$d" in *",$k,"*) ;; *) echo "$k";; esac; done | head -$CH | paste -sd, -; }
rc=0; first=1; SW0=$(swapmb); n=$(ls $O 2>/dev/null | grep -cE '^ifb12c[0-9]+_ids\.json$')
[ "$n" -gt 0 ] && first=0
run_chunk() {   # $1 = tag, $2 = keys; sets rc (0, or the stop / failure code)
  T=$1; K_=$2
  while :; do                                     # the gate before the chunk
    [ "$(date +%s)" -ge "$END" ] && { log "hard end reached before $T"; rc=5; return; }
    ok=0
    if [ ! -e $O/WAIT ] && [ ! -e "$PAUSEF" ] && [ ! -e "$HOLDF" ] && [ "$(nosw)" -ge 20 ]; then
      if eta_may && { [ $first -eq 0 ] || [ -e "$BGD" ]; }; then ok=1; mode=neighbour
      elif [ $first -eq 0 ] && [ "$(nbig)" -eq 0 ] && [ "$(nquiet)" -eq 0 ]; then ok=1; mode=alone; fi
    fi
    [ $ok -eq 1 ] && break
    sleep 30
  done
  [ "${IFB_NOLOCK:-0}" = 1 ] || peer lock share 90m --cpu 200 --gpu 50 "MEM 12 GB IFBench 12 GB arm (Fit12), quality" >/dev/null 2>&1
  SWc=$(swapmb); [ $first -eq 1 ] && SW0=$SWc
  log "$T ($mode): $(echo "$K_" | tr ',' '\n' | wc -l | tr -d ' ') keys from $(echo "$K_" | cut -d, -f1); NO-SWAP $(nosw) GB, swap $SWc MB (run start $SW0)"
  echo "$(date +%H:%M:%S) SCR ifb12 $T start ($mode)" >> $LOG
  env F12_OUT=$O F12_T3ARMS=ex F12_CAP=item F12_CMCL=47,32 F12_W1=cf:0.9:2 F12_W1SEED=1 F12_W1TAIL=1 F12_PR1=1 \
    F12_IDS_OUT=$O/${T}_ids.json F12_KEYS=$K_ $PY -B $H $T > $O/$T.txt 2>&1 &
  P=$!; first=0; why=""
  while kill -0 $P 2>/dev/null; do
    sleep 10
    pk=$(fpeak $P)
    if [ -n "$pk" ] && awk -v p="$pk" -v g="${IFB_PEAKGB:-12.0}" 'BEGIN {exit !(p > g)}'; then why="footprint peak $pk GB > ${IFB_PEAKGB:-12.0}"; rc=126
    elif [ $(( $(swapmb) - SW0 )) -gt 512 ]; then why="swap $(swapmb) MB > start $SW0 + 512"; rc=127
    elif eta_stop; then why="the ETA file says passes 13:30: yes"; rc=124; fi
    [ -n "$why" ] && { log "$T: STOP: $why"; pkill -TERM -P $P 2>/dev/null; kill -TERM $P; sleep 5; pkill -KILL -P $P 2>/dev/null; kill -KILL $P 2>/dev/null; break; }
    [ -n "$pk" ] && echo "$pk" > $O/.peak_$T
  done
  wait $P; w=$?
  [ "${IFB_NOLOCK:-0}" = 1 ] || peer unlock >/dev/null 2>&1
  log "$T end rc $w (footprint peak seen $(cat $O/.peak_$T 2>/dev/null || echo n/a) GB; harness: $(grep -o 'done; lifetime [0-9.]* GB' $O/$T.txt | tail -1))"
  echo "$(date +%H:%M:%S) SCR ifb12 $T end (rc $w)" >> $LOG
  [ $rc -eq 0 ] && [ $w -ne 0 ] && { rc=$w; log "$T failed: stopping the run"; }
}
log "ifb12 armed: $(echo "$KEYS" | tr ',' '\n' | wc -l | tr -d ' ') keys, chunks of $CH, $n chunk file(s) already, reverse check of $NREV ($REVK); swap $SW0 MB; held while $O/WAIT exists; hard end $(date -r $END '+%a %H:%M')"
while [ $rc -eq 0 ]; do
  K_=$(left); [ -z "$K_" ] && { log "all $(echo "$KEYS" | tr ',' '\n' | wc -l | tr -d ' ') keys done"; break; }
  n=$((n + 1)); run_chunk "$(printf 'ifb12c%02d' $n)" "$K_"
done
if [ $rc -eq 0 ] && [ ! -s $O/ifb12rev_ids.json ]; then run_chunk ifb12rev "$REVK"; fi
$PYU -I -B $F/scripts/ifb12_merge.py $O "$KEYS" "$REVK" > $O/ifb12_merge.txt 2>&1; cat $O/ifb12_merge.txt
echo "rc $rc ended $(date '+%a %H:%M:%S')" > $O/DONE
echo "$(date +%H:%M:%S) SCR ifb12 run end (rc $rc)" >> $LOG
exit $rc
