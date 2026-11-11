#!/bin/sh
# Fit12 chw19 (Sun 8 Nov; pre-registered in /Users/kamen/Fit12/PLAN_12GB_90_fit12.md 'T2 C2c: STRATA'S CHAT WIDTH'): ONE share
# lock (MEM 12 GB), three processes of the live scripts (t3_twopass 7d424eb4a4fe with F12_WL; fit_t3 42266985905b), ex +
# PR1 + tail:
#  chw19c: C2c = F12_W1=2 F12_WL=2 (one draft per check, then 2-row calls until the first miss), 12 IFEval, CAP 128
#  chw19p: P2' (cf:0.9:2, WL 4), the same prompts (the same-lock reference; tokens + lsha must = chw14c2 12 / 12)
#  chw19h: C2c on 16 HumanEval, CAP 256 (vs chw14ha / chw14h2)
# GATE as chw18.sh: 6 CLEAR minutes (1 once $O/GO exists; held while $O/WAIT exists), then no quiet ticket for 5 min, the
# share lock, an idle GPU. Starts by DL (arg 1, default "2026-11-08 22:15") or not at all. Stops on swap +1 GB or 25 min.
# Reports: $O/chw19_chat.txt (chw19c vs chw19p), chw19_rep.txt (chw19p vs chw14c2), chw19_code.txt (chw19h vs chw14ha),
# chw19_rows.txt (whole decode / whole request / first token / peak per run, paper_rows.py), chw19_widths.txt.
F=/Users/kamen/Fit12; PY=/Users/kamen/BigMoE/.venv/bin/python; LOG=$F/logs/summary.txt; O=$F/data/scr/chw19
Q=/Users/kamen/.local/share/quiet; PAUSEF=$Q/PAUSE; HOLDF=$Q/HOLD; HEALTH=$Q/HEALTH.md; SN=$F/snap/chw1108
IK=ifeval/1220,ifeval/3280,ifeval/1561,ifeval/2853,ifeval/374,ifeval/3407,ifeval/2239,ifeval/2590,ifeval/2870,ifeval/2439,ifeval/1219,ifeval/1262
K=humaneval/HumanEval
HK=$K/156,$K/42,$K/68,$K/91,$K/16,$K/6,$K/79,$K/19,$K/161,$K/17,$K/56,$K/141,$K/101,$K/63,$K/114,$K/77
DL=$(date -j -f "%Y-%m-%d %H:%M" "${1:-2026-11-08 22:15}" +%s) || exit 9
log() { echo "[$(date '+%a %H:%M:%S')] $*"; }
swapmb() { sysctl -n vm.swapusage | sed -E 's/.*used = ([0-9.]+)M.*/\1/' | cut -d. -f1; }
freegb() { vm_stat | awk '/Pages free|Pages inactive|Pages speculative|Pages purgeable/ {gsub(/\./,"",$NF); s+=$NF} END {printf "%d", s*16384/1e9}'; }
others() { peer ls 2>/dev/null | grep "^- "; }
locks() { grep -oE "· (shared run \(others may work\)|benchmarking|waiting for a quiet window)[^·]*"; }
nquiet() { others | locks | grep -cE "^· (benchmarking|waiting for a quiet window)"; }
nbig() { others | locks | grep -oE "MEM [~]?[0-9]+ GB" | tr -d '~' | awk '$2 > 5' | wc -l | tr -d " "; }
gpu() { peer top 2>/dev/null | head -1 | grep -o 'GPU [0-9]*%' | grep -o '[0-9]*'; }
act() { grep -q '^\*\*Verdict: ACT' "$HEALTH" 2>/dev/null; }
ramp() { pgrep -f '^/bin/zsh [^ ]*soft_end\.sh' >/dev/null; }
lid_closed() { ioreg -r -k AppleClamshellState -d 4 2>/dev/null | grep -m1 '"AppleClamshellState"' | grep -q Yes; }
blocked() { [ -e "$PAUSEF" ] || [ -e "$HOLDF" ] || [ -e "$O/WAIT" ] || act || ramp || lid_closed; }
cd $F || exit 9
mkdir -p $O
rm -f $O/DONE
log "chw19 armed: 6 clear minutes (1 with $O/GO; held while $O/WAIT exists), start by $(date -r $DL '+%a %H:%M')"
while :; do
  clear=0
  while [ "$(date +%s)" -lt "$DL" ]; do
    q=$(nquiet); b=$(nbig); a=$(freegb)
    if [ "$q" -eq 0 ] && [ "$b" -eq 0 ] && ! blocked && [ "$a" -ge 20 ]; then clear=$((clear + 1)); else clear=0; fi
    need=6; [ -e $O/GO ] && need=1
    [ $clear -ge $((need * 2)) ] && break
    sleep 30
  done
  if [ "$(date +%s)" -ge "$DL" ]; then
    log "chw19: no clear slot by the deadline: NOT STARTED"
    echo "$(date +%H:%M:%S) SCR chw19 NOT STARTED: no clear slot by the deadline" >> $LOG
    echo "rc 5 not started $(date '+%a %H:%M:%S')" > $O/DONE
    exit 5
  fi
  log "chw19: clear for $((clear / 2)) min ($a GB free): no quiet ticket for 5 min, then the share lock"
  peer wait --timeout 5m >/dev/null 2>&1 || { log "chw19: a quiet ticket is waiting: back to the gate"; continue; }
  peer lock share 20m --cpu 200 --gpu 100 "Fit12 12 GB chat width C2c (Strata's one-draft width): 12 IFEval x2 + 16 HumanEval, MEM 12 GB, ~10 min, rough" >/dev/null 2>&1
  sleep 3; g=$(gpu); gp=100; gn=1
  while { [ "${g:-0}" -gt 10 ] || [ "$gp" -gt 10 ]; } && [ $gn -lt 24 ]; do gp=${g:-0}; sleep 5; g=$(gpu); gn=$((gn + 1)); done
  if blocked || [ "$(freegb)" -lt 20 ] || [ "${g:-0}" -gt 10 ] || [ "$(nquiet)" -gt 0 ] || [ "$(nbig)" -gt 0 ]; then
    log "chw19: locked, but blocked / busy ($(freegb) GB free, GPU ${g}%, quiet $(nquiet), big $(nbig)): lock given back"; peer unlock >/dev/null 2>&1; continue
  fi
  break
done
SW0=$(swapmb); T0=$(date +%s); rc=0
echo "$(date +%H:%M:%S) SCR chw19 start: C2c (W1 2, WL 2) chat + P2' chat + C2c code, PR1 + tail, share lock ($(freegb) GB free, swap $SW0 MB, GPU ${g}%)" >> $LOG
{ echo "chw19 manifest $(date '+%a %d %b %H:%M:%S')"; shasum -a 256 $F/scripts/fit_t3.py $F/scripts/t3_twopass.py $F/scripts/t2_twopass.py \
    $F/scripts/fit_pr1.py $F/pinned/8db7/bt_b12.py /Users/kamen/Leap/scripts/d7b_live.py; sed -n '/^ENV = /,/^for k in/p' $F/scripts/fit_t3.py; } > $O/MANIFEST.txt 2>&1
for J in "chw19c|2|2|128|$IK" "chw19p|cf:0.9:2|4|128|$IK" "chw19h|2|2|256|$HK"; do
  T=$(echo "$J" | cut -d'|' -f1); W=$(echo "$J" | cut -d'|' -f2); WL=$(echo "$J" | cut -d'|' -f3); C=$(echo "$J" | cut -d'|' -f4); KS=$(echo "$J" | cut -d'|' -f5)
  [ $rc -ne 0 ] && break
  log "chw19: $T (W1 $W, WL $WL, CAP $C) start"
  env F12_OUT=$O F12_CAP=$C F12_W1=$W F12_WL=$WL F12_W1SEED=1 F12_W1TAIL=1 F12_PR1=1 F12_PHLOG=1 F12_RERUN=1 F12_T3ARMS=ex F12_CMCL=47,32 \
    F12_IDS_OUT=$O/${T}_ids.json F12_KEYS=$KS $PY -B $F/scripts/fit_t3.py $T > $O/$T.txt 2>&1 &
  P=$!; n=0
  while kill -0 $P 2>/dev/null; do
    sleep 5; n=$((n + 1))
    [ $((n % 12)) -eq 0 ] && echo "$(date +%H:%M:%S) $T GPU $(gpu)% | $(others | locks | grep -v '^$' | tr '\n' ' ')" >> $O/locks.txt
    [ $(( $(swapmb) - SW0 )) -gt 1024 ] && { rc=127; log "chw19: swap +1 GB: stopping"; }
    [ $(( $(date +%s) - T0 )) -ge 1500 ] && { rc=124; log "chw19: 25 min: stopping"; }
    [ $rc -ne 0 ] && { kill -TERM $P; sleep 5; kill -KILL $P 2>/dev/null; break; }
  done
  wait $P; w=$?; [ $rc -eq 0 ] && rc=$w
  log "chw19: $T end rc $w"
  echo "$(date +%H:%M:%S) SCR chw19 $T end (rc $w)" >> $LOG
done
peer unlock >/dev/null 2>&1
echo "$(date +%H:%M:%S) SCR chw19 end (rc $rc, $(( $(date +%s) - T0 )) s, swap $(swapmb) MB)" >> $LOG
echo "rc $rc started $(date -r $T0 '+%a %H:%M:%S') ended $(date '+%a %H:%M:%S')" > $O/DONE
grep -E 'Error|Traceback|AssertionError' $O/chw19c.txt $O/chw19p.txt $O/chw19h.txt 2>/dev/null | head -3
for f in $F/data/scr/chw14/chw14c2* $F/data/scr/chw14/chw14ha*; do ln -sf $f $O/; done      # stage 2's P2' chat + A' code
$PY -I -B $F/scripts/chw13_read.py $O chw19p chw19c > $O/chw19_chat.txt 2>&1; cp $O/chw13_report.json $O/chw19_chat.json
$PY -I -B $F/scripts/chw13_read.py $O chw14c2 chw19p > $O/chw19_rep.txt 2>&1; cp $O/chw13_report.json $O/chw19_rep.json
$PY -I -B $F/scripts/chw13_read.py $O chw14ha chw19h /nonexistent.json > $O/chw19_code.txt 2>&1; cp $O/chw13_report.json $O/chw19_code.json
$PY -I -B $F/scripts/paper_rows.py $O/chw19_rows "$O/chw19c.txt|chw19c|ex|C2c chat" "$O/chw19p.txt|chw19p|ex|P2' chat" "$O/chw19h.txt|chw19h|ex|C2c code" > /dev/null 2>&1
$PY -I -B -c "
import glob, json, collections
for t in ['chw19c', 'chw19p', 'chw19h']:
    c = collections.Counter()
    for f in glob.glob('$O/' + t + '_*_ex_calls.json'):
        for r in json.load(open(f)): c[('first' if r[1] == 0 else 'later', r[2])] += 1
    print(t, dict(sorted(c.items())))" > $O/chw19_widths.txt 2>&1
cut -c1-230 $O/chw19_chat.txt $O/chw19_rep.txt $O/chw19_code.txt $O/chw19_rows.txt $O/chw19_widths.txt
exit $rc
