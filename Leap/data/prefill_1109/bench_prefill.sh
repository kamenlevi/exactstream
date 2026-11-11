#!/bin/bash
# Prefill at 32 GB (orchestrator 21:1x; tonight's slot 23:2x on Kamen's yes): the chat-32 config through run.py, one
# long-document prompt per process (9,988 and 29,988 tokens), 64 new tokens, the default prefill vs the exact
# layer-major paths PF-A (R57_PFLM=1) and PF-B (R57_PFLM=3). Per run: first token (= the prefill wall), whole decode,
# lifetime peak (run.py's own line), and the 64 tokens (the same tokens on every path = the exactness check).
# Order: 10k default, 10k pfa, 30k default, 30k pfa, 10k pfb, 30k pfb (each size gets default + one PF path first).
# Before each run: last start (10K 02:00, 30K 01:56; every run is capped at 02:04), no chat.py / serve.py of Kamen's,
# Kamen idle >= 600 s, NO-SWAP >= the run's peak + 8 GB: measured peaks where known (a PF path at 10K: the 10K default
# peak + 1.5; 30K: the same path's 10K peak + 2.5), else 43 / 45 (10K default / PF) and 46 / 47 (30K).
# During a run: swap +2 GB, Kamen back (input within 5 s on 5 polls in a row) or the cap stop it.
P=$1; CL=/Users/kamen/Leap/release/cleanclone/final/clone; cd $CL
ep() { date -j -f "%Y-%m-%d %H:%M:%S" "$1" +%s; }
END=$(ep "2026-11-10 02:04:00"); L10=$(ep "2026-11-10 02:00:00"); L30=$(ep "2026-11-10 01:56:00")
noswap() { vm_stat | awk '/page size of/ {ps=$8} /^Pages (free|speculative|purgeable):/ || /^File-backed pages:/ {v=$NF; sub("[.]","",v); s+=v} END {printf "%d", s*ps/1e9}'; }
swap() { sysctl -n vm.swapusage | sed -E 's/.*used = ([0-9.]+)M.*/\1/' | cut -d. -f1; }
idle() { ioreg -c IOHIDSystem | awk '/HIDIdleTime/ {print int($NF/1000000000); exit}'; }
other() { ps -axo pid=,comm=,args= | awk '$2 ~ /[Pp]ython/ && /[ \/](chat|serve)[.]py( |$)/ && !/--max-tokens 2048/ {printf "%s ", $1}'; }
killtree() { for c in $(pgrep -P $1); do killtree $c $2; done; kill -$2 $1 2>/dev/null; }
say() { echo "$(date +%T) $*" | tee -a $P/summary.txt; }
up() { awk -v a="$1" -v b="$2" 'BEGIN {x = a + b; printf "%d", (x == int(x)) ? x : int(x) + 1}'; }
S0=$(swap); say "prefill bench start: swap $S0 MB, NO-SWAP $(noswap) GB"
for cell in "10k default" "10k pfa" "30k default" "30k pfa" "10k pfb" "30k pfb"; do
  set -- $cell; SZ=$1; PT=$2; TAG=${SZ}_${PT}
  ENV=""; [ $PT = pfa ] && ENV="--env R57_PFLM=1"; [ $PT = pfb ] && ENV="--env R57_PFLM=3"
  if [ $SZ = 10k ]; then
    LS=$L10; pk=$(cat $P/peak_10k_default 2>/dev/null)
    if [ $PT = default ]; then NEED=43; elif [ -n "$pk" ]; then NEED=$(up $pk 9.5); else NEED=45; fi
  else
    LS=$L30; pk=$(cat $P/peak_10k_$PT 2>/dev/null)
    if [ -n "$pk" ]; then NEED=$(up $pk 10.5); elif [ $PT = default ]; then NEED=46; else NEED=47; fi
  fi
  [ "$(date +%s)" -ge "$LS" ] && { say "SKIP $SZ $PT: past its last start"; continue; }
> (a quote from the project owner, kept private)
> (a quote from the project owner, kept private)
  A=$(noswap); [ "$A" -lt "$NEED" ] && { say "SKIP $SZ $PT: NO-SWAP $A GB < $NEED GB"; continue; }
  CAP=$(( END - $(date +%s) )); [ $CAP -gt 720 ] && CAP=720
  say "run $SZ $PT (NO-SWAP $A GB, need $NEED; swap $(swap) MB; cap $CAP s)"
  .venv/bin/python run.py --config chat --mem 32 --prompt-file $P/prompt_$SZ.txt --max-tokens 64 $ENV > $P/run_$TAG.out 2>&1 &
  RP=$!; T0=$(date +%s); SMAX=$S0; ACT=0; WHY=""; PZ=""
  while kill -0 $RP 2>/dev/null; do
    s=$(swap); [ "$s" -gt "$SMAX" ] && SMAX=$s
    i=$(idle); if [ "$i" -lt 5 ]; then ACT=$((ACT + 1)); else ACT=0; fi
    [ -e ~/.local/share/quiet/PAUSE ] && PZ=", heat PAUSE seen"
    if [ $((s - S0)) -gt 2048 ]; then WHY="swap $s MB (start $S0)"
> (a quote from the project owner, kept private)
    elif [ $(( $(date +%s) - T0 )) -gt $CAP ]; then WHY="cap $CAP s"
    fi
    [ -n "$WHY" ] && { say "STOP $SZ $PT: $WHY"; killtree $RP TERM; break; }
    sleep 3
  done
  wait $RP 2>/dev/null
  L=$(grep -a -E "^whole decode [0-9.]+ tok/s" $P/run_$TAG.out | tail -1)
  TK=$(grep -a -o "^tokens: [^;]*" $P/run_$TAG.out | tail -1 | cut -c9-); echo "$TK" > $P/tokens_$TAG
  pk=$(echo "$L" | sed -n 's/.*lifetime peak \([0-9.]*\) GB.*/\1/p'); [ $SZ = 10k ] && [ -n "$pk" ] && echo $pk > $P/peak_10k_$PT
  say "  $SZ $PT: ${L:-no result (see run_$TAG.out)} | wall $(( $(date +%s) - T0 )) s | swap max $SMAX MB$PZ"
  case "$WHY" in swap*|Kamen*) break;; esac
done
/opt/homebrew/opt/python@3.12/bin/python3.12 -I - "$P" <<'EOF' | tee -a $P/summary.txt
import json, os, sys
P = sys.argv[1]
def toks(x):                                  # the first list of ints inside run.py's tokens JSON
    if isinstance(x, list) and x and all(isinstance(t, int) for t in x):
        return x
    for v in (x.values() if isinstance(x, dict) else x if isinstance(x, list) else []):
        t = toks(v)
        if t:
            return t
    return None
for sz in ("10k", "30k"):
    got = {}
    for pt in ("default", "pfa", "pfb"):
        f = os.path.join(P, f"tokens_{sz}_{pt}")
        path = open(f).read().strip() if os.path.exists(f) else ""
        if path and os.path.exists(path):
            got[pt] = toks(json.load(open(path)))
    if "default" in got and len(got) > 1:
        print(f"exactness {sz}: " + ", ".join(f"{pt} {'== default' if t == got['default'] else 'DIFFERS from default'} ({len(t or [])} tokens)"
                                             for pt, t in got.items() if pt != "default"))
    else:
        print(f"exactness {sz}: not checked (runs with tokens: {', '.join(got) or 'none'})")
EOF
say "prefill bench end: swap $(swap) MB (start $S0), NO-SWAP $(noswap) GB"
