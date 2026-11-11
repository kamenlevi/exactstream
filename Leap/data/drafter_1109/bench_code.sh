#!/bin/bash
# One realistic code request (2,048 tokens) through chat.py (the unchanged loop) at 32 GB: drafter dfl vs mtp.
# Tonight (orchestrator 23:2x on Kamen's yes; NewIdeas 23:12): the 32 GB pair only, the 45 GB pair stays out.
# Before each run: no start after 01:58 (a run ends by 02:05), no chat.py / serve.py of Kamen's, Kamen idle >= 600 s,
# NO-SWAP (free + speculative + purgeable + file-backed) >= 39 GB. During a run: swap +2 GB, Kamen back (input within
# 5 s on 5 polls in a row), another chat.py, or a 6-min cap stop it. Per run: the loop's own log line and chat.py's
# summary line -> summary.txt.
B=$1; CL=/Users/kamen/Leap/release/cleanclone/final/clone; cd $CL
REQ="Write a complete Python command-line tool, csvreport.py, that reads a CSV of bank transactions with columns date, description, amount and category. It should validate each row (skipping bad rows with a logged warning), compute monthly totals per category, flag outlier transactions with a z-score above 3 within their category, and write a Markdown report with tables plus a bar chart PNG using matplotlib. Use argparse, logging, type hints and dataclasses, and include a pytest test file covering parsing, totals and outlier detection."
LAST=$(date -j -f "%Y-%m-%d %H:%M:%S" "2026-11-10 01:58:00" +%s)
noswap() { vm_stat | awk '/page size of/ {ps=$8} /^Pages (free|speculative|purgeable):/ || /^File-backed pages:/ {v=$NF; sub("[.]","",v); s+=v} END {printf "%d", s*ps/1e9}'; }
swap() { sysctl -n vm.swapusage | sed -E 's/.*used = ([0-9.]+)M.*/\1/' | cut -d. -f1; }
idle() { ioreg -c IOHIDSystem | awk '/HIDIdleTime/ {print int($NF/1000000000); exit}'; }
other() { ps -axo pid=,comm=,args= | awk '$2 ~ /[Pp]ython/ && /[ \/](chat|serve)[.]py( |$)/ && !/--max-tokens 2048/ {printf "%s ", $1}'; }
killtree() { for c in $(pgrep -P $1); do killtree $c $2; done; kill -$2 $1 2>/dev/null; }
stopit() { killtree $1 INT; for k in 1 2 3 4 5 6 7 8 9 10; do kill -0 $1 2>/dev/null || return; sleep 2; done; killtree $1 TERM; }
say() { echo "$(date +%T) $*" | tee -a $B/summary.txt; }
S0=$(swap); say "code bench start: swap $S0 MB, NO-SWAP $(noswap) GB"
for cell in "32 dfl" "32 mtp"; do
  set -- $cell; MEM=$1; DR=$2; NEED=39
  [ "$(date +%s)" -ge "$LAST" ] && { say "SKIP $MEM $DR: past the 01:58 last start"; continue; }
> (a quote from the project owner, kept private)
> (a quote from the project owner, kept private)
  A=$(noswap); [ "$A" -lt "$NEED" ] && { say "SKIP $MEM $DR: NO-SWAP $A GB < $NEED GB"; continue; }
  say "run mem $MEM drafter $DR (NO-SWAP $A GB, swap $(swap) MB)"
  printf '%s\n/exit\n' "$REQ" | .venv/bin/python chat.py --mem $MEM --drafter $DR --max-tokens 2048 > $B/run_${MEM}_${DR}.out 2>&1 &
  CP=$!; T0=$(date +%s); SMAX=$S0; ACT=0; WHY=""; PZ=""
  while kill -0 $CP 2>/dev/null; do
    s=$(swap); [ "$s" -gt "$SMAX" ] && SMAX=$s
    i=$(idle); if [ "$i" -lt 5 ]; then ACT=$((ACT + 1)); else ACT=0; fi
    [ -e ~/.local/share/quiet/PAUSE ] && PZ=", heat PAUSE seen"
    if [ $((s - S0)) -gt 2048 ]; then WHY="swap $s MB (start $S0)"
> (a quote from the project owner, kept private)
    elif [ -n "$(other)" ]; then WHY="another chat.py started ($(other))"
    elif [ $(( $(date +%s) - T0 )) -gt 360 ]; then WHY="6-min cap"
    fi
    [ -n "$WHY" ] && { say "STOP $MEM $DR: $WHY"; stopit $CP; break; }
    sleep 3
  done
  wait $CP 2>/dev/null
  LOG=$(grep -a -o "log: .*" $B/run_${MEM}_${DR}.out | tail -1 | cut -c6-)
  /opt/homebrew/opt/python@3.12/bin/python3.12 -I - "$LOG" "$B/run_${MEM}_${DR}.out" "$MEM" "$DR" <<'EOF' | tee -a $B/summary.txt
import os, re, sys
log, out, mem, dr = sys.argv[1:5]
L = [l for l in open(log, errors="replace") if re.search(r"/chat_\d+_\d+_001 \[", l)] if os.path.isfile(log) else []
S = re.findall(r"\[(\d+) tokens \| whole decode ([\d.]+) tok/s \| first token ([\d.]+|None) s \| lifetime peak ([\d.]+|nan) GB \| drafter (\w+), ([\d.]+) tokens per check", open(out, errors="replace").read())
if not L or not S:
    print(f"{mem} GB {dr}: no result (log {log or '-'}: {len(L)} lines; summaries {len(S)})"); sys.exit()
l = L[-1]
g = lambda pat: (re.search(pat, l) or [None, "?"])[1]
n, tps, ft, peak, drafter, kept = S[-1]
print(f"{mem} GB {dr:3s}: {n} tokens, {tps} tok/s whole decode, kept {kept} per check, {g(r'in (\d+) checks')} checks, "
      f"median cycle {g(r'median cycle ([\d.]+) ms')} ms (check + commit {g(r'check \+ commit ([\d.]+) ms')}, draft {g(r'draft ([\d.]+) ms')}), "
      f"misses per check {g(r'\(medians\) (\d+) misses')}, first token {ft} s, peak {peak} GB, drafter {drafter}, "
      f"type {g(r'\[(\w+): \w+\]')}, log {log}")
EOF
  say "  $MEM $DR: wall $(( $(date +%s) - T0 )) s, swap max $SMAX MB$PZ"
  [ -n "$WHY" ] && [ "$WHY" != "6-min cap" ] && break
done
say "code bench end: swap $(swap) MB (start $S0), NO-SWAP $(noswap) GB"
