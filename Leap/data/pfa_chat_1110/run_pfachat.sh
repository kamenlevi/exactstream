#!/bin/bash
# PF-A in chat.py at --mem 32 (Kamen Tue 11:0x via the orchestrator): one short and one long prompt (~6K tokens, so the
# prefill takes PF-A's layer-major path), chat.py with PF-A (its new default at 32 GB) against chat.py with R57_PFLM=0
# (the default prefill); max 256 tokens; /clear between, so each prompt stands alone. Compares answers, first token and
# lifetime peak from chat.py's per-answer JSONL. Gate: no quiet lock, no other MEM > 5 GB lock, no foreign engine, no
# chat.py / serve.py, NO-SWAP >= 43 GB (peak ~35 + 8); share lock 'MEM 36 GB'; swap guard +2 GB; 6-min cap per run.
SP=/private/tmp/claude-501/-Users-kamen-OjoX/f2971937-6c95-4e86-a049-05beff6a8e32/scratchpad/pfachat
CL=/Users/kamen/Leap/release/cleanclone/final/clone; cd $CL; LOG=$SP/summary.txt
noswap() { vm_stat | awk '/page size of/ {ps=$8} /^Pages (free|speculative|purgeable):/ || /^File-backed pages:/ {v=$NF; sub("[.]","",v); s+=v} END {printf "%d", s*ps/1e9}'; }
swap() { sysctl -n vm.swapusage | sed -E 's/.*used = ([0-9.]+)M.*/\1/' | cut -d. -f1; }
others() { peer ls 2>/dev/null | grep "^- "; }
locks() { grep -oE "· (shared run \(others may work\)|benchmarking|waiting for a quiet window)[^·]*"; }
busy() { others | locks | grep -cE "^· (benchmarking|waiting for a quiet window)|MEM ([6-9]|[1-9][0-9]) GB"; }
engines() { ps -axo pid=,comm=,args= | awk '$2 ~ /[Pp]ython/ && /r57_screen_|run_map[.]py|fit_onecall[.]py|[ \/](chat|serve)[.]py( |$)/ {printf "%s ", $1}'; }
say() { echo "$(date +%T) $*" | tee -a $LOG; }
desc() { local c; for c in $(pgrep -P "$1"); do desc "$c"; echo "$c"; done; }
LOCKED=0
trap '[ $LOCKED = 1 ] && peer unlock >/dev/null 2>&1' EXIT
n=0; until [ "$(busy)" = 0 ] && [ -z "$(engines)" ] && [ "$(noswap)" -ge 43 ]; do
  n=$((n + 1)); [ $n -gt 60 ] && { say "SKIP: not clear for 30 min (busy $(busy), engines '$(engines)', NO-SWAP $(noswap))"; exit 0; }
  sleep 30
done
peer lock share 20m --cpu 300 --gpu 100 "MEM 36 GB: Leap chat.py PF-A check at 32 GB (2 processes x 2 prompts, ~4 min)" > /dev/null 2>&1 && LOCKED=1
sleep 15; [ "$(busy)" = 0 ] && [ -z "$(engines)" ] || { say "SKIP: another job appeared after the lock"; exit 0; }
S0=$(swap); say "GO: NO-SWAP $(noswap) GB, swap $S0 MB"
LONG="$(cat $SP/long_body.txt) Summarize the main findings of the document above in five bullet points."
for arm in pfa default; do
  if [ $arm = pfa ]; then unset R57_PFLM; else export R57_PFLM=0; fi
  printf '%s\n/clear\n%s\n/exit\n' "Explain in three sentences how a refrigerator keeps food cold." "$LONG" \
    | .venv/bin/python chat.py --mem 32 --max-tokens 256 > $SP/run_$arm.out 2>&1 &
  P=$!; T0=$(date +%s)
  while kill -0 $P 2>/dev/null; do
    s=$(swap)
    if [ $((s - S0)) -gt 2048 ] || [ $(( $(date +%s) - T0 )) -gt 360 ]; then
      say "STOP $arm: swap $s MB (start $S0) or 6 min"; for c in $(desc $P); do kill -TERM $c 2>/dev/null; done; kill -TERM $P; break
    fi
    sleep 3
  done
  wait $P 2>/dev/null
  say "  $arm: wall $(( $(date +%s) - T0 )) s; jsonl $(grep -a -o 'out/chat_[0-9_]*\.jsonl' $SP/run_$arm.out | tail -1)"
done
unset R57_PFLM
peer unlock >/dev/null 2>&1; LOCKED=0; say "unlocked; NO-SWAP $(noswap) GB, swap $(swap) MB"
ls -t $CL/out/chat_*.jsonl | head -2 > $SP/jsonls.txt
/opt/homebrew/opt/python@3.12/bin/python3.12 -I - $SP <<'EOF' | tee -a $SP/summary.txt
import json, sys
SP = sys.argv[1]
fs = open(f"{SP}/jsonls.txt").read().split()          # newest first: default, then pfa
rows = {name: [json.loads(l) for l in open(f)] for name, f in zip(("default", "pfa"), fs)}
for i, label in enumerate(("short", "long")):
    a, d = rows["pfa"][i], rows["default"][i]
    same = a["answer"] == d["answer"]
    print(f"{label}: prompt {a.get('P')} tokens; answers {'IDENTICAL' if same else 'DIFFER'} ({a['n']} / {d['n']} tokens); "
          f"first token PF-A {a.get('pfs')} s vs default {d.get('pfs')} s; peak PF-A {a.get('peak')} vs default {d.get('peak')} GB; "
          f"whole decode {a.get('tps', 0):.1f} vs {d.get('tps', 0):.1f} tok/s")
EOF
