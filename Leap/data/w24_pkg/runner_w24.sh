#!/bin/bash
# 24 GB package check (orchestrator Tue 08:4x): the packaged lossy-24 (= w13b as gated, engine/loop/r57_screen_w24.py) through
# the clean clone's run.py. (1) replay 3 IFBench + 3 GPQA gate rows (no QOM there) at the gate's CAP 2048: tokens must equal
# ~/EvalSuite/results/card_w24_fwd/decode.jsonl, prompt tokens n_prompt; (2) BigMoE W24d's 12 HumanEval keys at CAP 256:
# whole decode vs 130.33 pooled, lifetime peak vs 23.78-23.82 GB.
# Order (orchestrator): Fit12's ifb12 -> BigMoE's W24e -> Breakthrough's c47 -> this. Gate, every 30 s: W24e's report written
# after 08:41 today; no c47_day.sh process; no python process running an r57_screen_ / run_map.py / fit_onecall.py; no quiet
# lock held or queued; no other MEM > 5 GB lock or >= 50 % GPU shared run; no chat.py / serve.py of Kamen's; no PAUSE / HOLD;
# NO-SWAP >= 34 GB (25.1 peak + 8) on 3 reads; clear on 2 polls. Hard end 14:55 (the apps reopen at 15:00); share lock
# 'MEM 26 GB'. The GPQA / IFBench items go into the clone only for this run and are removed at the end (never published).
SP=/private/tmp/claude-501/-Users-kamen-OjoX/f2971937-6c95-4e86-a049-05beff6a8e32/scratchpad/w24check
CL=/Users/kamen/Leap/release/cleanclone/final/clone; IT=$CL/harness/data/items; cd $CL
ep() { date -j -f "%Y-%m-%d %H:%M:%S" "$1" +%s; }
END=$(ep "2026-11-10 14:55:00"); T0841=$(ep "2026-11-10 08:41:00"); REP=/Users/kamen/BigMoE/data/w24/w24e_report.txt
noswap() { vm_stat | awk '/page size of/ {ps=$8} /^Pages (free|speculative|purgeable):/ || /^File-backed pages:/ {v=$NF; sub("[.]","",v); s+=v} END {printf "%d", s*ps/1e9}'; }
swap() { sysctl -n vm.swapusage | sed -E 's/.*used = ([0-9.]+)M.*/\1/' | cut -d. -f1; }
hischat() { ps -axo pid=,comm=,args= | awk '$2 ~ /[Pp]ython/ && /[ \/](chat|serve)[.]py( |$)/ {printf "%s ", $1}'; }
engines() { ps -axo pid=,comm=,args= | awk '$2 ~ /[Pp]ython/ && /r57_screen_|run_map[.]py|fit_onecall[.]py|dm_oea_masks[.]py/ {printf "%s ", $1}'; }
c47() { ps -axo pid=,args= | awk '/c47_da[y][.]sh/ {printf "%s ", $1}'; }
others() { peer ls 2>/dev/null | grep "^- "; }
locks() { grep -oE "· (shared run \(others may work\)|benchmarking|waiting for a quiet window)[^·]*"; }
quiet_n() { others | locks | grep -cE "^· (benchmarking|waiting for a quiet window)"; }
bigmem_n() { others | locks | grep -oE "MEM [0-9]+ GB" | awk '$2 > 5 {n++} END {print n + 0}'; }
gpu_n() { others | sed -n 's/.*shared run[^:]*: about [0-9]*% CPU, \([0-9]*\)% GPU.*/\1/p' | awk '$1 >= 50 {n++} END {print n + 0}'; }
held() { [ -e ~/.local/share/quiet/PAUSE ] || [ -e ~/.local/share/quiet/HOLD ]; }
say() { echo "$(date +%T) $*" | tee -a $SP/summary.txt; }
gate() {
  { [ -f $REP ] && [ "$(stat -f %m $REP)" -ge "$T0841" ]; } || { echo "waiting for W24e's report"; return; }
  [ -n "$(c47)" ] && { echo "c47's runner is still running ($(c47))"; return; }
  [ -n "$(engines)" ] && { echo "another engine run ($(engines))"; return; }
  [ "$(quiet_n)" -gt 0 ] && { echo "a quiet lock is held or queued"; return; }
  [ "$(bigmem_n)" -gt 0 ] && { echo "another session holds a MEM > 5 GB lock"; return; }
  [ "$(gpu_n)" -gt 0 ] && { echo "another shared run uses >= 50 % GPU"; return; }
  [ -n "$(hischat)" ] && { echo "a chat.py / serve.py is running"; return; }
  held && { echo "PAUSE / HOLD"; return; }
  local a b c; a=$(noswap); sleep 5; b=$(noswap); sleep 5; c=$(noswap)
  [ $a -lt 34 ] || [ $b -lt 34 ] || [ $c -lt 34 ] && { echo "NO-SWAP $a / $b / $c GB < 34"; return; }
}
LOCKED=0; RP=""
finish() {
  [ -n "$RP" ] && kill -0 $RP 2>/dev/null && { for c in $(pgrep -P $RP); do kill -TERM $c; done; kill -TERM $RP; }
  rm -f $IT/dec_ifbench.jsonl $IT/dec_gpqad.jsonl
  [ $LOCKED = 1 ] && { peer unlock >/dev/null 2>&1; say "unlocked"; }
  say "exit (test items removed; NO-SWAP $(noswap) GB, swap $(swap) MB)"
}
trap finish EXIT
trap 'say "stopped by a signal"; exit 1' INT TERM
say "armed (pid $$): replay 3 IFBench + 3 GPQA (CAP 2048), then 12 HumanEval (CAP 256); after W24e + c47; hard end 14:55"
CLR=0; NOTE=""
while :; do
  [ $(( $(date +%s) + 420 )) -gt $END ] && { say "SKIP: no start that ends by 14:55 (last wait: ${NOTE:-none})"; exit 0; }
  why=$(gate)
  if [ -z "$why" ]; then CLR=$((CLR + 1)); else CLR=0; fi
  [ -z "$why" ] && [ $CLR -lt 2 ] && why="clear once: one more poll"
  if [ -z "$why" ]; then
    out=$(peer lock share 20m --cpu 300 --gpu 100 "MEM 26 GB: Leap 24 GB package check (replay + HumanEval x12), ~6 min" 2>&1); rc=$?
    say "share lock rc $rc: $(echo "$out" | tr '\n' ' ' | cut -c1-240)"
    if [ $rc = 0 ]; then
      LOCKED=1; sleep 20
      w2=""; [ "$(noswap)" -lt 33 ] && w2="NO-SWAP $(noswap)"; [ -n "$(engines)" ] && w2="$w2 engine $(engines)"; [ "$(quiet_n)" -gt 0 ] && w2="$w2 quiet"; [ "$(bigmem_n)" -gt 0 ] && w2="$w2 MEM lock"
      [ -z "$w2" ] && break
      say "after the lock: $w2: unlock and wait"; peer unlock >/dev/null 2>&1; LOCKED=0; CLR=0; why="after the lock: $w2"
    else
      why="share lock refused"
    fi
  fi
  [ "$why" != "$NOTE" ] && say "wait: $why"; NOTE=$why
  sleep 30
done
S0=$(swap); say "GO: NO-SWAP $(noswap) GB, swap $S0 MB"
cp $SP/dec_ifbench.jsonl $SP/dec_gpqad.jsonl $IT/
run1() {   # name, keys, cap
  local T0=$(date +%s) SMAX=$S0 s
  .venv/bin/python run.py --config lossy --mem 24 --keys "$2" --max-tokens $3 > $SP/$1.out 2>&1 &
  RP=$!
  while kill -0 $RP 2>/dev/null; do
    s=$(swap); [ "$s" -gt "$SMAX" ] && SMAX=$s
    if [ $((s - S0)) -gt 2048 ] || [ $(date +%s) -gt $END ] || [ -n "$(hischat)" ]; then
      say "STOP $1: swap $s MB (start $S0), the 14:55 end, or a chat.py started"; for c in $(pgrep -P $RP); do kill -TERM $c; done; kill -TERM $RP; break
    fi
    sleep 3
  done
  wait $RP 2>/dev/null; RP=""
  say "  $1: $(grep -a -E '^whole decode' $SP/$1.out | tail -1) | wall $(( $(date +%s) - T0 )) s | swap max $SMAX MB"
}
run1 replay "$(cat $SP/replay_keys.txt)" 2048
[ $(( $(date +%s) + 180 )) -le $END ] && run1 he12 "$(cat $SP/he_keys.txt)" 256 || say "SKIP he12: cannot end by 14:55"
peer unlock >/dev/null 2>&1; LOCKED=0; say "unlocked"
/opt/homebrew/opt/python@3.12/bin/python3.12 -I - "$SP" <<'EOF' | tee -a $SP/summary.txt
import json, os, re, statistics, sys
SP = sys.argv[1]
def files(name):
    t = open(f"{SP}/{name}.out", errors="replace").read()
    m = re.search(r"^tokens: ([^;]+)", t, re.M)
    if not m: return None, None
    tp = m.group(1).strip(); return json.load(open(tp)), json.load(open(tp.replace("_tokens.json", "_runs.json")))
ref = json.load(open(f"{SP}/replay_ref.json"))
toks, runs = files("replay")
if toks:
    P = {r["key"]: r.get("P") for r in runs}
    for k, v in ref.items():
        got = toks.get(k)
        got = (list(got.values())[0][-1] if isinstance(got, dict) else got) if got else None
        eq = got == v["ids"]
        fd = None if eq or got is None else next((i for i, (a, b) in enumerate(zip(got, v["ids"])) if a != b), min(len(got), len(v["ids"])))
        print(f"replay {k}: tokens {'EQUAL' if eq else 'MISSING' if got is None else f'DIFFER from token {fd + 1}'} ({len(got or [])} vs gate {len(v['ids'])}); prompt tokens {P.get(k)} vs gate {v['n_prompt']}")
else:
    print("replay: no tokens (see replay.out)")
toks, runs = files("he12")
if runs:
    n = sum(r["n"] for r in runs); pf = [r.get("pfs") for r in runs if r.get("pfs") is not None]
    print(f"he12: {len(runs)} prompts, {n} tokens (BigMoE w13eh_100: 2,610); first token median {statistics.median(pf):.2f} s (BigMoE 3.54)")
EOF
