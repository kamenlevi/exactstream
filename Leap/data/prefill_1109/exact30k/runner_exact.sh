#!/bin/bash
# PF-B exact check at 30K in tonight's gap (orchestrator 00:2x: only if NewIdeas fits it; Editor 00:5x: the staged PF-B
# erratum waits on its logits hashes): the EXACT 32 GB config (exact-32) through run.py on the 29,988-token prompt,
# 64 tokens, R57_LOGHASH=1 (each check's verify logits hashed). Exact decoding drops nothing, so a token or hash difference
# is the prefill's. Arms, in order: PF-A (R57_PFLM=1, the reference), PF-B at R57_PFLM_TOK=8192 (the published setting:
# 4 MoE calls per layer at 30K), PF-B at the default 4,096 (tonight's chat-32 run: 8 calls per layer).
# Gate, every 15 s: no quiet lock held or queued; no other MEM > 5 GB lock or >= 50 % GPU shared run; Kamen idle >= 600 s,
# no chat.py / serve.py of his; no PAUSE / HOLD; lid open; NO-SWAP >= 49 GB on 3 reads 5 s apart; clear on 2 polls.
# Hard END (epoch, argument 2): a run starts only if it can end by END (100 s each); each run is capped at END. Share lock
# 'MEM 43 GB' to END, re-checked at +20 s and +40 s. Touches NO done-file. Usage: runner_exact.sh <outdir> <END epoch>
P=$1; END=$2; CL=/Users/kamen/Leap/release/cleanclone/final/clone; PR=/Users/kamen/Leap/data/prefill_1109/prompt_30k.txt
mkdir -p $P; cd $CL
noswap() { vm_stat | awk '/page size of/ {ps=$8} /^Pages (free|speculative|purgeable):/ || /^File-backed pages:/ {v=$NF; sub("[.]","",v); s+=v} END {printf "%d", s*ps/1e9}'; }
swap() { sysctl -n vm.swapusage | sed -E 's/.*used = ([0-9.]+)M.*/\1/' | cut -d. -f1; }
idle() { ioreg -c IOHIDSystem | awk '/HIDIdleTime/ {print int($NF/1000000000); exit}'; }
hischat() { ps -axo pid=,comm=,args= | awk '$2 ~ /[Pp]ython/ && /[ \/](chat|serve)[.]py( |$)/ && !/--max-tokens 2048/ {printf "%s ", $1}'; }
others() { peer ls 2>/dev/null | grep "^- "; }
locks() { grep -oE "· (shared run \(others may work\)|benchmarking|waiting for a quiet window)[^·]*"; }
quiet_n() { others | locks | grep -cE "^· (benchmarking|waiting for a quiet window)"; }
bigmem_n() { others | locks | grep -oE "MEM [0-9]+ GB" | awk '$2 > 5 {n++} END {print n + 0}'; }
gpu_n() { others | sed -n 's/.*shared run[^:]*: about [0-9]*% CPU, \([0-9]*\)% GPU.*/\1/p' | awk '$1 >= 50 {n++} END {print n + 0}'; }
held() { [ -e ~/.local/share/quiet/PAUSE ] || [ -e ~/.local/share/quiet/HOLD ]; }
lid_closed() { ioreg -r -k AppleClamshellState -d 1 | grep -q '"AppleClamshellState" = Yes'; }
killtree() { for c in $(pgrep -P $1); do killtree $c $2; done; kill -$2 $1 2>/dev/null; }
say() { echo "$(date +%T) $*" | tee -a $P/summary.txt; }
gate() {   # $P/GO = NewIdeas' yes (touched by Leap by hand); without it nothing starts
  [ -f $P/GO ] || { echo "waiting for NewIdeas' yes (no $P/GO)"; return; }
  [ "$(quiet_n)" -gt 0 ] && { echo "a quiet lock is held or queued"; return; }
  [ "$(bigmem_n)" -gt 0 ] && { echo "another session holds a MEM > 5 GB lock"; return; }
  [ "$(gpu_n)" -gt 0 ] && { echo "another shared run uses >= 50 % GPU"; return; }
  held && { echo "PAUSE / HOLD"; return; }
  lid_closed && { echo "lid closed"; return; }
> (a quote from the project owner, kept private)
> (a quote from the project owner, kept private)
  local a b c; a=$(noswap); sleep 5; b=$(noswap); sleep 5; c=$(noswap)
  [ $a -lt 49 ] || [ $b -lt 49 ] || [ $c -lt 49 ] && { echo "NO-SWAP $a / $b / $c GB < 49"; return; }
}
recheck() {
  local w=""
  [ "$(noswap)" -lt 47 ] && w="NO-SWAP $(noswap) GB"
  [ "$(quiet_n)" -gt 0 ] && w="$w; a quiet lock"
  [ "$(bigmem_n)" -gt 0 ] && w="$w; a MEM > 5 GB lock"
  [ "$(gpu_n)" -gt 0 ] && w="$w; a >= 50 % GPU shared run"
  echo "$w"
}
LOCKED=0; RP=""
finish() { [ -n "$RP" ] && kill -0 $RP 2>/dev/null && killtree $RP TERM; [ $LOCKED = 1 ] && { peer unlock >/dev/null 2>&1; say "unlocked"; }; say "exit (NO-SWAP $(noswap) GB, swap $(swap) MB)"; }
trap finish EXIT
trap 'say "stopped by a signal"; exit 1' INT TERM
say "armed (pid $$): exact-32 30K PF-A / PF-B 8192 / PF-B 4096, R57_LOGHASH=1; hard end $(date -r $END +%H:%M:%S)"
CLR=0; NOTE=""
while :; do
  [ $(( $(date +%s) + 100 )) -gt $END ] && { say "SKIP: no start that can end by $(date -r $END +%H:%M:%S) (last wait: ${NOTE:-none})"; echo "SKIP" > $P/result; exit 0; }
  why=$(gate)
  if [ -z "$why" ]; then CLR=$((CLR + 1)); else CLR=0; fi
  [ -z "$why" ] && [ $CLR -lt 2 ] && why="clear once: one more poll"
  if [ -z "$why" ]; then
    M=$(( (END - $(date +%s)) / 60 + 2 ))
    out=$(peer lock share ${M}m --cpu 300 --gpu 100 "MEM 43 GB: Leap PF-B exact check, exact-32 30K x3, ends by $(date -r $END +%H:%M)" 2>&1); rc=$?
    say "share lock (${M}m) rc $rc: $(echo "$out" | tr '\n' ' ' | cut -c1-300)"
    if [ $rc = 0 ]; then
      LOCKED=1; sleep 20; w2=$(recheck); [ -z "$w2" ] && { sleep 20; w2=$(recheck); }
      [ -z "$w2" ] && break
      say "after the lock: $w2: unlock and wait"; peer unlock >/dev/null 2>&1; LOCKED=0; CLR=0; why="after the lock: $w2"
    else
      why="share lock refused"
    fi
  fi
  [ "$why" != "$NOTE" ] && say "wait: $why"; NOTE=$why
  sleep 15
done
S0=$(swap); say "GO: NO-SWAP $(noswap) GB, swap $S0 MB, idle $(idle) s"
for arm in "pfa 1 0 47" "pfb8k 3 8192 49" "pfb4k 3 4096 49"; do
  set -- $arm; A_=$1; MODE=$2; TOK=$3; NEED=$4
  [ $(( $(date +%s) + 100 )) -gt $END ] && { say "SKIP $A_: cannot end by $(date -r $END +%H:%M:%S)"; continue; }
> (a quote from the project owner, kept private)
  N_=$(noswap); [ "$N_" -lt "$NEED" ] && { say "SKIP $A_: NO-SWAP $N_ GB < $NEED GB"; continue; }
  EXTRA=""; [ $TOK != 0 ] && EXTRA="--env R57_PFLM_TOK=$TOK"
  CAP=$(( END - $(date +%s) ))
  say "run $A_ (R57_PFLM=$MODE${TOK:+, TOK $TOK}; NO-SWAP $N_ GB; cap $CAP s)"
  .venv/bin/python run.py --config exact --mem 32 --prompt-file $PR --max-tokens 64 --env R57_PFLM=$MODE $EXTRA --env R57_LOGHASH=1 > $P/run_$A_.out 2>&1 &
  RP=$!; T0=$(date +%s); SMAX=$S0; ACT=0; WHY=""
  while kill -0 $RP 2>/dev/null; do
    s=$(swap); [ "$s" -gt "$SMAX" ] && SMAX=$s
    i=$(idle); if [ "$i" -lt 5 ]; then ACT=$((ACT + 1)); else ACT=0; fi
    if [ $((s - S0)) -gt 2048 ]; then WHY="swap $s MB (start $S0)"
> (a quote from the project owner, kept private)
    elif [ $(( $(date +%s) - T0 )) -gt $CAP ]; then WHY="hard end"
    fi
    [ -n "$WHY" ] && { say "STOP $A_: $WHY"; killtree $RP TERM; break; }
    sleep 3
  done
  wait $RP 2>/dev/null; RP=""
  L=$(grep -a -E "^whole decode [0-9.]+ tok/s" $P/run_$A_.out | tail -1)
  grep -a -o "^tokens: [^;]*" $P/run_$A_.out | tail -1 | cut -c9- > $P/tokens_$A_
  say "  $A_: ${L:-no result (see run_$A_.out)} | wall $(( $(date +%s) - T0 )) s | swap max $SMAX MB"
  [ -n "$WHY" ] && break
done
peer unlock >/dev/null 2>&1; LOCKED=0; say "unlocked"
/opt/homebrew/opt/python@3.12/bin/python3.12 -I - "$P" <<'EOF' | tee -a $P/summary.txt
import json, os, sys
P = sys.argv[1]
def toks(x):
    if isinstance(x, list) and x and all(isinstance(t, int) for t in x): return x
    for v in (x.values() if isinstance(x, dict) else x if isinstance(x, list) else []):
        t = toks(v)
        if t: return t
got = {}
for a in ("pfa", "pfb8k", "pfb4k"):
    f = os.path.join(P, f"tokens_{a}")
    path = open(f).read().strip() if os.path.exists(f) else ""
    if path and os.path.exists(path):
        r = json.load(open(path.replace("_tokens.json", "_runs.json")))[0]
        got[a] = (toks(json.load(open(path))), r.get("lh") or [], r.get("pflm"), r.get("pfs"))
if "pfa" not in got:
    print("exact check: no PF-A reference"); sys.exit()
ta, ha, _, _ = got["pfa"]
for a in ("pfb8k", "pfb4k"):
    if a not in got:
        print(f"{a}: not run"); continue
    tb, hb, pl, pfs = got[a]
    ft = next((i for i, (x, y) in enumerate(zip(ta, tb)) if x != y), None)
    eq = sum(x == y for x, y in zip(ha, hb)); fh = next((i for i, (x, y) in enumerate(zip(ha, hb)) if x != y), None)
    print(f"exact-32 30K {a} vs PF-A: tokens {'EQUAL' if ta == tb else f'DIFFER from token {ft + 1} (after {ft} equal)'} "
          f"({len(ta)} / {len(tb)}); verify-logits hashes equal on {eq} of {min(len(ha), len(hb))} checks"
          f"{'' if fh is None else f' (first difference at check {fh + 1})'} ({len(ha)} / {len(hb)}); pflm {pl}; prefill {pfs} s")
EOF
echo DONE > $P/result
