#!/bin/bash
# rel121_diag3.sh (Leap 11-14 01:2x): the 5 check-1 diagnosis runs that did not fit at gate 51 (rel121_diag1.sh /
# rel121_diag2.sh, 01:02-01:13: only "1.2.0 again" ran, = check 1's 1.2.0 on tokens, logits hashes and PF-D counts).
# The orchestrator 01:2x: run them tonight at GATE 50 (peak 46.78 + 3.2; its margin rule is peak + 3-4; these runs compare
# tokens and hashes only; the floor is 49-50 with idle pages already compressed; swap flat at 5.8 GB, +2 GB still stops).
# One quiet ticket (BigMoE's sequencer counts lapsed and queued locks since 1cc3c85, 01:13:27), runs back to back, all
# within 12 min of the grant. Same prompt (1,972 tokens), cap 256, R57_LOGHASH=1:
#   1.2.1 R57_P2=0        -> the same warm-up path as 1.2.0: = 1.2.0 means check 1's difference is P2's warm-up history
#   1.2.1 again           -> 1.2.1 reproducible (warm)?
#   1.2.1 / 1.2.0 / 1.2.1 --no-warmup -> cold (empty history): the long-prompt path alone, and its reproducibility
# Usage: rel121_diag3.sh "<CUTOFF YYYY-MM-DD HH:MM:SS>"
V=/Users/kamen/Leap/release/cleanclone/v121; CL=$V/clone; V0=/Users/kamen/Leap/release/cleanclone/v120b/clone
SCR=/private/tmp/claude-501/-Users-kamen-OjoX/f2971937-6c95-4e86-a049-05beff6a8e32/scratchpad/agent
P=$SCR/r121; HFH=$HOME/.cache/huggingface; Q=$HOME/.local/share/quiet; H2=$V/home2; O=$V/out; LOG=$V/rel121_diag3.log
PY=/opt/homebrew/opt/python@3.12/bin/python3.12
CUT=$(date -j -f "%Y-%m-%d %H:%M:%S" "${1:?CUTOFF}" +%s); G=50
noswap() { vm_stat | awk '/page size of/ {ps=$8} /^Pages (free|speculative|purgeable):/ || /^File-backed pages:/ {v=$NF; sub("[.]","",v); s+=v} END {printf "%d", s*ps/1e9}'; }
swap() { sysctl -n vm.swapusage | sed -E 's/.*used = ([0-9.]+)M.*/\1/' | cut -d. -f1; }
idle() { ioreg -c IOHIDSystem | awk '/HIDIdleTime/ {print int($NF/1000000000); exit}'; }
procs() { ps -axo pid=,args= | awk '/[Pp]ython/ && /r57_screen_|run_map[.]py|fit_onecall[.]py|[ \/](chat|serve|run)[.]py( |$)/ && !/ awk / {printf "%s ", $1}'; }
lid_closed() { ioreg -r -k AppleClamshellState -d 1 | grep -q '"AppleClamshellState" = Yes'; }
say() { echo "$(date '+%m-%d %H:%M:%S') $*" | tee -a $LOG; }
killtree() { local c; for c in $(pgrep -P $1 2>/dev/null); do killtree $c; done; kill -TERM $1 2>/dev/null; }
LOCKED=0; CP=""; GUARD=""; RC=""; DL=0
cleanup() { [ -n "$GUARD" ] && kill $GUARD 2>/dev/null
            [ -n "$CP" ] && kill -0 $CP 2>/dev/null && killtree $CP
            [ $LOCKED = 1 ] && { peer unlock >/dev/null 2>&1; say "ticket released"; }
            say "exit (NO-SWAP $(noswap) GB, swap $(swap) MB)"; }
trap cleanup EXIT; trap 'say "stopped by a signal"; exit 1' INT TERM
for f in $P/long_2k.txt $CL/run.py $V0/run.py $O/long2k_121 $O/long2k_120; do [ -e $f ] || { say "STOP: no $f"; exit 2; }; done
mkdir -p $H2 $O
say "armed (pid $$): check-1 diagnosis, the 5 remaining runs at gate $G (1.2.1 P2=0, 1.2.1 again, 1.2.1 / 1.2.0 / 1.2.1 cold); every run ends within 12 min of the grant and by $1"
TO=$(( CUT - $(date +%s) - 600 )); [ $TO -lt 120 ] && { say "SKIP: under 12 min to the cutoff"; exit 0; }
say "queue the quiet ticket (PEER_QUEUE_TIMEOUT $TO s; NO-SWAP $(noswap) GB)"
if PEER_QUEUE_TIMEOUT=$TO peer lock quiet --queue 15m "Leap: 1.2.1 token check 1 diagnosis (5 short runs at 45 GB, 1,972 tokens), MEM 50 GB, ~6 min" >> $LOG 2>&1; then
  LOCKED=1
else
  say "the ticket left the queue without a grant"; exit 0
fi
GR=$(date +%s); DL=$(( GR + 720 )); [ $DL -gt $CUT ] && DL=$CUT
say "granted (NO-SWAP $(noswap) GB); settle 30 s; every run ends by $(date -r $DL +%T)"; sleep 30
while [ "$(noswap)" -lt $G ] && [ $(date +%s) -lt $(( GR + 240 )) ]; do sleep 10; done
[ "$(noswap)" -lt $G ] && { say "STOP: NO-SWAP $(noswap) < $G at grant + $(( $(date +%s) - GR )) s"; exit 0; }
S0=$(swap); ( while :; do [ $(( $(swap) - S0 )) -gt 2048 ] && { echo "$(date +%T) SWAP GUARD: $(swap) MB" >> $LOG; kill -TERM $$; exit; }; sleep 5; done ) &
GUARD=$!
run() {   # run <tag> <clone> [extra args]
  local t=$1 c=$2; shift 2
  [ -e $Q/HOLD ] && { say "STOP: HOLD"; exit 0; }
  while [ -e $Q/PAUSE ] && [ $(( $(date +%s) + 75 )) -le $DL ]; do sleep 5; done
  [ -e $Q/PAUSE ] && { say "SKIP $t: PAUSE"; return 1; }
  lid_closed && { say "STOP: lid closed"; exit 0; }
  grep -q "Verdict: ACT" $Q/HEALTH.md 2>/dev/null && { say "STOP: HEALTH ACT"; exit 0; }
> (a quote from the project owner, kept private)
  while [ -n "$(procs)" ] && [ $(( $(date +%s) + 75 )) -le $DL ]; do sleep 5; done
  [ -n "$(procs)" ] && { say "SKIP $t: an engine runs ($(procs))"; return 1; }
  while [ "$(noswap)" -lt $G ] && [ $(( $(date +%s) + 75 )) -le $DL ]; do sleep 5; done
  [ "$(noswap)" -lt $G ] && { say "SKIP $t: NO-SWAP $(noswap) < $G"; return 1; }
  [ $(( $(date +%s) + 75 )) -gt $DL ] && { say "SKIP $t: it could end after $(date -r $DL +%T) (grant + 12 min)"; return 1; }
  say "run $t ($c $*) (NO-SWAP $(noswap) GB)"
  env HOME=$H2 HF_HOME=$HFH $c/.venv/bin/python $c/run.py --config chat --mem 45 --prompt-file $P/long_2k.txt --max-tokens 256 \
      --env R57_LOGHASH=1 "$@" --out $O/$t > $O/$t.txt 2>&1 &
  CP=$!; wait $CP; RC=$?; CP=""
  say "  $t rc $RC; $(grep -o 'lifetime peak [0-9.]* GB' $O/$t.txt | tail -1)"
}
run diag_121_nop2 $CL --env R57_P2=0
run diag_121_r3 $CL
run diag_121_cold $CL --no-warmup
run diag_120_cold $V0 --no-warmup
run diag_121_cold2 $CL --no-warmup
say "runs done at grant + $(( $(date +%s) - GR )) s"
peer unlock >/dev/null 2>&1; LOCKED=0; say "ticket released (before the compare)"
$PY - $O <<'EOF' | tee -a $LOG
import glob, json, sys
O = sys.argv[1]
def newest(t, k):
    f = sorted(glob.glob(f"{O}/{t}/cli_*_{k}.json")); return f[-1] if f else None
def load(t):
    tk, rr = newest(t, "tokens"), newest(t, "runs")
    if not tk or not rr: return None
    a = json.load(open(tk)); r = json.load(open(rr))
    toks = [v for k in sorted(a) for arm in sorted(a[k]) for v in a[k][arm]][0]
    pfd = r[0].get("pfd") or {}
    return dict(t=toks, lh=r[0].get("lh"), abc=(pfd.get("A"), pfd.get("B"), pfd.get("C")), p2=r[0].get("p2"))
names = [("1.2.1 (check 1)", "long2k_121"), ("1.2.0 (check 1)", "long2k_120"), ("1.2.0 again", "diag_120_r1"),
         ("1.2.1 P2=0", "diag_121_nop2"), ("1.2.1 again", "diag_121_r3"), ("1.2.1 cold", "diag_121_cold"),
         ("1.2.0 cold", "diag_120_cold"), ("1.2.1 cold again", "diag_121_cold2")]
runs = [(n, load(t)) for n, t in names]
for n, r in runs:
    print(f"{n:16s}: " + ("no run" if r is None else f"PF-D A/B/C {r['abc']}, p2 {r['p2']}, first tokens {r['t'][:4]}"))
def cmp(x, y):
    if x is None or y is None: return "n/a"
    d = next((i for i, (p, q) in enumerate(zip(x["t"], y["t"])) if p != q), None)
    return ("tokens EQUAL" if d is None and len(x["t"]) == len(y["t"]) else f"tokens differ at {d}") + \
           (", logits hashes EQUAL" if x["lh"] == y["lh"] else ", logits hashes differ")
for i in range(len(runs)):
    for j in range(i + 1, len(runs)):
        print(f"  {runs[i][0]} vs {runs[j][0]}: {cmp(runs[i][1], runs[j][1])}")
EOF
say "DONE"
exit 0
