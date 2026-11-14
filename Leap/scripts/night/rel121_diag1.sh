#!/bin/bash
# rel121_diag1.sh (Leap 11-14 01:0x): diagnosis of 1.2.1's FAILED token check 1 (rel121_check_v2.sh part D, 00:46-00:47:
# the 1,972-token prompt, 1.2.1 chat-45 vs 1.2.0 chat-45, tokens differ from token 1, first logits hash differs).
# Both runs were fresh processes, but the warm-up (gsm8k/708, short) takes P2 in 1.2.1 and PF-D in 1.2.0, so the long
# prompt starts from a different expert state; c45p2's loop is rel1's plus only the P2 additions (README, static analysis).
# The orchestrator 01:0x: run it now, these THREE runs back to back, all ending within 8 min of the grant (after 8 min, a
# quiet lock with no bench process running shows as lapsed). Same prompt, cap 256, R57_LOGHASH=1, one quiet ticket:
#   R1  1.2.0 again            -> is the reference (with its warm-up) reproducible run to run?
#   R2  1.2.1 with R57_P2=0    -> the same warm-up path as 1.2.0: bit-identical to 1.2.0 = check 1's difference is P2's
#                                 warm-up history (the fix would be a matched-history check 1)
#   R3  1.2.1 again            -> is 1.2.1 reproducible run to run?
# Gate: NO-SWAP >= 51 (the check's peak 46.78 GB + 4). Every wait and run is bounded by DL = grant + 8 min: the first
# memory wait stops at grant + 4 min, and a run that could not end by DL (est 75 s) is skipped. Kamen-idle, lid, HOLD,
# PAUSE, HEALTH ACT are checked before each run; swap +2 GB stops; nothing may end after CUTOFF.
# Usage: rel121_diag1.sh "<FROM YYYY-MM-DD HH:MM:SS>" "<CUTOFF YYYY-MM-DD HH:MM:SS>"
V=/Users/kamen/Leap/release/cleanclone/v121; CL=$V/clone; V0=/Users/kamen/Leap/release/cleanclone/v120b/clone
SCR=/private/tmp/claude-501/-Users-kamen-OjoX/f2971937-6c95-4e86-a049-05beff6a8e32/scratchpad/agent
P=$SCR/r121; HFH=$HOME/.cache/huggingface; Q=$HOME/.local/share/quiet; H2=$V/home2; O=$V/out; LOG=$V/rel121_diag1.log
PY=/opt/homebrew/opt/python@3.12/bin/python3.12
FROM=$(date -j -f "%Y-%m-%d %H:%M:%S" "${1:?FROM}" +%s); CUT=$(date -j -f "%Y-%m-%d %H:%M:%S" "${2:?CUTOFF}" +%s)
noswap() { vm_stat | awk '/page size of/ {ps=$8} /^Pages (free|purgeable):/ || /^File-backed pages:/ {v=$NF; sub("[.]","",v); s+=v} END {printf "%d", s*ps/1e9}'; }
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
say "armed (pid $$): check-1 diagnosis (R1 1.2.0 again, R2 1.2.1 R57_P2=0, R3 1.2.1 again); from $1, every run ends within 8 min of the grant and by $2"
while [ $(date +%s) -lt $FROM ]; do sleep 20; done
TO=$(( CUT - $(date +%s) - 600 )); [ $TO -lt 120 ] && { say "SKIP: under 12 min to the cutoff"; exit 0; }
say "queue the quiet ticket (PEER_QUEUE_TIMEOUT $TO s; NO-SWAP $(noswap) GB)"
if PEER_QUEUE_TIMEOUT=$TO peer lock quiet --queue 15m "Leap: 1.2.1 token check 1 diagnosis (3 short runs at 45 GB, 1,972 tokens), MEM 51 GB, ~6 min" >> $LOG 2>&1; then
  LOCKED=1
else
  say "the ticket left the queue without a grant"; exit 0
fi
GR=$(date +%s); DL=$(( GR + 480 )); [ $DL -gt $CUT ] && DL=$CUT
say "granted (NO-SWAP $(noswap) GB); settle 30 s; every run ends by $(date -r $DL +%T)"; sleep 30
while [ "$(noswap)" -lt 51 ] && [ $(date +%s) -lt $(( GR + 240 )) ]; do sleep 10; done
[ "$(noswap)" -lt 51 ] && { say "STOP: NO-SWAP $(noswap) < 51 at grant + $(( $(date +%s) - GR )) s"; exit 0; }
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
  while [ "$(noswap)" -lt 51 ] && [ $(( $(date +%s) + 75 )) -le $DL ]; do sleep 5; done
  [ "$(noswap)" -lt 51 ] && { say "SKIP $t: NO-SWAP $(noswap) < 51"; return 1; }
  [ $(( $(date +%s) + 75 )) -gt $DL ] && { say "SKIP $t: it could end after $(date -r $DL +%T) (grant + 8 min)"; return 1; }
  say "run $t ($c $*) (NO-SWAP $(noswap) GB)"
  env HOME=$H2 HF_HOME=$HFH $c/.venv/bin/python $c/run.py --config chat --mem 45 --prompt-file $P/long_2k.txt --max-tokens 256 \
      --env R57_LOGHASH=1 "$@" --out $O/$t > $O/$t.txt 2>&1 &
  CP=$!; wait $CP; RC=$?; CP=""
  say "  $t rc $RC; $(grep -o 'lifetime peak [0-9.]* GB' $O/$t.txt | tail -1)"
}
run diag_120_r1 $V0
run diag_121_nop2 $CL --env R57_P2=0
run diag_121_r3 $CL
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
         ("1.2.1 P2=0", "diag_121_nop2"), ("1.2.1 again", "diag_121_r3")]
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
