#!/bin/bash
# onebuild_45_runs.sh (Leap 11-13 01:5x): onebuild_45.sh v2's runs and compare WITHOUT its own gate loop and share lock, for
# night45_1113.sh, which holds one quiet lock for the 45 GB clean-clone stage and this (the orchestrator 01:4x). Under that
# lock the old gate loop would wait forever: busy() counts other sessions' queued quiet tickets, e.g. the ones queued behind
# this run. Same runs, order and compare as onebuild_45.sh (pre-registered in data/prefill15_1112.md section 3):
#   R0 = as shipped (PF-D; no hook): the reference;  R1 = R57_PFLM=3 R57_PFD=0 with the one-build pflm.py (hook);
#   R2 = R1 + R57_PFLM_QD=2;  R3 = R1 + R57_PFLM_QD=1 R57_PFLM_CC=1.  10K R0 R1 R2 R3, then 30K R3 R2 R1 R0.
# v2 (Fri 02:4x; the orchestrator 02:3x: under a quiet lock the margin is peak + 4, and 56 never came on a normal night):
# per-run gates = the run's lifetime peak + 4 GB, highest-peak runs last. Measured peaks: R0 = chat-45 with PF-D 47.46 GB
# (10K) / 48.56 GB (30K) (BigMoE data/pfd_long, 11-10). PF-B at 45 GB is unmeasured; estimated as PF-D off (45.17 / 46.65 GB)
# plus PF-B's extra over the default at 32 GB (+1.8 / +5.4 GB, data/prefill_1109.md): R1 ~47.0 / ~52 GB; the QD trims R2 / R3
# sit between R0 and R1 at 30K (~49-50 GB). Gates: every 10K run 52; 30K R0 53, R3 and R2 54, R1 56. The 30K block now runs
# R0 FIRST (its gate is the lowest, and R1-R3 need it as the bitwise reference): 30K R0 R3 R2 R1 (was R3 R2 R1 R0).
# v3 (Fri 15:5x; the orchestrator's start rule: nothing may END after Sat 10:00): CUTOFF=<epoch> skips a run that could end
# after it (est 420 s per 10K run, 900 s per 30K run). BLOCKS (default "10k 30k") picks the blocks. Before each run: no engine and NO-SWAP >= its gate, else that run is skipped;
# swap +2 GB stops. Usage: [BLOCKS="10k"] onebuild_45_runs.sh <work dir with prompt_10k.txt and prompt_30k.txt>
W=$1; CL=/Users/kamen/Leap/release/cleanclone/cache/clone; HOOK=/Users/kamen/Leap/scripts/prefill15/onebuild/hook; LOG=$W/onebuild.log
noswap() { vm_stat | awk '/page size of/ {ps=$8} /^Pages (free|speculative|purgeable):/ || /^File-backed pages:/ {v=$NF; sub("[.]","",v); s+=v} END {printf "%d", s*ps/1e9}'; }
swap() { sysctl -n vm.swapusage | sed -E 's/.*used = ([0-9.]+)M.*/\1/' | cut -d. -f1; }
procs() { ps -axo pid=,args= | awk '/[Pp]ython/ && /r57_screen_|run_map[.]py|fit_onecall[.]py|[ \/](chat|serve|run)[.]py( |$)/ && !/ awk / {printf "%s ", $1}'; }
say() { echo "$(date '+%m-%d %H:%M:%S') $*" | tee -a $LOG; }
RP=""; GUARD=""
cleanup() { [ -n "$GUARD" ] && kill $GUARD 2>/dev/null; [ -n "$RP" ] && kill -0 $RP 2>/dev/null && kill -TERM $RP
            touch $W/onebuild.done; say "exit (NO-SWAP $(noswap) GB, swap $(swap) MB)"; }
trap cleanup EXIT; trap 'say "stopped by a signal"; exit 1' INT TERM
for n in 10k 30k; do [ -f $W/prompt_$n.txt ] || { say "no $W/prompt_$n.txt"; exit 2; }; done
[ -f $HOOK/sitecustomize.py ] && [ -f $HOOK/../pflm.py ] || { say "no hook or candidate pflm.py"; exit 2; }
say "runs (pid $$, under the caller's quiet lock): PF-B one-build at 45 GB, R0 (PF-D) vs R1-R3 on 10K / 30K, tokens + LOGHASH + peaks; candidate pflm.py sha256 $(shasum -a 256 $HOOK/../pflm.py | cut -c1-12)"
S0=$(swap); ( while :; do [ $(( $(swap) - S0 )) -gt 2048 ] && { echo "$(date +%T) SWAP GUARD: $(swap) MB" >> $LOG; kill -TERM $$; exit; }; sleep 5; done ) &
GUARD=$!
BLOCKS=${BLOCKS:-10k 30k}
for job in "10k R0 52" "10k R1 52" "10k R2 52" "10k R3 52" "30k R0 53" "30k R3 54" "30k R2 54" "30k R1 56"; do
  set -- $job; N=$1; R=$2; NEED=$3
  [[ " $BLOCKS " == *" $N "* ]] || continue
  [ -n "$(procs)" ] && { say "SKIP $N $R: an engine runs"; continue; }
  [ "$(noswap)" -lt $NEED ] && { say "SKIP $N $R: NO-SWAP $(noswap) < $NEED (its peak + 4)"; continue; }
  EST=$( [ $N = 10k ] && echo 420 || echo 900 )        # v3: a generous wall per run (load, prefill, 32 tokens, heat pauses)
  [ -n "${CUTOFF:-}" ] && [ $(( $(date +%s) + EST )) -gt "$CUTOFF" ] && { say "SKIP $N $R: it could end after the cutoff (est $EST s)"; continue; }
  case $R in
    R0) PP=""; EX="" ;;
    R1) PP=$HOOK; EX="--env R57_PFLM=3 --env R57_PFD=0" ;;
    R2) PP=$HOOK; EX="--env R57_PFLM=3 --env R57_PFD=0 --env R57_PFLM_QD=2" ;;
    R3) PP=$HOOK; EX="--env R57_PFLM=3 --env R57_PFD=0 --env R57_PFLM_QD=1 --env R57_PFLM_CC=1" ;;
  esac
  say "run $N $R (NO-SWAP $(noswap) GB)"
  ( cd $CL && exec env ${PP:+PYTHONPATH=$PP} .venv/bin/python run.py --config chat --mem 45 --prompt-file $W/prompt_$N.txt \
      --max-tokens 32 --trace $EX --env R57_LOGHASH=1 ) > $W/ob_${N}_$R.out 2>&1 &
  RP=$!; wait $RP; rc=$?; RP=""
  tag=$(grep -o 'out/cli_[0-9_]*_tokens.json' $W/ob_${N}_$R.out | head -1 | sed 's|out/||; s|_tokens.json||')
  if [ -n "$tag" ]; then for k in runs tokens modules; do cp $CL/out/${tag}_$k.json $W/ob_${N}_${R}_$k.json 2>/dev/null; done; fi
  say "  rc $rc; $(grep -o 'first token [0-9.]* s' $W/ob_${N}_$R.out | head -1); $(grep -o 'lifetime peak [0-9.]* GB' $W/ob_${N}_$R.out | head -1)"
done
/opt/homebrew/opt/python@3.12/bin/python3.12 -I - $W <<'EOF' | tee -a $LOG
import json, os, sys
W = sys.argv[1]
def load(n, r):
    try:
        tk = json.load(open(f"{W}/ob_{n}_{r}_tokens.json")); rr = json.load(open(f"{W}/ob_{n}_{r}_runs.json"))
    except Exception:
        return None
    lh = []
    def walk(o):
        if isinstance(o, dict):
            for k, v in o.items():
                if k == "lh": lh.append(v)
                else: walk(v)
        elif isinstance(o, list):
            for v in o: walk(v)
    walk(rr)
    peak = max((x["mem"][6] for x in rr if x.get("mem") and len(x["mem"]) > 6), default=None)
    pf = None
    try:
        m = json.load(open(f"{W}/ob_{n}_{r}_modules.json"))
        pf = next((p for name, p in (m["modules"] if isinstance(m, dict) else m) if name == "pflm"), "not imported")
    except Exception:
        pf = "?"
    return dict(tok=json.dumps([v for k in sorted(tk) for a in sorted(tk[k]) for v in tk[k][a]]), lh=json.dumps(lh), nlh=len(lh), peak=peak, pflm=pf)
for r in ("R0", "R1", "R2", "R3"):
    parts = []
    for n in ("10k", "30k"):
        a, b = load(n, "R0"), load(n, r)
        if b is None:
            parts.append(f"{n}: no run"); continue
        src = "onebuild" if "onebuild" in str(b["pflm"]) else ("engine" if "engine/loop" in str(b["pflm"]) else b["pflm"])
        if r == "R0" or a is None:
            parts.append(f"{n}: peak {b['peak']} GB, pflm {src}, lh {b['nlh']}"); continue
        parts.append(f"{n}: tokens {'EQUAL' if a['tok'] == b['tok'] else 'DIFFER'}, lh {'EQUAL' if a['lh'] == b['lh'] else 'DIFFER'} "
                     f"({b['nlh']}), peak {b['peak']} GB "
                     f"({(b['peak'] - a['peak']) if (b['peak'] is not None and a['peak'] is not None) else float('nan'):+.2f} vs R0), pflm {src}")
    print(f"onebuild {r}: " + " | ".join(parts))
EOF
