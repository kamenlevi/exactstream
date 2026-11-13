#!/bin/bash
# onebuild_45_runs.sh (Leap 11-13 01:5x): onebuild_45.sh v2's runs and compare WITHOUT its own gate loop and share lock, for
# night45_1113.sh, which holds one quiet lock for the 45 GB clean-clone stage and this (the orchestrator 01:4x). Under that
# lock the old gate loop would wait forever: busy() counts other sessions' queued quiet tickets, e.g. the ones queued behind
# this run. Same runs, order and compare as onebuild_45.sh (pre-registered in data/prefill15_1112.md section 3):
#   R0 = as shipped (PF-D; no hook): the reference;  R1 = R57_PFLM=3 R57_PFD=0 with the one-build pflm.py (hook);
#   R2 = R1 + R57_PFLM_QD=2;  R3 = R1 + R57_PFLM_QD=1 R57_PFLM_CC=1.  10K R0 R1 R2 R3, then 30K R3 R2 R1 R0.
# Kept: before each run no engine and NO-SWAP >= 52, else that run is skipped; swap +2 GB stops. The caller checks NO-SWAP
# >= 56 before calling. Usage: onebuild_45_runs.sh <work dir with prompt_10k.txt and prompt_30k.txt>
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
for job in "10k R0" "10k R1" "10k R2" "10k R3" "30k R3" "30k R2" "30k R1" "30k R0"; do
  set -- $job; N=$1; R=$2
  [ -n "$(procs)" ] && { say "SKIP $N $R: an engine runs"; continue; }
  [ "$(noswap)" -lt 52 ] && { say "SKIP $N $R: NO-SWAP $(noswap) < 52"; continue; }
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
