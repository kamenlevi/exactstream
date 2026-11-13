#!/bin/bash
# prof_sys.sh (Leap 11-13 01:1x): is the 30K full-attention layers' ~12.5 s of main-thread CPU kernel time or user time?
# (data/prefill15_1112.md sections 6-8: quadratic in the context, in the eval after the self_attn build; the dense band and a
# 4 GB MLX cache do not change it; process counters could not tell.) PF-B (R57_PFLM=3) PIPE=0 at 32 GB through prof5's copy:
# R57_PFLM_PROF_CPU=sys|user turns every stage's CPU column into the main thread's kernel or user time (Mach thread_info).
# Runs: 10K sys, 30K sys, 30K user. Kernel time ~ the thread CPU = fresh pages zero-filled / wired (the zero-fill hypothesis);
# user time = MLX's encoding or CPU-side compute. Tokens + R57_LOGHASH vs prof_ab's PIPE=0 runs (token lists only). Gates as
# prof_ab.sh (start NO-SWAP >= 45, per run >= 44, two clear polls, a share lock re-checked 20 s after the grant); swap +2 GB
# stops. Usage: prof_sys.sh <work dir with prompt_10k.txt, prompt_30k.txt and prof_ab's ab1 / ab6 files> "<LAST ...>"
W=$1; P5=/Users/kamen/Leap/scripts/prefill15/prof5; CL=/Users/kamen/Leap/release/cleanclone/cache/clone; LOG=$W/sys.log
LAST=$(date -j -f "%Y-%m-%d %H:%M:%S" "$2" +%s)
noswap() { vm_stat | awk '/page size of/ {ps=$8} /^Pages (free|speculative|purgeable):/ || /^File-backed pages:/ {v=$NF; sub("[.]","",v); s+=v} END {printf "%d", s*ps/1e9}'; }
swap() { sysctl -n vm.swapusage | sed -E 's/.*used = ([0-9.]+)M.*/\1/' | cut -d. -f1; }
idle() { ioreg -c IOHIDSystem | awk '/HIDIdleTime/ {print int($NF/1000000000); exit}'; }
procs() { ps -axo pid=,args= | awk '/[Pp]ython/ && /r57_screen_|[ \/](chat|serve|run)[.]py( |$)/ && !/ awk / {printf "%s ", $1}'; }
busy() { peer ls 2>/dev/null | grep "^- " | grep -oE "· (shared run \(others may work\)|benchmarking|waiting for a quiet window)[^·]*" | grep -cE "^· (benchmarking|waiting for a quiet window)|MEM ([6-9]|[1-9][0-9]) GB"; }
held() { [ -e ~/.local/share/quiet/PAUSE ] || [ -e ~/.local/share/quiet/HOLD ]; }
say() { echo "$(date '+%m-%d %H:%M:%S') $*" | tee -a $LOG; }
RP=""; GUARD=""; LOCKED=0
cleanup() { [ -n "$GUARD" ] && kill $GUARD 2>/dev/null; [ -n "$RP" ] && kill -0 $RP 2>/dev/null && kill -TERM $RP
            [ $LOCKED = 1 ] && peer unlock >/dev/null 2>&1; touch $W/sys.done; say "exit (NO-SWAP $(noswap) GB, swap $(swap) MB)"; }
trap cleanup EXIT; trap 'say "stopped by a signal"; exit 1' INT TERM
for n in 10k 30k; do [ -f $W/prompt_$n.txt ] || { say "no $W/prompt_$n.txt"; exit 2; }; done
[ -f $P5/pflm_prof.py ] && [ -f $P5/sitecustomize.py ] || { say "no prof5 copy"; exit 2; }
say "armed (pid $$): PF-B PIPE=0 at 32 GB, main-thread sys / user per stage: 10K sys, 30K sys, 30K user; last start $2"
NOTE=""; CLR=0
while :; do
  now=$(date +%s); [ $now -ge $LAST ] && { say "SKIP: no free window by $2 (last wait: $NOTE)"; exit 0; }
  why=""
> (a quote from the project owner, kept private)
  [ -z "$why" ] && [ -n "$(procs)" ] && why="an engine runs"
  [ -z "$why" ] && [ "$(busy)" -gt 0 ] && why="another lock"
  [ -z "$why" ] && held && why="PAUSE / HOLD"
  [ -z "$why" ] && [ "$(noswap)" -lt 45 ] && why="NO-SWAP $(noswap) < 45"
  if [ -z "$why" ]; then CLR=$((CLR + 1)); else CLR=0; fi
  [ -z "$why" ] && [ $CLR -lt 2 ] && why="clear once"
  if [ -z "$why" ]; then
    if peer lock share 30m --cpu 400 --gpu 100 "MEM 46 GB: Leap prefill sys / user split (32 GB, 10K / 30K, 3 runs), ~5 min" > /dev/null 2>&1; then
      LOCKED=1; sleep 20
      if [ "$(busy)" = 0 ] && [ -z "$(procs)" ] && [ "$(noswap)" -ge 45 ]; then say "lock granted; clear after 20 s (NO-SWAP $(noswap) GB)"; break; fi
      peer unlock >/dev/null 2>&1; LOCKED=0; CLR=0; why="after the lock: another lock / an engine / NO-SWAP $(noswap)"
    else
      CLR=0; why="share lock not granted"
    fi
  fi
  [ "$why" != "$NOTE" ] && say "wait: $why"; NOTE=$why
  sleep 60
done
S0=$(swap); ( while :; do [ $(( $(swap) - S0 )) -gt 2048 ] && { echo "$(date +%T) SWAP GUARD: $(swap) MB" >> $LOG; kill -TERM $$; exit; }; sleep 5; done ) &
GUARD=$!
for job in "10k sys" "30k sys" "30k user"; do
  set -- $job; N=$1; K=$2
  [ -n "$(procs)" ] && { say "SKIP $N $K: an engine runs"; continue; }
  [ "$(noswap)" -lt 44 ] && { say "SKIP $N $K: NO-SWAP $(noswap) < 44"; continue; }
  T=sys_${N}_$K; J=$W/$T.jsonl; rm -f $J
  say "run $N CPU=$K (NO-SWAP $(noswap) GB)"
  ( cd $CL && exec env PYTHONPATH=$P5 R57_PFLM_PROF=$J R57_PFLM_PROF_SYNC=0 R57_PFLM_PROF_CPU=$K R57_PFLM_PIPE=0 .venv/bin/python run.py \
      --config chat --mem 32 --prompt-file $W/prompt_$N.txt --max-tokens 32 --env R57_PFLM=3 --env R57_LOGHASH=1 ) > $W/$T.out 2>&1 &
  RP=$!; wait $RP; rc=$?; RP=""
  tag=$(grep -o 'out/cli_[0-9_]*_tokens.json' $W/$T.out | head -1 | sed 's|out/||; s|_tokens.json||')
  [ -n "$tag" ] && cp $CL/out/${tag}_runs.json $W/${T}_runs.json 2>/dev/null && cp $CL/out/${tag}_tokens.json $W/${T}_tokens.json 2>/dev/null
  say "  rc $rc; $(grep -o '"total": \[[^]]*\]' $J 2>/dev/null); $(grep -o 'first token [0-9.]* s' $W/$T.out | head -1); $(grep -o 'lifetime peak [0-9.]* GB' $W/$T.out | tail -1); tag $tag"
done
/opt/homebrew/opt/python@3.12/bin/python3.12 -I - $W <<'PY' | tee -a $LOG
import json, sys, collections
W = sys.argv[1]
def toks(T):
    try:
        tk = json.load(open(f"{W}/{T}_tokens.json")); rr = json.load(open(f"{W}/{T}_runs.json"))
    except Exception:
        return None
    lh = []
    def walk(o):
        if isinstance(o, dict):
            for k, v in o.items():
                (lh.append(v) if k == "lh" else walk(v))
        elif isinstance(o, list):
            for v in o: walk(v)
    walk(rr)
    return json.dumps([v for k in sorted(tk) for a in sorted(tk[k]) for v in tk[k][a]]), json.dumps(lh)
def sums(T):
    S = collections.defaultdict(lambda: [0.0, 0.0]); F = collections.defaultdict(lambda: [0.0, 0.0]); tot = None
    try:
        for line in open(f"{W}/{T}.jsonl"):
            d = json.loads(line)
            if "total" in d: tot = d["total"]; continue
            for k, (w, c) in d["t"].items():
                S[k][0] += w; S[k][1] += c
                if not d["linear"]: F[k][0] += w; F[k][1] += c
    except Exception:
        return None
    return tot, S, F
for n, ref in (("10k", "ab1_10k_p0"), ("30k", "ab6_30k_p0")):
    for k in ("sys", "user"):
        T = f"sys_{n}_{k}"; r = sums(T)
        if r is None: continue
        a, b = toks(ref), toks(T)
        eq = "no ref" if a is None or b is None else f"tokens {'EQUAL' if a[0] == b[0] else 'DIFFER'} | hashes {'EQUAL' if a[1] == b[1] else 'DIFFER'}"
        tot, S, F = r
        print(f"sys-split {n} CPU={k} ({eq} vs {ref}): whole prefill wall {tot[0]:.2f} s, main-thread {k} {tot[1]:.2f} s; "
              + ", ".join(f"{s} {w:.2f}/{c:.2f}" for s, (w, c) in sorted(S.items(), key=lambda x: -x[1][0])))
        print(f"   the 12 full-attention layers ({k}): " + ", ".join(f"{s} {w:.2f}/{c:.2f}" for s, (w, c) in sorted(F.items(), key=lambda x: -x[1][0])))
PY
