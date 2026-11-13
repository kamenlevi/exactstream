#!/bin/bash
# prof_ab.sh (Leap 11-13 00:5x): two checks in one 32 GB slot, after the night's prof chain and before the 45 GB clean-clone
# stage (the orchestrator's night queue, 11-12; record: data/prefill15_1112.md sections 6-7).
# (1) PF-B PIPE=2 (layer L+1's expert reads in a background thread through bigmoe_ext_ov, GIL released) vs PIPE=0, order-
#     balanced. prof_pipe3 ran PIPE=2 after prof_pipe2's PIPE=0 runs and saw first token 10K 14.77 vs 18.34 s, 30K 52.58 vs
#     57.43 s (bitwise EQUAL); a later back-to-back slot can favour an arm by 12-21 %, so here ABBA: 10K p0 p2 p2 p0, then
#     30K p2 p0 p0 p2. Verdict on the arm means; timings are information only (share lock, fans at the night cap).
# (2) Where the 30K main-thread CPU goes. Every run runs under /usr/bin/time -l (process user / sys CPU, page reclaims).
#     Hypothesis (prof_pipe2, section 6): the 12 full-attention layers' ~12.9 s of main-thread CPU at 30K is the kernel
#     zero-filling fresh O(context) planes every chunk (MLX's unfused attention for head dim 256 materializes the
#     [heads, rows, T] score plane), because the engine keeps MLX's buffer cache at 1 GB (BIGMOE_MLX_CACHE_MB) and clears it
#     often. It predicts sys CPU of about that size, growing ~quadratically from 10K to 30K. Last run: 30K PIPE=0 with
#     BIGMOE_MLX_CACHE_MB=4096 (does a bigger cache keep the planes and cut sys CPU?).
# All runs: the prof3 instrumented pflm copy (timestamps only), --max-tokens 32, R57_LOGHASH=1; tokens + verify-logit hashes
# vs this series' first PIPE=0 run of the same length (token lists only: item keys carry timestamps). Gates as prof_pipe3
# (no engine / other lock / PAUSE / HOLD, Kamen idle >= 600 s, start NO-SWAP >= 45, two clear polls, a share lock re-checked
# 20 s after the grant); before each run NO-SWAP >= 44 (>= 47 for the 4 GB cache run) or that run is skipped; swap +2 GB
# stops. The run's process is /usr/bin/time; a stop kills its child (the engine) by pid first.
# Usage: prof_ab.sh <work dir with prompt_10k.txt and prompt_30k.txt> "<LAST YYYY-MM-DD HH:MM:SS>"
W=$1; P3=/Users/kamen/Leap/scripts/prefill15/prof3; CL=/Users/kamen/Leap/release/cleanclone/cache/clone; LOG=$W/ab.log
SO=/private/tmp/claude-501/-Users-kamen-OjoX/f2971937-6c95-4e86-a049-05beff6a8e32/scratchpad/agent/ext_ov_build/bigmoe_ext_ov.cpython-312-darwin.so
LAST=$(date -j -f "%Y-%m-%d %H:%M:%S" "$2" +%s)
noswap() { vm_stat | awk '/page size of/ {ps=$8} /^Pages (free|speculative|purgeable):/ || /^File-backed pages:/ {v=$NF; sub("[.]","",v); s+=v} END {printf "%d", s*ps/1e9}'; }
swap() { sysctl -n vm.swapusage | sed -E 's/.*used = ([0-9.]+)M.*/\1/' | cut -d. -f1; }
idle() { ioreg -c IOHIDSystem | awk '/HIDIdleTime/ {print int($NF/1000000000); exit}'; }
procs() { ps -axo pid=,args= | awk '/[Pp]ython/ && /r57_screen_|[ \/](chat|serve|run)[.]py( |$)/ && !/ awk / {printf "%s ", $1}'; }
busy() { peer ls 2>/dev/null | grep "^- " | grep -oE "· (shared run \(others may work\)|benchmarking|waiting for a quiet window)[^·]*" | grep -cE "^· (benchmarking|waiting for a quiet window)|MEM ([6-9]|[1-9][0-9]) GB"; }
held() { [ -e ~/.local/share/quiet/PAUSE ] || [ -e ~/.local/share/quiet/HOLD ]; }
say() { echo "$(date '+%m-%d %H:%M:%S') $*" | tee -a $LOG; }
RP=""; GUARD=""; LOCKED=0
killrun() { [ -n "$RP" ] || return 0; for c in $(pgrep -P $RP 2>/dev/null); do kill -TERM $c 2>/dev/null; done
            kill -0 $RP 2>/dev/null && kill -TERM $RP; return 0; }
cleanup() { [ -n "$GUARD" ] && kill $GUARD 2>/dev/null; killrun
            [ $LOCKED = 1 ] && peer unlock >/dev/null 2>&1; touch $W/ab.done; say "exit (NO-SWAP $(noswap) GB, swap $(swap) MB)"; }
trap cleanup EXIT; trap 'say "stopped by a signal"; exit 1' INT TERM
[ -f $SO ] || { say "no $SO"; exit 2; }
for n in 10k 30k; do [ -f $W/prompt_$n.txt ] || { say "no $W/prompt_$n.txt"; exit 2; }; done
say "armed (pid $$): PIPE=2 vs PIPE=0 ABBA at 10K / 30K + a 30K 4 GB-cache run, all under time -l (9 runs); last start $2"
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
    if peer lock share 30m --cpu 400 --gpu 100 "MEM 47 GB: Leap prefill PIPE=2 A/B + sys-time check (32 GB, 10K / 30K, 9 runs), ~12 min" > /dev/null 2>&1; then
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
i=0
for job in "10k 0" "10k 2" "10k 2" "10k 0" "30k 2" "30k 0" "30k 0" "30k 2" "30k 0c"; do
  set -- $job; N=$1; A=$2; i=$((i + 1)); PI=${A%c}; MIN=44; CE=""
  [ "$A" = "0c" ] && { MIN=47; CE="BIGMOE_MLX_CACHE_MB=4096"; }
  [ -n "$(procs)" ] && { say "SKIP run $i ($N $A): an engine runs"; continue; }
  [ "$(noswap)" -lt $MIN ] && { say "SKIP run $i ($N $A): NO-SWAP $(noswap) < $MIN"; continue; }
  T=ab${i}_${N}_p$A; J=$W/$T.jsonl; rm -f $J
  say "run $i: $N PIPE=$PI${CE:+ $CE} (NO-SWAP $(noswap) GB)"
  ( cd $CL && exec /usr/bin/time -l env $CE PYTHONPATH=$P3 R57_PFLM_PROF=$J R57_PFLM_PROF_SYNC=0 R57_PFLM_PIPE=$PI R57_PFLM_EXTOV=$SO \
      .venv/bin/python run.py --config chat --mem 32 --prompt-file $W/prompt_$N.txt --max-tokens 32 --env R57_PFLM=3 \
      --env R57_LOGHASH=1 ${CE:+--env $CE} ) > $W/$T.out 2>&1 &
  RP=$!; wait $RP; rc=$?; RP=""
  tag=$(grep -o 'out/cli_[0-9_]*_tokens.json' $W/$T.out | head -1 | sed 's|out/||; s|_tokens.json||')
  [ -n "$tag" ] && cp $CL/out/${tag}_runs.json $W/${T}_runs.json 2>/dev/null && cp $CL/out/${tag}_tokens.json $W/${T}_tokens.json 2>/dev/null
  say "  rc $rc; $(grep -o '"total": \[[^]]*\]' $J 2>/dev/null); $(grep -o 'first token [0-9.]* s' $W/$T.out | head -1); $(grep -o 'lifetime peak [0-9.]* GB' $W/$T.out | tail -1); $(grep -E '^ *[0-9.]+ real' $W/$T.out | awk '{print "user " $3 " s, sys " $5 " s"}'); $(grep -E 'page reclaims' $W/$T.out | awk '{print $1 " page reclaims"}'); tag $tag"
done
/opt/homebrew/opt/python@3.12/bin/python3.12 -I - $W <<'PY' | tee -a $LOG
import json, os, re, sys, collections, statistics
W = sys.argv[1]
runs = []
for f in sorted(os.listdir(W)):
    m = re.fullmatch(r"ab(\d+)_(10k|30k)_p(0c|0|2)\.out", f)
    if m: runs.append((int(m.group(1)), m.group(2), m.group(3), f[:-4]))
runs.sort()
def info(T):
    d = {}
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
    d["tok"] = json.dumps([v for k in sorted(tk) for a in sorted(tk[k]) for v in tk[k][a]]); d["lh"] = json.dumps(lh)
    out = open(f"{W}/{T}.out").read()
    g = lambda p: (lambda m: float(m.group(1)) if m else None)(re.search(p, out))
    d["ft"] = g(r"first token ([0-9.]+) s"); d["peak"] = g(r"lifetime peak ([0-9.]+) GB")
    d["user"] = g(r"(?m)^\s*[0-9.]+ real\s+([0-9.]+) user"); d["sys"] = g(r"(?m)^\s*[0-9.]+ real\s+[0-9.]+ user\s+([0-9.]+) sys")
    d["recl"] = g(r"(?m)^\s*(\d+)\s+page reclaims")
    full = [0.0, 0.0]
    try:
        for line in open(f"{W}/{T}.jsonl"):
            j = json.loads(line)
            if "total" in j: d["pf"] = j["total"]; continue
            if not j["linear"]:
                for k, (w, c) in j["t"].items():
                    if k in ("attn_submit", "hc_write"): full[0] += w; full[1] += c
    except Exception:
        pass
    d["full"] = full
    return d
ref = {}
arms = collections.defaultdict(list)
for i, n, a, T in runs:
    d = info(T)
    if d is None:
        print(f"ab run {i} {n} p{a}: no result files"); continue
    if a == "0" and n not in ref: ref[n] = d
    r = ref.get(n)
    eq = "ref" if r is d else (f"tokens {'EQUAL' if r['tok'] == d['tok'] else 'DIFFER'} | hashes {'EQUAL' if r['lh'] == d['lh'] else 'DIFFER'}" if r else "no p0 ref")
    arms[(n, a)].append(d["ft"])
    print(f"ab run {i} {n} p{a}: {eq}; first token {d['ft']} s; prefill wall/cpu {d.get('pf')}; full-attn submit wall/cpu "
          f"{d['full'][0]:.2f}/{d['full'][1]:.2f}; process user {d['user']} s, sys {d['sys']} s; page reclaims {d['recl']}; peak {d['peak']} GB")
for (n, a), v in sorted(arms.items()):
    v = [x for x in v if x is not None]
    if v: print(f"ab mean {n} p{a}: first token {statistics.mean(v):.2f} s over {len(v)} run(s) {v}")
for n in ("10k", "30k"):
    a0, a2 = [x for x in arms.get((n, "0"), []) if x], [x for x in arms.get((n, "2"), []) if x]
    if a0 and a2:
        m0, m2 = statistics.mean(a0), statistics.mean(a2)
        print(f"ab {n}: PIPE=2 vs PIPE=0 first token {m2:.2f} vs {m0:.2f} s ({(m2 - m0) / m0 * 100:+.1f} %)")
PY
