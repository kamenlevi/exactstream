#!/bin/bash
# prof_wired.sh (Leap 11-13 01:3x): does R57_WIRED=1 (mx.set_wired_limit(max working set), on in every shipped config) cause
# the 30K full-attention layers' ~12.4 s of main-thread KERNEL time? (prof_sys, data/prefill15_1112.md section 9.) The
# standalone SDPA bench: one layer's 29-chunk stock sweep spends 0.40 s in the kernel with a wired limit set and 0.00 s
# without, at the same wall time; wiring each fresh O(T) plane is the suspect. R57_WIRED=1 was Leap's first-token fix (FCP
# 038bfa1: IOGPU stopped unwiring the expert pools in each prefill), so turning it off can cost elsewhere: this measures both.
# PF-B (R57_PFLM=3) at 32 GB through prof5's copy with R57_PFLM_PROF_CPU=sys. Runs: 30K PIPE=0 W0 W1 W1 W0 (order-balanced),
# then 10K PIPE=0 W0 and 30K PIPE=2 W0 (W0 = run.py --env R57_WIRED=0). Tokens + R57_LOGHASH vs prof_ab's PIPE=0 runs.
# Gates as prof_sys.sh (start NO-SWAP >= 45, per run >= 44, two clear polls, a share lock re-checked 20 s after the grant);
# swap +2 GB stops. Usage: prof_wired.sh <work dir with prompt_10k.txt, prompt_30k.txt and prof_ab's ab1 / ab6 files> "<LAST ...>"
W=$1; P5=/Users/kamen/Leap/scripts/prefill15/prof5; CL=/Users/kamen/Leap/release/cleanclone/cache/clone; LOG=$W/wired.log
SO=/private/tmp/claude-501/-Users-kamen-OjoX/f2971937-6c95-4e86-a049-05beff6a8e32/scratchpad/agent/ext_ov_build/bigmoe_ext_ov.cpython-312-darwin.so
LAST=$(date -j -f "%Y-%m-%d %H:%M:%S" "$2" +%s)
noswap() { vm_stat | awk '/page size of/ {ps=$8} /^Pages (free|purgeable):/ || /^File-backed pages:/ {v=$NF; sub("[.]","",v); s+=v} END {printf "%d", s*ps/1e9}'; }
swap() { sysctl -n vm.swapusage | sed -E 's/.*used = ([0-9.]+)M.*/\1/' | cut -d. -f1; }
idle() { ioreg -c IOHIDSystem | awk '/HIDIdleTime/ {print int($NF/1000000000); exit}'; }
procs() { ps -axo pid=,args= | awk '/[Pp]ython/ && /r57_screen_|[ \/](chat|serve|run)[.]py( |$)/ && !/ awk / {printf "%s ", $1}'; }
busy() { peer ls 2>/dev/null | grep "^- " | grep -oE "· (shared run \(others may work\)|benchmarking|waiting for a quiet window)[^·]*" | grep -cE "^· (benchmarking|waiting for a quiet window)|MEM ([6-9]|[1-9][0-9]) GB"; }
held() { [ -e ~/.local/share/quiet/PAUSE ] || [ -e ~/.local/share/quiet/HOLD ]; }
say() { echo "$(date '+%m-%d %H:%M:%S') $*" | tee -a $LOG; }
RP=""; GUARD=""; LOCKED=0
cleanup() { [ -n "$GUARD" ] && kill $GUARD 2>/dev/null; [ -n "$RP" ] && kill -0 $RP 2>/dev/null && kill -TERM $RP
            [ $LOCKED = 1 ] && peer unlock >/dev/null 2>&1; touch $W/wired.done; say "exit (NO-SWAP $(noswap) GB, swap $(swap) MB)"; }
trap cleanup EXIT; trap 'say "stopped by a signal"; exit 1' INT TERM
for n in 10k 30k; do [ -f $W/prompt_$n.txt ] || { say "no $W/prompt_$n.txt"; exit 2; }; done
[ -f $P5/pflm_prof.py ] && [ -f $P5/sitecustomize.py ] && [ -f $SO ] || { say "no prof5 copy or ext_ov"; exit 2; }
say "armed (pid $$): R57_WIRED 0 vs 1, PF-B at 32 GB, kernel time per stage: 30K W0 W1 W1 W0, 10K W0, 30K PIPE=2 W0; last start $2"
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
    if peer lock share 30m --cpu 400 --gpu 100 "MEM 46 GB: Leap prefill R57_WIRED A/B (32 GB, 10K / 30K, 6 runs), ~10 min" > /dev/null 2>&1; then
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
for job in "30k 0 0" "30k 1 0" "30k 1 0" "30k 0 0" "10k 0 0" "30k 0 2"; do
  set -- $job; N=$1; WI=$2; PI=$3; i=$((i + 1))
  [ -n "$(procs)" ] && { say "SKIP run $i: an engine runs"; continue; }
  [ "$(noswap)" -lt 44 ] && { say "SKIP run $i: NO-SWAP $(noswap) < 44"; continue; }
  T=wired${i}_${N}_w${WI}_p$PI; J=$W/$T.jsonl; rm -f $J
  say "run $i: $N R57_WIRED=$WI PIPE=$PI (NO-SWAP $(noswap) GB)"
  ( cd $CL && exec env PYTHONPATH=$P5 R57_PFLM_PROF=$J R57_PFLM_PROF_SYNC=0 R57_PFLM_PROF_CPU=sys R57_PFLM_PIPE=$PI R57_PFLM_EXTOV=$SO \
      .venv/bin/python run.py --config chat --mem 32 --prompt-file $W/prompt_$N.txt --max-tokens 32 --env R57_PFLM=3 \
      --env R57_LOGHASH=1 --env R57_WIRED=$WI ) > $W/$T.out 2>&1 &
  RP=$!; wait $RP; rc=$?; RP=""
  tag=$(grep -o 'out/cli_[0-9_]*_tokens.json' $W/$T.out | head -1 | sed 's|out/||; s|_tokens.json||')
  [ -n "$tag" ] && cp $CL/out/${tag}_runs.json $W/${T}_runs.json 2>/dev/null && cp $CL/out/${tag}_tokens.json $W/${T}_tokens.json 2>/dev/null
  wl=$(grep -o 'R57_WIRED: mx.set_wired_limit([0-9.]* GB)' $CL/engine/data/r57/${tag}_screen.txt 2>/dev/null | head -1)
  say "  rc $rc; $(grep -o '"total": \[[^]]*\]' $J 2>/dev/null); $(grep -o 'first token [0-9.]* s' $W/$T.out | head -1); $(grep -o 'whole decode [0-9.]* tok/s' $W/$T.out | head -1); $(grep -o 'lifetime peak [0-9.]* GB' $W/$T.out | tail -1); ${wl:-wired limit not set}; tag $tag"
done
/opt/homebrew/opt/python@3.12/bin/python3.12 -I - $W <<'EOF' | tee -a $LOG
import json, os, re, sys, statistics
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
rows = []
for f in sorted(os.listdir(W)):
    m = re.fullmatch(r"wired(\d+)_(10k|30k)_w([01])_p([02])\.jsonl", f)
    if not m:
        continue
    T = f[:-6]; i, n, wi, pi = int(m.group(1)), m.group(2), m.group(3), m.group(4)
    tot = None; full = [0.0, 0.0]
    for line in open(f"{W}/{f}"):
        d = json.loads(line)
        if "total" in d:
            tot = d["total"]; continue
        if not d["linear"]:
            for k, (w, c) in d["t"].items():
                if k in ("attn_submit", "hc_write"):
                    full[0] += w; full[1] += c
    out = open(f"{W}/{T}.out").read() if os.path.exists(f"{W}/{T}.out") else ""
    g = lambda p: (lambda mm: float(mm.group(1)) if mm else None)(re.search(p, out))
    ref = "ab1_10k_p0" if n == "10k" else "ab6_30k_p0"
    a, b = toks(ref), toks(T)
    eq = "no ref" if a is None or b is None else f"tokens {'EQUAL' if a[0] == b[0] else 'DIFFER'} | hashes {'EQUAL' if a[1] == b[1] else 'DIFFER'}"
    rows.append((i, n, wi, pi, g(r"first token ([0-9.]+) s"), tot, full, g(r"whole decode ([0-9.]+) tok/s"), g(r"lifetime peak ([0-9.]+) GB"), eq))
rows.sort()
for i, n, wi, pi, ft, tot, full, dec, pk, eq in rows:
    print(f"wired run {i} {n} R57_WIRED={wi} PIPE={pi}: {eq}; first token {ft} s; prefill wall {tot[0] if tot else None} s, "
          f"main-thread kernel {tot[1] if tot else None} s; full-attn wall {full[0]:.2f} s, kernel {full[1]:.2f} s; decode {dec} tok/s; peak {pk} GB")
for wi in ("0", "1"):
    v = [r[4] for r in rows if r[1] == "30k" and r[2] == wi and r[3] == "0" and r[4] is not None]
    if v:
        print(f"wired mean 30k PIPE=0 R57_WIRED={wi}: first token {statistics.mean(v):.2f} s over {len(v)} {v}")
EOF
