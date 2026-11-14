#!/bin/bash
# S45-T runner (NewIdeas 18:2x, 45 GB first): the live exact loop at C 224's RAM, REF2b on the gate (A) vs REF2-T tiers on
# the gate (B: 4433333333 / 3c, C_l 112, C_m from the byte budget, TIER_COMBINE off), A-B-A processes, S45's 6 prompts,
# warm-up discarded; tokens compared across runs of an arm. usage: HAVE_LOCK=1 nohup bash scripts/s45t_runner.sh > data/r57/s45t_runner.out 2>&1 &
export PATH="$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:$PATH"
set -m   # each run is its own process group: time-up or a lost lock stops the whole tree (RULES, orchestrator 20:5x)
cd "$HOME/BigMoE" || exit 1
OUT="$HOME/BigMoE/data/r57"; mkdir -p "$OUT"
LOG="$OUT/s45t.log"
PY="$HOME/BigMoE/.venv/bin/python"
REF="BIGMOE_DUPROW=1 BIGMOE_COMPILE_HC=1 BIGMOE_HC_VREAD=1 BIGMOE_ASYNC_EVERY=1 BIGMOE_PF_JIT=1 BIGMOE_PF_QD=4 BIGMOE_PREFETCH=12"
LOCKF="$HOME/.claude-peers/locks/${CLAUDE_PID}.json"
keepalive() { "$PY" -I -S -c '
import json, os, sys, time
p = sys.argv[1]
try: d = json.load(open(p))
except Exception: sys.exit(1)
if d.get("mode") != "quiet" or d.get("until", 0) <= time.time(): sys.exit(1)
d["active_at"] = time.time(); t = "%s.tmp%d" % (p, os.getpid()); json.dump(d, open(t, "w")); os.replace(t, p)' "$LOCKF"; }
say() { echo "$(date +%H:%M:%S) $*" >> "$LOG"; }
source "$HOME/BigMoE/scripts/ssd_gate.sh"
avail_gb() { vm_stat | awk '/page size of/ {ps=$8} /^Pages (free|purgeable):/ || /^File-backed pages:/ {v=$NF; sub("[.]","",v); s+=v} END {printf "%d", s*ps/1e9}'; }   # NO-SWAP (the orchestrator 23:5x, RULES): free + purgeable + file-backed (speculative sits inside file-backed: the orchestrator 11-14 05:5x)
run() {  # run NAME TIMEOUT OUTFILE CMD...  (memory gate at most 30 s: the lock is only 15 min)
  local name=$1 to=$2 of=$3; shift 3
  keepalive || { say "lost the lock before $name: stopping the batch"; exit 1; }
  while [ -e "$HOME/.local/share/quiet/PAUSE" ]; do sleep 5; keepalive || { say "lock ended during a PAUSE"; exit 1; }; done
  ssd_wait 66
  local i=0
  # a C 224 load peaks ~41 GB: NO-SWAP >= 45 GB (peak + 4 under a quiet lock, the orchestrator 23:5x), else skip
  while { [ "$(avail_gb)" -lt 45 ] || [ "$(ps -axo rss=,pid= | awk -v me=$$ '$1 > 8000000 && $2 != me' | wc -l)" -gt 0 ]; } && [ $i -lt 24 ]; do sleep 5; i=$((i+1)); done
  [ $i -ge 24 ] && { say "memory busy at the start of $name: skipped"; return 9; }
  say "run $name"
  perl -e 'alarm shift; exec @ARGV' "$to" "$@" >> "$OUT/$of" 2>&1 &
  local jp=$!
  while kill -0 $jp 2>/dev/null; do
    for k in $(seq 36); do kill -0 $jp 2>/dev/null || break; sleep 5; done
    kill -0 $jp 2>/dev/null || break
    keepalive || { say "lost the lock during $name: stopping"; kill -- -$jp; sleep 5; kill -9 -- -$jp 2>/dev/null; exit 1; }
  done
  wait $jp; local rc=$?; kill -9 -- -$jp 2>/dev/null; say "done $name rc=$rc"   # the job's whole group (RULES)
  ssd_check || say "SSD hot after $name"
  return $rc
}
say "s45t start (CLAUDE_PID=$CLAUDE_PID)"
if [ -z "$HAVE_LOCK" ]; then
  PEER_QUEUE_TIMEOUT=21600 peer lock quiet --queue 15m "BigMoE S45-T: live exact loop at C 224 RAM, REF2b gate vs REF2-T tiers on the gate, A-B-A, ~10 min" >> "$LOG" 2>&1 || { say "no lock"; exit 1; }
fi
swap_gb() { sysctl -n vm.swapusage | awk '{gsub("M","",$6); print $6 / 1024}'; }
S0=$(swap_gb); say "swap at start: $S0 GB"
( while sleep 5; do
    S=$(swap_gb)
    if awk -v s="$S" -v s0="$S0" 'BEGIN { exit !(s - s0 > 2) }'; then
      say "SWAP GUARD: swap $S GB (start $S0): stopping the screen"
      P=$(pgrep -P $$ -f "scripts/r57_screen.py" | head -1)     # the runner's own job, by PID (RULES: no pkill)
      [ -n "$P" ] && kill -TERM -- -"$P"; break
    fi
    kill -0 $$ 2>/dev/null || break
  done ) &
GUARD=$!
G="BIGMOE_MAXK=160 BIGMOE_EXT_BUILD=build_next R57_ARMS=gate R57_PROMPTS=6 R57_WARM=1"
run A1 330 s45t_A1_run.txt env $G R57_REPS=1 R57_TAG=s45t_A1 $PY scripts/r57_screen.py 224 256 || say "A1 failed"
run B 480 s45t_B_run.txt env $G R57_ARMS=gate,gate+t R57_TIER=1 R57_REPS=1 R57_TAG=s45t_B $PY scripts/r57_screen.py 224 256 || say "B failed"
run A2 330 s45t_A2_run.txt env $G R57_REPS=1 R57_TAG=s45t_A2 $PY scripts/r57_screen.py 224 256 || say "A2 failed"
kill $GUARD 2>/dev/null
[ -n "$KEEP_LOCK" ] || peer unlock >> "$LOG" 2>&1
say "lock released"
OMP_NUM_THREADS=2 taskpolicy -b $PY - "$OUT" > "$OUT/s45t_screen.txt" 2>&1 <<'PYEOF2'
import json, os, re, statistics as S, sys
D = sys.argv[1]
def runs(tag):
    out = {}
    try:
        for l in open(os.path.join(D, f"{tag}_run.txt")):
            m = re.match(r"\s+(\S+) \[(gate|gate\+t)\]: (\d+) tokens .*?, ([\d.]+) tok/s \(steady state, checks >= 2: ([\d.]+)\), median cycle ([\d.]+) ms, check \+ commit ([\d.]+) ms; per check \(medians\) (\d+) misses, demand SSD ([\d.]+) ms", l)
            if m and "PAUSED" not in l and "warm-up" not in l:
                out.setdefault((m.group(1), m.group(2)), []).append(tuple(float(x) for x in m.groups()[3:]))
    except FileNotFoundError:
        pass
    return out
def toks(tag):
    try:
        return json.load(open(os.path.join(D, f"{tag}_tokens.json")))
    except Exception:
        return {}
A1, B, A2 = runs("s45t_A1"), runs("s45t_B"), runs("s45t_A2")
TR = 0.3
TA1, TB, TA2 = toks("s45t_A1"), toks("s45t_B"), toks("s45t_A2")
print("S45-T: live exact DFlash2 loop at C 224's RAM, 256 tokens; A = REF2b on the gate (C 224), B = REF2-T 4433333333 / 3c on the gate")
print("(C_l 112, C_m from the byte budget, TIER_COMBINE off); A-B-A processes, warm-up discarded; columns: tok/s, steady tok/s, cycle, check+commit, misses, SSD ms")
ra, rb, rcy = [], [], []
rt = []
for key in sorted({k for k, _ in set(A1) | set(B) | set(A2)}):
    a = A1.get((key, "gate"), []) + A2.get((key, "gate"), []); b = B.get((key, "gate"), []); bt = B.get((key, "gate+t"), [])
    if not a or not b:
        print(f"  {key}: missing runs (A {len(a)}, B {len(b)})"); continue
    f = lambda rs, i: S.mean(r[i] for r in rs)
    ra.append(f(a, 1)); rb.append(f(b, 1)); rcy.append(f(b, 2) / f(a, 2))
    eqa = len({tuple(t) for t in TA1.get(key, {}).get("gate", []) + TA2.get(key, {}).get("gate", [])}) == 1
    eqb = len({tuple(t) for arm in ("gate", "gate+t") for t in TB.get(key, {}).get(arm, [])}) == 1
    if bt:
        rt.append(f(bt, 1))
    print(f"  {key}: A steady {f(a, 1):.1f} tok/s (whole {f(a, 0):.1f}), cycle {f(a, 2):.0f} ms, misses {f(a, 4):.0f}, SSD {f(a, 5):.0f} ms | "
          f"B steady {f(b, 1):.1f} (whole {f(b, 0):.1f}), cycle {f(b, 2):.0f} ms, misses {f(b, 4):.0f}, SSD {f(b, 5):.0f} ms | "
          f"B/A steady {f(b, 1) / f(a, 1) - 1:+.1%}" + (f" | B + t {TR} steady {f(bt, 1):.1f} (whole {f(bt, 0):.1f}), cycle {f(bt, 2):.0f} ms, "
          f"{f(bt, 1) / f(a, 1) - 1:+.1%} vs A" if bt else "") + f"; tokens equal across A runs {eqa}, across B runs (both arms) {eqb}")
if ra:
    print(f"S45-T: steady-state tok/s (prompt means) A {S.mean(ra):.1f} -> B {S.mean(rb):.1f} ({S.mean(rb) / S.mean(ra) - 1:+.1%}); median "
          f"per-prompt cycle ratio {S.median(rcy) - 1:+.1%}" + (f"; B + t {TR} {S.mean(rt):.1f} ({S.mean(rt) / S.mean(ra) - 1:+.1%} vs A)" if len(rt) == len(ra) else ""))
PYEOF2
say "ab rc=$?"
say "s45t end"
