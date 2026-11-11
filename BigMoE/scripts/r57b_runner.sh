#!/bin/bash
# R-57b runner (NewIdeas 01:2x GO; Fit12's R-57b-R replay): reserved prefetch slots + the corrected L+1 forecast in the
# live exact REF2b gate loop at equal RAM C 176: A = no reserve (C_main 176) vs B = C_main 160 + 16 reserved (BIGMOE_SPARE 16,
# PF_VICTIM 0, JIT, QD 4, ranked 32 with Fit12's held-out maps); A-B-A processes, S45's 6 prompts. usage: via turn45.sh
export PATH="$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:$PATH"
set -m   # each run is its own process group: time-up or a lost lock stops the whole tree (RULES, orchestrator 20:5x)
cd "$HOME/BigMoE" || exit 1
OUT="$HOME/BigMoE/data/r57"; mkdir -p "$OUT"
LOG="$OUT/r57b.log"
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
avail_gb() { vm_stat | awk '/page size of/ {ps=$8} /^Pages (free|speculative|purgeable):/ || /^File-backed pages:/ {v=$NF; sub("[.]","",v); s+=v} END {printf "%d", s*ps/1e9}'; }   # NO-SWAP (the orchestrator 23:5x, RULES): free + speculative + purgeable + file-backed
run() {  # run NAME TIMEOUT OUTFILE CMD...  (memory gate at most 30 s: the lock is only 15 min)
  local name=$1 to=$2 of=$3; shift 3
  keepalive || { say "lost the lock before $name: stopping the batch"; exit 1; }
  while [ -e "$HOME/.local/share/quiet/PAUSE" ]; do sleep 5; keepalive || { say "lock ended during a PAUSE"; exit 1; }; done
  ssd_wait 66
  local i=0
  # a C 176 load peaks ~35 GB: NO-SWAP >= 39 GB (peak + 4 under a quiet lock), else skip
  while { [ "$(avail_gb)" -lt 39 ] || [ "$(ps -axo rss=,pid= | awk -v me=$$ '$1 > 8000000 && $2 != me' | wc -l)" -gt 0 ]; } && [ $i -lt 24 ]; do sleep 5; i=$((i+1)); done
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
say "r57b start (CLAUDE_PID=$CLAUDE_PID)"
if [ -z "$HAVE_LOCK" ]; then
  PEER_QUEUE_TIMEOUT=21600 peer lock quiet --queue 15m "BigMoE R-57b: reserved prefetch slots + corrected forecast, live exact loop at C 176, A-B-A" >> "$LOG" 2>&1 || { say "no lock"; exit 1; }
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
G="BIGMOE_MAXK=160 BIGMOE_EXT_BUILD=build_next R57_PROMPTS=6 R57_WARM=1 R57_REPS=1"
PF="R57_SPARE=16 BIGMOE_PF_VICTIM=0 BIGMOE_PF_JIT=1 BIGMOE_PF_QD=4 R57_CORR=$HOME/Fit12/data/r57b"
run A1 330 r57b_A1_run.txt env $G R57_ARMS=gate R57_TAG=r57b_A1 $PY scripts/r57_screen.py 176 256 || say "A1 failed"
run B 360 r57b_B_run.txt env $G $PF R57_ARMS=gate+pf R57_TAG=r57b_B $PY scripts/r57_screen.py 176 256 32 || say "B failed"
run A2 330 r57b_A2_run.txt env $G R57_ARMS=gate R57_TAG=r57b_A2 $PY scripts/r57_screen.py 176 256 || say "A2 failed"
kill $GUARD 2>/dev/null
[ -n "$KEEP_LOCK" ] || peer unlock >> "$LOG" 2>&1
say "lock released"
OMP_NUM_THREADS=2 taskpolicy -b $PY - "$OUT" > "$OUT/r57b_screen.txt" 2>&1 <<'PYEOF2'
import json, os, re, statistics as S, sys
D = sys.argv[1]
def runs(tag):
    out = {}
    try:
        for l in open(os.path.join(D, f"{tag}_run.txt")):
            m = re.match(r"\s+(\S+) \[(gate|gate\+pf)\]: (\d+) tokens .*?, ([\d.]+) tok/s \(steady state, checks >= 2: ([\d.]+)\), median cycle ([\d.]+) ms, check \+ commit ([\d.]+) ms; per check \(medians\) (\d+) misses, demand SSD ([\d.]+) ms", l)
            if m and "PAUSED" not in l and "INVALID" not in l:
                out.setdefault(m.group(1), []).append(tuple(float(x) for x in m.groups()[3:]))
    except FileNotFoundError:
        pass
    return out
def toks(tag):
    try:
        return json.load(open(os.path.join(D, f"{tag}_tokens.json")))
    except Exception:
        return {}
A1, B, A2 = runs("r57b_A1"), runs("r57b_B"), runs("r57b_A2")
T = [toks(t) for t in ("r57b_A1", "r57b_B", "r57b_A2")]
print("R-57b: live exact REF2b gate loop at C 176's RAM, S45's six prompts, 256 tokens; A = no reserve (C_main 176), B = C_main 160 +")
print("16 reserved slots (PF_VICTIM 0), JIT, QD 4, ranked 32 per layer with Fit12's held-out corrected L+1 forecast; A-B-A processes")
dio, dcy, dck, dtw, eq_all = [], [], [], [], True
for key in sorted(set(A1) | set(B) | set(A2)):
    a = A1.get(key, []) + A2.get(key, []); b = B.get(key, [])
    if not a or not b:
        print(f"  {key}: missing runs (A {len(a)}, B {len(b)})"); continue
    f = lambda rs, i: S.mean(r[i] for r in rs)
    eq = len({tuple(t) for d in T for arm in d.get(key, {}).values() for t in arm}) == 1
    eq_all &= eq
    dio.append(f(b, 5) / f(a, 5) - 1); dcy.append(f(b, 2) / f(a, 2) - 1); dck.append(f(b, 3) / f(a, 3) - 1); dtw.append(f(b, 1) / f(a, 1) - 1)
    print(f"  {key}: A steady {f(a, 1):.1f} tok/s, cycle {f(a, 2):.0f} ms, misses {f(a, 4):.0f}, demand SSD {f(a, 5):.0f} ms | B steady "
          f"{f(b, 1):.1f}, cycle {f(b, 2):.0f} ms, misses {f(b, 4):.0f}, demand SSD {f(b, 5):.0f} ms | SSD {dio[-1]:+.1%}, cycle {dcy[-1]:+.1%}, "
          f"steady tok/s {dtw[-1]:+.1%}; tokens equal across A1 / B / A2: {eq}")
if dio:
    mi, mc = S.median(dio), S.median(dcy)
    v = "FAIL (tokens differ)" if not eq_all else "PASS" if mi <= -0.25 and mc <= -0.10 else "KILL" if mc > -0.04 else "BETWEEN"
    print(f"R-57b verdict (PASS: demand SSD per check -25 % AND cycle -10 %, medians over prompts; KILL: cycle under -4 %): {v} "
          f"(SSD {mi:+.1%}, cycle {mc:+.1%}, check + commit {S.median(dck):+.1%}, steady tok/s {S.median(dtw):+.1%})")
PYEOF2
say "ab rc=$?"
say "r57b end"
