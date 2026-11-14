#!/bin/bash
# Sat 7 Nov 07:0x (the orchestrator 07:0x; NewIdeas QUEUE FIRSTCHK db21669, ~/NewIdeas/scripts/firstcheck_price.py): the kill test
# of the first-check lead. In v1' (S1b's B arm) each prompt's check 0 takes ~306 ms vs ~58 ms later (graph build 287 vs 53 ms;
# fewer misses than later checks) = 15.8 % of the decode wall; ceiling +21 % whole decode [P].
# On v1' (final_env dd7ad0f: stackb + S2 + KEEP, C 256; QOM sets humaneval), 3 HumanEval prompts per arm, CAP 64, one process
# per arm, main-thread CPU per check (R57_TCPU=1: time.thread_time around build and wait; MLX's Metal encoding counts there):
#   fcctl: R57_A6=ctl  (the control: check 0 as in v1', its host / non-host split)
#   fcrep: R57_A6=repf (copy 1 of check 0 under the real check's FAST gate, rolled back; then the real check 0 = copy 2).
#          ('rep' ran copy 1 ungated: on a FAST arm it would leave the FAST path's first use in copy 2.)
# LINE (NewIdeas QUEUE, written here before arming), on the median over the 3 prompts:
#   KILL if copy 2 (fcrep's check 0, build + wait) is also >= 200 ms AND the host CPU in the excess (copy 1's thread CPU minus
#   check 2's) is <= 20 ms: the cost is check 0's content, not a one-time cost.
#   Else the split names the piece: ONE-TIME if copy 2 < 200 ms -> HOST (thread CPU excess > 20 ms: Python setup / retrace)
#   or GPU (first use); then scope a fix with Leap (exact-safe lever for v1' and the exact mode; bitwise check before ~20:00).
# usage: NOTAFTER=hh:mm NOWAIT45=1 GAP0=0 GAP=0 MARGIN=4 bash scripts/turn45.sh 33 15 "..." /bin/bash scripts/fc_kill.sh
export PATH="$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:$PATH"
cd "${FC_ROOT:-$HOME/BigMoE}" || exit 1
mkdir -p data/fc data/r57/snap
LOG="$PWD/data/fc/fc.log"
say() { echo "$(date '+%m-%d %H:%M:%S') $*" >> "$LOG"; }
PY=${FC_PY:-$HOME/BigMoE/.venv/bin/python}
avail_gb() { vm_stat | awk '/page size of/ {ps=$8} /^Pages (free|purgeable):/ || /^File-backed pages:/ {v=$NF; sub("[.]","",v); s+=v} END {printf "%.1f", s*ps/1e9}'; }
A=${FC_AVAIL:-$(avail_gb)}
awk -v a="$A" 'BEGIN {exit !(a >= 47.2)}' || { say "NO-SWAP $A GB < 47.2: C 256 does not fit, not run"; exit 0; }
[ -n "$FC_DRYRUN" ] || sleep 20                          # look again (gated waiters start together)
A=${FC_AVAIL:-$(avail_gb)}
awk -v a="$A" 'BEGIN {exit !(a >= 47.2)}' || { say "NO-SWAP $A GB < 47.2 at the 20 s look-again: not run"; exit 0; }
CC=256
FZ="$PWD/data/r57/snap/frozen_fc"; rm -rf "$FZ"; mkdir -p "$FZ"
SD=data/r57/snap/fc; mkdir -p "$SD"
cp scripts/r57_screen_fast.py "$SD/r57_screen_fc.py" && cp scripts/relax_delta.py scripts/relax_span.py scripts/inert_span.py \
  "$SD/" && cp scripts/c45_runner.sh data/r57/snap/c45_runner_fc.sh || exit 2
[ -n "$FC_DRYRUN" ] || { cp -R "$HOME/Leap/scripts" "$FZ/Leap_scripts" && cp -R "$HOME/Leap/third_party" "$FZ/Leap_third_party" \
  && cp -R "$HOME/TrainLab/sticky" "$FZ/TrainLab_sticky" && cp -R "$HOME/Breakthrough/keepset" "$FZ/Breakthrough_keepset" || exit 2; }
find "$FZ" -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null
sed -i '' -e "s#\"~/Leap/scripts\"#\"$FZ/Leap_scripts\"#g" -e "s#\"~/Leap/third_party\"#\"$FZ/Leap_third_party\"#g" \
  -e "s#\"~/TrainLab/sticky\"#\"$FZ/TrainLab_sticky\"#g" -e "s#\"/Users/kamen/Breakthrough/keepset\"#\"$FZ/Breakthrough_keepset\"#g" \
  "$SD/r57_screen_fc.py"
RUN="/bin/bash data/r57/snap/c45_runner_fc.sh"
export C45_OUT="$PWD/data/fc" C45_SWAP0=$(sysctl -n vm.swapusage | awk '{gsub("M","",$6); print $6 / 1024}') C45_ALARM=900 \
  MG=4 C45_DEADLINE=$(date -v+30M +%H%M) C45_SCREEN="$SD/r57_screen_fc.py" C45_REFTOK= C45_CAP=64
K="humaneval/HumanEval/68,humaneval/HumanEval/16,humaneval/HumanEval/91"
V1P=$(grep -v '^#' data/stack/final_env.txt | grep . | tail -1 | sed 's/,R57_QOM_SETS=humaneval,mbpp$/,R57_QOM_SETS=humaneval/')
echo "$V1P" | grep -q 'NEVERWAIT' && { say "final_env.txt carries NW: not v1', not run"; exit 2; }
echo "$V1P" | tr ',' '\n' | grep -vq '=' && { say "v1' env still has a comma inside a value: not run"; exit 2; }
[ -n "$FC_DRYRUN" ] && { echo "dry run: $(grep -c frozen_fc "$SD/r57_screen_fc.py") screen paths; v1' $(echo "$V1P" | tr ',' '\n' | grep -c =) K=V"; exit 0; }
for t in fcctl fcrep; do rm -f data/r57/${t}_${CC}_* data/fc/${t}_${CC}.txt; done
rm -f data/fc/DONE
say "FIRSTCHK kill test start (pid $$, NO-SWAP $A GB, C $CC); screen md5 $(md5 -q scripts/r57_screen_fast.py)"
for a in "fcctl:$V1P,R57_A6=ctl,R57_TCPU=1" "fcrep:$V1P,R57_A6=repf,R57_TCPU=1"; do
  R57_KEYS=$K C45_ARMS="${a%%:*}_$CC:$CC:${a#*:}" $RUN
  say "${a%%:*} rc=$?"
done
"$PY" -I - "$CC" >> "$LOG" 2>&1 <<'EOF'
import json, statistics as S, sys
C = sys.argv[1]
med = lambda xs: S.median(xs) if xs else float("nan")
L = [f"FIRSTCHK kill test, C {C} (data/fc; v1' = final_env dd7ad0f; 3 HumanEval, CAP 64): wall ms (build + wait) and "
     f"main-thread CPU ms (build + wait) per check; medians over the prompts"]
D = {}
for t in ("fcctl", "fcrep"):
    try:
        R = {r["key"]: r for r in json.load(open(f"data/r57/{t}_{C}_runs.json"))}
        Cs = {c["key"]: c for c in json.load(open(f"data/r57/{t}_{C}_checks.json"))}
    except (OSError, ValueError) as e:
        L.append(f"{t}: MISSING ({e})"); continue
    rows = {}
    for k, c in Cs.items():
        tc = c.get("tc") or []
        for i in range(min(4, len(c["bld"]))):
            rows.setdefault(f"check {i}", []).append((c["bld"][i] + c["wt"][i], sum(tc[i]) if i < len(tc) else float("nan"),
                                                      c["bld"][i], c["wt"][i], c["miss"][i]))
        a6 = (R.get(k) or {}).get("a6") or {}
        if "rep1" in a6 and len(a6["rep1"]) >= 7:
            r1 = a6["rep1"]
            rows.setdefault("copy 1", []).append((r1[0] + r1[1], r1[5] + r1[6], r1[0], r1[1], r1[2]))
    D[t] = rows
    for name in ("copy 1", "check 0", "check 1", "check 2", "check 3"):
        if name in rows:
            v = rows[name]
            L.append(f"  {t} {name:8s}: wall {med([x[0] for x in v]):6.1f} ms (build {med([x[2] for x in v]):6.1f}, wait "
                     f"{med([x[3] for x in v]):5.1f}); thread CPU {med([x[1] for x in v]):6.1f} ms; misses {med([x[4] for x in v]):4.0f}"
                     f"; per prompt wall {[round(x[0]) for x in v]}, CPU {[round(x[1]) for x in v]}")
r = D.get("fcrep", {})
if "copy 1" in r and "check 0" in r and "check 2" in r:
    c1w, c2w = med([x[0] for x in r["copy 1"]]), med([x[0] for x in r["check 0"]])
    k2w, k2c = med([x[0] for x in r["check 2"]]), med([x[1] for x in r["check 2"]])
    c1c = med([x[1] for x in r["copy 1"]])
    host = c1c - k2c
    if c2w >= 200 and host <= 20:
        v = "KILL (copy 2 still slow, host CPU not the excess: the cost is check 0's content)"
    elif c2w < 200:
        v = (f"ONE-TIME, {'HOST' if host > 20 else 'GPU'} (copy 2 {c2w:.0f} ms < 200: the excess is paid once per prompt; "
             f"host CPU in it {host:.0f} ms of {c1w - k2w:.0f}) -> scope a fix with Leap")
    else:
        v = f"CONTENT + HOST (copy 2 {c2w:.0f} ms >= 200 but host CPU in the excess {host:.0f} ms > 20)"
    L.append(f"Verdict FIRSTCHK (pre-registered, NewIdeas QUEUE db21669): {v}. copy 1 {c1w:.0f} ms, copy 2 {c2w:.0f} ms, check 2 "
             f"{k2w:.0f} ms; thread CPU copy 1 {c1c:.0f} vs check 2 {k2c:.0f} ms")
else:
    L.append("Verdict FIRSTCHK: MISSING (no copy 1 / check 0 / check 2 in fcrep)")
c = D.get("fcctl", {})
if "check 0" in c and "check 2" in c:
    L.append(f"Control (fcctl): check 0 wall {med([x[0] for x in c['check 0']]):.0f} ms, thread CPU {med([x[1] for x in c['check 0']]):.0f}"
             f" ms; check 2 wall {med([x[0] for x in c['check 2']]):.0f}, CPU {med([x[1] for x in c['check 2']]):.0f} ms")
open("data/fc/fc_report.txt", "w").write("\n".join(L) + "\n"); print("\n".join(L))
EOF
say "report rc=$? (data/fc/fc_report.txt)"
touch data/fc/DONE
say "FIRSTCHK kill test end"
