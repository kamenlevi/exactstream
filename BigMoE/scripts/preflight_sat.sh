#!/bin/bash
> (a quote from the project owner, kept private)
# scripts/r57_screen_fast.py (the relaxed copy + GOAL's INERT + MoonShots' relax v2 + FULL's NOCOM / COPYPROBE + Fable B's
# FAST-tax; FAST-NAX via Leap's ref2 eb43fcc2), for tonight's stacked FAST headline. Two prompts (HumanEval/68 + ifeval/1220)
# at CAP 32, C 160 uniform, under a share lock "MEM 36 GB", once no quiet lock and no other > 5 GB job runs, start by 14:15.
# Arms: pq qall | pi + R57_INERT=CD16 | pp + R57_NOCOM=1 | pt + R57_FASTTAX=1 | pn + R57_NAX=1,REF2_NAX_1CQ=1 |
#       pr + R57_RELAX_DELTA=0.5 (v2) | ps + R57_RELAX_SPAN=1 (v2) | prs + both | pk the stack: FASTTAX + NAX + DELTA + SPAN |
#       pki the stack with INERT in place of the relaxed rules (INERT does not run with them)
# Checks per arm: it ran both prompts, a clean log (no Traceback / AssertionError / Error:), rlx / spn where set.
export PATH="$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:$PATH"
cd "${PS_ROOT:-$HOME/BigMoE}" || exit 1
mkdir -p data/pf2 data/r57/snap
LOG="$PWD/data/pf2/pf2.log"
say() { echo "$(date '+%m-%d %H:%M:%S') $*" >> "$LOG"; }
LAST=${PS_LAST:-1415}; LOCKED=
big_other() { [ -n "$PS_NOBIG" ] && return 1; [ "$(ps -axo rss= | awk '$1 > 5000000' | wc -l | tr -d ' ')" -gt 0 ] \
  || peer ls 2>/dev/null | grep -qE 'benchmarking: quiet lock|waiting for a quiet window' \
  || peer ls 2>/dev/null | grep -E 'shared run' | grep -qE 'MEM ([5-9]|[1-9][0-9]+) GB'; }
say "pre-flight (sat) armed: waiting for no quiet lock and no other > 5 GB job, start by $LAST"
nc=0                                                     # Fri 07:5x: clear for 3 polls in a row (90 s) and NO-SWAP >= 40 GB:
while [ $nc -lt 3 ]; do                                  # the 07:55 try raced EvalSuite's E8 share run and hit the swap rule
  if big_other || [ "$(vm_stat | awk '/page size of/ {ps=$8} /^Pages (free|purgeable):/ || /^File-backed pages:/ {v=$NF; sub("[.]","",v); s+=v} END {printf "%d", s*ps/1e9}')" -lt "${PS_MINFREE:-40}" ]; then nc=0; else nc=$((nc + 1)); fi
  [ "$(date +%H%M)" -ge "$LAST" ] && { say "SKIPPED: not clear by $LAST"; exit 0; }
  [ $nc -lt 3 ] && sleep 30
done
[ -n "$PS_NOLOCK" ] || peer lock share 25m --cpu 300 --gpu 100 "BigMoE pre-flight (stacked FAST levers): 2 prompts per arm, CAP 32, C 160, MEM 36 GB" >> "$LOG" 2>&1 \
  && LOCKED=$([ -n "$PS_NOLOCK" ] || echo 1) || { say "share lock refused: skipped"; exit 0; }
trap '[ -n "$LOCKED" ] && peer unlock >> "$LOG" 2>&1' EXIT
cp scripts/r57_screen_fast.py data/r57/snap/r57_screen_pf2.py && cp scripts/relax_delta.py scripts/relax_span.py scripts/inert_span.py \
  data/r57/snap/ && cp scripts/c45_runner.sh data/r57/snap/c45_runner_pf2.sh || exit 2
RUN="/bin/bash data/r57/snap/c45_runner_pf2.sh"
export C45_OUT="$PWD/data/pf2" C45_SWAP0=$(sysctl -n vm.swapusage | awk '{gsub("M","",$6); print $6 / 1024}') C45_ALARM=600 MG=4 \
  C45_CAP=32 KEEP_LOCK=1 R57_KEYS=humaneval/HumanEval/68,ifeval/1220 C45_DEADLINE=$(date -v+30M +%H%M) \
  C45_SCREEN=data/r57/snap/r57_screen_pf2.py C45_REFTOK=
QA="R57_LAYSZ=0,BIGMOE_TIER_CL=64,BIGMOE_EXT_BUILD=build_qom,R57_REFTOK=,R57_FAST=QOM,R57_D17=1,R57_D7B=code,R57_HC8=1"
QA="$QA,REF2_HC8_FAST=1,R57_DNR=1,R57_HCB=1,REF2_SMALL_NATIVE=1,R57_D7BNAX=1,REF2_OOO_1C=1,R57_VTIER="   # no volatile tier: a
rm -f data/r57/f2*_160_*                                 # code-path check; f2 tags, so no stale same-named file is read (07:5x)
NX="R57_NAX=1,REF2_NAX_1CQ=1"; RL="R57_RELAX_DELTA=0.5,R57_RELAX_LOG=$PWD/data/pf2/relax.jsonl"
SP="R57_RELAX_SPAN=1,R57_RELAX_LOG=$PWD/data/pf2/span.jsonl"
say "pre-flight (sat) start (pid $$)"
for a in "f2q:$QA" "f2i:$QA,R57_INERT=CD16" "f2p:$QA,R57_NOCOM=1" "f2t:$QA,R57_FASTTAX=1" "f2n:$QA,$NX" "f2r:$QA,$RL" "f2s:$QA,$SP" \
         "f2rs:$QA,R57_RELAX_DELTA=0.5,R57_RELAX_SPAN=1,R57_RELAX_LOG=$PWD/data/pf2/rs.jsonl" \
         "f2k:$QA,R57_FASTTAX=1,$NX,REF2_GDN_LAZY=1,REF2_DN_KS=1,R57_RELAX_DELTA=0.5,R57_RELAX_SPAN=1,R57_RELAX_LOG=$PWD/data/pf2/k.jsonl" \
         "f2ki:$QA,R57_FASTTAX=1,$NX,REF2_GDN_LAZY=1,REF2_DN_KS=1,R57_INERT=CD16"; do
  C45_ARMS="${a%%:*}_160:160:${a#*:}" $RUN
  say "${a%%:*} rc=$?"
done
"${PS_PY:-$HOME/BigMoE/.venv/bin/python}" -I - >> "$LOG" 2>&1 <<'EOF'
import json
L = ["PRE-FLIGHT (sat; data/pf2): HumanEval/68 + ifeval/1220, CAP 32, C 160 uniform, r57_screen_fast.py"]
for t, extra in (("f2q", None), ("f2i", None), ("f2p", None), ("f2t", None), ("f2n", None), ("f2r", "rlx"), ("f2s", "spn"),
                 ("f2rs", "spn"), ("f2k", "spn"), ("f2ki", None)):
    v = []
    try:
        rs = json.load(open(f"data/r57/{t}_160_runs.json")); v.append(f"ran {len(rs)} / 2")
        if len(rs) < 2:
            v.append("SHORT")
        if extra:
            v.append(f"{extra} {'present' if any(r.get(extra) for r in rs) else 'MISSING'}")
        v.append("tok/s " + " / ".join(f"{r['tpsw']:.0f}" for r in rs))
    except OSError:
        v.append("NO runs.json")
    try:
        txt = open(f"data/pf2/{t}_160.txt").read()
        bad = [x for x in ("Traceback", "AssertionError", "Error:") if x in txt]
        v.append("log clean" if not bad else "LOG HAS " + "/".join(bad))
    except OSError:
        v.append("no log")
    ok = not any(("MISSING" in x or "NO " in x or "LOG HAS" in x or "SHORT" in x or x == "no log") for x in v)
    L.append(f"{t:4s} {'OK  ' if ok else 'FAIL'} " + "; ".join(v))
open("data/pf2/pf2_report.txt", "w").write("\n".join(L) + "\n"); print("\n".join(L))
EOF
say "pre-flight (sat) end"
