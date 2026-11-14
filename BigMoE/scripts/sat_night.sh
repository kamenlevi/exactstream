#!/bin/bash
> (a quote from the project owner, kept private)
# NewIdeas QUEUE ae954df / 4b2c45c + Fri 21:3x-21:5x; the orchestrator Fri 21:2x). Arms on scripts/r57_screen_fast.py (the merged
# FAST screen; qall = its flags off), the outside code frozen at the start. Phases, in this order:
#   0. MATCH's copy check at 16 rows (R57_COPY=g2: an exact copy check, never run under R57_FAST before): a 2-prompt check of
#      stackr + g2 (CAP 32). Clean (2 / 2, no error line) -> stackrm runs and stackt carries g2; else neither carries it.
#   1. T16 (GOAL's guarded INERT: comment / docstring rows kept only if the 125B ranks the draft top-3 with p >= 0.1 p1), ONLY if
#      data/sat/T16_OK exists (written after GOAL's module + its CPU self-test are in, by 23:45): MBPP12 quality pair
#      mq / mt / mtb / mqb (GOAL's 12, CAP 384, R57_QOM_SETS in the ENVIRONMENT: the runner splits arm envs on commas), read on
#      GOAL's line by scripts/sat_mbpp_verdicts.py. No fallback form: G16 / D16 / C16 are OUT (GOAL's log read, NewIdeas).
#   2. The palindrome on the 12 HumanEval at C 224 (ladder by NO-SWAP): stq / sts / str / srm / stt / stqm / sttb / srmb / strb /
#      stsb / stqb (srm / srmb with a clean phase 0; stt / sttb when T16 PASSed MBPP12). Chat and d7x are dropped.
#   3. C 256, only if NO-SWAP >= 47.2 GB at the grant (and again before its first arm): sh / wh / whb / shb = the stack and the
#      winner (the fastest of stackr / stackrm / stackt among those that PASS their lines; none -> the stack's two passes only).
#   4. never-wait's ceiling (snw, pre-registered Fri 09:0x), if the deadline (start + 60 min) allows.
# stack   = qall + FAST-NAX + DN-KS + GDN lazy + D7BNAX_32=1 + FAST-tax (CD16 dropped Fri 16:0x after its MBPP12 quality KILL)
# stackr  = the stack + MoonShots' relax v2 (DELTA 0.5 + SPAN);  stackrm = stackr + R57_COPY=g2
# stackt  = the stack + R57_INERT=T16 (+ R57_COPY=g2 with a clean phase 0) = NewIdeas' "stackg"; INERT and relax v2 exclude
#           each other (no combined arm). Passes overlapping a fanwatch freeze are dropped and listed.
# Lines (pre-registered before arming; pair-mean geo steady tok/s on the 12 HumanEval unless said):
#   headline  stack vs qall: PASS >= +20 %, KILL < +10 %, else BETWEEN (Fri 08:3x; its chat clause is moot tonight). Below C 224
#             every speed verdict is INDICATIVE.
#   stackr / stackrm / stackt  vs the stack: PASS >= x1.03, KILL < x0.97, else BETWEEN (Fri 08:4x; NewIdeas Fri 21:2x / 21:4x).
#   MBPP12 T16: GOAL's line in scripts/sat_mbpp_verdicts.py (set when GOAL sends it; no line = no PASS = no stackt).
#   C 256     per arm (NewIdeas Fri 21:3x): PASS = >= +3 % vs the same arm at C 224 AND lifetime peak phys_footprint (RUSAGE_INFO_V4,
#             the runner's done line) <= 45.0 GB; KILL otherwise.
#   ceiling   snw: median (cold16 - warm16) >= 0.6 x S PASS, < 0.3 x S KILL, else BETWEEN (Fri 09:0x).
# code_env.txt for EvalSuite: the fastest of the ELIGIBLE arms that PASS (stackt; stackb when armed), else the stack. stackr and
# stackrm are SPEED ONLY (the orchestrator Fri 21:4x): STACKR = relax v2 with DELTA and no R57_RELAX_GATE relaxes CODE tokens too,
# the C_E form that failed decision 4 (EvalSuite docs/RESULT_relax_2026-11-05.md). stackb (only with data/sat/SHIPB_OK) = the
# stack + ship B's relax form (R57_RELAX_SPAN=1, R57_RELAX_SPAN_RULE=R2:2:0.25, no DELTA; + g2 with a clean phase 0), eligible,
# line (NewIdeas Fri 21:5x) PASS >= stack x 1.02, KILL < x0.98. When the pick's C 256 line PASSes, a comment records it; the header keeps this run's C
# (EvalSuite's gate runs at its own C). The C 256 winner is chosen among the eligible arms only.
# usage: NOTBEFORE=00:30 NOTAFTER=03:00 NOWAIT45=1 GAP0=0 GAP=0 MARGIN=4 bash scripts/turn45.sh 33 65 "..." /bin/bash scripts/sat_night.sh
#   -> data/sat/sat.log, data/sat/sat_report.txt, data/sat/mbpp_forms.json, data/stack/*_env.txt, data/stack/code_env.txt
export PATH="$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:$PATH"
cd "${SN_ROOT:-$HOME/BigMoE}" || exit 1
mkdir -p data/sat data/stack data/r57/snap
LOG="$PWD/data/sat/sat.log"
say() { echo "$(date '+%m-%d %H:%M:%S') $*" >> "$LOG"; }
PY=${SN_PY:-$HOME/BigMoE/.venv/bin/python}
EPY=${SN_EPY:-$HOME/EvalSuite/.venv/bin/python}           # EvalSuite's venv (evalplus): the MBPP test verdicts
STACK=${SN_STACK-R57_NAX=1,REF2_NAX_1CQ=1,REF2_DN_KS=1,REF2_GDN_LAZY=1,D7BNAX_32=1,R57_FASTTAX=1}
STACKR=${SN_STACKR-R57_NAX=1,REF2_NAX_1CQ=1,REF2_DN_KS=1,REF2_GDN_LAZY=1,D7BNAX_32=1,R57_FASTTAX=1,R57_RELAX_DELTA=0.5,R57_RELAX_SPAN=1}
avail_gb() { vm_stat | awk '/page size of/ {ps=$8} /^Pages (free|purgeable):/ || /^File-backed pages:/ {v=$NF; sub("[.]","",v); s+=v} END {printf "%.1f", s*ps/1e9}'; }
A=${SN_AVAIL:-$(avail_gb)}
fits() { awk -v a="$A" -v n="$1" 'BEGIN {exit !(n <= a)}'; }
CC=; for c in 224:44.0 208:41.9 192:39.7 176:37.6 160:35.4; do fits ${c#*:} && { CC=${c%%:*}; break; }; done   # Fri C 224 peaks 38.4-38.9 GB + 4
[ -z "$CC" ] && { say "NO-SWAP $A GB < 35.4: nothing fits, nothing runs"; exit 0; }
C256=; [ "$CC" = 224 ] && fits 47.2 && C256=256          # C 256's peak ~43.2 GB (38.9 + 32 x 0.134) + 4
T16=; [ -f data/sat/T16_OK ] && grep -q '"T16"' scripts/inert_span.py && T16=1
SHIPB=; [ -f data/sat/SHIPB_OK ] && SHIPB=1                # NewIdeas' call on the orchestrator's option (2)
# Fri 23:4x (the orchestrator): SN_PART=A runs phases 0-1 now, at the C that fits (cache-independent: MATCH-16's check, T16's
# quality pair); SN_PART=B runs phases 2-4 + the report, ONLY at C 224 (its turn45 waits for NO-SWAP >= 44 before queueing),
# on part A's frozen snapshot and part A's results (data/sat/part_a.env). Unset = all phases in one ticket.
PART=${SN_PART:-AB}
[ "$PART" = B ] && [ "$CC" != 224 ] && { say "part B: the palindrome needs C 224, NO-SWAP $A GB < 44: not run";
  peer say --to "AI/LLM project recommendations" "BigMoE part B: NO-SWAP $A GB < 44 at the grant: the C 224 palindrome did not run; free memory and I re-arm" >/dev/null 2>&1; exit 0; }
FZ="$PWD/data/r57/snap/frozen_sat"
if [ "$PART" = B ] && [ -s data/sat/part_a.env ] && [ -d "$FZ/Leap_scripts" ] && [ -s data/r57/snap/r57_screen_sat.py ]; then
  say "part B: reusing part A's frozen snapshot ($FZ) and screen copy"
else
cp scripts/r57_screen_fast.py data/r57/snap/r57_screen_sat.py && cp scripts/relax_delta.py scripts/relax_span.py scripts/inert_span.py \
  data/r57/snap/ && cp scripts/c45_runner.sh data/r57/snap/c45_runner_sat.sh || exit 2
# Fri 09:4x (EvalSuite's warning: the screen loads LIVE outside code): freeze it for every arm of tonight's run. Copies of Leap's
# scripts + third_party, TrainLab's sticky and Breakthrough's keepset; the snapshot screen points at the copies.
rm -rf "$FZ"; mkdir -p "$FZ"
[ -n "$SN_DRYRUN" ] || { cp -R "$HOME/Leap/scripts" "$FZ/Leap_scripts" && cp -R "$HOME/Leap/third_party" "$FZ/Leap_third_party" \
  && cp -R "$HOME/TrainLab/sticky" "$FZ/TrainLab_sticky" && cp -R "$HOME/Breakthrough/keepset" "$FZ/Breakthrough_keepset" || exit 2; }
find "$FZ" -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null
sed -i '' -e "s#\"~/Leap/scripts\"#\"$FZ/Leap_scripts\"#g" -e "s#\"~/Leap/third_party\"#\"$FZ/Leap_third_party\"#g" \
  -e "s#\"~/TrainLab/sticky\"#\"$FZ/TrainLab_sticky\"#g" -e "s#\"/Users/kamen/Breakthrough/keepset\"#\"$FZ/Breakthrough_keepset\"#g" \
  data/r57/snap/r57_screen_sat.py
fi
LIVE0=$(md5 -q "$HOME/Leap/scripts/ref2.py" "$HOME/Leap/scripts/d7b_nax.py" "$HOME/Breakthrough/keepset/qom_keep.py" | tr '\n' ' ')
say "frozen outside code in $FZ ($(grep -c frozen_sat data/r57/snap/r57_screen_sat.py) screen paths): ref2 / d7b_nax / qom_keep md5 $LIVE0; inert_span md5 $(md5 -q data/r57/snap/inert_span.py)"
RUN="/bin/bash data/r57/snap/c45_runner_sat.sh"
export C45_OUT="$PWD/data/sat" C45_SWAP0=$(sysctl -n vm.swapusage | awk '{gsub("M","",$6); print $6 / 1024}') C45_ALARM=1200 \
  MG=${SN_MG:-4} C45_DEADLINE=${SN_DEADLINE:-$(date -v+60M +%H%M)} C45_SCREEN=data/r57/snap/r57_screen_sat.py C45_REFTOK=
K="humaneval/HumanEval/68,humaneval/HumanEval/16,humaneval/HumanEval/91,humaneval/HumanEval/6,humaneval/HumanEval/79"
K="$K,humaneval/HumanEval/19,humaneval/HumanEval/161,humaneval/HumanEval/17,humaneval/HumanEval/56,humaneval/HumanEval/141"
K="$K,humaneval/HumanEval/101,humaneval/HumanEval/63"
KM="mbpp/16,mbpp/100,mbpp/146,mbpp/161,mbpp/183,mbpp/208,mbpp/224,mbpp/246,mbpp/268,mbpp/296,mbpp/305,mbpp/319"   # GOAL's 12
QA="BIGMOE_EXT_BUILD=build_qom,R57_REFTOK=,R57_FAST=QOM,R57_D17=1,R57_D7B=code,R57_HC8=1,REF2_HC8_FAST=1,R57_DNR=1,R57_HCB=1"
QA="$QA,REF2_SMALL_NATIVE=1,R57_D7BNAX=1,REF2_OOO_1C=1"
QAC="$QA,R57_QCNT1C=1"            # every arm carries Leap's one-call counters (the same small cost everywhere); env files keep QA
RR="$QAC,$STACKR,R57_RELAX_LOG=$PWD/data/sat/stackr_relax.jsonl"
HEAD=$(git rev-parse --short HEAD)
envf() { printf '%s\n' "# Sat 7 Nov $2 (BigMoE sat_night.sh, $HEAD): screen scripts/r57_screen_fast.py; env (comma list; *_LOG optional):" \
  "$3,R57_QOM_SETS=humaneval,mbpp" > "data/stack/$1_env.txt"; }   # QOM on the code sets only (NewIdeas 70a4951)
envf stack "stacked FAST" "$QA,$STACK"
envf stackr "stacked FAST + relax v2 for CODE" "$QA,$STACKR"
if [ "$PART" != B ]; then
  rm -f data/sat/part_a.env data/sat/mbpp_forms.json
  for t in pfg mq mt mtb mqb; do rm -f data/r57/${t}_${CC}_* data/sat/${t}_${CC}.txt; done
fi
if [ "$PART" != A ]; then
  rm -f data/stack/stackrm_env.txt data/stack/stackt_env.txt data/stack/stackg_env.txt data/stack/stackb_env.txt data/sat/winner.txt
  for t in stq sts str srm stt stb stqm stbb sttb srmb strb stsb stqb snw; do rm -f data/r57/${t}_${CC}_* data/sat/${t}_${CC}.txt; done
  for t in sh wh whb shb; do rm -f data/r57/${t}_256_* data/sat/${t}_256.txt; done
fi
say "200 STACK start (pid $$, NO-SWAP $A GB -> C $CC, C 256 ${C256:-off (needs 47.2 GB)}, T16 ${T16:-off (no data/sat/T16_OK or no T16 in inert_span.py)}, deadline $C45_DEADLINE, stack '$STACK')"
if [ "$PART" = B ]; then                                  # part B: part A's results
  G2=; TP=; C1=
  [ -s data/sat/part_a.env ] && . data/sat/part_a.env
  say "part B (C $CC): part A's results: MATCH g2 ${G2:-not clean or not run}, T16 PASS ${TP:-no} (part A at C ${C1:-?})"
else
# 0. MATCH's copy check at 16 rows under R57_FAST: a 2-prompt check of stackr + g2
R57_KEYS="humaneval/HumanEval/68,humaneval/HumanEval/16" C45_CAP=32 C45_ARMS="pfg_$CC:$CC:$RR,R57_COPY=g2" $RUN
n=$(grep -o '"key"' "data/r57/pfg_${CC}_runs.json" 2>/dev/null | wc -l | tr -d ' ')
bad=$(grep -cE 'Traceback|AssertionError|Error:' "data/sat/pfg_${CC}.txt" 2>/dev/null)
G2=; [ "${n:-0}" = 2 ] && [ "${bad:-1}" = 0 ] && G2=1
say "MATCH g2 at 16 rows (stackr + R57_COPY=g2, 2 prompts, CAP 32): ran ${n:-0} / 2, error lines ${bad:-none} -> $([ -n "$G2" ] && echo "CLEAN: stackrm runs, stackt carries g2" || echo "NOT CLEAN: no stackrm, stackt without g2")"
# 1. T16's MBPP12 quality pair (GOAL's 12, CAP 384, R57_QOM_SETS in the environment)
TP=
if [ -n "$T16" ]; then
  for a in "mq:$QAC" "mt:$QAC,R57_INERT=T16,R57_INERT_LOG=$PWD/data/sat/mt_inert.jsonl" \
           "mtb:$QAC,R57_INERT=T16,R57_INERT_LOG=$PWD/data/sat/mtb_inert.jsonl" "mqb:$QAC"; do
    R57_QOM_SETS=humaneval,mbpp C45_CAP=384 R57_KEYS=$KM C45_ARMS="${a%%:*}_$CC:$CC:${a#*:}" $RUN
    say "${a%%:*} rc=$?"
  done
  PK=$("$EPY" -I scripts/sat_mbpp_verdicts.py "$CC" data/sat/mbpp_forms.json T16 2>> "$LOG" | tee -a "$LOG" | awk '/^PICK/ {print $2}')
  [ "$PK" = T16 ] && TP=1
fi
say "T16: $([ -z "$T16" ] && echo "not run (not cleared by 23:45): no stackt" || { [ -n "$TP" ] && echo "MBPP12 PASS: stackt runs" || echo "MBPP12 not PASS: no stackt"; })"
printf 'G2=%s\nTP=%s\nC1=%s\n' "$G2" "$TP" "$CC" > data/sat/part_a.env
fi
[ "$PART" = A ] && { say "part A end (MATCH g2 ${G2:-0}, T16 PASS ${TP:-0}, C $CC): part B runs the palindrome at C 224"; exit 0; }
SRM="$RR,R57_COPY=g2"
STT="$QAC,$STACK,R57_INERT=T16,R57_INERT_LOG=$PWD/data/sat/stt_inert.jsonl${G2:+,R57_COPY=g2}"
[ -n "$G2" ] && envf stackrm "stacked FAST + relax v2 + MATCH g2 at 16 rows for CODE" "$QA,$STACKR,R57_COPY=g2"
[ -n "$TP" ] && envf stackt "stacked FAST + T16${G2:+ + MATCH g2 at 16 rows} for CODE" "$QA,$STACK,R57_INERT=T16${G2:+,R57_COPY=g2}"
SB="R57_RELAX_SPAN=1,R57_RELAX_SPAN_RULE=R2:2:0.25${G2:+,R57_COPY=g2}"   # ship B's relax form: comment rows only, no DELTA
STB="$QAC,$STACK,$SB,R57_RELAX_LOG=$PWD/data/sat/stb_relax.jsonl"
[ -n "$SHIPB" ] && envf stackb "stacked FAST + ship B relax (span R2:2:0.25)${G2:+ + MATCH g2 at 16 rows} for CODE" "$QA,$STACK,$SB"
# 2. the palindrome on the 12 HumanEval at C $CC
AR=("stq:$QAC" "sts:$QAC,$STACK" "str:$RR")
[ -n "$G2" ] && AR+=("srm:$SRM")
[ -n "$TP" ] && AR+=("stt:$STT")
[ -n "$SHIPB" ] && AR+=("stb:$STB")
AR+=("stqm:$QAC")
[ -n "$SHIPB" ] && AR+=("stbb:$STB")
[ -n "$TP" ] && AR+=("sttb:$STT")
[ -n "$G2" ] && AR+=("srmb:$SRM")
AR+=("strb:$RR" "stsb:$QAC,$STACK" "stqb:$QAC")
for a in "${AR[@]}"; do
  R57_KEYS=$K C45_ARMS="${a%%:*}_$CC:$CC:${a#*:}" $RUN
  say "${a%%:*} rc=$?"
done
# the winner for C 256: the fastest of stackr / stackrm / stackt among those that PASS (>= x1.03 vs the stack), else none
WIN=$("$PY" -I - "$CC" 2>> "$LOG" <<'EOF'
import json, math, sys
C = sys.argv[1]
def pm(ts):
    out = {}
    for t in ts:
        try:
            rs = json.load(open(f"data/r57/{t}_{C}_runs.json"))
        except (OSError, ValueError):
            continue
        for r in rs:
            if "PAUSED" not in r.get("bad", "") and r.get("tpsw"):
                out.setdefault(r["key"], []).append(r["tpsw"])
    return {k: sum(v) / len(v) for k, v in out.items()}
def ratio(a, b):
    ks = [k for k in a if k in b]
    return math.exp(sum(math.log(a[k] / b[k]) for k in ks) / len(ks)) if ks else 0.0
s_ = pm(("sts", "stsb"))   # ELIGIBLE arms only (stackr / stackrm relax code tokens: speed only), each on its own PASS line
c_ = {n: (ratio(pm(ts), s_), th) for n, ts, th in (("stackt", ("stt", "sttb"), 1.03), ("stackb", ("stb", "stbb"), 1.02))}
ok = {n: r for n, (r, th) in c_.items() if r >= th}
print(max(ok, key=ok.get) if ok else "none")
EOF
)
case "$WIN" in stackt) WE="$STT" ;; stackb) WE="$STB" ;; *) WIN=none; WE= ;; esac
echo "$WIN" > data/sat/winner.txt; say "winner for C 256: $WIN"
# 3. C 256 (NO-SWAP >= 47.2 GB at the grant and now)
if [ -n "$C256" ]; then
  A2=${SN_AVAIL2:-$(avail_gb)}
  if awk -v a="$A2" 'BEGIN {exit !(a >= 47.2)}'; then
    AR3=("sh:$QAC,$STACK"); [ -n "$WE" ] && AR3+=("wh:$WE" "whb:$WE"); AR3+=("shb:$QAC,$STACK")
    for a in "${AR3[@]}"; do
      R57_KEYS=$K C45_ARMS="${a%%:*}_256:256:${a#*:}" $RUN
      say "${a%%:*} rc=$?"
    done
  else
    say "C 256 skipped: NO-SWAP $A2 GB < 47.2 before its first arm"
  fi
else
  say "C 256 skipped: NO-SWAP $A GB < 47.2 at the grant (or C $CC < 224)"
fi
# 4. never-wait's ceiling (Fri 09:0x; ~/FableGod/far/NEVERWAIT_TEST.md section 3): the stack with R57_WARMSPLIT=4, one pass
R57_KEYS=$K C45_ARMS="snw_$CC:$CC:$QAC,$STACK,R57_WARMSPLIT=4" $RUN
say "snw rc=$?"
LIVE1=$(md5 -q "$HOME/Leap/scripts/ref2.py" "$HOME/Leap/scripts/d7b_nax.py" "$HOME/Breakthrough/keepset/qom_keep.py" | tr '\n' ' ')
[ "$LIVE1" = "$LIVE0" ] && say "live outside code unchanged during the run" \
  || say "NOTE: live outside code CHANGED during the run ($LIVE0 -> $LIVE1); every arm used the frozen copies"
"$PY" -I - "$CC" "$(date +%F)" "${SN_FANLOG:-$HOME/Library/Logs/fanwatch.log}" "${G2:-0}" "${TP:-0}" >> "$LOG" 2>&1 <<'EOF'
import datetime as D, json, math, re, statistics as S, sys
C, day, fanlog, g2, tp_ = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4] == "1", sys.argv[5] == "1"
g = lambda xs: (lambda ys: math.exp(sum(map(math.log, ys)) / len(ys)) if ys else float("nan"))([x for x in xs if x and x > 0])
win, t0 = [], None
try:
    for ln in open(fanlog):
        m = re.match(r"(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) (froze|cool)", ln)
        if m and m.group(1).startswith(day):
            t = D.datetime.fromisoformat(m.group(1))
            if m.group(2) == "froze":
                t0 = t
            elif t0 is not None:
                win.append((t0, t)); t0 = None
except OSError:
    pass
arm_t, peak = {}, {}
for ln in open("data/sat/c45.log"):
    m = re.match(r"(\d\d:\d\d:\d\d) (run|done) (\S+)", ln)
    if m:
        arm_t.setdefault(m.group(3), {})[m.group(2)] = D.datetime.fromisoformat(f"{day} {m.group(1)}")
    m = re.search(r"done (\S+) rc=\d+:.*lifetime peak ([\d.]+) GB", ln)
    if m:
        peak[m.group(1)] = float(m.group(2))
hit = {t for t, v in arm_t.items() if "run" in v and "done" in v and any(a < v["done"] and b > v["run"] for a, b in win)}
CT = lambda t: "256" if t in ("sh", "shb", "wh", "whb") else C
P = {"qall": ("stq", "stqm", "stqb"), "stack": ("sts", "stsb"), "stackr": ("str", "strb"), "stackrm": ("srm", "srmb"),
     "stackt": ("stt", "sttb"), "stackb": ("stb", "stbb"), "stack256": ("sh", "shb"), "winner256": ("wh", "whb")}
runs = {}
for t in sum(P.values(), ()):
    try:
        runs[t] = {r["key"]: r for r in json.load(open(f"data/r57/{t}_{CT(t)}_runs.json"))}
    except (OSError, ValueError):
        runs[t] = {}
ok = lambda t, k: k in runs[t] and "PAUSED" not in runs[t][k].get("bad", "") and f"{t}_{CT(t)}" not in hit
kept = lambda r: r["kept"] if isinstance(r["kept"], (int, float)) else S.mean(r["kept"])
def pm(ps, f):
    out = {}
    for k in set().union(*[runs[t] for t in ps]):
        v = [f(runs[t][k]) for t in ps if ok(t, k)]
        if v:
            out[k] = sum(v) / len(v)
    return out
def ratio(a, b):
    ks = [k for k in a if k in b]
    return (g([a[k] / b[k] for k in ks]), len(ks)) if ks else (float("nan"), 0)
tp = lambda b: pm(P[b], lambda r: r["tpsw"])
ind = " INDICATIVE at C < 224" if int(C) < 224 else ""
L = [f"200 STACK, C {C} (data/sat): MATCH g2 pre-check {'CLEAN' if g2 else 'NOT CLEAN or not run'}; T16 {'PASSed MBPP12' if tp_ else 'not in'}; "
     f"the 12 HumanEval palindrome qall / stack / stackr{' / stackrm' if g2 else ''}{' / stackt' if tp_ else ''}{' / stackb' if runs['stb'] else ''}, then C 256; "
     f"freeze windows {[(a.strftime('%H:%M:%S'), b.strftime('%H:%M:%S')) for a, b in win]}; passes dropped for a freeze: "
     f"{sorted(hit) or 'none'}"]
try:
    mf = json.load(open("data/sat/mbpp_forms.json"))
    for f_ in ([mf["form"]] if "form" in mf else [k for k in ("T16", "G16", "D16") if k in mf]):   # flat (T16 reader) or keyed
        x = mf if "form" in mf else mf[f_]
        L.append(f"MBPP12 {f_} vs qall (GOAL's line): {x.get('verdict', 'MISSING')}: median S {x.get('S', float('nan')):.3f}, T "
                 f"{x.get('T', float('nan')):.3f}, " + (f"pairs {[(p_['pair'], p_['equal'], p_['flips']) for p_ in x['pairs']]}, K ratio "
                 f"{x.get('K', float('nan')):.3f}, tests passed {x.get('arm_passed')} vs qall {x.get('control_passed')}" if "pairs" in x else
                 f"verdicts equal {x.get('equal')} / {x.get('tasks')}, flips to fail {x.get('flips') or 'none'}, tests passed "
                 f"{x.get('passed')} vs qall {mf.get('control_passed')}"))
except (OSError, ValueError):
    L.append("MBPP12 T16: not run (no data/sat/mbpp_forms.json)")
for b, ps in P.items():
    t_ = tp(b)
    if not t_:
        continue
    L.append(f"{b:9s} geo steady {g(list(t_.values())):6.1f} tok/s over {len(t_)}; passes " +
             ", ".join(f"{t} {g([r['tpsw'] for k, r in runs[t].items() if ok(t, k)]):.1f}" for t in ps) +
             f"; kept per check {g(list(pm(ps, kept).values())):.2f}; tokens per task "
             f"{S.median(pm(ps, lambda r: r['n']).values()) if pm(ps, lambda r: r['n']) else float('nan'):.0f}; lifetime peak "
             f"{max([peak.get(f'{t}_{CT(t)}', 0.0) for t in ps]) or float('nan'):.1f} GB")
qm, qe = pm(("stqm",), lambda r: r["tpsw"]), pm(("stq", "stqb"), lambda r: r["tpsw"])
nf, nn = ratio(qm, qe)
L.append(f"noise floor: qall middle pass vs its end passes {100 * (nf - 1):+.1f} % over {nn}")
rc, nc = ratio(tp("stack"), tp("qall"))
v = "MISSING" if nc == 0 else "PASS" if rc >= 1.20 else "KILL" if rc < 1.10 else "BETWEEN"
L.append(f"Verdict stacked FAST (pre-registered Fri 08:3x){ind}: {v}: code {100 * (rc - 1):+.1f} % vs qall over {nc} (PASS >= +20 %; "
         f"KILL < +10 %; the chat clause is moot tonight)")
vv = {}
for b, when, hi, lo, elig in (("stackr", "Fri 08:4x", 1.03, 0.97, False), ("stackrm", "NewIdeas Fri 21:4x", 1.03, 0.97, False),
                              ("stackt", "NewIdeas Fri 21:4x", 1.03, 0.97, True), ("stackb", "NewIdeas Fri 21:5x", 1.02, 0.98, True)):
    r_, n_ = ratio(tp(b), tp("stack"))
    q_, _ = ratio(tp(b), tp("qall"))
    vv[b] = ("MISSING" if n_ == 0 else "PASS" if r_ >= hi else "KILL" if r_ < lo else "BETWEEN", r_)
    if n_ or b == "stackr":
        L.append(f"Verdict {b} (pre-registered {when}){ind}: {vv[b][0]}: {100 * (r_ - 1):+.1f} % vs the stack over {n_}, vs qall "
                 f"{100 * (q_ - 1):+.1f} % (PASS >= {100 * (hi - 1):+.0f} % vs the stack; KILL < {100 * (lo - 1):+.0f} %)"
                 + ("" if elig else "; SPEED ONLY: relaxes code tokens (the C_E form that failed decision 4), never in code_env"))
try:
    wn = open("data/sat/winner.txt").read().strip()
except OSError:
    wn = "none"
c256 = {}
for b, ref in (("stack256", "stack"), ("winner256", wn)):
    if not any(runs[t] for t in P[b]):
        continue
    r_, n_ = ratio(tp(b), tp(ref)) if ref in P else (float("nan"), 0)
    pk = max([peak.get(f"{t}_256", 0.0) for t in P[b]]) or float("nan")
    c256[ref] = "PASS" if n_ and r_ >= 1.03 and pk <= 45.0 else "KILL"
    L.append(f"Verdict C 256 for {ref} (pre-registered NewIdeas Fri 21:3x): {c256[ref]}: {100 * (r_ - 1):+.1f} % vs C {C} over {n_}, "
             f"lifetime peak {pk:.1f} GB (PASS >= +3 % AND peak <= 45.0 GB)")
if not c256:
    L.append("C 256: not run (see data/sat/sat.log for the NO-SWAP gate)")
try:                                                     # never-wait's speed ceiling (R57_WARMSPLIT samples of arm snw)
    ws = [s for e in json.load(open(f"data/r57/snw_{C}_warmsplit.json")) for s in e.get("samples", [])]
    ck = json.load(open(f"data/r57/snw_{C}_checks.json"))
    live = [(c, x) for e in ck if not e.get("paused") for c, x in zip(e.get("svc", []), e.get("miss", []))]
    S_ = S.median([c for c, _ in live]) if live else float("nan")
    d_ = [s["cold"][1] - w[1] for s in ws if s.get("cold") for w in s.get("warm", []) if w[0] == s["cold"][0]]
    mc = S.median([s["cold"][2] for s in ws if s.get("cold")]) if ws else float("nan")
    ml = S.median([x for _, x in live]) if live else float("nan")
    cd = S.median(d_) if d_ else float("nan")
    vn = "MISSING" if not d_ else ("PASS" if cd >= 0.6 * S_ else "KILL" if cd < 0.3 * S_ else "BETWEEN")
    L.append(f"Verdict never-wait ceiling (pre-registered Fri 09:0x){ind}: {vn}: median cold16 - warm16 {cd:.1f} ms over {len(d_)} "
             f"samples vs S {S_:.1f} ms (PASS >= 0.6 S, KILL < 0.3 S); sample misses {mc:.0f} vs live {ml:.0f} per check")
except (OSError, ValueError, KeyError, TypeError) as e_:
    L.append(f"never-wait ceiling: MISSING ({type(e_).__name__})")
for b in ("stackr", "stackrm", "stackb"):
    rl = [r.get("rlx") for t in P[b] for r in runs[t].values() if r.get("rlx")]
    cp = [r.get("copy") for t in P[b] for r in runs[t].values() if r.get("copy")]
    if rl or cp:
        L.append(f"{b}: relaxed accepts {sum(x.get('relaxed', 0) for x in rl)}" + (
            f"; copy: positions {sum(x.get('pos', 0) for x in cp)}, used {sum(x.get('used', 0) for x in cp)}, changed the draft "
            f"{sum(x.get('chg', 0) for x in cp)}, accepted {sum(x.get('ok', 0) for x in cp)}" if cp else ""))
ins = [r.get("inert") for t in P["stackt"] for r in runs[t].values() if r.get("inert")]
if ins:
    L.append("stackt INERT counters: " + ", ".join(f"{k} {sum(i.get(k, 0) for i in ins)}" for k in
                                                    ("rows", "kx", "k", "whole", "ruled", "com", "doc", "open")))
L.append("lifetime peak phys_footprint per arm (GB): " + ", ".join(f"{t} {p:.1f}" for t, p in sorted(peak.items())
                                                                   if t.rsplit("_", 1)[0] in sum(P.values(), ()) + (
                                                                       "snw", "pfg", "mq", "mt", "mtb", "mqb")))
cand = {b: vv[b][1] for b in ("stackt", "stackb") if vv[b][0] == "PASS"}   # eligible arms only (NewIdeas / the orchestrator 21:5x)
pick = max(cand, key=cand.get) if cand else "stack"
opts = [(pick, C), ("stack", C)] + ([("winner256", "256")] if pick == wn and pick != "stack" else []) + [("stack256", "256")]
best = max(((g(list(tp(b).values())), b, c_) for b, c_ in opts if tp(b)), default=(float("nan"), "none", C))
L.append(f"FINAL stacked number (quality gate pending: EvalSuite same-scores): {best[1]} at C {best[2]}: {best[0]:.1f} tok/s geo "
         f"steady on the 12 HumanEval (lifetime peaks above)")
try:                                                     # EvalSuite's same-scores runner reads ONE code file (Fri 09:2x rule)
    envl = lambda f: [x for x in open(f).read().splitlines() if x and not x.startswith("#")][-1]
    extra = (f"# C 256 PASSED for {pick} tonight (for the record; EvalSuite's gate keeps its own C)\n"
             if c256.get(pick) == "PASS" else "")
    open("data/stack/code_env.txt", "w").write(
        f"# CODE profile for EvalSuite's same-scores run (sat_night.sh, Sat {C}): {pick} (stackt {vv['stackt'][0]}, stackb "
        f"{vv['stackb'][0]}; rule: the fastest of those two that PASS, else the stack; stackr / stackrm are speed only). Screen "
        f"scripts/r57_screen_fast.py. One comma-separated env line:\n" + extra + envl(f"data/stack/{pick}_env.txt") + "\n")
    L.append(f"code_env.txt = {pick}{'; C 256 recorded' if extra else ''}")
except (OSError, IndexError) as e_:
    L.append(f"code_env.txt NOT written ({type(e_).__name__})")
open("data/sat/sat_report.txt", "w").write("\n".join(L) + "\n"); print("\n".join(L))
EOF
say "report rc=$? (data/sat/sat_report.txt)"
say "200 STACK end"
