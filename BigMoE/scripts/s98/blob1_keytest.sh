#!/bin/bash
# BLOB 1 KEY TEST (BigMoE, Wed 18 Nov 2026; NewIdeas 11:3x / EvalSuite 11:4x): blob 1 decoded end to end on the real loop
# before EvalSuite's tier 1 #1 arms. usage: SNAP=<snapshot> ENVF=<blob 1 env file, one 'export K=V' per line> \
#   WAIT_FOR=<marker to wait for, newer than arming> DEADLINE="YYYY-mm-dd HH:MM" bash scripts/s98/blob1_keytest.sh
# Gates (polled every 20 s until DEADLINE, after WAIT_FOR exists): no HOLD / PAUSE, the lid open, away_secs >= 900, no other
# session's quiet lock, NO-SWAP >= GATE (34) on two reads 30 s apart.
# 1. ENGAGEMENT (share lock): one process, 4 keys (humaneval/HumanEval/45, humanevalx/HumanEval/23, mbpp/16, ifeval/2239) x
#    512 tokens. PASS: rc 0; every key decoded; summed over the code keys tree.sib_accepted > 0 and close_call.accepted > 0;
#    every run's xrw: cand > 0, orig == 0, fallback_not_p11 == 0, mode == the env's R57_XRW; IFEval: close_call absent or 0.
# 2. TIMING (information, quiet lock): R57_XRW as in the env (ON) vs unset (OFF) on 4 code keys x 256 tokens, A B B A, one
#    process each: median ms per check and expert MB per check (checks.json ck / mb).
# EVERY end writes data/r57/BLOB1_KEYTEST_END: 'BLOB1 KEYTEST PASS|FAIL|NOT RUN <mm-dd HH:MM:SS>: <evidence>' + details beside it
# (data/r57/blob1kt/). SIM=1: PYCMD replaces each process (args: tag), FAKE_* as cert98_run.sh, no locks.
cd ~/BigMoE || exit 9
Q=$HOME/.local/share/quiet; SIM=${SIM:-0}; O=${SIMD:-data/r57/blob1kt}; mkdir -p "$O" || exit 2
MK=${SIMD:-data/r57}/BLOB1_KEYTEST_END; L=$O/keytest.log
SNAP=${SNAP:?SNAP (the blob 1 snapshot name) is required}; ENVF=${ENVF:?ENVF (the blob 1 env file) is required}; [ -s "$ENVF" ] || exit 2
SCR=data/r57/snap/$SNAP/r57_screen_$SNAP.py; [ "$SIM" = 1 ] || [ -f "$SCR" ] || { echo "no $SCR" >&2; exit 2; }
DL=$(date -j -f "%Y-%m-%d %H:%M:%S" "${DEADLINE:?DEADLINE (YYYY-mm-dd HH:MM) is required}:00" +%s) || exit 2
GATE=${GATE:-34}; T_ARM=$(date +%s)
XRW=$(grep -oE 'R57_XRW=[^ ]*' "$ENVF" | tail -1 | cut -d= -f2 | tr -d "'\"")
say() { echo "$(date '+%m-%d %H:%M:%S') $*" >> $L; }
lid_closed() { ioreg -r -k AppleClamshellState -d 1 | grep -q '"AppleClamshellState" = Yes'; }
away_ok() { local a; if [ -n "${FAKE_AWAYF:-}" ]; then a=$(cat "$FAKE_AWAYF"); else a=$(bash $Q/away_secs.sh 2>/dev/null); fi
  AW=${a:-none}; case "$a" in ""|*[!0-9]*) return 1;; esac; [ "$a" -ge 900 ]; }
quiet_held() { { if [ -n "${FAKE_PEERLS:-}" ]; then cat "$FAKE_PEERLS"; else peer ls 2>/dev/null; fi; } | grep "^- " | grep -q "· benchmarking"; }
noswap() { if [ -n "${FAKE_NOSWAP:-}" ]; then cat "$FAKE_NOSWAP"; return; fi
  vm_stat | awk '/page size of/ {ps=$8} /^Pages free/ || /^Pages purgeable/ || /^File-backed pages/ {s+=$3} END {printf "%d", s*ps/1e9}'; }
VERDICT="NOT RUN"; WHY="ended before the gates opened"; LOCK=""
fin() { [ -n "$LOCK" ] && [ "$SIM" != 1 ] && peer unlock > /dev/null 2>&1
  echo "BLOB1 KEYTEST $VERDICT $(date '+%m-%d %H:%M:%S'): $WHY" > $MK; say "END: BLOB1 KEYTEST $VERDICT: $WHY"; }
trap fin EXIT; trap 'WHY="$WHY; stopped by a signal"; exit 0' INT TERM
rm -f $MK; say "armed: snap $SNAP, env $ENVF (R57_XRW=$XRW), wait for ${WAIT_FOR:-nothing}, deadline $DEADLINE$( [ "$SIM" = 1 ] && echo '; SIMULATION')"
if [ -n "${WAIT_FOR:-}" ]; then
  until [ -e "$WAIT_FOR" ] && [ $(stat -f %m "$WAIT_FOR") -ge $T_ARM ]; do
    [ "$(date +%s)" -ge $DL ] && { WHY="no $WAIT_FOR by the deadline"; exit 0; }; sleep 20; done
  say "$WAIT_FOR seen: $(head -c 160 "$WAIT_FOR")"; fi
gates() { local w=""
  while :; do
> (a quote from the project owner, kept private)
    elif quiet_held; then w="another quiet lock"; else a=$(noswap); sleep ${GS:-30}; b=$(noswap); [ $a -ge $GATE ] && [ $b -ge $GATE ] && return 0; w="NO-SWAP $a / $b < $GATE"; fi
    [ "$(date +%s)" -ge $DL ] && { WHY="gates not open by the deadline: $w"; exit 0; }
    sleep 20; done; }
run1() { # $1 tag, $2 tokens, $3 keys, $4 extra env (KEY=VAL ...), $5 1 = unset R57_XRW
  local t=$1; rm -f data/r57/${t}_runs.json data/r57/${t}_tokens.json data/r57/${t}_checks.json
  if [ "$SIM" = 1 ]; then ( exec bash -c "$PYCMD $t" ) > $O/$t.log 2>&1
  else ( source "$ENVF"; [ "$5" = 1 ] && unset R57_XRW; for kv in R57_PRESET=45 R57_REPS=1 R57_WARM=1 R57_KEYS=$3 R57_TAG=$t $4; do export "$kv"; done
         exec .venv/bin/python scripts/kt_launch.py $SCR 160 $2 ) > $O/$t.log 2>&1; fi
  local rc=$?; for x in runs tokens checks; do mv -f data/r57/${t}_$x.json $O/ 2>/dev/null; done; say "$t rc $rc"; return $rc; }
gates; say "gates open (away $AW)"
if [ "$SIM" != 1 ]; then peer lock share 40m --cpu 200 --gpu 100 "MEM 34 GB: BigMoE blob 1 key test, engagement (~5 min)" > /dev/null 2>&1; LOCK=1; fi
run1 b1kt 512 humaneval/HumanEval/45,humanevalx/HumanEval/23,mbpp/16,ifeval/2239 "" 0; RC=$?
[ "$SIM" != 1 ] && peer unlock > /dev/null 2>&1; LOCK=""
EV=$(python3 - "$O" "$XRW" "$RC" <<'PY'
import json, os, sys
O, xrw, rc = sys.argv[1], sys.argv[2], int(sys.argv[3])
try:
    R = json.load(open(os.path.join(O, "b1kt_runs.json")))
except Exception as e:
    print(f"FAIL: no runs.json ({e}; rc {rc})"); raise SystemExit
why, code = [], [r for r in R if not str(r.get("key", "")).startswith("ifeval/")]
if rc != 0: why.append(f"rc {rc}")
if len({r.get("key") for r in R}) < 4 or any((r.get("n") or 0) <= 0 for r in R): why.append("not every key decoded")
sib = sum((r.get("tree") or {}).get("sib_accepted", 0) for r in code); cc = sum((r.get("close_call") or {}).get("accepted", 0) for r in code)
if sib <= 0: why.append("tree.sib_accepted 0 on code")
if cc <= 0: why.append("close_call.accepted 0 on code")
for r in R:
    x = r.get("xrw") or {}
    if not (x.get("cand", 0) > 0 and x.get("orig", 1) == 0 and x.get("fallback_not_p11", 1) == 0 and x.get("mode") == xrw):
        why.append(f"xrw {r.get('key')}: {x}")
    if str(r.get("key", "")).startswith("ifeval/") and (r.get("close_call") or {}).get("accepted", 0):
        why.append("close calls on IFEval")
print(("FAIL: " + "; ".join(why)) if why else f"PASS: 4 keys decoded, code sib_accepted {sib}, close calls {cc}, xrw {xrw} on every run (orig 0, fallback 0)")
PY
)
say "engagement: $EV"
case "$EV" in PASS*) VERDICT=PASS;; *) VERDICT=FAIL;; esac; WHY="${EV#*: }"
[ "$VERDICT" = PASS ] || exit 0
gates; if [ "$SIM" != 1 ]; then peer lock quiet --queue 20m "BigMoE blob 1 key test: tgl16eu ON vs OFF timing (~10 min)" >> $L 2>&1 && LOCK=1; fi
K4=humaneval/HumanEval/68,humaneval/HumanEval/16,mbpp/16,mbpp/30
run1 b1on1 256 $K4 "" 0; run1 b1off1 256 $K4 "" 1; run1 b1off2 256 $K4 "" 1; run1 b1on2 256 $K4 "" 0
TM=$(python3 - "$O" <<'PY'
import json, os, sys, statistics as st
O = sys.argv[1]
def get(t):
    try: C = json.load(open(os.path.join(O, f"{t}_checks.json")))
    except Exception: return None, None
    ck = [x for r in C for x in r.get("ck", [])]; mb = [x for r in C for x in r.get("mb", [])]
    return (st.median(ck) if ck else None), (st.median(mb) if mb else None)
on = [get(t) for t in ("b1on1", "b1on2")]; off = [get(t) for t in ("b1off1", "b1off2")]
f = lambda v: [x for x in v if x is not None]
ck_on, ck_off = f([x[0] for x in on]), f([x[0] for x in off]); mb_on, mb_off = f([x[1] for x in on]), f([x[1] for x in off])
if ck_on and ck_off:
    print(f"tgl16eu ON vs OFF (A B B A, median per check): {st.mean(ck_on):.2f} vs {st.mean(ck_off):.2f} ms ({st.mean(ck_on) - st.mean(ck_off):+.2f}); "
          f"expert MB {st.mean(mb_on):.0f} vs {st.mean(mb_off):.0f} ({st.mean(mb_on) - st.mean(mb_off):+.0f})")
else:
    print("timing: records missing")
PY
)
say "$TM"; WHY="$WHY | $TM"
