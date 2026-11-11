#!/bin/bash
# The records3 conversion's token checks for BigMoE (PLAN_records3_convert section 5 step 3; HANDOVER Tue 3 Nov 17:3x).
# REF2-T re-decodes six IFEval rows and their tokens are compared with reference runs' rows: the two IFX reproduction
# rows (ifeval/3409, ifeval/2439) and four 308-525-token IFX rows (results/r3check_keys.txt; 1,614 tokens stored).
# Check 1 = before the conversion (row-major store), check 2 = after it (converted store). Both use variant
# ref2tf0 = ref2t on snap/leap-c805fd0 with the FAST flags off (fatt_verify, fatt_r8, hc8_fast; PREREG_f1 rev. 2):
# Leap's per-layer layout router and tierglu_3cn / p11 (byte-identical to 7f43a3f's) for the P11PAD format (7624eb0,
# 22:52: the padded native + p11 records that BigMoE's A/B and conversion write). 90456ce's kernels predate that
# format, so ref2tn is dropped; snapshot: snap/bigmoe-da8e7a6. The same variant is F1's control
# (PREREG_f1), so the checks validate that code on both layouts. Only the store differs between them, so check 2 ==
# check 1 is the conversion's test; check 1 == ref2t.r2 (the stored IFX rows, snap/bigmoe-6c64148 + snap/leap-faff8e5)
# is information on the newer code. Everything else is runner 38's ref2t.r2 recipe (TIERENV, C 192). Shared GPU,
# attached, in the orchestrator's slot. Check 1 after T0_DONE (row-major); check 2 with the permanent conversion
# (BigMoE 22:3x: its B / B2 token checks cover the A/B night).
#   tools/r3check.sh <bigmoe snap dir> <ref2tf0.rN> <rowmajor|converted> <ref run[,ref run]> ["extra stage env k=v,k=v"]
#   check 1: tools/r3check.sh snap/bigmoe-<sha7> ref2tf0.r1 rowmajor ref2t.r2
#   check 2: tools/r3check.sh snap/bigmoe-<sha7> ref2tf0.r2 converted ref2tf0.r1,ref2t.r2
# Exit: 0 = token-identical to the FIRST ref run on all six rows, 5 = different, 2 = refused (no snapshot or variant,
# the store not as expected, the run exists), else the stage's rc. Log: results/r3check.log.
export PATH="$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:$PATH"
cd "$HOME/EvalSuite" || exit 1
SNAP=$1; RUN=$2; WANT=$3; REFS=$4; EXTRA=$5
LOG=results/r3check.log
C=${R3_C:-192}                         # pool size (tokens do not depend on C; runners pick it by free memory)
R3IDX=$HOME/BigMoE/models/bare/records3/index.json
TIERENV="BIGMOE_TIER=4433333333,BIGMOE_TIER_GLU=3c,BIGMOE_TIER_CL=112"   # runner 38's, for ref2t.r2
say() { echo "$(date +%H:%M:%S) $*" | tee -a "$LOG"; }
store() { if grep -q '"layout"' "$R3IDX"; then echo converted; else echo rowmajor; fi; }
[ -f "$SNAP/SOURCE" ] && [ -d "$SNAP/bigmoe" ] || { say "no snapshot at $SNAP (SOURCE + bigmoe/)"; exit 2; }
case "$RUN" in ref2tf0.r[1-9]) ;; *) say "run name must be ref2tf0.r1 .. ref2tf0.r9, not $RUN"; exit 2 ;; esac
grep -q '"ref2tf0"' evalsuite/model.py || { say "variant ref2tf0 is not in evalsuite/model.py yet (after T0_DONE)"; exit 2; }
[ -e "results/$RUN" ] && { say "results/$RUN exists: take the next run name"; exit 2; }
[ -n "$REFS" ] || { say "no reference run given"; exit 2; }
[ "$(store)" = "$WANT" ] || { say "records3 is $(store), this check wants $WANT: not running"; exit 2; }
while [ -e "$HOME/.local/share/quiet/PAUSE" ] || [ -e "$HOME/.local/share/quiet/HOLD" ]; do sleep 30; done
export EVALSUITE_BIGMOE_CODE="$(cd "$SNAP" && pwd)"   # absolute, as the runners pass it
say "r3check $RUN: records3 $(store); snapshot $SNAP ($(head -c 140 "$SNAP/SOURCE")); refs $REFS; C $C; extra env: ${EXTRA:-none}"
EVALSUITE_STAGE_ENV="$TIERENV${EXTRA:+,$EXTRA}" ./bin/evalsuite gpu decode "$RUN" --sets ifeval,ifevalx \
  --keys results/r3check_keys.txt --chunk-minutes 30 --C "$C" --share >> "$LOG" 2>&1; rc=$?
say "decode $RUN rc=$rc"
[ "$rc" = 0 ] || exit "$rc"
.venv/bin/python - "$RUN" "$REFS" <<'EOF' 2>&1 | tee -a "$LOG"; r=${PIPESTATUS[0]}
import json, os, sys
run, refs = sys.argv[1], sys.argv[2].split(",")
keys = [l.strip() for l in open("results/r3check_keys.txt") if l.strip()]
def rows(v):
    p = os.path.join("results", v, "decode.jsonl")
    return {r["key"]: r for r in (json.loads(l) for l in open(p)) if "key" in r} if os.path.exists(p) else {}
got = rows(run)
if not all(k in got for k in keys):
    print(f"{run}: rows missing: {[k for k in keys if k not in got]}"); sys.exit(3)
first = None
for ref in refs:
    want = rows(ref)
    bad = [k for k in keys if k not in want or want[k]["completion_ids"] != got[k]["completion_ids"]]
    print(f"{run} vs {ref}: " + (f"DIFFERENT on {bad}" if bad else f"token-identical on {', '.join(keys)}")
          + f" ({sum(len(got[k]['completion_ids']) for k in keys)} tokens)")
    first = bad if first is None else first
sys.exit(5 if first else 0)
EOF
say "r3check $RUN end (rc $r)"
exit "$r"
