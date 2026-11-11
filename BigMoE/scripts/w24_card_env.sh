#!/bin/bash
# Mon 9 Nov 19:xx. Writes data/stack/card_w24_env.txt for EvalSuite's 24 GB card gate (runner 61 at C 100, its 18:59 format:
# like card_lcoh_env.txt; comment = w24 + C + MEM 24 + the screen path; env = the arm's keys + R57_QOM_SETS with lcb6, last).
# The arm's keys come from scripts/w24.sh's own armenv table, so the gated env is the measured one. R57_HARD=1200: the
# gate's processes run twice as long before a restart (EvalSuite's stop must be >= 1260 s). Does not commit.
# usage: bash scripts/w24_card_env.sh w12|w16u|w12pf|w10|w8|w12s
cd "$HOME/BigMoE" || exit 1
ARM=${1:?arm}
OUT=${W24_CARD_OUT:-data/stack/card_w24_env.txt}   # W24_CARD_OUT: a test path (EvalSuite starts on the committed file)
eval "$(sed -n '/^armenv() {/,/esac; }/p' scripts/w24.sh)"
read -r cm cl rw ex <<< "$(armenv "$ARM")" || { echo "unknown arm $ARM"; exit 2; }
LC=$(grep -v '^#' data/stack/lcoh_env.txt | grep . | tail -1 | sed 's/,R57_QOM_SETS=humaneval,mbpp$//')
echo "$LC" | grep -q 'REF2_NAX_COH=1' || { echo "lcoh_env.txt's line has no REF2_NAX_COH=1"; exit 2; }
SNAP=$(git log -1 --format=%h -- data/r57/snap/w24 bigmoe)
git diff --quiet HEAD -- data/r57/snap/w24 bigmoe scripts/w24.sh || { echo "w24 snapshot / bigmoe / w24.sh not committed"; exit 2; }
X="R57_VTIER=,BIGMOE_TIER_CM=$cm,BIGMOE_TIER_CL=$cl,R57_ROWCAP=$rw,REF2_TIER_CHUNK=1${ex:+,$ex},R57_HARD=1200"
# W24c (Mon 23:1x): a key the arm sets (e.g. R57_D7B=all over the L + COH line's R57_D7B=code) leaves the L + COH part, so
# every key appears once whatever the reader does with repeats (runner 61's parse_env keeps order: the last would win)
LC=$(python3 - "$LC" "$X" <<'PY'
import sys
def toks(line):                       # runner 61's parse_env: a token without '=' continues the previous value
    out = []
    for t in line.split(","):
        if "=" in t or not out:
            out.append(t)
        else:
            out[-1] += "," + t
    return out
xk = {t.split("=", 1)[0] for t in toks(sys.argv[2])}
print(",".join(t for t in toks(sys.argv[1]) if t.split("=", 1)[0] not in xk))
PY
)
REP=${W24_CARD_REPORT:-data/w24/w24_report.txt}   # W24c: the report the arm was measured in; W24_CARD_NOTE: appended text
case "$ex" in *U3W*) RULE="U3W (U3 at thr 0 on humaneval, mbpp, lcb6: ranks 0-1 + the check's union on sure rows, no optional uses; 16-row checks admitted by R57_GATEREQ)";;
  *) RULE="QOM as shipped (L + COH: optional uses served only when resident) on humaneval, mbpp + lcb6";; esac
{
  echo "# CARD w24 = arm $ARM of W24 ($REP): L + COH below 24 GB on the fast path, for EvalSuite's card gates (IFBench / GPQA diamond / LCB v6, runner 61): Sat 100, C 100, MEM 24 GB, screen data/r57/snap/w24/r57_screen_w24.py (BigMoE snapshot + bigmoe/ at $SNAP; rel1's screen + R57_ROWCAP / R57_GATEREQ / U3W / R57_HARD). Pools: $cm base + $cl sidecar slots ($(awk -v m=$cm -v l=$cl 'BEGIN {printf "%.2f", m * 0.0991 + l * 0.0346}') GB), R57_VTIER off, rows <= $rw where the rule caps them. Code rule: $RULE. No QOM / U3 on ifbench, gpqad (exact expert reads there). ${W24_CARD_NOTE:+ $W24_CARD_NOTE.} One comma-separated env line:"
  echo "$LC,$X,R57_QOM_SETS=humaneval,mbpp,lcb6"
} > "$OUT"
cut -c1-400 "$OUT"
echo "keys: $(sed -n 2p "$OUT" | tr ',' '\n' | grep -c =) -> $OUT"
