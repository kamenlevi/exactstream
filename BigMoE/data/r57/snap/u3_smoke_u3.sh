#!/bin/bash
# U3 = Q1-UNION 0.9 + 3-bit sure rows (the map's q1u_cf09_t3, PASS on code / IFEval / multi; the orchestrator 11-03 11:01: "a
# per-row 'no LSB' mask in the 3c GLU / OOO tier call, so drafter-sure rows read MSB-3 only"; FAST, quality by EvalSuite's
# RELAX t3 after Thu). The engine's form (scripts/r57_screen.py R57_FAST=U3 / U3A): rows the drafter is >= 0.9 sure of keep
# ranks 0-1 at 3 bits (their sidecar request is an unsure row's rank-0 expert, so no LSB read; LP1's kernel test runs them
# at tier 3) and ranks 2-9 only where the expert is in the check's union; unsure rows and the bonus row exact.
# Arms (one ticket, 12 HumanEval prompts, CAP 256, C ${U3_C:-224}):
#   u3c   the default (control)               u3n  the null control: U3 with no sure rows (R57_U3_THR=2, R57_U3_FORCE=1)
#   u3    U3 on code                           (the copied OOO path + LP1 kernels must give REF2-T's tokens exactly)
# Line (pre-registered 11-03 15:5x): u3n == REF2-T on every token (else the copy is wrong: stop); u3: SSD MB per kept token
# <= -15 % vs u3c on code = PASS (a byte saving worth EvalSuite's quality run), > -5 % = KILL, else BETWEEN; tok/s and the
# sure-row share beside it. Tokens of u3 differ by design (FAST).
# usage: inside turn45 (HAVE_LOCK=1 KEEP_LOCK=1), with frozen copies (data/r57/snap/*_u3.*)
export PATH="$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:$PATH"
cd "$HOME/BigMoE" || exit 1
mkdir -p data/u3
LOG="$HOME/BigMoE/data/u3/u3.log"
say() { echo "$(date '+%m-%d %H:%M:%S') $*" >> "$LOG"; }
RUN="/bin/bash data/r57/snap/c45_runner_u3.sh"
export C45_SCREEN=data/r57/snap/r57_screen_u3.py C45_DEADLINE=${U3_DEADLINE:-2350} C45_OUT="$HOME/BigMoE/data/u3"
CC=${U3_C:-224}
K="humaneval/HumanEval/68,humaneval/HumanEval/16,humaneval/HumanEval/91,humaneval/HumanEval/6,humaneval/HumanEval/79"
K="$K,humaneval/HumanEval/19,humaneval/HumanEval/161,humaneval/HumanEval/17,humaneval/HumanEval/56,humaneval/HumanEval/141"
K="$K,humaneval/HumanEval/101,humaneval/HumanEval/63"
say "U3 smoke start (pid $$, C $CC)"
R57_KEYS=$K MG=${U3_MG:-6} C45_ALARM=900 C45_REFTOK="$HOME/BigMoE/data/r57/chat23_ref_tokens.json:gate+t" C45_ARMS="u3c_$CC:$CC u3n_$CC:$CC:R57_FAST=U3,R57_U3_THR=2,R57_U3_FORCE=1" $RUN
say "controls rc=$?"
# the gate (the orchestrator 11-03 15:5x): u3n must equal u3c (the default = REF2-T) on every token of every prompt (most of
# these prompts have no stored chat23 reference, so the default run beside it is the reference); else U3 stops here
if "$HOME/BigMoE/.venv/bin/python" -I -c "
import json, sys
a = json.load(open('$HOME/BigMoE/data/r57/u3c_${CC}_tokens.json')); b = json.load(open('$HOME/BigMoE/data/r57/u3n_${CC}_tokens.json'))
bad = [k for k in a if a[k].get('gate+t') != b.get(k, {}).get('gate+t')]
print('u3n vs u3c: %d prompts, differ: %s' % (len(a), bad)); sys.exit(1 if bad or len(a) < 12 else 0)" >> "$LOG" 2>&1; then
  R57_KEYS=$K MG=${U3_MG:-6} C45_ALARM=900 C45_ARMS="u3_$CC:$CC:R57_FAST=U3,R57_REFTOK=" $RUN
  say "U3 rc=$?"
else
  say "u3n differs from u3c (REF2-T) or a control is missing: U3 stopped (the copied OOO path is not exact)"
fi
say "U3 smoke end"
