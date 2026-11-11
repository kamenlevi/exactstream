#!/bin/bash
# THE records3 CONVERSION A/B NIGHT (docs/PLAN_records3_convert.md section 4, pre-registered with the plan 1336043; M-27b's
# line in ~/NewIdeas/QUEUE.md). The orchestrator 11-03 22:4x adopted EvalSuite's proposal + BigMoE's bracketed order: the
# night does the conversion itself and ENDS ROW-MAJOR (T0's leftover and FLOOR night 2 read the row-major store; the
# permanent conversion after a PASS waits for T0_DONE; EvalSuite's records3 checks are NOT in this night).
# One quiet ticket, the same C for every arm (AB_C, default 224), 40 prompts (chat23's 30, which hold S45's six, + 10 more
# HumanEval), CAP 256, every token vs data/ab/ref_tokens.json:
#   abA    row-major, M-17 sizes
#   conv   md5 of the 48 originals -> data/records3_md5_rowmajor.txt, then Leap's convert_records_p11.py --pad
#   abB    native+p11+pad, M-17 sizes         abB2  native+p11+pad, MoonShots' packed sizes (data/seed45/packed45_C.json)
#   abB2z  abB2 + zero-copy reads (BIGMOE_RECORDS_DIRECT: aligned pieces straight into the slots; MoonShots' M-29)
#   inv    --inverse --md5: all 48 files must match the originals
#   abA2   row-major, M-17 sizes (A brackets B)
# If anything fails between conv and inv, the script still runs the inverse, so the store ends row-major.
# AB_T0GO=1: touch ~/EvalSuite/results/t0/GO after abA2, once the store is row-major (EvalSuite, if T0's leftover follows).
# Preconditions: ~/EvalSuite/results/RELAX_DONE, no T0 chunk running (the tool's lsof check: no records3 file open), the
# store row-major, the 1 h heads-up sent when the night was armed, Leap >= 90456ce, the live bigmoe/ with the layout patch.
# Lines (scripts/ab_records3_report.py): mine = PASS if every token equals the references in every arm AND abB2's code SSD
# MB per kept token <= -10 % vs A (the mean of abA / abA2) AND abB2's steady tok/s not below A's; KILL if any token differs
# or code MB per kept token > -3 %; else BETWEEN. M-27b = PASS if tokens are bitwise AND code MB/kept <= -10 % AND median
# steady >= +3 %; KILL if any token differs, > -5 % or < +1 %.
# usage: inside turn45 (HAVE_LOCK=1 KEEP_LOCK=1), with frozen copies (data/r57/snap/*_ab.*); CLAUDE_PID set
export PATH="$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:$PATH"
cd "$HOME/BigMoE" || exit 1
mkdir -p data/ab
LOG="$HOME/BigMoE/data/ab/ab.log"
say() { echo "$(date '+%m-%d %H:%M:%S') $*" >> "$LOG"; }
RUN="/bin/bash data/r57/snap/c45_runner_ab.sh"
export C45_SCREEN=data/r57/snap/r57_screen_ab.py C45_DEADLINE=${AB_DEADLINE:-2400} C45_OUT="$HOME/BigMoE/data/ab"
CC=${AB_C:-224}
R3=models/bare/records3
PY="$HOME/BigMoE/.venv/bin/python"
TOOL="$HOME/Leap/scripts/convert_records_p11.py"
LOCKF="$HOME/.claude-peers/locks/${CLAUDE_PID}.json"
MD5A=data/records3_md5_rowmajor.txt; MD5B=data/records3_md5_p11.txt
keepalive() { "$PY" -I -S -c '
import json, os, sys, time
p = sys.argv[1]
d = json.load(open(p))
if d.get("mode") != "quiet" or d.get("until", 0) <= time.time(): sys.exit(1)
d["active_at"] = time.time(); t = "%s.tmp%d" % (p, os.getpid()); json.dump(d, open(t, "w")); os.replace(t, p)' "$LOCKF"; }
layout() { "$PY" -I -c "
import json; ls = {r.get('layout', 'row-major') for r in json.load(open('$R3/index.json'))['layers'].values()}
print(ls.pop() if len(ls) == 1 else 'mixed')"; }
K="ifeval/1220,ifeval/1561,ifeval/3280,multi/ar21,multi/ar22,multi/bg0,multi/bg2,multi/de3,multi/de4,multi/es10,multi/es11"
K="$K,multi/fr7,multi/fr8,multi/ja18,multi/ja19,multi/ru12,multi/ru13,multi/zh15,multi/zh16,agent/bash17,agent/edit00"
K="$K,agent/text23,agent/write11,agent/bash18,humaneval/HumanEval/68,humaneval/HumanEval/91,math500/test/prealgebra/1640.json"
K="$K,math500/test/algebra/694.json,math500/test/algebra/1282.json,gsm8k/708,humaneval/HumanEval/16,humaneval/HumanEval/6"
K="$K,humaneval/HumanEval/79,humaneval/HumanEval/19,humaneval/HumanEval/161,humaneval/HumanEval/17,humaneval/HumanEval/56"
K="$K,humaneval/HumanEval/141,humaneval/HumanEval/101,humaneval/HumanEval/63"
REF="$HOME/BigMoE/data/ab/ref_tokens.json"
PK="$HOME/BigMoE/data/seed45/packed45_$CC.json"
arm() {  # arm TAG [ENV...]: one arm on the store as it is
  local tag=$1; shift
  R57_KEYS=$K MG=${AB_MG:-6} C45_ALARM=900 C45_REFTOK="$REF:gate+t" C45_ARMS="${tag}_$CC:$CC${1:+:$1}" $RUN
  say "$tag rc=$? (store $(layout))"
}
back_to_rowmajor() {  # the inverse with the md5 check; the store must end row-major
  keepalive
  say "inverse (--md5 $MD5A)"
  "$PY" "$TOOL" "$R3" --inverse --md5 "$MD5A" --yes >> "$LOG" 2>&1; local rc=$?
  if [ $rc = 0 ] && [ "$(layout)" = row-major ] && tail -20 "$LOG" | grep -q "md5 after --inverse: 48 of 48 files match"; then
    say "store back to row-major, md5s == the originals"; return 0
  fi
  say "INVERSE PROBLEM (rc $rc, store $(layout)): look before anything else touches records3"; return 4
}
say "A/B night start (pid $$, C $CC, store $(layout))"
[ "$(layout)" = row-major ] || { say "the store is not row-major at the start: STOP"; exit 2; }
[ -e "$HOME/EvalSuite/results/RELAX_DONE" ] || { say "no RELAX_DONE: STOP"; exit 2; }
[ -s "$REF" ] && [ -s "$PK" ] || { say "missing $REF or $PK: STOP"; exit 2; }
git -C "$HOME/Leap" merge-base --is-ancestor 7624eb0 HEAD 2>/dev/null || { say "Leap is older than 7624eb0 (P11PAD): STOP"; exit 2; }
"$PY" -I -c "import sys; sys.path.insert(0, '$HOME/BigMoE'); import bigmoe.tier as T; assert hasattr(T, 'LAYOUTS')" 2>> "$LOG" \
  || { say "the live bigmoe/ has no layout patch: STOP"; exit 2; }
[ -e "$MD5A" ] && mv "$MD5A" "$MD5A.$(date +%m%d%H%M).old"     # each night writes its own list of the originals
arm abA
keepalive || { say "lock gone before the conversion: STOP (store row-major)"; exit 3; }
say "convert (md5 of the originals first)"
if "$PY" "$TOOL" "$R3" --pad --md5 "$MD5A" --yes >> "$LOG" 2>&1 && [ "$(layout)" = native+p11+pad ]; then
  (cd "$R3" && md5 -r L*.bin) > "$MD5B"
  say "converted; md5s of the converted files -> $MD5B"
  keepalive
  arm abB
  arm abB2 "BIGMOE_TIER_SIZES=$PK"
  arm abB2z "BIGMOE_TIER_SIZES=$PK,BIGMOE_RECORDS_DIRECT=1"   # zero-copy: the 4 KB-aligned pieces preadv straight into the slots
else
  say "the conversion stopped (store $(layout)): skipping B / B2"
fi
back_to_rowmajor || exit 4
keepalive
arm abA2
if [ -n "$AB_T0GO" ] && [ "$(layout)" = row-major ]; then
  touch "$HOME/EvalSuite/results/t0/GO" && say "touched ~/EvalSuite/results/t0/GO (store row-major)"
fi
"$PY" scripts/ab_records3_report.py "$CC" >> "$LOG" 2>&1; say "report rc=$? (data/ab/ab_report.txt)"
say "A/B night end (store $(layout))"
