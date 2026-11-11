#!/bin/bash
# The ONE in-place records3 conversion (Kamen approved Tue 3 Nov ~17:3x via the orchestrator; docs/PLAN_records3_convert.md):
# models/bare/records3 row-major -> native+p11+pad (4 KB-aligned pieces, Tue 22:5x) with Leap's ~/Leap/scripts/convert_records_p11.py (90456ce; its output ==
# BigMoE's scripts/r3layout_ref.py byte for byte on L0 / L20 / L47). Run BY HAND, never armed, and only when all of these hold
# (the script refuses otherwise):
#   - ~/EvalSuite/results/RELAX_DONE exists (the orchestrator 11-03 17:2x), and runner 42 ended with a NAX6 GO:
#     results/NAX6_DONE reads "D4X: PASS" or "D4X: DIFFERS (HIGHER)" (a set gains, p < 0.05, nothing fails: GO like PASS,
#     the orchestrator Wed 19:2x decision (a), as runners 43 / 44 read it; Kamen may veto that before Thu 10:00), and
#     results/CHECK1_DONE (records3 check 1, row-major) reads "check 1 rc 0 ...".
#     (Wed 19:1x, Leap: EvalSuite's Wed 18:05 order runs T0 AFTER the conversion and T0 reads records3m only, so the old
#     T0_DONE precondition would have refused Thu ~10:55; Kamen approved converting on the NAX6 PASS);
#   - the 1 h heads-up went out (peer say to all + the orchestrator): CONV_HEADSUP=<epoch seconds of it>, >= 60 min ago;
#   - this session holds a quiet lock (the conversion moves ~200 GB through the SSD in ~5 min: nobody measures meanwhile);
#   - nothing holds a records3 file open, free disk >= 112 GB + one layer (the tool checks both again).
# Steps: (1) read-only --check: every s3 / b3 / b piece of every layer packs; (2) the tool writes the md5 of all 48 row-major
# files to data/records3_md5_rowmajor.txt, then converts layer by layer (temp + fsync + re-read + inverse compare + atomic
# rename + index entry; the "rowmajor" entry is kept for --inverse); (3) md5 of the converted files ->
# data/records3_md5_p11.txt (the A/B night's re-conversion must match it); (4) the index check: every layer native+p11 with
# its "rowmajor" entry, bigmoe.tier.load_index refuses without layout_ok. Back: `convert_records_p11.py
# models/bare/records3 --inverse --md5 data/records3_md5_rowmajor.txt --yes` (the md5s must then match).
# usage: CONV_HEADSUP=<epoch> bash scripts/convert_records3.sh [--check-only]
# Test mode: CONV_MODEL=<a model dir under ~/BigMoE/scratch> skips the EvalSuite / heads-up / lock preconditions only there,
# and writes data/scratch_records3_md5_*.txt.
export PATH="$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:$PATH"
cd "$HOME/BigMoE" || exit 1
MODEL=${CONV_MODEL:-models/bare}
R3=$MODEL/records3
TEST=0; case "$(cd "$MODEL" 2>/dev/null && pwd -P)" in "$HOME/BigMoE/scratch/"*) TEST=1 ;; esac
P=$([ $TEST = 1 ] && echo scratch_)
MD5A=data/${P}records3_md5_rowmajor.txt; MD5B=data/${P}records3_md5_p11.txt
LOG=data/${P}conversion.log
PY=.venv/bin/python
TOOL="$HOME/Leap/scripts/convert_records_p11.py"
say() { echo "$(date '+%m-%d %H:%M:%S') $*" | tee -a "$LOG"; }
die() { say "REFUSED: $*"; exit 2; }
[ $TEST = 1 ] && say "TEST MODE on $MODEL (a scratch dir): EvalSuite / heads-up / lock preconditions skipped"
if [ $TEST = 0 ]; then
[ -e "$HOME/EvalSuite/results/RELAX_DONE" ] || die "no ~/EvalSuite/results/RELAX_DONE"
NAX=$(head -1 "$HOME/EvalSuite/results/NAX6_DONE" 2>/dev/null); CK1=$(head -1 "$HOME/EvalSuite/results/CHECK1_DONE" 2>/dev/null)
case "$NAX" in "D4X: PASS"|"D4X: DIFFERS (HIGHER)") ;; *) die "runner 42's NAX6 verdict is not a GO: '${NAX:-no ~/EvalSuite/results/NAX6_DONE}' (GO = PASS or DIFFERS (HIGHER); FAIL / VOID / none = no conversion)" ;; esac
case "$CK1" in "check 1 rc 0 "*) ;; *) die "records3 check 1 (row-major) did not pass: '${CK1:-no ~/EvalSuite/results/CHECK1_DONE}'" ;; esac
[ -n "$CONV_HEADSUP" ] && [ $(( $(date +%s) - CONV_HEADSUP )) -ge 3600 ] || die "the 1 h heads-up (CONV_HEADSUP=epoch) is missing or < 60 min old"
$PY -I -S -c 'import json, sys, time; d = json.load(open(sys.argv[1])); sys.exit(0 if d.get("mode") == "quiet" and d.get("until", 0) > time.time() + 600 else 1)' \
  "$HOME/.claude-peers/locks/${CLAUDE_PID:-none}.json" 2>/dev/null || die "no quiet lock with >= 10 min left for CLAUDE_PID=${CLAUDE_PID:-unset}"
fi
git -C "$HOME/Leap" merge-base --is-ancestor 7624eb0 HEAD 2>/dev/null || die "~/Leap is not at 7624eb0 (P11PAD) or later"
[ -n "$(lsof +D "$R3" 2>/dev/null)" ] && die "records3 files are open: $(lsof +D "$R3" 2>/dev/null | awk 'NR>1 {print $1"("$2")"}' | sort -u | tr '\n' ' ')"
say "preflight OK ($([ $TEST = 1 ] && echo test mode || echo "RELAX_DONE, $NAX, check 1 rc 0, heads-up $(( ($(date +%s) - CONV_HEADSUP) / 60 )) min ago"), nothing open); --check"
CK=$($PY "$TOOL" "$R3" 2>&1); echo "$CK" >> "$LOG"   # the check mode always exits 1 ("read-only check done"): read its lines
echo "$CK" | grep -q '^check: 0 scale pieces' && echo "$CK" | grep -q 'read-only check done' || die "the tool's read-only check failed (see $LOG)"
[ "$1" = "--check-only" ] && { say "check only: done"; exit 0; }
[ -e $MD5A ] && die "$MD5A exists: a conversion ran before; look first"
# Overlay stores (Leap, Wed 19:0x): a dir of symlinks into the real records3 with its OWN index.json keeps those layers
# marked row-major, so after the conversion a load there decodes converted bytes as row-major with no error. Before the
# first byte moves: mine (~/BigMoE/scratch/*/records3, e.g. model_mix) get their index renamed (loads fail loudly; mv back
# to undo); other sessions' overlays are listed here and in the DONE line for their owners (not touched).
OVL=$(find "$HOME/FableGod" "$HOME/Breakthrough" "$HOME/NewIdeas" "$HOME/Leap" "$HOME/MoonShots" "$HOME/TrainLab" \
  "$HOME/EvalSuite" "$HOME/BigMoE/scratch" -maxdepth 4 -type l -lname "*/BigMoE/models/bare/records3/L*" 2>/dev/null \
  | xargs -n1 dirname 2>/dev/null | sort -u)
FOREIGN=""
for d in $OVL; do
  case "$d" in
    "$HOME/BigMoE/scratch/"*) [ $TEST = 0 ] && [ -e "$d/index.json" ] && mv "$d/index.json" "$d/index.json.rowmajor-stale" \
      && say "overlay $d: index.json -> index.json.rowmajor-stale (loads there now fail loudly)" ;;
    *) FOREIGN="$FOREIGN $d"
       say "overlay $d (another session's) links into records3 with its own index: its owner must re-index or delete it" ;;
  esac
done
say "converting (md5 of the row-major files first)"
t0=$(date +%s)
$PY "$TOOL" "$R3" --pad --md5 $MD5A --yes >> "$LOG" 2>&1 || die "the conversion stopped (rc $?): the store may be mixed; re-run resumes, or --inverse"
say "converted in $(( $(date +%s) - t0 )) s; md5 of the converted files"
(cd "$R3" && md5 -r L*.bin) > $MD5B
$PY - "$MODEL" >> "$LOG" 2>&1 <<'EOF' || die "index check failed"
import json, os, sys
sys.path.insert(0, os.environ.get("CONV_BIGMOE") or os.path.expanduser("~/BigMoE"))   # (test: the staged patch)
ix = json.load(open(os.path.join(sys.argv[1], "records3", "index.json")))["layers"]
bad = [L for L, r in ix.items() if r.get("layout") != "native+p11+pad" or "rowmajor" not in r or r["recsz"] != 2785280]
assert not bad, f"layers not native+p11+pad / without their rowmajor entry: {bad}"
import bigmoe.tier as T
try:
    T.load_index(sys.argv[1]); raise SystemExit("load_index took a converted store")
except RuntimeError:
    pass
print(f"index OK: {len(ix)} layers native+p11+pad, rowmajor entries kept; load_index refuses row-major readers")
EOF
say "DONE: $R3 is native+p11+pad (md5s: $MD5A before, $MD5B after). Tell all sessions.${FOREIGN:+ OVERLAYS TO FIX (other sessions):$FOREIGN}"
