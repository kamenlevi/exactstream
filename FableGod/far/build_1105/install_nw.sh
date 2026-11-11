#!/bin/bash
# NEVER-WAIT into the decision map (Fable FAR's build ~/FableGod/far/build_1105 for NEVERWAIT_TEST.md v2, Thu 2026-11-05;
# run by the map's owner, Breakthrough, or the orchestrator). In Breakthrough's installers' style (install_codeonly.sh,
# install_c3.sh): everything is patched and tested on scratch copies first, then moved in atomically.
#   1. the build folder equals its MD5SUMS (the files that were tested and reviewed);
#   2. scratch copies of the map's dm_common.py / dm_ablations.py, patched by nw_patch.py (anchored edits; a file that
#      already holds the patch is left as it is);
#   3. the CPU self-test on them: test_nw.py --base on the unpatched copies (stores their outputs), then test_nw.py on
#      the patched ones (ALL OK needed: the old arms bit-equal, the three self-checks, the reference forward, measure()),
#      in float64 and again in bfloat16 (the 125B's dtype: the exact checks);
#   4. the masks snapshotted into <map>/out/far/<arm> (a fixed copy, so a later change in the build folder cannot change
#      what runs; an older differing snapshot is renamed aside, nothing is deleted), md5s against MD5SUMS;
#   5. check_install.py on the patched copies: every arm resolves and its 36 masks load through dm_common._nw_late;
#   6. moved in with no map process running and the map's files unchanged since the copy. The old files stay beside
#      them as dm_common.py.pre_nw / dm_ablations.py.pre_nw (written once).
# Exit 0 = in (or already in), 1 = failed (the map's .py files unchanged), 2 = a map process is running (unchanged).
# CPU only (MLX's CPU device), < 1 GB, about 15 s. No GPU, no model load.
# NW_DM=<dir>: install into another copy of the map (the build's own dry run). NW_SP=<dir>: the scratch root; each
# run works in a new <scratch root>/run.<time> folder and deletes nothing.
B=$(cd "$(dirname "$0")" && pwd); DM=${NW_DM:-$HOME/FableGod/decision_map}; PY=$HOME/BigMoE/.venv/bin/python
SNAP=$DM/out/far; SP=${NW_SP:-$HOME/Breakthrough/data/abl_scratch/nw}
pgrep -f "run_ma[p].py" > /dev/null && { echo "nw: a map process is running; unchanged"; exit 2; }
(cd "$B" && md5 -r $(awk '{print $2}' MD5SUMS) | diff -q - MD5SUMS > /dev/null) || { echo "nw: the build folder does not match its MD5SUMS; unchanged"; exit 1; }
SP=$SP/run.$(date +%Y%m%d-%H%M%S); mkdir -p "$SP/base" "$SP/new" "$SP/test" "$SNAP" \
  && cp "$DM/dm_common.py" "$DM/dm_ablations.py" "$SP/base/" || { echo "nw: scratch copy failed; unchanged"; exit 1; }
already=0; grep -q "NEVER-WAIT" "$SP/base/dm_common.py" && grep -qF 'def nw_mask(' "$SP/base/dm_ablations.py" && already=1
NW_SNAP=$SNAP $PY -I -B "$B/nw_patch.py" "$SP/base" "$SP/new" || { echo "nw: patch FAILED; unchanged"; exit 1; }
for f in dm_t3.py dm_gate.py to2bit.py run_map.py records3_index_rowmajor.json; do [ -e "$SP/new/$f" ] || ln -sf "$DM/$f" "$SP/new/$f"; done
if [ $already = 0 ]; then
  NW_TEST_SCRATCH=$SP/test $PY -I -B "$B/test_nw.py" --base "$SP/base" > "$SP/test_base.log" 2>&1 \
    || { echo "nw: the base run of the self-test FAILED ($SP/test_base.log); unchanged"; exit 1; }
  NW_SNAP=$SNAP NW_TEST_SCRATCH=$SP/test $PY -I -B "$B/test_nw.py" "$SP/new" > "$SP/test_nw.log" 2>&1 \
    || { echo "nw: the CPU self-test FAILED ($SP/test_nw.log); unchanged"; grep -E "FAIL|Error" "$SP/test_nw.log" | head -5; exit 1; }
  echo "nw: CPU self-test $(tail -1 "$SP/test_nw.log") ($(grep -c '^  ok' "$SP/test_nw.log") checks; $SP/test_nw.log)"
  NW_TEST_DTYPE=bfloat16 NW_TEST_SCRATCH=$SP/test $PY -I -B "$B/test_nw.py" --base "$SP/base" > "$SP/test_base_bf16.log" 2>&1 \
    && NW_TEST_DTYPE=bfloat16 NW_SNAP=$SNAP NW_TEST_SCRATCH=$SP/test $PY -I -B "$B/test_nw.py" "$SP/new" > "$SP/test_nw_bf16.log" 2>&1 \
    || { echo "nw: the CPU self-test in bfloat16 FAILED ($SP/test_nw_bf16.log); unchanged"; grep -E "FAIL|Error" "$SP/test_nw_bf16.log" | head -5; exit 1; }
  echo "nw: the same in bfloat16 (the 125B's dtype; exact checks) $(tail -1 "$SP/test_nw_bf16.log") ($(grep -c '^  ok' "$SP/test_nw_bf16.log") checks)"
else echo "nw: the map already holds the patch; checking the snapshots and the arms only"; fi
for d in "$B"/masks/*/; do
  a=$(basename "$d")
  if [ -d "$SNAP/$a" ] && diff -q <(cd "$SNAP/$a" && md5 -r *.npz) <(cd "$d" && md5 -r *.npz) > /dev/null; then continue; fi
  t=$(mktemp -d "$SNAP/.$a.new.XXXX") && cp -R "$d." "$t/" || { echo "nw: snapshot of $a failed"; exit 1; }
  [ -e "$SNAP/$a" ] && mv "$SNAP/$a" "$SNAP/.$a.old.$(date +%s)"   # nothing deleted: a differing older snapshot is renamed aside
  mv "$t" "$SNAP/$a" || { echo "nw: snapshot of $a failed"; exit 1; }
done
grep ' masks/.*\.npz$' "$B/MD5SUMS" | sed 's# masks/# #' > "$SNAP/MD5SUMS"
(cd "$SNAP" && md5 -r $(awk '{print $2}' MD5SUMS) | diff -q - MD5SUMS > /dev/null) || { echo "nw: the snapshots do not match the build's md5s; the map's .py files are unchanged"; exit 1; }
echo "nw: $(ls -d "$SNAP"/*/ | wc -l | tr -d ' ') mask folders in $SNAP, md5s equal the build's"
NW_SNAP=$SNAP NW_TEST_SCRATCH=$SP/test $PY -I -B "$B/check_install.py" "$SP/new" "$SNAP" > "$SP/check_install.log" 2>&1 \
  || { echo "nw: the install check FAILED ($SP/check_install.log); unchanged"; grep FAIL "$SP/check_install.log" | head -5; exit 1; }
echo "nw: install check $(tail -1 "$SP/check_install.log") ($SP/check_install.log)"
[ $already = 1 ] && { echo "nw: already in"; exit 0; }
pgrep -f "run_ma[p].py" > /dev/null && { echo "nw: a map process appeared; unchanged"; exit 2; }
cmp -s "$SP/base/dm_common.py" "$DM/dm_common.py" && cmp -s "$SP/base/dm_ablations.py" "$DM/dm_ablations.py" \
  || { echo "nw: the map's files changed since the copy; unchanged (run again)"; exit 1; }
[ -e "$DM/dm_common.py.pre_nw" ] || cp -p "$DM/dm_common.py" "$DM/dm_common.py.pre_nw"
[ -e "$DM/dm_ablations.py.pre_nw" ] || cp -p "$DM/dm_ablations.py" "$DM/dm_ablations.py.pre_nw"
cp "$SP/new/dm_common.py" "$DM/.dm_common.py.new" && cp "$SP/new/dm_ablations.py" "$DM/.dm_ablations.py.new" \
  && mv "$DM/.dm_common.py.new" "$DM/dm_common.py" && mv "$DM/.dm_ablations.py.new" "$DM/dm_ablations.py" \
  && echo "nw: installed (dm_common.py $(md5 -q "$DM/dm_common.py"), dm_ablations.py $(md5 -q "$DM/dm_ablations.py"); the old files are *.pre_nw)"
