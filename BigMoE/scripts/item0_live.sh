#!/bin/bash
# WED_TESTS item 0, pulled forward to ~09:00 (the orchestrator Wed 06:2x; Fit12 06:2x: its 13:05 runs load neither
# ~/BigMoE/bigmoe nor ~/BigMoE/ext/build, only the .venv and models/bare, which this does not touch): the records3 layout
# patch goes live in ~/BigMoE/bigmoe/ and ext/build is rebuilt from the current bigmoe_ext.cpp. The patch
# (patches/records3_layout.diff: tier.py + stream.py) is a no-op on today's row-major store (Breakthrough's layout check
# PASS Wed 03:28). Each step must pass, or the live files go back:
#   1. no process has a ~/BigMoE ext build loaded (lsof), so none has bigmoe/ imported (tier.py loads the ext);
#   2. tier.py / stream.py md5 == scratch/stage_layout/base_md5.txt; copies -> scratch/stage_layout/pre_live/;
#   3. patch -p0 (a dry run first); the patched files == the staged scratch/stage_layout/bigmoe/ ones (md5);
#   4. test_layout_engine.py on the LIVE package (ext build_w1: the current source, with set_direct_split, which the patched tier.py needs for native+p11+pad; build_nat lacks it, Wed 07:41 rc 3); ext/build rebuilt (cmake, -j 4, background QoS; the old
#      .so kept in pre_live/); the test again on ext/build, and its layout reader present (layout_max 2, mix_e2e's check);
#   5. the log gets the .so sha256 prefix and Leap's HEAD for EvalSuite. The commit is made by hand after a look.
# usage: bash scripts/item0_live.sh   (outside any run of this session) -> data/item0.log; rc 0 = live, 2 = not started
# (nothing changed), 3 = patch / test failed (files restored), 4 = the ext/build rebuild failed (the patch stays live)
export PATH="$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:$PATH"
cd "$HOME/BigMoE" || exit 1
LOG="$HOME/BigMoE/data/item0.log"
say() { echo "$(date '+%m-%d %H:%M:%S') $*" | tee -a "$LOG"; }
PY="$HOME/BigMoE/.venv/bin/python"
ST=scratch/stage_layout
SO=bigmoe_ext.cpython-312-darwin.so
loaded() { lsof -nP 2>/dev/null | grep -F "$HOME/BigMoE/ext/build" | awk '{print $1"("$2")"}' | sort -u | tr '\n' ' '; }
restore() { cp "$ST/pre_live/tier.py" bigmoe/tier.py && cp "$ST/pre_live/stream.py" bigmoe/stream.py \
  && say "RESTORED the pre-live tier.py / stream.py (md5 $(md5 -q bigmoe/tier.py | cut -c1-8) / $(md5 -q bigmoe/stream.py | cut -c1-8))"; }
say "item 0 start (pid $$)"
U=$(loaded); [ -z "$U" ] || { say "NOT STARTED: a ~/BigMoE ext build is loaded by: $U"; exit 2; }
want_t=$(awk '$2 == "tier.py" {print $1}' $ST/base_md5.txt); want_s=$(awk '$2 == "stream.py" {print $1}' $ST/base_md5.txt)
[ -n "$want_t" ] && [ "$(md5 -q bigmoe/tier.py)" = "$want_t" ] && [ "$(md5 -q bigmoe/stream.py)" = "$want_s" ] \
  || { say "NOT STARTED: the live tier.py / stream.py differ from the patch base ($ST/base_md5.txt)"; exit 2; }
git -C "$HOME/Leap" merge-base --is-ancestor 7624eb0 HEAD 2>/dev/null && git -C "$HOME/Leap" merge-base --is-ancestor 90456ce HEAD 2>/dev/null \
  || { say "NOT STARTED: ~/Leap lacks 7624eb0 (P11PAD) or 90456ce (the ref2 router)"; exit 2; }
patch -p0 --dry-run < patches/records3_layout.diff >> "$LOG" 2>&1 || { say "NOT STARTED: the patch does not apply (dry run)"; exit 2; }
mkdir -p $ST/pre_live && cp bigmoe/tier.py bigmoe/stream.py $ST/pre_live/ && cp ext/build/$SO $ST/pre_live/build_$SO || exit 2
patch -p0 < patches/records3_layout.diff >> "$LOG" 2>&1 || { say "the patch failed"; restore; exit 3; }
for f in tier.py stream.py; do
  [ "$(md5 -q bigmoe/$f)" = "$(md5 -q $ST/bigmoe/$f)" ] || { say "the patched bigmoe/$f != the staged one"; restore; exit 3; }
done
say "patch live: tier.py / stream.py == the staged copies"
engine_test() { local out rc; out=$(TEST_EXT_BUILD=$1 $PY scripts/test_layout_engine.py . 2>&1); rc=$?; echo "$out" >> "$LOG"
  [ $rc = 0 ] && echo "$out" | grep -q '^PASS$'; }
engine_test build_w1 || { say "test_layout_engine on the live package (build_w1) FAILED"; restore; exit 3; }
say "test_layout_engine on the live package (build_w1): PASS"
U=$(loaded); [ -z "$U" ] || { say "ext/build NOT rebuilt (the patch stays live): loaded by: $U"; exit 4; }
t0=$(date +%s)
{ cmake -S ext -B ext/build -DCMAKE_BUILD_TYPE=Release -DPython_EXECUTABLE="$PY" && taskpolicy -b cmake --build ext/build -j 4; } \
  > data/item0_build.log 2>&1 || { say "the ext/build rebuild FAILED (data/item0_build.log; old .so in $ST/pre_live/build_$SO)"; exit 4; }
[ ext/build/$SO -nt ext/bigmoe_ext.cpp ] || { say "ext/build is still older than bigmoe_ext.cpp"; exit 4; }
say "ext/build rebuilt in $(( $(date +%s) - t0 )) s"
engine_test build || { say "test_layout_engine on ext/build FAILED (the patch stays live; look first)"; exit 4; }
$PY -I -c "import sys; sys.path.insert(0, '$HOME/BigMoE/ext/build'); import bigmoe_ext as e; assert e.layout_max() == 2" >> "$LOG" 2>&1 \
  || { say "ext/build has no layout reader (layout_max != 2)"; exit 4; }
say "test_layout_engine on ext/build: PASS; layout_max 2"
say "ITEM 0 LIVE: ext/build sha256 $(shasum -a 256 ext/build/$SO | cut -c1-12), Leap $(git -C "$HOME/Leap" rev-parse --short HEAD), bigmoe tier.py $(md5 -q bigmoe/tier.py | cut -c1-8) stream.py $(md5 -q bigmoe/stream.py | cut -c1-8)"
