#!/bin/bash
# dm_t3 learns BigMoE's padded converted layout before Thu's records3 conversion (~10:55; Leap's 18:5x records3 readers
# triage, ~/Leap/data/records3_readers_triage.md: dm_t3.py:21 LAYOUT lacks "native+p11+pad", so every t3 arm would raise
# on the converted store; r3_overlay is trap T2). Breakthrough, Wed 2026-11-04 19:0x.
#   prep          copy decision_map's dm_t3.py to scratch, add "native+p11+pad": 2 to LAYOUT (bigmoe.tier.LAYOUTS maps it
#                 the same; nothing else changes), test it (test_t3_pad.py: the padded copy == records3 bit for bit; CPU)
#   install       with no map process (checked twice, 10 s apart): the tested copy in by cp + mv, if decision_map's
#                 dm_t3.py is still its base; then r3_overlay (45 row-major entries that link to the records3 files the
#                 conversion rewrites) moved aside to r3_overlay.STALE-before-records3-conversion, not deleted
#                 (mk_r3_overlay.py rebuilds it), so a DM_R3=r3_overlay run fails loudly instead of misreading.
#   wait-install  install, retried every 60 s while a map process runs
# Today's store is row-major, so the new name changes nothing before the conversion. After it, runs need BigMoE's
# layout-aware ext (dm_t3._ext: BIGMOE_EXT_BUILD=build_nat).
# Exit 0 = done, 1 = failed (unchanged), 2 = a map process is running (unchanged).
DM=~/FableGod/decision_map; PY=~/BigMoE/.venv/bin/python; SP=~/Breakthrough/data/abl_scratch/t3pad
HERE=$(cd "$(dirname "$0")" && pwd)
case "$1" in
prep)
  mkdir -p $SP && rm -f $SP/TESTED && cp $DM/dm_t3.py $SP/base.py && cp $SP/base.py $SP/dm_t3.py || exit 1
  grep -qF '"native+p11+pad"' $SP/dm_t3.py || sed -i '' '/^LAYOUT = /s/"native+p11": 2}/"native+p11": 2, "native+p11+pad": 2}/' $SP/dm_t3.py
  grep -q '^LAYOUT = .*"native+p11+pad": 2}' $SP/dm_t3.py || { echo "t3pad: the LAYOUT line did not take the name; unchanged"; exit 1; }
  diff <(grep -v '^LAYOUT = ' $SP/base.py) <(grep -v '^LAYOUT = ' $SP/dm_t3.py) > /dev/null || { echo "t3pad: more than LAYOUT changed"; exit 1; }
  ln -sf "$(readlink -f $DM/records3_index_rowmajor.json)" $SP/records3_index_rowmajor.json
  (cd $SP && T3_MODULE=$SP/dm_t3.py $PY $HERE/test_t3_pad.py) || { echo "t3pad: test FAILED; unchanged"; exit 1; }
  touch $SP/TESTED; echo "t3pad: prepared and tested ($SP/dm_t3.py)";;
install)
  pgrep -f "run_ma[p].py" > /dev/null && exit 2; sleep 10; pgrep -f "run_ma[p].py" > /dev/null && exit 2
  [ -e $SP/TESTED ] || { echo "t3pad: not tested (prep first)"; exit 1; }
  if grep -qF '"native+p11+pad"' $DM/dm_t3.py; then echo "t3pad: dm_t3 already has the name"
  else
    cmp -s $SP/base.py $DM/dm_t3.py || { echo "t3pad: decision_map's dm_t3.py changed since prep; unchanged"; exit 1; }
    cp $SP/dm_t3.py $DM/.dm_t3.py.new && mv $DM/.dm_t3.py.new $DM/dm_t3.py && cmp -s $SP/dm_t3.py $DM/dm_t3.py \
      && echo "t3pad: dm_t3 installed $(date '+%a %H:%M:%S')" || { echo "t3pad: install failed"; exit 1; }
  fi
  if [ -d $DM/r3_overlay ]; then mv $DM/r3_overlay $DM/r3_overlay.STALE-before-records3-conversion && echo "t3pad: r3_overlay moved aside"; fi
  exit 0;;
wait-install)
  while :; do bash $0 install; rc=$?; [ $rc = 2 ] || exit $rc; sleep 60; done;;
*) echo "usage: $0 prep | install | wait-install"; exit 1;;
esac
