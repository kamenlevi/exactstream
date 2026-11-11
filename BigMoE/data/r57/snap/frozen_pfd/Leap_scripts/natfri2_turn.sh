#!/bin/bash
# Leap, Fri 2026-11-06 01:5x. The orchestrator 01:4x: MTP Head's L1-20 first, then NATIVE-FRI's re-run (>= 3 GB beyond
# the rung's need), then BigMoE's qall / s99 / s90 chunk after the split's GPU check. A PAUSE (fanwatch heat) blocked
# the check before MTP Head, and a share lock's `peer unlock` would drop NATIVE-FRI's queued ticket (same session).
# So ONE quiet turn of this session runs:
#   1. the QOM-SPLIT-1C GPU check (scripts/qomsplit_check.py; ~3-5 min, MEM ~4 GB, bitwise checks, no timing);
#   2. on RESULT PASS the candidate goes live: ~/Leap/data/split1c/ref2_split.py -> scripts/ref2.py, atomically, only if
#      the live file is still FAST-NAX's (md5 eda11b8b); then ref2.self_test(), restoring eda11b8b on any failure.
#      The split is off by default (R57_QOM_SPLIT / REF2_QOM_SPLIT unset), and NATIVE-FRI's exact arms never take the
#      one-call path, so they run the same code either way;
#   3. NATIVE-FRI's re-run: natfri_c.sh with NATFRI_MARGIN=3 (turn45's HAVE_LOCK / KEEP_LOCK pass through).
# Log: data/qomsplit_check.log (+ data/qomsplit_check.txt, data/natfri/natfri.log).
cd "$HOME/Leap" || exit 1
LOG="$HOME/Leap/data/qomsplit_check.log"
say() { echo "$(date '+%m-%d %H:%M:%S') $*" | tee -a "$LOG"; }
PY="$HOME/BigMoE/.venv/bin/python"
R2="$HOME/Leap/scripts/ref2.py"; CANDF="$HOME/Leap/data/split1c/ref2_split.py"; BK="$HOME/Leap/data/split1c/ref2_before_split.py"
LIVEMD5=eda11b8be7a6e97331df9b61af86a173
n0=$(wc -l < "$LOG" 2>/dev/null || echo 0)
say "split check start (pid $$, candidate md5 $(md5 -q "$CANDF"), live md5 $(md5 -q "$R2"))"
VECLIB_MAXIMUM_THREADS=4 QSC_COND="first in Leap's NATIVE-FRI quiet turn" perl -e 'alarm shift; exec @ARGV' 600 \
  "$PY" -B scripts/qomsplit_check.py >> "$LOG" 2>&1
rc=$?
res=$(tail -n +"$((n0 + 1))" "$LOG" | grep -E '^RESULT ' | tail -1)
say "split check rc=$rc: ${res:-no RESULT line}"
if [ "$rc" = 0 ] && [ "$res" = "RESULT PASS" ]; then
  if [ "$(md5 -q "$R2")" != "$LIVEMD5" ]; then
    say "ref2.py md5 $(md5 -q "$R2") is not FAST-NAX's eda11b8b (changed meanwhile): the candidate is NOT installed"
  elif cp "$R2" "$BK" && cp "$CANDF" "$R2.split_tmp" && mv "$R2.split_tmp" "$R2" && [ "$(md5 -q "$R2")" = "$(md5 -q "$CANDF")" ]; then
    say "INSTALLED the QOM-SPLIT-1C ref2 (md5 $(md5 -q "$R2"), split off by default); self_test next"
    VECLIB_MAXIMUM_THREADS=4 perl -e 'alarm shift; exec @ARGV' 300 "$PY" -B -c "
import os, sys
for p in ('~/BigMoE', '~/Leap/third_party/ms_gdn', '~/Leap/scripts'):
    sys.path.insert(0, os.path.expanduser(p))
import ref2
ref2.apply()          # self_test needs _ORIG from apply()
r = ref2.self_test()
ref2.remove()
print('ref2.self_test ->', r, flush=True)
assert ref2.CFG['qom_split'] is None and ref2.QCNT_1C is None, 'not off by default'
sys.exit(0 if r and all(r.values()) else 1)" >> "$LOG" 2>&1
    st=$?
    say "ref2.self_test rc=$st"
    if [ "$st" != 0 ]; then
      if [ "$(md5 -q "$BK")" = "$LIVEMD5" ] && cp "$BK" "$R2.restore_tmp" && mv "$R2.restore_tmp" "$R2"; then
        say "RESTORED ref2.py to FAST-NAX's eda11b8b (the self_test failed)"
      else
        say "RESTORE FAILED: restore $BK by hand"
      fi
    else
      say "QOM-SPLIT-1C LIVE: ref2.py = the candidate, self_test PASS"
    fi
  else
    say "INSTALL FAILED: ref2.py md5 $(md5 -q "$R2")"
    [ "$(md5 -q "$R2")" != "$LIVEMD5" ] && [ "$(md5 -q "$BK")" = "$LIVEMD5" ] && cp "$BK" "$R2.restore_tmp" && mv "$R2.restore_tmp" "$R2" \
      && say "RESTORED ref2.py to eda11b8b"
  fi
else
  say "the split check did not PASS: the candidate is NOT installed (ref2.py stays eda11b8b)"
fi
say "NATIVE-FRI re-run next (NATFRI_MARGIN=3)"
NATFRI_MARGIN=3 exec /bin/bash "$HOME/Leap/scripts/natfri_c.sh" "$@"
