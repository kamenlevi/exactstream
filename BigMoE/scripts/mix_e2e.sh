#!/bin/bash
# The records3 conversion's end-to-end exactness check (docs/PLAN_records3_convert.md section 3.6; HANDOVER 'CONVERSION
# ENGINE WORK'), BEFORE the real conversion: the 45 GB default (R57_PRESET=45) on scratch/model_mix, a model dir of symlinks
# whose records3 has layers 0 / 20 / 47 native+p11 (scratch/r3p11, == Leap's convert_records_p11.py byte for byte) and the
# rest row-major (models/bare/records3). Needs the per-layer layout patch live in bigmoe/ (patches/records3_layout.diff, Wed),
# ext/build rebuilt from 761cb3d+ (read_rebuilt_layout) and Leap's ref2 router (90456ce). Every token passes 3 converted
# layers: the prefill's CPU rebuild (read_rebuilt_layout), 3c-N with the p11 dequant in decode / verify, OOO + combine.
# C 160 (peak ~30 GB): tokens do not depend on C; the small peak fits a night that starts with apps open.
# Line (pre-registered Tue 3 Nov 17:4x): every token == the stored REF2-T tokens on all 11 prompts (cmb0_224's from Tue night,
# else chat23's where stored) = PASS (the converted path is exact end to end); any difference = STOP: no conversion until it
# is found. Then the row-major control on the same prompts (mixc), so a difference can be pinned on the converted layers.
# usage: inside turn45 (HAVE_LOCK=1 KEEP_LOCK=1), with frozen copies (data/r57/snap/*_e2e.*)
export PATH="$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:$PATH"
cd "$HOME/BigMoE" || exit 1
mkdir -p data/e2e
LOG="$HOME/BigMoE/data/e2e/e2e.log"
say() { echo "$(date '+%m-%d %H:%M:%S') $*" >> "$LOG"; }
RUN="/bin/bash data/r57/snap/c45_runner_e2e.sh"
export C45_SCREEN=data/r57/snap/r57_screen_e2e.py C45_DEADLINE=${E2E_DEADLINE:-0200} C45_OUT="$HOME/BigMoE/data/e2e"
CC=${E2E_C:-160}
KEYS="humaneval/HumanEval/68,humaneval/HumanEval/91,humaneval/HumanEval/16,humaneval/HumanEval/6"
KEYS="$KEYS,math500/test/prealgebra/1640.json,ifeval/1220,ifeval/1561,multi/de3,multi/zh15,agent/bash18,agent/text23"
REF="$HOME/BigMoE/data/d45/ref14_tokens.json"   # Wed 08:05: cmb0_224 never existed (the night ran C 160), so chat23 (no HumanEval 16 / 6) made a PASS read "False"
[ -s "$REF" ] || REF="$HOME/BigMoE/data/r57/chat23_ref_tokens.json"
MIX="$HOME/BigMoE/scratch/model_mix"
say "e2e start (pid $$, C $CC, reference $REF)"
"$HOME/BigMoE/.venv/bin/python" -I -c "
import sys; sys.path.insert(0, '$HOME/BigMoE')
import bigmoe.tier as T; assert hasattr(T, 'LAYOUTS'), 'bigmoe/ has no layout patch'
sys.path.insert(0, '$HOME/BigMoE/ext/build'); import bigmoe_ext as e; assert e.layout_max() == 2, 'ext/build has no layout reader'
print('engine OK: layout patch + ext/build read_rebuilt_layout')" >> "$LOG" 2>&1 || { say "engine not ready: STOP"; exit 2; }
R57_KEYS=$KEYS MG=${E2E_MG:-6} C45_ALARM=900 C45_REFTOK="$REF:gate+t" C45_ARMS="mix_$CC:$CC:R57_MODEL_DIR=$MIX mixc_$CC:$CC" $RUN
say "runs rc=$?"
for t in mix mixc; do
  f="$C45_OUT/${t}_$CC.txt"
  v=$(grep -o 'every token equal [A-Za-z]*' "$f" 2>/dev/null | tail -1)
  say "$t: ${v:-no verdict line} ($(grep -o 'converted layers \[.*\]' "$f" 2>/dev/null | head -1))"
done
say "e2e end"
