#!/bin/sh
# Breakthrough R-41 part 4: the new DEFAULTS (fold-exact on, B12_PFR_GRID keep, engine v3m) with NO switches set, on the
# candidate bt_b12 (BT_DIR = $1), before it replaces the live one. One quiet lock per item (scripts/lockrun.sh):
#   def_mtp  fold_check p2k, p8k, 128 tokens: must = runner_fx's fx_ship / fx_exact hashes for those prompts
#            (p2k 98c6bc1c0d 82995cdb4e 78a72d3ccb fb2087cab5 76ecab8796; p8k 61b2520e58 0b0d91cd43 0643da7828 f25e0ebd31 b6d20a341e)
#   def_ar   pfr_ab 8,474 tokens: must = L2-11's r8 (KV ecd389371d, logits c57fe16fd1, sha32 142abd5d7e)
# Log: data/fx/summary.txt
cd "$HOME/Breakthrough" || exit 1
OUT=$HOME/Breakthrough/data/fx; mkdir -p "$OUT"; LOG=$OUT/summary.txt
. scripts/lockrun.sh
SP=$1
[ -f "$SP/bt_b12.py" ] || { echo "usage: runner_def.sh <dir with the candidate bt_b12.py>"; exit 2; }
log "runner_def start (pid $$), candidate $SP/bt_b12.py, no B12_ENGINE / B12_FOLD / B12_PFR_GRID set"
caffeinate -i -w $$ &
item def_mtp 420 "R-41 new defaults, MTP p2k + p8k (~3 min)" \
  env -u B12_ENGINE -u B12_FOLD -u B12_PFR_GRID BT_DIR=$SP FC_PROMPTS=p2k,p8k FC_NDEC=128 $PY scripts/fold_check.py def_mtp
item def_ar 300 "R-41 new defaults, AR 8,474 tokens (~2 min)" \
  env -u B12_ENGINE -u B12_FOLD -u B12_PFR_GRID BT_DIR=$SP PA_N=8474 $PY scripts/pfr_ab.py def_ar
log "runner_def done"
