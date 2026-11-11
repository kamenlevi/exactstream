#!/bin/sh
# Breakthrough R-41 part 3 (NewIdeas Wed ~16:0x; lines written before the run, SCOREBOARD): verify the rebuilt pieces on
# the scratch bt_b12 (BT_DIR; B12_PFR_GRID default keep, MTP PFR only above 4096 tokens or to resize). One quiet lock
# per arm (scripts/lockrun.sh):
#   AR, PFR-keep (pfr_ab.py, KV hash on): 8,474 tokens must = L2-11's r8 (KV ecd389371d, logits c57fe16fd1, sha32
#     142abd5d7e); 65,000 tokens must = r64 (2227a92414 / 4dcabb778f / dc67a9270b); prefill time and peaks vs PFR
#     balanced (L2-11: 18.9 s / 208.1 s, prefill-phase 9.27 / 11.08 GB) and PFR off (27.5 / 265.1 s, 11.05 / 13.54 GB)
#   MTP fold-exact (fold_check.py, p2k, p8k, p16k, p32k in one process, 128 tokens): shipped (live, fold off) vs
#     fold-exact (B12_FOLD=1, AW / CM off): every hash equal per prompt, and every fold-exact lifetime peak <= 11.95 GB
#   write16 fold-exact (w16_bisect.py): must = the suite's OFF row
# Log: data/fx/summary.txt
cd "$HOME/Breakthrough" || exit 1
OUT=$HOME/Breakthrough/data/fx; mkdir -p "$OUT"; LOG=$OUT/summary.txt
. scripts/lockrun.sh
SP=$1
[ -f "$SP/bt_b12.py" ] || { echo "usage: runner_fx.sh <dir with the candidate bt_b12.py>"; exit 2; }
FX="B12_FOLD=1 B12_FOLD_AW=0 B12_FOLD_CM=0 B12_FOLD_ROWS=2048"
log "runner_fx start (pid $$), candidate $SP/bt_b12.py"
caffeinate -i -w $$ &
item ar_keep8 300 "R-41 AR PFR-keep, 8,474 tokens (~2 min)" \
  env B12_ENGINE=v3l BT_DIR=$SP PA_N=8474 $PY scripts/pfr_ab.py ar_keep8
item ar_keep64 600 "R-41 AR PFR-keep, 65,000 tokens (~6 min)" \
  env B12_ENGINE=v3l BT_DIR=$SP PA_N=65000 $PY scripts/pfr_ab.py ar_keep64
item fx_ship 600 "R-41 MTP shipped, p2k p8k p16k p32k (~6 min)" \
  env B12_ENGINE=v3l B12_FOLD=0 FC_PROMPTS=p2k,p8k,p16k,p32k FC_NDEC=128 $PY scripts/fold_check.py fx_ship
item fx_exact 600 "R-41 MTP fold-exact, p2k p8k p16k p32k (~6 min)" \
  env B12_ENGINE=v3m BT_DIR=$SP $FX FC_PROMPTS=p2k,p8k,p16k,p32k FC_NDEC=128 $PY scripts/fold_check.py fx_exact
item fx_w16 300 "R-41 write16 fold-exact (~2 min)" \
  env B12_ENGINE=v3m BT_DIR=$SP $FX $PY scripts/w16_bisect.py fx_w16
log "runner_fx done"
