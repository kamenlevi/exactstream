#!/bin/sh
# Breakthrough's detached runner for the fold's REGRESSION checks (fold OFF; no fold run before Fit12's R-01 passes,
# NewIdeas 10-27). One 10-min quiet lock per item (scripts/lockrun.sh):
#   fold_build    build the v3m engine (v3l + reset_layer with a new slot count + drain()) in v3m/ext/build
#   reg_ar_v3m    AR path, 8,474 tokens, PFR on: the NEW bt_b12 (B12_FOLD off; BT_DIR = the scratch copy) on v3m must
#                 give L2-11's references (KV ecd389371d, logits c57fe16fd1, sha32 142abd5d7e)
#   reg_mtp_old   MTP path (b12_prefill_state + spec125.generate, p2k then p8k, 256 tokens): the LIVE bt_b12 on v3l
#   reg_mtp_new   the same with the NEW bt_b12 (fold off) on v3m: every hash must equal reg_mtp_old's
# Log: data/fold/summary.txt
cd "$HOME/Breakthrough" || exit 1
OUT=$HOME/Breakthrough/data/fold; mkdir -p "$OUT"; LOG=$OUT/summary.txt
. scripts/lockrun.sh
SP=/private/tmp/claude-501/-Users-kamen-OjoX/2c7bb7fe-3ba1-4c03-91ba-f6be33886b82/scratchpad/btfold
log "runner start (pid $$)"
caffeinate -i -w $$ &
item fold_build 600 "fold prep: build the v3m engine (CPU, 2 cores, ~2 min)" \
  sh -c "cd $HOME/Breakthrough/v3m && cmake -S ext -B ext/build -DCMAKE_BUILD_TYPE=Release -DPython_EXECUTABLE=$PY \
-DNB_CMAKE_DIR=$HOME/BigMoE/.nb215/nanobind/cmake && cmake --build ext/build -j 2" || { log "build failed: stop"; exit 1; }
item reg_ar_v3m 300 "fold regression: AR 8.5K on v3m + the new bt_b12, fold off (~2 min)" \
  env B12_ENGINE=v3m BT_DIR=$SP PA_N=8474 $PY scripts/pfr_ab.py reg_ar_v3m
item reg_mtp_old 420 "fold regression: MTP p2k + p8k, live bt_b12 on v3l (~3 min)" \
  env B12_ENGINE=v3l B12_FOLD=0 $PY scripts/fold_check.py reg_mtp_old
item reg_mtp_new 420 "fold regression: MTP p2k + p8k, new bt_b12 (fold off) on v3m (~3 min)" \
  env B12_ENGINE=v3m BT_DIR=$SP B12_FOLD=0 $PY scripts/fold_check.py reg_mtp_new
log "runner done"
