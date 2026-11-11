#!/bin/sh
# Breakthrough's detached runner for NewIdeas' Tue 19:5x hand-outs (QUEUE 'RAPID SCREENS'; lines written before any run in
# SCOREBOARD 'R-20 KV8M + R-19 W24-12'), one 10-min quiet lock per item (scripts/lockrun.sh):
#   R-20 KV8M: one 64K prompt (65,000 tokens of ctx64k), AR path, PFR on, v3l, bf16 vs kv:64, fresh process each, no KV
#     hash: prefill-phase, decode-phase and lifetime peaks (scripts/pfr_ab.py)
#   R-19 W24-12: AR decode at 2.6K + 8.5K, cold (shipped C_m 51, PFR on) vs hot (C_m 160, PFR off, second decode of the
#     same prompt): SSD share of a decode step, MB / token, reads / token (scripts/r19_price.py); the hot arm (~25 GB)
#     only if >= 45 % of memory is free
#   AW-AR (NewIdeas GO ~20:1x): ALLWAIT vs shipped on the AR decode, 2.6K + 8.5K, ABBA rounds of 256 tokens from empty
#     pools (scripts/aw_ar.py; needs bt_b12's B12_ALLWAIT switch)
# Log: data/r1920/summary.txt
cd "$HOME/Breakthrough" || exit 1
OUT=$HOME/Breakthrough/data/r1920; mkdir -p "$OUT"; LOG=$OUT/summary.txt
. scripts/lockrun.sh
export B12_ENGINE=v3l
log "runner start (pid $$)"
caffeinate -i -w $$ &
item r20_bf16 480 "R-20 KV8M, bf16 arm: 64K prompt with PFR, memory peaks (~5 min)" \
  env PA_N=65000 PA_KVHASH=0 PA_DEC=32 $PY scripts/pfr_ab.py r20_bf16
item r20_kv64 480 "R-20 KV8M, kv:64 arm: 64K prompt with PFR, memory peaks (~5 min)" \
  env PA_N=65000 PA_KVHASH=0 PA_DEC=32 PA_KV8=kv:64 $PY scripts/pfr_ab.py r20_kv64
item aw_ar 540 "AW-AR: ALLWAIT vs shipped on the AR decode, 2.6K + 8.5K, ABBA x 256 tokens (~6 min)" \
  $PY scripts/aw_ar.py aw_ar
item r19_cold 300 "R-19 W24-12, cold arm: shipped decode at 2.6K + 8.5K (~2.5 min)" \
  env R19_ARM=cold $PY scripts/r19_price.py r19_cold
fr=$(memory_pressure -Q 2>/dev/null | awk -F': ' '/free percentage/ {print int($2)}')
if [ "${fr:-0}" -ge 45 ]; then
  item r19_hot 420 "R-19 W24-12, hot arm: C_m 160, each prompt decoded twice (~4 min, ~25 GB)" \
    env R19_ARM=hot B12_CM=160 B12_PFR=0 $PY scripts/r19_price.py r19_hot
else
  log "r19_hot skipped: memory free ${fr:-?} % < 45 % (it needs ~25 GB)"
fi
log "runner done"
