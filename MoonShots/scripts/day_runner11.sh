#!/bin/sh
# MoonShots day_runner11 (detached): round 29, per-flag A/B of MTPLX kernel-cutting switches (flag_ab125.py).
# Same lock discipline as day_runner6. Results: ~/MoonShots/data/day11/*.txt, one line per job in day7/summary.txt.
OUT=$HOME/MoonShots/data/day11; mkdir -p "$OUT"
PY=$HOME/BigMoE/.venv/bin/python
PL=$HOME/MoonShots/patches/planes125
export PEER_QUEUE_TIMEOUT=14400
log() { echo "$(date +%H:%M) $*" >> "$OUT/summary.txt"; }
DL=$(date -j -f "%Y-%m-%d %H:%M" "$(date +%Y-%m-%d) 23:59" +%s)   # no new job after 23:59
LOCKF=$HOME/.claude-peers/locks/$CLAUDE_PID.json
keepalive() { $PY -I -S -c '
import json, os, sys, time
p = sys.argv[1]
try: d = json.load(open(p))
except Exception: sys.exit(1)
if d.get("mode") != "quiet" or d.get("until", 0) <= time.time(): sys.exit(1)
if sys.argv[2] == "check": sys.exit(0 if time.time() - max(d.get("taken", 0), d.get("active_at", 0)) < 360 else 1)
d["active_at"] = time.time(); t = "%s.tmp%d" % (p, os.getpid()); json.dump(d, open(t, "w")); os.replace(t, p)' "$LOCKF" "${1:-stamp}"; }
job() {   # job NAME SECONDS_LIMIT CMD...   (sets P = the job's pid: don't use P for anything else)
  name=$1; lim=$2; shift 2
  [ "$(date +%s)" -lt "$DL" ] || { log "skip $name (past 23:59)"; return; }
  if keepalive check; then keepalive; log "lock already held for $name"
  else peer lock quiet --queue 90m "MoonShots day_runner11: $name" > /dev/null 2>&1 || { log "no lock for $name"; return; }; fi
  g=0
  while { [ "$(memory_pressure | tail -1 | grep -o '[0-9]*')" -lt 85 ] || [ "$(ps -axo rss=,pid= | awk -v me=$$ '$1 > 8000000 && $2 != me' | wc -l)" -gt 0 ]; } && [ $g -lt 120 ]; do
    sleep 5; g=$((g + 1)); [ $((g % 36)) -eq 0 ] && keepalive; done
  if [ $g -ge 120 ]; then log "skip $name (memory busy)"; peer unlock > /dev/null 2>&1; return; fi
  log "start $name"
  ( "$@" > "$OUT/$name.txt" 2>&1 ) & P=$!
  i=0
  while kill -0 $P 2>/dev/null && [ $i -lt "$lim" ]; do
    sleep 5; i=$((i + 5))
    if [ $((i % 180)) -eq 0 ]; then
      keepalive || { log "lost the quiet lock during $name: stopping it"; kill $P 2>/dev/null; sleep 5; kill -9 $P 2>/dev/null; break; }
    fi
  done
  kill -0 $P 2>/dev/null && { kill -9 $P; log "TIMEOUT $name"; }
  wait $P 2>/dev/null
  peer unlock > /dev/null 2>&1
  log "done $name: $(grep -E '^(125B|  vs|  noise|planes|bigmoe|fixed|adaptive|Traceback|[A-Za-z]*Error)' "$OUT/$name.txt" | tr '\n' ' ' | cut -c1-900)"
}
log "day_runner11 started (pid $$); free $(df -g / | tail -1 | awk '{print $4}') GB"
F=$HOME/MoonShots/scripts/flag_ab125.py
# round 29: per-flag A/B of MTPLX's own kernel-cutting switches (one process per config; base twice for noise)
job ab_base        900 $PY $F base
job ab_opd_rope    900 env MTPLX_QWEN4_OPDIET=1 MTPLX_QWEN4_OPDIET_ITEMS=rope $PY $F opd_rope
job ab_opd_resid   900 env MTPLX_QWEN4_OPDIET=1 MTPLX_QWEN4_OPDIET_ITEMS=resid $PY $F opd_resid
job ab_opd_bank    900 env MTPLX_QWEN4_OPDIET=1 MTPLX_QWEN4_OPDIET_ITEMS=bank $PY $F opd_bank
job ab_opd_k20     900 env MTPLX_QWEN4_OPDIET=1 MTPLX_QWEN4_OPDIET_ITEMS=k20 $PY $F opd_k20
job ab_glue        900 env MTPLX_QWEN4_VERIFY_GLUE=1 $PY $F glue
job ab_gdn_inproj  900 env MTPLX_FUSED_GDN_INPROJ=1 $PY $F gdn_inproj
job ab_gdn_convnorm 900 env MTPLX_FUSED_GDN_CONVNORM=1 $PY $F gdn_convnorm
job ab_convnorm_ver 900 env MTPLX_FUSED_CONVNORM_VERIFY=1 $PY $F convnorm_verify
job ab_gdn_out     900 env MTPLX_FUSED_GDN_OUT=1 $PY $F gdn_out
job ab_gdn_step    900 env MTPLX_FUSED_GDN_STEP=1 $PY $F gdn_step
job ab_qsa_qkv     900 env MTPLX_FUSED_QSA_QKV=1 $PY $F qsa_qkv
job ab_qsa_indexer 900 env MTPLX_FUSED_QSA_INDEXER=1 $PY $F qsa_indexer
job ab_hc          900 env MTPLX_FUSED_HC=1 $PY $F hc
job ab_hc_v3       900 env MTPLX_FUSED_HC_V3=1 $PY $F hc_v3
job ab_base2       900 $PY $F base2
log "day_runner11 finished; free $(df -g / | tail -1 | awk '{print $4}') GB"
