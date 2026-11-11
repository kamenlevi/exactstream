#!/bin/bash
# MTP Head, Wed 11-04 09:0x (orchestrator 09:0x: the GPU is free to 12:50; pull TTS1 forward; SAMELAYER-T may run too).
# Waits for BigMoE's 09:15 quiet run (C 256 old-vs-new confirmation, ~20 min) and any other quiet lock to clear, then ONE
# share lock and, one at a time: SAMELAYER-T (CPU, ~3 GB, ~15 min) -> TTS1 full (GPU, 4.4 GB peak in the smoke, ~35 min).
# Swap guard: a job is stopped if swap grows > 2 GB over its start. Hard end 12:45 (the window ends 12:50).
# Log: data/wed_chain.log (+ data/samelayer_t.log, data/tts_replay.log).
cd ~/TrainLab || exit 1
log() { echo "[$(date +%H:%M:%S)] $*"; }
swap_mb() { sysctl -n vm.swapusage | awk '{print $6}' | tr -d M; }
hm() { date +%H%M; }

seen=0
while :; do
  ls_=$(peer ls 2>/dev/null)
  bq=$(printf '%s\n' "$ls_" | grep "larger-than-RAM" | grep -c "quiet lock")
  aq=$(printf '%s\n' "$ls_" | grep -c "quiet lock")
  if [ "$bq" -gt 0 ] && [ $seen -eq 0 ]; then seen=1; log "BigMoE's quiet lock seen"; fi
  if [ "$aq" -eq 0 ] && { [ $seen -eq 1 ] || [ "$(hm)" -ge 0940 ]; }; then break; fi
  if [ "$(hm)" -ge 1150 ]; then log "GIVE UP: no free slot by 11:50"; exit 3; fi
  sleep 30
done
log "slot free (BigMoE quiet seen: $seen)"

ok=0
for t in 1 2 3 4 5; do
  out=$(peer lock share 75m --cpu 800 --gpu 50 "MEM 5 GB: MTP Head SAMELAYER-T (CPU ~3 GB) then TTS1 (GPU ~4.4 GB), one at a time, to ~10:4x" 2>&1); rc=$?
  log "lock try $t rc=$rc: $(printf '%s' "$out" | head -3 | tr '\n' ' ')"
  if [ $rc -eq 0 ] && printf '%s' "$out" | grep -q "reserved"; then ok=1; break; fi
  sleep 60
done
[ $ok -eq 1 ] || { log "NO LOCK after 5 tries"; exit 4; }

run_guarded() {   # label, output file, command...: stop it if swap grows > 2 GB or past 12:45
  local label=$1 outf=$2; shift 2
  local s0; s0=$(swap_mb)
  log "$label start (swap $s0 MB)"
  "$@" > "$outf" 2>&1 &
  local pid=$!
  while kill -0 $pid 2>/dev/null; do
    sleep 20
    local s; s=$(swap_mb)
    if awk -v a="$s" -v b="$s0" 'BEGIN{exit !(a-b>2048)}'; then
      log "$label: swap $s MB = +2 GB over its start: stopping"; kill $pid; break
    fi
    if [ "$(hm)" -ge 1245 ]; then log "$label: 12:45 hard end: stopping"; kill $pid; break; fi
  done
  wait $pid; local rc=$?
  log "$label rc=$rc; tail: $(tail -2 "$outf" | tr '\n' ' ')"
}

run_guarded SAMELAYER-T data/samelayer_t.log ~/BigMoE/.venv/bin/python sticky/samelayer_t.py
run_guarded TTS1 data/tts_replay.log /usr/bin/time -l ~/BigMoE/.venv/bin/python sticky/tts_replay.py
peer unlock 2>&1 | tail -1
log "CHAIN DONE"
