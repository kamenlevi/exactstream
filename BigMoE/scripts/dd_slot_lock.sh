# Sourced by dd_capture_slot.sh and dd_ttt_slot.sh (Fri 13 Nov). A session holds ONE peer lock ("peer unlock" drops it
# whole), so the two launchers share it: each writes "<pid> <MEM GB> <label>" to $SLOT_STATE/<job>.mem while its job runs,
# and slot_lock re-takes the session's share lock with the sum of the live jobs' MEM, or drops it when none runs.
SLOT_STATE=$HOME/FableGod/deepdraft/.slot; mkdir -p "$SLOT_STATE"
> (a quote from the project owner, kept private)
slot_lock() {
  local tot=0 labels="" f pid mem lab
  for f in "$SLOT_STATE"/*.mem; do
    [ -e "$f" ] || continue
    read -r pid mem lab < "$f"
    if kill -0 "$pid" 2>/dev/null; then tot=$((tot + mem)); labels="$labels${labels:+ + }$lab"; else rm -f "$f"; fi
  done
  if [ "$tot" -gt 0 ]; then
    peer lock share 720m --cpu 300 --gpu 60 "BigMoE DEEPDRAFT: MEM $tot GB; yields to quiet locks ($labels)" >/dev/null 2>&1
  else
    peer unlock >/dev/null 2>&1
  fi
}
