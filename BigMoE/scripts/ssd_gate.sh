# SSD heat gate for BigMoE runners (source it; health review 10-25 16:1x). The 24 GB streamed decode reads the SSD at
# ~10 GB/s; back-to-back arms took it to 80 C (b28, 13:46 health-monitor ACT cooldown: SSD >= 75 C for 3+ min) and
# 76-77 C (b29, b30). With FAN_CAP off nothing pauses jobs before that ACT, so runners gate themselves:
#   ssd_wait   before each arm: wait (keeping the lock alive) until the SSD is at or below the cool line (66 C)
#   ssd_check  after each arm: >= 73 C (RULES.md's safety line) -> say so; the caller stops the batch
# Temperature from fanwatch's latest sample (~/Library/Logs/fanwatch.log, every 20 s). If that is stale (fanwatch down:
# 10-25 19:27-22:1x its last line said 22 C for hours), HEALTH.md's "Now" line (health monitor, every ~5 min); unknown -> no wait.
ssd_t() {
  local f="$HOME/Library/Logs/fanwatch.log" h="$HOME/.local/share/quiet/HEALTH.md" now; now=$(date +%s)
  if [ -f "$f" ] && [ $((now - $(stat -f %m "$f"))) -le 120 ]; then tail -1 "$f" | grep -o 'ssd [0-9]*' | awk '{print $2}'; return; fi
  if [ -f "$h" ] && [ $((now - $(stat -f %m "$h"))) -le 900 ]; then grep -m1 '^\*\*Now:\*\*' "$h" | grep -o 'SSD [0-9]*' | awk '{print $2}'; fi
}
ssd_wait() {   # ssd_wait [cool=66]: needs keepalive and say from the runner
  local cool=${1:-66} t i=0
  t=$(ssd_t); [ -z "$t" ] && return 0
  [ "$t" -le "$cool" ] && return 0
  say "SSD $t C > $cool C: cooling before the next arm"
  while t=$(ssd_t); [ -n "$t" ] && [ "$t" -gt "$cool" ] && [ $i -lt 120 ]; do sleep 10; i=$((i+1)); [ $((i % 18)) -eq 0 ] && keepalive; done
  say "SSD $(ssd_t) C after $((i * 10)) s"
}
ssd_check() { local t; t=$(ssd_t); [ -n "$t" ] && [ "$t" -ge 73 ] && { say "SSD $t C >= 73 C after the arm"; return 1; }; return 0; }
