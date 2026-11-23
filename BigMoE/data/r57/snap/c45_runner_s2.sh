#!/bin/bash
# C45 (NewIdeas 11-01 06:1x): size the 45 GB default's C (R57_PRESET=45: REF2-T + gate + t 0.3) to the memory budget: one
# process per C (4-bit-slot bytes; C_l 112, C_m from the budget), S45's six prompts, warm-up discarded; tok/s + lifetime peak.
# Each C runs only if vm_stat NO-SWAP >= its expected peak + MG GB (4: the orchestrator's quiet-lock rule, 11-01 15:2x; NewIdeas'
# earlier 8 would have skipped C 224 itself at last nights' grants: NO-SWAP 42-47 GB). Peak = 38.0 GB at C 224 (S45-T's lifetime peak) + 0.1325 GB per C unit (48 layers x 2.76 MB
# of pool bytes; Leap 07:1x: 29.7 at C 160, 38.0 at C 224): ~45 GB at C ~277, 48.1 at C 300, 50.7 at C 320.
# Since NewIdeas 11-01 11:4x the preset carries OOO + TIER_COMBINE (EvalSuite's R-56d PASS). At the first C that runs:
# A c45_C = the no-OOO control (R57_COMBINE=0 BIGMOE_OOO=0); C45_L=1's L c45l_C = the default + MoonShots' M-17 LAYSZ45
# per-layer pool sizes (BIGMOE_TIER_SIZES=data/seed45/laysz45_C.json, the same pool bytes; NewIdeas 15:3x, vs B); B c45o_C
# = the default; C45_K73=1's K c45k_C = the default + Leap's exact-safe kernel stack (K73_EXACT=1: fused router + MoE tail +
# shared_cat + draft_fused + ref2 att_rope / idx_prep / hc_write; its every-token check, in place of Leap's runner 61); A2
# c45_Cr = the control again (the one controlled OOO A/B: A-L-B-K-A2 puts B in the middle, L and K next to it); then
# C45_PRIOR=1's P c45p_C (SEED45: ext/build_prior's BIGMOE_EVICT_PRIOR=30, LRU + popularity prior, each prompt's prior held
> (a quote from the project owner, kept private)
# the orchestrator: the default at C ~300 / ~320). Every later C of the list: c45o_C only (sizing on the default).
# Each run logs per check: demand SSD busy, gate service on misses, draft ms (NewIdeas 11:5x: SSD- or compute-bound at
# each C), and per kept token: SSD MB + misses. Every token checked against S45-T's REF2-T reference
# (data/r57/s45t_B_tokens.json, arm gate+t). A heat PAUSE (PAUSE.fanwatch) between runs is waited out (up to 10 min, the
# lock kept alive); any other PAUSE stops the turn.
# usage (inside a quiet memory turn): HAVE_LOCK=1 C45_BIG="300 320" bash scripts/c45_runner.sh "224 272 256 240"
export PATH="$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:$PATH"
set -m
cd "$HOME/BigMoE" || exit 1
OUT="${C45_OUT:-$HOME/BigMoE/data/c45}"; mkdir -p "$OUT"   # C50 (NewIdeas 11-02): C45_OUT=data/c50
LOG="$OUT/c45.log"
PY="$HOME/BigMoE/.venv/bin/python"
LOCKF="$HOME/.claude-peers/locks/${CLAUDE_PID}.json"
say() { echo "$(date +%H:%M:%S) $*" >> "$LOG"; }
keepalive() { [ -n "$HAVE_LOCK" ] || return 0; "$PY" -I -S -c '
import json, os, sys, time
p = sys.argv[1]
try: d = json.load(open(p))
except Exception: sys.exit(1)
if d.get("mode") != "quiet" or d.get("until", 0) <= time.time(): sys.exit(1)
d["active_at"] = time.time(); t = "%s.tmp%d" % (p, os.getpid()); json.dump(d, open(t, "w")); os.replace(t, p)' "$LOCKF"; }
source "$HOME/BigMoE/scripts/ssd_gate.sh"
noswap() { vm_stat | awk '/page size of/ {ps=$8} /^Pages (free|speculative|purgeable):/ || /^File-backed pages:/ {v=$NF; sub("[.]","",v); s+=v} END {printf "%d", s*ps/1e9}'; }
swap_gb() { sysctl -n vm.swapusage | awk '{gsub("M","",$6); print $6 / 1024}'; }
SWAP_RUN0=${C45_SWAP0:-$(swap_gb)}   # the swap rule is per RUN (health review Wed 04:4x: mission 20's C 224 arms grew swap
# 4.11 -> 7.50 GB in steps of +1.7 / +0.7 / +1.0, under the per-arm guard): growth > 2 GB since this runner (or the chain's
# exported C45_SWAP0) started stops the list before the next arm, and inside an arm next to the per-arm guard
one() {  # one C TAG ENV...: a run at C if NO-SWAP fits its peak + MG (default 4); returns 9 when skipped, 1 when the lock is gone
  local C=$1 tag=$2; shift 2
  local need; need=$(awk -v c=$C -v m=${MG:-4} 'BEGIN { printf "%d", 38.0 + (c - 224) * 0.1325 + m + 0.5 }')
  local w=0
  while [ -e "$HOME/.local/share/quiet/PAUSE" ]; do         # a heat pause clears in ~2-4 min: wait it out between runs
    if [ ! -e "$HOME/.local/share/quiet/PAUSE.fanwatch" ]; then   # fanwatch drops PAUSE.fanwatch just before PAUSE when it
      sleep 5; [ -e "$HOME/.local/share/quiet/PAUSE" ] || break     # clears (11-02 10:57 race): look again before stopping
      [ -e "$HOME/.local/share/quiet/PAUSE.fanwatch" ] || { say "PAUSE (not heat): stop"; return 1; }
    fi
    if [ $w -ge 60 ]; then say "heat PAUSE over 10 min: stop"; return 1; fi
    [ $w = 0 ] && say "heat PAUSE before $tag: waiting"
    sleep 10; w=$((w + 1))
    [ $((w % 12)) = 0 ] && { keepalive || { say "lost the lock in the pause before $tag: stop"; return 1; }; }
  done
  [ $w -gt 0 ] && say "PAUSE cleared after $((w * 10)) s"
  # Fri 23:4x fix: HHMM compared across midnight (a 23:42 start with deadline 0042 read as past it, part A VOID). Now
  # wrap-aware: past = now is within the 12 h after the deadline's clock time (runs are shorter than 12 h).
  if [ -n "$C45_DEADLINE" ] && [ $(( (10#$(date +%H) * 60 + 10#$(date +%M) - 10#${C45_DEADLINE:0:2} * 60 - 10#${C45_DEADLINE:2:2} + 1440) % 1440 )) -lt 720 ]; then
    say "$tag: past $C45_DEADLINE, stop (a run must end before 12:00)"; return 1; fi
  keepalive || { say "lost the lock before $tag: stop"; return 1; }
  ssd_wait 66
  if awk -v s="$(swap_gb)" -v r0="$SWAP_RUN0" 'BEGIN { exit !(s - r0 > 2) }'; then
    say "$tag: swap grew > 2 GB since the run's start ($SWAP_RUN0 -> $(swap_gb) GB): stop (the swap rule)"; return 1; fi
  local a; a=$(noswap)
  if [ "$a" -lt "$need" ]; then say "$tag skipped: NO-SWAP $a GB < $need"; return 9; fi
  local S0; S0=$(swap_gb); say "run $tag (NO-SWAP $a GB, swap $S0 GB)"
  # DAY9 (11-02): C45_CAP / C45_ALARM / C45_REFTOK for other prompt sets (R57_KEYS from the caller's environment: its commas
  # can't sit in an arm's env list); an arm's own ENV=V entries come last and win
  perl -e 'alarm shift; exec @ARGV' ${C45_ALARM:-420} env R57_PRESET=45 BIGMOE_EXT_BUILD=build R57_PROMPTS=6 R57_REPS=1 R57_WARM=1 \
    R57_TAG=$tag R57_REFTOK=${C45_REFTOK:-$HOME/BigMoE/data/r57/s45t_B_tokens.json:gate+t} \
    "$@" $PY ${C45_SCREEN:-scripts/r57_screen.py} $C ${C45_CAP:-256} >> "$OUT/$tag.txt" 2>&1 &   # C45_SCREEN: a frozen copy
  local jp=$! k=0
  while kill -0 $jp 2>/dev/null; do
    sleep 5; k=$((k + 1))
    if awk -v s="$(swap_gb)" -v s0="$S0" -v r0="$SWAP_RUN0" 'BEGIN { exit !(s - s0 > 2 || s - r0 > 2) }'; then say "SWAP GUARD in $tag"; kill -TERM -- -$jp; sleep 3; kill -9 -- -$jp 2>/dev/null; break; fi
    [ $((k % 36)) -eq 0 ] && { keepalive || { say "lost the lock during $tag: stop"; kill -TERM -- -$jp; sleep 3; kill -9 -- -$jp 2>/dev/null; return 1; }; }
  done
  wait $jp; local rc=$?; kill -9 -- -$jp 2>/dev/null
  say "done $tag rc=$rc: $(grep -E 'footprint now' "$OUT/$tag.txt" | tail -1 | cut -c1-120)"
  ssd_check || say "SSD hot after $tag"
  return 0
}
if [ -n "$C45_ARMS" ]; then   # C50 (NewIdeas 11-02 01:3x): explicit arms "tag:C[:ENV=V,ENV=V]" in order, no A/B block, on the
  say "c45 start (arm list): $C45_ARMS"                      # adopted default (M-17 sizes + K-73; R57_LAYSZ=0 = uniform pools)
  for a in $C45_ARMS; do
    IFS=: read -r tag C envs <<< "$a"
    one $C $tag ${envs//,/ }; [ $? = 1 ] && break
  done
  [ -n "$KEEP_LOCK" ] || peer unlock >> "$LOG" 2>&1
  say "c45 end"
  exit 0
fi
say "c45 start: C list ${1:-224 240 256 272}, A-L-B-K-A2 at the first C, L arm ${C45_L:-0}, K-73 arm ${C45_K73:-0}, prior arm ${C45_PRIOR:-0}, ~50 GB arms ${C45_BIG:-none}"
NOOOO="R57_COMBINE=0 BIGMOE_TIER_COMBINE=0 BIGMOE_OOO=0"
first=1
for C in ${1:-224 240 256 272}; do
  if [ $first = 1 ]; then
    one $C c45_$C $NOOOO; rc=$?
    [ $rc = 1 ] && break
    [ $rc = 9 ] && continue                                  # did not fit: the next C hosts the A/B
    first=0
    if [ "$C45_L" = 1 ]; then
      LS="$HOME/BigMoE/data/seed45/laysz45_$C.json"         # MoonShots' M-17 sizes for this C (copies of ~/MoonShots/data's)
      if [ -s "$LS" ]; then one $C c45l_$C BIGMOE_TIER_SIZES=$LS; [ $? = 1 ] && break; else say "no $LS: L arm skipped"; fi
    fi
    one $C c45o_$C; [ $? = 1 ] && break
    if [ "$C45_K73" = 1 ]; then one $C c45k_$C K73_EXACT=1; [ $? = 1 ] && break; fi
    one $C c45_${C}r $NOOOO; [ $? = 1 ] && break
    if [ "$C45_PRIOR" = 1 ]; then
      one $C c45p_$C BIGMOE_EXT_BUILD=build_prior BIGMOE_EVICT_PRIOR=30 R57_PRIOR=$HOME/BigMoE/data/seed45/prior45.npz
      [ $? = 1 ] && break
    fi
    rc=0
> (a quote from the project owner, kept private)
      one $CB c45o_$CB; rc=$?                                # NO-SWAP >= peak + 4 GB (52 / 55 GB), else skipped
      [ $rc = 1 ] && break
    done
    [ $rc = 1 ] && break
  else
    one $C c45o_$C; [ $? = 1 ] && break
  fi
done
[ -n "$KEEP_LOCK" ] || peer unlock >> "$LOG" 2>&1
say "c45 end"
