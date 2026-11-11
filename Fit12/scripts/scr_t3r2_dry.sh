#!/bin/sh
# Fit12 (Tue 3 Nov 22:4x): dry test of scr_t3r2's 13:05 / 13:39 deadline logic (screens.sh), CPU only, seconds. screens.sh is
# sourced as a library, then every helper that touches the GPU, the lock, the disk or the clock-waits is replaced by a stub
# (lock / unlock / arm / es_wait / bigrun / memgate / avail_gb / ssd_t / keepalive / gpu / sleep); the deadline is moved
# with F12_T3R2_DL and the 13:05 wait is skipped with F12_T3R2_AT=000000. Stubs "advance the clock" by moving T2DL back.
# (Tue 23:1x) K12 cases J-N: F12_T3R2_O = a scratch dir with a fake r47.txt (r47's main PAIR line for 156), a stubbed
# lock-seconds-left (STUB_LEFT); the real k12_verdict.py runs on the scratch dir (VOID there: no k12 output).
# Log: the scratchpad's scr_t3r2_dry.log, never logs/summary.txt.
D=${1:-/private/tmp/claude-501/-Users-kamen-OjoX/7dd61b7e-2c45-43ec-b373-7768d2c96a18/scratchpad}
export CLAUDE_PID=${CLAUDE_PID:-18778}
SCREENS_LIB=1; . /Users/kamen/Fit12/scripts/screens.sh
LOG=$D/scr_t3r2_dry.log; : > "$LOG"
PAUSEF=$D/scr_t3r2_dry.PAUSE; rm -f "$PAUSEF"   # a scratch PAUSE path: the real ~/.local/share/quiet/PAUSE is never touched
F12_T3R2_AT=000000
sleep() { :; }
es_wait() { return 0; }; bigrun() { echo ""; }; memgate() { return 0; }; avail_gb() { echo 20; }; keepalive() { return 0; }
gpu() { echo 0; }; ssd_t() { echo "${STUB_SSD:-50}"; }; left() { echo "${STUB_LEFT:-1500}"; }
lock() { log "STUB lock DUR $2 NEED $3 GPU_MAX $4 | $1"; T2DL=$((T2DL - ${STUB_LOCK_SHIFT:-0})); return 0; }
unlock() { log "STUB unlock"; }
arm() { log "STUB arm $1/$2 need $3 cap $6"; [ "$2" = r47 ] && T2DL=$((T2DL - ${STUB_R47_SHIFT:-0})); [ "$2" = r47 ] && return "${STUB_RC:-0}"; return 0; }
O=/Users/kamen/Fit12/data/scr/t3r2; [ -e "$O/r47.txt" ] && { echo "r47.txt exists: the screen would return at once; test not valid"; exit 1; }
case_() {  # case_ NAME DL_FROM_NOW_S [VAR=VALUE ...]
  n=$1 d=$2; shift 2
  F12_T3R2_DL=$(date -r $(( $(date +%s) + d )) +%Y%m%d%H%M%S)
  unset STUB_SSD STUB_LOCK_SHIFT STUB_R47_SHIFT STUB_RC STUB_LEFT; for kv in "$@"; do eval "$kv"; done
  echo "== $n (deadline in $d s; $*)" >> "$LOG"; scr_t3r2; echo "   -> return $?" >> "$LOG"
}
case_ A_normal 1800
case_ B_too_late_for_the_lock 700
case_ C_late_grant_r47_cut 1000 STUB_LOCK_SHIFT=300
case_ D_late_grant_no_r47 1000 STUB_LOCK_SHIFT=450
case_ E_hot_ssd 1800 STUB_SSD=60
case_ F_hot_ssd_near_the_deadline 1000 STUB_SSD=60 STUB_LOCK_SHIFT=395
case_ G_r47_failed 1800 STUB_RC=124
case_ H_no_time_for_p47 1800 STUB_R47_SHIFT=1560
touch "$PAUSEF"; case_ I_pause_held 1800; rm -f "$PAUSEF"
KO=$D/scr_t3r2_dry_O; rm -rf "$KO"; mkdir -p "$KO"
echo "[r47] PAIR humaneval/HumanEval/156: SSD per arm (C) t80ck2 50->52 | PAUSE 0.0 s inside -> ok" > "$KO/r47.txt"
export F12_T3R2_O=$KO
case_ J_k12_runs 1800
case_ K_k12_cap_by_deadline 1800 STUB_R47_SHIFT=1300
case_ L_k12_no_time 1800 STUB_R47_SHIFT=1500
case_ M_k12_lock_short 1800 STUB_LEFT=200
case_ M2_k12_cap_by_lock 1800 STUB_LEFT=330
rm "$KO/r47.txt"; case_ N_k12_no_r47_data 1800
unset F12_T3R2_O
sed 's/^[0-9:]* SCR //' "$LOG" | cut -c1-230
