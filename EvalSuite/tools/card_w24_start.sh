#!/bin/bash
# PREREG_card_gates amendment 12: start the w24 card arm in the queue owner's slot (NewIdeas 18:59: Tue ~01:30 if the env is
# ready, else the Tue-night slot). Waits until START, then for BigMoE's data/stack/card_w24_env.txt to be committed (tracked,
# equal to HEAD) and written after amendment 12; starts runner 61 only if that happens by LATEST, so the arm has its
# ~10-12 h before the 15:00 cutoff. Otherwise it starts nothing and says so.
# Amendment 12b (Mon 9 Nov 23:2x): START 02:10:00 (Kamen, via the orchestrator 23:2x); times carry seconds (BSD date -j -f
# keeps the current seconds otherwise). The env file must also be one of the dry-run blobs in W24S_BLOBS (git hash-object):
# any other content waits, says so once, and starts nothing by LATEST. Runner 61 reads a frozen copy (cp -p, same blob), so
# a later edit of BigMoE's file cannot reach the running arm. CARD_YIELD_GB=12: the arm yields at a process start only to
# another session's shared MEM run above 12 GB or to two above 5 GB, so one <= 12 GB neighbour can run beside it.
# Amendment 12c (Mon 9 Nov 23:4x; NewIdeas 23:30): it may start before START, not before EARLY, once every AFTER file
# (Leap's done files, colon-separated) is newer than NEWER and AFTER_S s old, and only if GO exists (written by EvalSuite
# when NewIdeas' W24c verdict has arrived, so the config is settled first). W24S_EARLY= (empty) turns this path off.
# Test: W24S_START=<now, with seconds> W24S_LATEST=<now + 2 min> W24S_ENV=<path> W24S_BLOBS=<blob> W24S_RUNENV=<tmp path>
#       W24S_LOG=<tmp log> W24S_DRY=1 [W24S_EARLY= | W24S_EARLY=<t> W24S_AFTER=<f1:f2> W24S_AFTER_S=<s> W24S_AFTER_NEWER=<t>
#       W24S_GO=<f>] tools/card_w24_start.sh
export PATH="$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:$PATH"
cd "$HOME/EvalSuite" || exit 1
START=${W24S_START:-2026-11-10 02:10:00}
EARLY=${W24S_EARLY-2026-11-10 01:25:00}
AFTER=${W24S_AFTER-$HOME/Leap/data/drafter_1109.done:$HOME/Leap/data/prefill_1109.done}
AFTER_S=${W24S_AFTER_S:-300}
NEWER=${W24S_AFTER_NEWER:-2026-11-09 23:30:00}
GO=${W24S_GO-$HOME/EvalSuite/results/w24c_verdict.txt}
LATEST=${W24S_LATEST:-2026-11-10 05:00:00}
CUT=${W24S_CUTOFF:-2026-11-10 15:00}
ENVF=${W24S_ENV:-$HOME/BigMoE/data/stack/card_w24_env.txt}
BLOBS=${W24S_BLOBS:-757b768c492ff3032001f0eff4b82f8627374347}   # w13t (BigMoE 8f76fc3; dry runs Mon 21:11 and 23:13)
RUNENV=${W24S_RUNENV:-results/card_w24_env_run.txt}
case "$RUNENV" in /*) ;; *) RUNENV=$PWD/$RUNENV;; esac
YIELD=${W24S_YIELD_GB:-12}
L=${W24S_LOG:-results/card_w24_start.log}
log() { echo "$(date '+%m-%d %H:%M:%S') w24 start: $*" | tee -a "$L"; }
t_start=$(date -j -f "%Y-%m-%d %H:%M:%S" "$START" +%s) || exit 2
t_late=$(date -j -f "%Y-%m-%d %H:%M:%S" "$LATEST" +%s) || exit 2
t_reg=$(git log -1 --format=%ct d5429d4) || { log "amendment 12's commit d5429d4 is not in this repo"; exit 2; }
if [ -n "$EARLY" ]; then
  t_early=$(date -j -f "%Y-%m-%d %H:%M:%S" "$EARLY" +%s) || exit 2
  t_newer=$(date -j -f "%Y-%m-%d %H:%M:%S" "$NEWER" +%s) || exit 2
  case "$AFTER_S" in ''|*[!0-9]*) log "W24S_AFTER_S '$AFTER_S' is not whole seconds"; exit 2;; esac
  [ -n "$AFTER" ] || { log "an early start needs W24S_AFTER files"; exit 2; }
  EARLYTXT="; or earlier, not before $EARLY, once $AFTER (newer than $NEWER) are $AFTER_S s old${GO:+ and $GO exists}"
fi
log "armed (pid $$): start at $START${EARLYTXT:-}; env $ENVF committed in BigMoE, newer than amendment 12, blob in {$BLOBS}; latest $LATEST, cutoff $CUT, yield above $YIELD GB"
early_ok() {    # amendment 12c: every AFTER file newer than NEWER and AFTER_S s old, GO present, not before EARLY
  local f m last=0
  [ -n "$EARLY" ] && [ "$(date +%s)" -ge "$t_early" ] || return 1
  [ -z "$GO" ] || [ -e "$GO" ] || return 1
  local IFS=:
  for f in $AFTER; do
    [ -e "$f" ] || return 1
    m=$(stat -f %m "$f"); [ "$m" -gt "$t_newer" ] || return 1
    [ "$m" -gt "$last" ] && last=$m
  done
  [ "$(date +%s)" -ge $(( last + AFTER_S )) ] || return 1
  EARLYWHY="the files' last mtime $(date -r "$last" '+%H:%M:%S') + $AFTER_S s${GO:+, $GO present}"
}
until [ "$(date +%s)" -ge "$t_start" ] || early_ok; do
  w=$(( t_start - $(date +%s) )); [ "$w" -gt 20 ] && w=20; [ "$w" -gt 0 ] && sleep "$w"
done
if [ -n "${EARLYWHY:-}" ]; then log "start: the early path ($EARLYWHY)"; else log "start: $START"; fi
said=""
ready() {
  [ -s "$ENVF" ] || return 1
  [ "$(stat -f %m "$ENVF")" -gt "$t_reg" ] || return 1
  b=$(git hash-object --no-filters "$ENVF") || return 1
  case " $BLOBS " in *" $b "*) ;; *) [ "$said" = "$b" ] || { log "env blob $b is not a dry-run blob: waits"; said=$b; }; return 1;; esac
  [ -n "${W24S_DRY:-}" ] && return 0
  git -C "$HOME/BigMoE" ls-files --error-unmatch "${ENVF#$HOME/BigMoE/}" > /dev/null 2>&1 || return 1
  git -C "$HOME/BigMoE" diff --quiet HEAD -- "${ENVF#$HOME/BigMoE/}"
}
until ready; do
  [ "$(date +%s)" -lt "$t_late" ] || { log "no committed dry-run env by $LATEST: nothing started (the Tue-night slot)"; exit 3; }
  sleep 60
done
cp -p "$ENVF" "$RUNENV" && [ "$(git hash-object --no-filters "$RUNENV")" = "$b" ] \
  || { log "the frozen copy $RUNENV of $ENVF failed or differs: nothing started"; exit 2; }
log "env ready: blob $b, frozen as $RUNENV: $(grep '^#' "$RUNENV" | head -1 | cut -c1-160)"
if [ -n "${W24S_DRY:-}" ]; then log "DRY: would run CARD_YIELD_GB=$YIELD CARD_TAG=w24 CARD_FENV=$RUNENV CARD_CUTOFF=$CUT tools/day_runner61.sh"; exit 0; fi
CARD_YIELD_GB="$YIELD" CARD_TAG=w24 CARD_FENV="$RUNENV" CARD_CUTOFF="$CUT" tools/day_runner61.sh
rc=$?
log "runner 61 (w24) ended rc=$rc"
exit "$rc"
