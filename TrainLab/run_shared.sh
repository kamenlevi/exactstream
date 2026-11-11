#!/bin/sh
# TrainLab: run one proof-of-concept job under a SHARED lock (FAST INTEGRATION, Kamen Fri 10-30 18:4x): take
# `peer lock share` (waiting while a quiet lock blocks shared runs), run the job with a time limit, then unlock.
# Its numbers are "shared GPU, rough": acceptance / tokens per pass / exactness only, never a speed result.
# usage: run_shared.sh <minutes> <cpu%> <gpu%> <job> "<why>"      log: logs/turns.log, job output: logs/turn_job.out
T=$HOME/TrainLab; LOG=$T/logs/turns.log; MIN=${1:-15}; CPU=${2:-200}; GPU=${3:-50}; JOB=$4; WHY=${5:-$4}
log() { echo "$(date '+%m-%d %H:%M:%S') $*" >> "$LOG"; }
until peer lock share ${MIN}m --cpu $CPU --gpu $GPU "MTP Head (shared): $WHY" > /dev/null 2>&1; do
  peer wait > /dev/null 2>&1; sleep 5
done
log "share lock held; running $JOB"
LIM=$(( MIN * 60 - 60 ))
set -m
( TL_LIMIT=$LIM "$JOB" >> "$T/logs/turn_job.out" 2>&1 ) & P=$!
i=0
while kill -0 $P 2>/dev/null && [ $i -lt $LIM ]; do sleep 5; i=$((i + 5)); done
if kill -0 $P 2>/dev/null; then kill -TERM -- -$P; sleep 20; kill -9 -- -$P 2>/dev/null; log "turn time up: job stopped"; fi
wait $P; rc=$?
peer unlock > /dev/null 2>&1; log "job done (rc $rc), share lock released"
