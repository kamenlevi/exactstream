#!/bin/sh
# NT-1 runner (NewIdeas, Mon 2026-10-26): wait for a quiet GPU slot, replay L1-20 at block 32 (~1 min), release.
cd "$HOME/NewIdeas" || exit 1
PEER_QUEUE_TIMEOUT=14400 peer lock quiet --queue 6m "NewIdeas NT-1: L1-20 drafter replay at block 32 + the 125B's top-32 at first misses (~1 min GPU, ~2 GB)" || exit 2
echo "lock $(date +%H:%M:%S)" > data/nt/run.log
NT_DEV=gpu "$HOME/BigMoE/.venv/bin/python" scripts/nt_accept.py >> data/nt/run.log 2>&1
echo "rc $? $(date +%H:%M:%S)" >> data/nt/run.log
peer unlock
