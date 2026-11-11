#!/bin/zsh
# SG12 step 0: wait out PAUSE (slock gates on it), shared lock, the synthetic micro-test, release.
source /Users/kamen/Fit12/scripts/screens.sh
slock "Fit12 SG12 step 0: synthetic MoE micro-test, <2 min, ~3.5 GB, no SSD" 10m 120 0000 12 || { log "sg0: no lock"; exit 1; }
log "sg0 start (NO-SWAP $(avail_gb) GB)"
$PY /Users/kamen/Fit12/scripts/sg12_micro.py > /Users/kamen/Fit12/data/sg12/step0.txt 2>&1; rc=$?
unlock
log "sg0 done (rc $rc): $(tail -1 /Users/kamen/Fit12/data/sg12/step0.txt)"
