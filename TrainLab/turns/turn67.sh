#!/bin/sh
# MTP Head 23:2x: the DLOG ticket the session queued by hand (the peers hook blocks starting a chain while others measure):
# turn66 (C 160 / 128, to 20K answer tokens per language), then night9_chain detached for the rest (its NIGHT9_AFTER mode
# first waits for this ticket's release): more DLOG tickets if short, E1 + E5 (night8_train), DLOG top-ups.
cd $HOME/TrainLab
sh turns/turn66.sh; rc=$?
NIGHT9_AFTER=1 nohup sh turns/night9_chain.sh > /dev/null 2>&1 &
exit $rc
