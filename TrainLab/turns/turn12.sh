#!/bin/sh
# NewIdeas 10-22 09:16: the 35B layer-local sticky-router kill test, betas in the useful range (turn 11's 2 / 8 / 32
# overshot to U16 13-17 by step 200), 14 min of training
cd $HOME/TrainLab
$HOME/BigMoE/.venv/bin/python sticky35.py run --betas 0.05,0.1,0.2,0.44,0.2r --train-min 14 2>&1 | grep --line-buffered -v "Token indices"
