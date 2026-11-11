#!/bin/sh
# NewIdeas 10-22 09:16: the 35B layer-local sticky-router kill test (sticky35.py; ~35 min of a 55 min turn, ~25 GB)
cd $HOME/TrainLab
$HOME/BigMoE/.venv/bin/python sticky35.py run --betas 0.44,2,8,32,8r --train-min 16 2>&1 | grep --line-buffered -v "Token indices"
