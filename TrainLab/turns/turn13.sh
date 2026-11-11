#!/bin/sh
# NewIdeas 10-22 11:12 (approved): real 125B KLD of the per-pass union budget at 4-row verify passes on the Eval suite's
# reference engine: exact 4-row floor, budget U 14 / 16 / 18, OEA k0 4 / 5; the suite's four tf parts, every 4th sequence
cd $HOME/TrainLab
$HOME/BigMoE/.venv/bin/python sticky/budget125_tf.py 4 exact,b16,o4,b14,b18,o5 tf_gsm,tf_ifeval,tf_code,tf_multi 4 2>&1 | grep --line-buffered -v "Token indices"
