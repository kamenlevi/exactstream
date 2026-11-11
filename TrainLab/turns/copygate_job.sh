#!/bin/sh
# COPYGATE job (MTP Head runs it, Mon 11-02): Fable's copygate session's gated MATCH walk, inside a held SHARE lock.
# The script and the walk copy live in ~/FableGod/copygate (reviewed 18:4x: gate_walk.sh, d02b_walk_gate.patch, learn()
# uses only positions <= k, i.e. what a check reveals); outputs: ~/TrainLab/data/d02b_walk_g_<arm>.*, data/gate_read.txt.
exec sh $HOME/FableGod/copygate/gate_walk.sh
