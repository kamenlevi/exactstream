#!/bin/sh
# Returns when the overnight download has finished (night.pid gone and "done" logged).
D=$HOME/BigMoE/models
until [ ! -f $D/night.pid ] && grep -q "done;" $D/night.log 2>/dev/null; do sleep 300; done
tail -3 $D/night.log; du -sh $D/bare | cut -f1; df -h / | tail -1
