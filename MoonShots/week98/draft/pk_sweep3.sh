#!/bin/sh
# PK skeleton v3 GEMV config sweep (MoonShots, Wed 18 Nov): one config per line, ms per pop at S1 192 (share lock)
P=$HOME/BigMoE/.venv/bin/python
for cfg in "--r8 4,2" "--r8 2,4" "--r8 4,1" "--r8 1,5" "--r8 2,2" "--rh 4,1" "--rh 1,5" "--rh 2,5" "--rh 4,5" "--re 2,5" "--re 4,1" "--rs 2,2" "--rs 1,4" "--rs 4,2"; do
  r=$($P -I -B pk_skel3.py --s1 192 --rounds 6 --trees 2 $cfg 2>&1 | grep "ms per pop" | sed 's/.*skel \([0-9.]*\).*fails \([0-9]*\).*/\1 fails \2/')
  echo "$cfg -> $r"
done
