#!/bin/zsh
# M-29b runner (~/NewIdeas/QUEUE.md M-29b): ssdmicro2.m in its "big" mode (4 / 8 / 15 records per call), quiet lock,
# ~10 s, unlock, summary appended to data/ssdmicro2.txt (raw calls: scratch/ssdmicro2b_raw.tsv).
# usage: zsh scripts/run_ssdmicro2b.sh [rounds, default 150]
cd ~/MoonShots || exit 1
while [ -e ~/.local/share/quiet/PAUSE ]; do sleep 20; done
mkdir -p scratch
clang -O2 -fobjc-arc -framework Metal -framework Foundation -o scratch/ssdmicro2b scripts/ssdmicro2.m || exit 1
peer lock quiet --queue 30m "M-29b ZEROCOPY-24: ~10 s of SSD micro reads (records3, read-only); needs a quiet SSD" || exit 1
scratch/ssdmicro2b ${1:-150} big > scratch/ssdmicro2b_raw.tsv
rc=$?
peer unlock
[ $rc -eq 0 ] && ~/BigMoE/.venv/bin/python scripts/ssdmicro2b_sum.py scratch/ssdmicro2b_raw.tsv
exit $rc
