#!/bin/zsh
# M-28 SSDMICRO runner (~/NewIdeas/QUEUE.md M-28): build, wait out PAUSE, take the quiet lock, ~1 min of SSD micro calls
# (scripts/ssdmicro.c), unlock, summarize against the lines -> data/ssdmicro.txt (raw calls: scratch/ssdmicro_raw.tsv).
# Default QoS on purpose (taskpolicy -b throttles disk I/O). usage: zsh scripts/run_ssdmicro.sh [rounds, default 100]
cd ~/MoonShots || exit 1
while [ -e ~/.local/share/quiet/PAUSE ]; do sleep 20; done
mkdir -p scratch
clang -O2 -o scratch/ssdmicro scripts/ssdmicro.c || exit 1
peer lock quiet --queue 30m "M-28 SSDMICRO: ~1 min of SSD micro reads (records3, read-only); needs a quiet SSD" || exit 1
scratch/ssdmicro ${1:-100} > scratch/ssdmicro_raw.tsv
rc=$?
peer unlock
[ $rc -eq 0 ] && ~/BigMoE/.venv/bin/python scripts/ssdmicro_sum.py scratch/ssdmicro_raw.tsv
exit $rc
