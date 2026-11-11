#!/bin/zsh
# M-31 HEADPING runner (~/NewIdeas/QUEUE.md M-31): build, wait out PAUSE, take the quiet lock, ~10 s of SSD micro calls
# (scripts/ssdmicro4.c), unlock, summarize -> data/ssdmicro4.txt (raw calls:
# scratch/ssdmicro4_raw.tsv). Default QoS on purpose (taskpolicy -b throttles disk I/O).
# usage: zsh scripts/run_ssdmicro4.sh [rounds, default 150]
cd ~/MoonShots || exit 1
while [ -e ~/.local/share/quiet/PAUSE ]; do sleep 20; done
mkdir -p scratch
clang -O2 -o scratch/ssdmicro4 scripts/ssdmicro4.c || exit 1
peer lock quiet --queue 30m "M-31 HEADPING: ~10 s of SSD micro reads (records3, read-only); needs a quiet SSD" || exit 1
scratch/ssdmicro4 ${1:-150} > scratch/ssdmicro4_raw.tsv
rc=$?
peer unlock
[ $rc -eq 0 ] && ~/BigMoE/.venv/bin/python scripts/ssdmicro4_sum.py scratch/ssdmicro4_raw.tsv
exit $rc
