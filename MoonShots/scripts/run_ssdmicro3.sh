#!/bin/zsh
# M-30 QUEUEMIX runner (~/NewIdeas/QUEUE.md M-30): build, wait out PAUSE, take the quiet lock, ~10 s of SSD micro calls
# (scripts/ssdmicro3.c), unlock, summarize -> data/ssdmicro3.txt (raw calls:
# scratch/ssdmicro3_raw.tsv). Default QoS on purpose (taskpolicy -b throttles disk I/O).
# usage: zsh scripts/run_ssdmicro3.sh [rounds, default 150]
cd ~/MoonShots || exit 1
while [ -e ~/.local/share/quiet/PAUSE ]; do sleep 20; done
mkdir -p scratch
clang -O2 -o scratch/ssdmicro3 scripts/ssdmicro3.c || exit 1
peer lock quiet --queue 30m "M-30 QUEUEMIX: ~10 s of SSD micro reads (records3, read-only); needs a quiet SSD" || exit 1
scratch/ssdmicro3 ${1:-150} > scratch/ssdmicro3_raw.tsv
rc=$?
peer unlock
[ $rc -eq 0 ] && ~/BigMoE/.venv/bin/python scripts/ssdmicro3_sum.py scratch/ssdmicro3_raw.tsv
exit $rc
