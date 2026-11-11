#!/bin/zsh
# M-29 SSDMICRO2 runner (~/NewIdeas/QUEUE.md M-29): build, wait out PAUSE, take the quiet lock, ~20 s of SSD micro calls
# (scripts/ssdmicro2.m; ~1.1 GB of pre-faulted slot memory), unlock, summarize -> data/ssdmicro2.txt (raw calls:
# scratch/ssdmicro2_raw.tsv). Default QoS on purpose (taskpolicy -b throttles disk I/O).
# usage: zsh scripts/run_ssdmicro2.sh [rounds, default 150]
cd ~/MoonShots || exit 1
while [ -e ~/.local/share/quiet/PAUSE ]; do sleep 20; done
mkdir -p scratch
clang -O2 -fobjc-arc -framework Metal -framework Foundation -o scratch/ssdmicro2 scripts/ssdmicro2.m || exit 1
peer lock quiet --queue 30m "M-29 SSDMICRO2: ~20 s of SSD micro reads (records3, read-only); needs a quiet SSD" || exit 1
scratch/ssdmicro2 ${1:-150} > scratch/ssdmicro2_raw.tsv
rc=$?
peer unlock
[ $rc -eq 0 ] && ~/BigMoE/.venv/bin/python scripts/ssdmicro2_sum.py scratch/ssdmicro2_raw.tsv
exit $rc
