#!/bin/sh
# fetch_one.sh FILE : resumable download of one file of the Bare-Speed pack into the cwd
B=https://huggingface.co/Youssofal/Qwen3.8-Flash-Next-MTPLX-Bare-Speed/resolve/main
mkdir -p "$(dirname "$1")"
for i in 1 2 3 4 5 6; do
  curl -sSL --retry 10 --retry-delay 5 --speed-limit 100000 --speed-time 120 -C - -o "$1" "$B/$1" && exit 0
  sleep 5
done
echo "FAILED $1"
