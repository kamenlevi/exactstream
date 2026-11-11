#!/bin/bash
# NATIVE-CK runner (NewIdeas Sun 8 Nov 16:1x): waits for the orchestrator's Strata arena-33 re-run and BigMoE's 32 GB chat
# arm to finish (a new 'done rc' line in turn45.log after L0, or 16:50), then for 2 consecutive minutes with no quiet lock held or
# queued by anyone; then one share lock (~1 min, < 1 GB), M-B (native) and M-A (MLX), both under taskpolicy -b; unlock.
# Gives up at 17:00 (EvalSuite resumes 17:07; chw18 needs the gap) and says so.
export PATH="$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:$PATH"
D=$HOME/NewIdeas/scripts/native_ck; OUT=$HOME/NewIdeas/data/native_ck.txt
L0=${L0:-$(wc -l < $HOME/BigMoE/data/turn45.log)}
END=$(date -j -f "%Y-%m-%d %H:%M" "2026-11-08 17:00" +%s); SOFT=$(date -j -f "%Y-%m-%d %H:%M" "2026-11-08 16:50" +%s)
quietbusy() { peer ls 2>/dev/null | grep -E "^- " | grep -v "NewIdeas" | grep -cE "benchmarking: quiet|waiting for a quiet"; }
armdone() { tail -n +$((L0 + 1)) $HOME/BigMoE/data/turn45.log | grep -q "done rc" || [ $(date +%s) -ge $SOFT ]; }
stratadone() { tail -1 $HOME/StrataEval/h2h/logs/waiter.log | grep -q "WAITER END"; }
clear=0
while [ $(date +%s) -lt $END ]; do
  if stratadone && armdone && [ "$(quietbusy)" = "0" ] && [ ! -e $HOME/.local/share/quiet/PAUSE ]; then clear=$((clear + 1)); else clear=0; fi
  [ $clear -ge 2 ] && break
  sleep 60
done
if [ $clear -lt 2 ]; then echo "$(date '+%H:%M:%S') NATIVE-CK: no clear gap by 17:00, not run" | tee -a $OUT; exit 3; fi
peer lock share 5m --cpu 100 --gpu 20 "NewIdeas NATIVE-CK microbench (<1 GB, ~1 min)" > /dev/null 2>&1
{ echo "== NATIVE-CK run $(date '+%a %d %b %H:%M:%S') (taskpolicy -b; share lock; vm NO-SWAP ok)"
  cd $D && swiftc -O mb_native.swift -o mb_native 2>&1 | tail -2
  taskpolicy -b ./mb_native
  taskpolicy -b ./mb_native
  taskpolicy -b $HOME/BigMoE/.venv/bin/python -I ma_mlx.py
  echo "== end $(date '+%H:%M:%S')"; } >> $OUT 2>&1
peer unlock > /dev/null 2>&1
tail -8 $OUT
