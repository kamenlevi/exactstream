#!/bin/sh
# MTP Head is in pacer SLEEP (Mon 11-02 10:0x -> ~Tue 15:09): forward D-23e's and D-23x's result lines to NewIdeas with
# `peer say --to ojox-bd` (no Claude usage) as soon as after_hpd2.sh logs each ticket done. Deadline 18:30.
cd $HOME/TrainLab
LOG=logs/turns.log
STOPAT=$(date -j -f "%Y-%m-%d %H:%M" "2026-11-02 18:30" +%s)
N0=$(wc -l < $LOG)
waitfor() { while [ $(date +%s) -lt $STOPAT ]; do tail -n +$((N0 + 1)) $LOG | grep -q -E "$1" && return 0; sleep 60; done; return 1; }
if waitfor "after_hpd2: (D-23e ticket done|D-23e not granted|18:00 without)"; then
  if tail -n +$((N0 + 1)) $LOG | grep -q "D-23e ticket done"; then
    H=$(grep 'D-23e CHATSWITCH read-out' data/d23d_read.txt | tail -1 | sed -E 's/.*: ([0-9]+ of [0-9]+) prompts clean in both arms; every token equal across arms: (True|False).*/\1 clean, tokens equal \2/')
    P=$(grep 'PRIMARY chat' data/d23d_read.txt | tail -1 | sed -E 's/.*time-weighted G ([+-][0-9.]+) %, geo-mean ratio ([0-9.]+), mtp15e faster on ([0-9]+).*\): ([A-Z]+)(.*)$/chat G \1 %, geo \2, faster on \3 -> \4\5/')
    A=$(grep 'SECONDARY agent' data/d23d_read.txt | tail -1 | sed -E 's/.*G ([+-][0-9.]+) %, geo-mean ([0-9.]+) -> agent (joins|stays).*/agent G \1 %, geo \2, \3/')
    M=$(grep 'own rates' data/d23d_read.txt | tail -1 | sed -E 's/.*the switch ([+-][0-9.]+) % whole set.*/whole set \1 %/')
    peer say --to ojox-bd "MTP Head (auto; I sleep to Tue 15:09 per pacer): D-23e = chat switch at C 224: $P; $A; $M; $H. REPORT verdict on wake; data/d23d_read.txt" > /dev/null 2>&1
  else
    peer say --to ojox-bd "MTP Head (auto): D-23e did not run today (no 47 GB NO-SWAP / no grant by 18:00); I queue it on wake (Tue 15:09)." > /dev/null 2>&1
  fi
fi
if waitfor "after_hpd2: (D-23x ticket done|D-23x not granted|18:00 without 44)"; then
  if tail -n +$((N0 + 1)) $LOG | grep -q "D-23x ticket done"; then
    for w in "check 1 " "check 2-6"; do
      X=$(grep -E "^    agent  (dfl|mtp15e) +check $w" data/d23x_read.txt | tail -2 | awk '{gsub(/ +/, " "); sub(/^ /, ""); print}' | sed -E 's/base demand\+joint/base d+j/; s/base prefetch/pf/; s/sidecar demand\+joint/side d+j/; s/sidecar prefetch/pf/' | tr '\n' ' ')
      peer say --to ojox-bd "MTP Head (auto) D-23x agent $w(reads per check): $X" > /dev/null 2>&1
    done
  fi
fi
