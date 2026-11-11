#!/bin/sh
# Overnight resume of the Bare-Speed pack (Kamen: run at 01:00). Keeps >= 200 GB free on /:
# before each file, (free space - bytes still to fetch for it) must stay >= 200 GB, else stop.
# Removes its own pid file when done; nothing persistent is installed.
D=$HOME/BigMoE/models; cd "$D/bare" || exit 1
LOG=$D/night.log; echo $$ > $D/night.pid
now=$(date +%s); start=$(date -j -f "%Y-%m-%d %H:%M:%S" "$(date -v+1d +%Y-%m-%d) 01:00:00" +%s)
[ "$(date +%H)" -lt 1 ] && start=$(date -j -f "%Y-%m-%d %H:%M:%S" "$(date +%Y-%m-%d) 01:00:00" +%s)
echo "$(date) waiting $(( (start-now)/60 )) min until 01:00" >> $LOG
sleep $(( start - now ))
FLOOR=$((200 * 1000 * 1000 * 1000))
ok_to_fetch() {   # $1 file, $2 size
  have=$(stat -f %z "$1" 2>/dev/null || echo 0); need=$(( $2 - have ))
  free=$(( $(df -k / | tail -1 | awk '{print $4}') * 1024 ))
  [ $(( free - need )) -ge $FLOOR ]
}
export -f ok_to_fetch 2>/dev/null
echo "$(date) start; free $(df -h / | tail -1 | awk '{print $4}')" >> $LOG
# 4 at a time; each worker checks the floor right before its file
while read f size; do
  while [ "$(pgrep -f fetch_one.sh | wc -l)" -ge 4 ]; do sleep 5; done
  if ! ok_to_fetch "$f" "$size"; then echo "$(date) STOP: $f would cross the 200 GB floor" >> $LOG; break; fi
  "$HOME/BigMoE/scripts/fetch_one.sh" "$f" >> $LOG 2>&1 &
done < "$D/files_sizes.txt"
wait
echo "$(date) done; $(du -sh . | cut -f1) on disk; free $(df -h / | tail -1 | awk '{print $4}')" >> $LOG
rm -f $D/night.pid
