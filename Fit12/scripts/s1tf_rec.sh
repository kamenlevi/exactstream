#!/bin/sh
# Fit12 s1tf_rec.sh (Sun 15 Nov 01:3x): sub2's window closes ~01:45 (orchestrator 01:31), so this records S1 DF-TF's result
# without me: when /Users/kamen/Fit12/data/scr/s1tf/DONE appears (by 06:00), it appends s1tf.txt verbatim (+ DONE) to the TRIED
# file and the plan. s1tf.py computes the verdict itself (its LINE in its header).
O=/Users/kamen/Fit12/data/scr/s1tf; TE=$(date -j -f "%Y-%m-%d %H:%M" "2026-11-15 06:00" +%s)
while [ ! -e $O/DONE ] && [ "$(date +%s)" -lt "$TE" ]; do sleep 30; done
r=$(cat $O/s1tf.txt 2>/dev/null | tr '\n' ' '); d=$(cat $O/DONE 2>/dev/null)
echo "- WEEK98 S1 DF-TF [K, auto-recorded $(date '+%m-%d %H:%M') by s1tf_rec.sh] (DFlash2 l120_ft teacher-forced at block 32 on the 125B bank; MBPP 100 + HE+ 24; LINE: PASS if TF HE+ 23-30 >= 95.0 within 3 pt of depth 0 and MBPP 23-30 >= 85; BETWEEN 90-95; KILL < 90): ${r:-NO RESULT FILE} | DONE: ${d:-none} | data: $O/s1tf.txt, s1tf.json | [fit12]" >> /Users/kamen/TRIED/fit12.md
printf '\n## WEEK98 S1 DF-TF RESULT (auto-recorded %s by s1tf_rec.sh; the script s1tf.py computes the verdict)\n- %s\n- DONE: %s; data %s/s1tf.txt, s1tf.json, s1tf_run.txt\n' "$(date '+%a %d %b %H:%M')" "${r:-NO RESULT FILE}" "${d:-none}" "$O" >> /Users/kamen/Fit12/PLAN_12GB_90_fit12.md
echo "$(date +%H:%M:%S) SCR s1tf recorded (TRIED + PLAN)" >> /Users/kamen/Fit12/logs/summary.txt
