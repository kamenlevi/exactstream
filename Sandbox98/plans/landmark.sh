# LANDMARK-ANS screen 1 (Breakthrough's pre-registration ~/Breakthrough/docs/IDEAS_1117_landmark.md + ADDENDUM + the 08:05
# alignment note; NewIdeas' queue after B98S1; results/LEVERS_A.md section LANDMARK): eval-only on the frozen s30 head,
# HE 60 + MBPP 100, modes own / true / lm1 / lm05 / stale1 (+ copy (i) / (ii) / copy1 from own's cells), covered-cell
# walk-strict G by match length; then the verdict.
NAME=landmark
plan() {
  step lm logs/landmark_s1.log 40 $T -l $PY sb/landmark.py --sets he,mbpp --nh 60 --nm 100 --depth 31 --out results/landmark_s1.json
  [ "$DRY" = 1 ] && { VERDICT="(dry)"; return; }
  $PY -I sb/landmark_verdict.py > logs/landmark_verdict.log 2>&1
  VERDICT=$(grep -E "^VERDICT" logs/landmark_verdict.log | awk '{print $2, $3}' | tr '\n' ' ' | cut -c1-250)
}
