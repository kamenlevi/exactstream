# DFlash2 teacher-forced at block 32 is not a 98 % corrector (Sun 15 Nov 06:02; WEEK98 S1 DF-TF)
- **What:** the 12 GB engine's DFlash2 drafter (l120_ft), shown the TRUE earlier block tokens, on the 125B bank (HE+ 24,
  MBPP 100), block 32. Strict conditional accuracy counts a depth only when all earlier ones hit.
- **Result [K]:** HE+ strict falls from 97.3 % (depth 0) to 89.0 % (7-10) and 88.1 % (23-30). MBPP falls from 91.4 % to
  ~73 % (7-14); no MBPP anchor reaches depth 23.
  - Teacher forcing barely moves kept/32: 11.76 vs 11.46 native (HE+); 6.25 vs 6.62 (MBPP).
- **So:** a 5-layer target-fed block drafter does not compose 30 tokens at 98 %. KILL on the pre-registered line.
- Sources: /Users/kamen/Fit12/data/scr/s1tf; ~/FableGod/deepdraft/WEEK98.md (Sun 15 02:41); /Users/kamen/TRIED/fit12.md.
