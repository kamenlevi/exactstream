# FULL, round 2: the little things (Wed 4 Nov ~21:4x -> ~23:00)

> (a quote from the project owner, kept private)

## 1. Your bet, re-registered
**M-39** (MoonShots, `~/MoonShots/data/span_relax_check.txt`): KILL by the letter.
- 4 of rule A's 70 relaxed accepts (5.7 %; B 6.8 %) sat where the region scan and the token view disagree.
- In all 4, the 125B was about to open a comment (top token ' #'), the scan already said "comment", and rule A accepted code (' for', ' if').
- Otherwise: A 12.65, B 12.78 kept at 0.14 % code-token deviation.
- With the engine's causal tagger (EvalSuite `relax_core.CodeScan`): A 12.69 with 0 code drafts, B 12.78 at 0.11 %. That is post hoc.

Deliver:
- the rule with the causal tagger, written exactly;
- fresh data not used in M-39, and its cost. The other 12 windows lack the 125B's top-8: name a teacher-forced pass and its GPU minutes, or MBPP walks;
- PASS / KILL lines;
- the length-inflation check for the engine test: tokens per task and seconds per task vs the plain FAST stack. arXiv 2610.23818 reports 1.34x more accepted tokens but 1.54x longer completions under relaxed verification.

## 2. R1, parts of one answer in one pass: its first tests
- **(a) CPU, on recorded HumanEval+ / MBPP answers:** how many split into >= 2 independent parts, and what share of tokens sits in the parallel parts.
- **(b) The quality risk:**
  - what prompt or decoding constraint makes the 125B write skeleton-first;
  - the EvalSuite same-scores test (GPU minutes, PASS / KILL).
- **(c) Speed:** the arithmetic on Leap's par32 numbers at 45 GB, for today's ~90 ms check and for an SSD-free trellis check.
  - Context: Leap's TRL-NAX is 0.87x NAX6-N; screen 7 tonight judges the trellis store's quality.
- **Prior art:** Skeleton-of-Thought, APAR, Hogwild! inference, PASTA. Say what would be new here.

## Deliver (by ~23:00)
- Append "## Round 2" to ~/FableGod/full/FULL.md.
- Then send one message: the re-registered rule's PASS line, and R1's first test, in two lines.
