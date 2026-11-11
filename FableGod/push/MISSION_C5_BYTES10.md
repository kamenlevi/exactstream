# MISSION C5 (Fable C): BYTES10. From ~62 to <= 10 MB of SSD reads per kept token.

(Draft by the orchestrator for Kamen, Tue 3 Nov 2026 23:5x. Starts on Kamen's go.)

## Why

Kamen's bar for the SSD wall: it counts as broken when RAM stops deciding the speed. Two numbers:
- the GPU never waits for the SSD (exposed SSD ~0);
- SSD MB per kept token <= ~10, first at 24 GB, then 12 GB. Today it is 62.4 [K] at 45 GB (C 256, exact).

At ~13 GB/s, 10 MB per kept token leaves >= 1,300 tok/s of disk headroom. A 125B would then run at its compute speed from a fraction of its size in RAM. That is the "10x less RAM" claim behind the project's level 3.

> (a quote from the project owner, kept private)

Your own C1-2 idea (QOM) is tonight's best FAST arm: -60 / -37 / -33 % [P] bytes on code / IFEval / multi, with the fewest flips of any FAST rule. One 2.5x factor exists. A ~6x product needs two more.

## Inputs

- NewIdeas' BYTES10 track. It has two files:
  - ~/NewIdeas/BYTES10.md (88be71b): the scoreboard per RAM line and track, the owners, and the factor ledger with overlap flags;
  - ~/NewIdeas/research/TEN_X_LEDGER.md: the ledger's numbers. Its replay starts ~04:30, after BigMoE's quiet lock. Partial rows land by ~06:00 (the kept-per-check curve, per-request working sets, QOM reads at 45 / 24 / 12 GB), and the full ledger well before Wed 13:30.
  WALL1_SCOREBOARD.md's BROKEN section points to both. The ledger measures each factor on existing captures:
  - kept-per-check -> union curve;
  - per-request working sets (top-N coverage at 90 / 95 / 99 %);
  - the priced skip / precision factors (QOM, t3 sure rows, COLD, 11-bit, M-01, the volatile tier);
  - their product per domain.
- FLOOR night 1 (Breakthrough, runs to Wed 13:00; ~/Breakthrough/data/floor1). The decision map at scale, 42 held-out windows.
- TRIED/NOVELTY.md, for prior-art tags. Note: the drafter-certainty gate lost its NEW tag tonight to PSR (2610.34721).
- Earlier work: push/SSD_WALL2.md and push/WALL1_D1_SPEC.md (Fable A); fablec/UNION_STACK.md (your C4); lunatic/LUNACY2.md; ~/NewIdeas/FAST_TRACK.md.

## What to deliver: ~/FableGod/fablec/BYTES10.md

1. **The ledger audit.** Which factors truly multiply, and which overlap (double counting)? Per domain, EXACT vs FAST.
2. **The floor.** The minimum SSD MB per kept token at 45 / 24 / 12 GB, given the measured routing statistics. Cover:
   - per-request working sets;
   - union growth with kept rows;
   - drafter acceptance;
   - the 4.0 / 3.x-bit record sizes.
   Say what binds at each RAM size.
3. **A ranked path to <= 10 MB** at 24 GB, then 12 GB. Give each lever:
   - its halved gain;
   - its odds;
   - its cost in builds and GPU minutes;
   - its exactness;
   - its novelty tag.
4. **At least one genuinely new mechanism**, aimed at "RAM size is a false limit". Check it against NOVELTY.md and a web prior-art pass, and say how it would be shown in a paper.
5. **The first three pre-registered screens** for Kamen's research day (Wed 13:45). Each <= 10 min of GPU, with its line written before it runs.

## Rules

- No "impossible". Every finding gives the best measured number and the next step. Floors are targets.
- Mark [K] measured (source), [P] arithmetic, [G] judgement. Price EXACT and FAST both.
- CPU only: no GPU runs, no downloads. Write only ~/FableGod/fablec/ (BYTES10.md, plus scripts there).
- Keep any personal context out of files.
- Deliver by Wed 13:00. Read the ledger and FLOOR's read-out when they land; start from the captures before then.

## Cost

About 1-1.5 % of main's weekly usage (CPU reading and arithmetic, like C3 / C4). No machine time.
