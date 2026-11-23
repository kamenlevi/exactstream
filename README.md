# ExactStream research data

This branch holds the research behind ExactStream: the measurements, tests, ideas, scripts and records the papers cite.
The engine is on the `main` branch. The papers are at https://kamenlevi.github.io/exactstream.

Research paused on 19 November 2026. The 23 November snapshot holds everything measured up to the pause; the latest
results and predictions are written up at https://kamenlevi.github.io/exactstream/latest-results.html.

It is a snapshot of the research folders, refreshed every hour while the work goes on. Session notes, personal notes and
third-party code are left out, and private details are removed before each push.

The papers' source keys are paths below this root. For example, LP:data/x.md is `Leap/data/x.md` and BM:docs/x.md is
`BigMoE/docs/x.md`. Each paper lists all of its keys in its sources section.

- `BigMoE`: the streaming engine and its records
- `Leap`: kernels, prefill and the release builds
- `EvalSuite`: the quality gates, with every pre-registration and result
- `Breakthrough`, `NewIdeas`, `MoonShots`, `Fit12`, `TrainLab`: the research tracks
- `Sandbox98`: a 2B model with its own draft head, a minutes-long sandbox for drafter tests (from 23 November)
- `FableGod`: mission analyses, dated findings, and the TRIED ledger (one line per idea, with its number and verdict)
- `StrataEval`, `PrefillKernels`: the head-to-head harness and the prefill kernel findings

The snapshots from 2 to 11 November 2026 are also at https://github.com/kamenlevi/350-tps, the address the papers'
1.1.0 edition cites.

To download only this branch:

    git clone --branch research --single-branch --depth 1 https://github.com/kamenlevi/exactstream exactstream-research
