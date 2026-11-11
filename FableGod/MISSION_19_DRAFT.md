# MISSION 19: THE DATA SAFARI. Find the next "copies" in data we already have.

(Draft by the orchestrator for Kamen, Mon 2 Nov 2026 ~17:5x.)

> (a quote from the project owner, kept private)

**Rules of the hunt**
- CPU only, on files we already have. No GPU, no downloads, writes only in ~/FableGod.
- Every finding needs a measured size, from a script, before any pricing.
- Price gains conservatively. Today your profile model ran ~2x high against MATCH's real walk (x1.038 vs x1.071), so treat every modelled gain as an upper bound and say which real walk or run would confirm it.
- Check ~/TRIED/TRIED.md for each lever. Name the closest entry, and why its kill reason does or doesn't apply.
- Mark every number as measured / arithmetic / judgement.

**Hypotheses to measure** (keep the ones that hold, add up to two of your own)
1. **The 125B reuses its experts.** Inside a copied span, does it route to the same experts per layer as at the source? Beyond copies: do repeated tokens or repeated short contexts reuse experts, and how far ahead is that known?
   - Data: the decision map's routing capture (~/FableGod/decision_map), MoonShots' verify_trace125 / mtp_route_pred125, BigMoE's b31 S10 data.
   - The lever: exact expert prefetch (S10's perfect forecasts were worth +35-44 % at 24 GB).
2. **Sure runs.** The 125B's sure positions come in runs. How long are they on code, math and chat, and what predicts where a run ends: copy length, line ends, the drafter's confidence, token kind?
   - Data: relax_positions.json, the walks.
   - The lever: the check length per check (rows only where tokens will be kept).
3. **Where the GPU waits.** Per-check timing: draft, build, wait, check and commit, misses, SSD ms, the GPU's spin time on misses. How much of each check is the GPU idle by day, and what could run there (the next draft, a cheap certificate pass, prefetch)?
   - Data: ~/BigMoE/data/r57/a6?_160_checks.json, a6*_160.txt, ~/Leap/data/k80, c0.
4. **An answer's working set.** Do the experts used in an answer's first ~100 tokens predict its later experts? A6 found the cold-expert burst is about half of the start-up cost.
   - Data: as for 1.
   - The lever: warm the cache at the start (A6), and an eviction policy per answer.
5. **Token kinds.** Do some kinds of tokens (whitespace, punctuation, identifiers, numbers) need fewer experts, or flip less under cuts? A gate by the draft's token kind is known before the check.
   - Data: the decision map's per-row tables, relax_positions.json, vocab.json.

**Deliver**
1. A table: each hypothesis, the measured size, and the lever it implies (exact or quality-defined).
2. The top 3 by expected gain on code at 45 GB (and at 24 / 12 GB where the lever is SSD-side). For each: the mechanism, a conservative estimate, the closest TRIED entry, and a test of one night or less with PASS / KILL lines and an owner (MTP Head drafters, BigMoE engine, Leap kernels, MoonShots / BigMoE routing).
3. Anything surprising you saw that doesn't fit a hypothesis: one line each.

**Inputs**
- ~/FableGod/decision_map/ (routing capture, per-row tables), relax_positions.json, drafter_anatomy.json, drafter_memory_probe.py, FLOOR_125B.md, DRAFTER_WALL.md, ANE_DRAFTER.md, DAY9_2026-11-02.md
- ~/MoonShots/data/ (verify_trace125, mtp_route_pred125, band45_capture.txt)
- ~/BigMoE/data/ (r57/a6*, b31/s10b*, trace files)
- ~/Leap/data/ (k80, c0, ane_probe.txt)
- ~/TrainLab/data/ (the walks, match_read.txt)
- ~/BigMoE/models/bare/vocab.json
- ~/TRIED/TRIED.md
