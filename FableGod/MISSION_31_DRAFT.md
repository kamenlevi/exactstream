# MISSION 31: THE ASSUMPTION HUNT. Find the wrong assumption the whole project rests on.

> (a quote from the project owner, kept private)

**Why.** Tonight's verdict on novelty (NewIdeas, ~/TRIED/NOVELTY.md): of 215 levers in use or passing, 30 are published, 55 adapted, 127 engineering and 3 new. Your screen of cloud's six (NEWIDEAS_SCREEN.md) found most were new forms of known moves. Incremental search keeps finding incremental ideas.

Breakthroughs usually come from an assumption everyone makes that turns out false for a specific setup. Ours is unusual:
- one user and one Mac
- unified memory, plus an SSD that wakes slowly (MoonShots' M-20)
- a Neural Engine and media engines sitting idle
- a hybrid GDN / attention model with 512 fine-grained experts
- speculative 16-row checks
- a drafter that is right 90-98 % of the time

**Deliver**
1. **The assumptions.** List the 12-15 assumptions this project's design rests on, without being told what they are. Starting points, not limits:
   - every verified row needs its top-10 experts
   - each expert's weights are read once per check
   - each token's expert outputs are computed fresh
   - the dense trunk runs fully for every row
   - verification runs all 48 layers on all rows
   - the drafter is a separate network
   - experts live on the SSD as files read on demand
   - tokens come one check at a time, left to right
   - the weights are fixed at run time
   - the GPU does the math
   - exactness is bitwise against the 1-row path
2. **Is each one actually true here?** Measure on recorded data wherever you can (routing captures, walks, taps, the decision map, Leap's timelines). For example:
   - How often does a token's expert output nearly repeat a recent token's?
   - How many rows are already decided (certified) by layer 24?
   - How much of an expert's input-times-weights work is shared across the rows of a check?
3. **For each assumption that breaks:** the idea that exploits it, with:
   - the closest prior art (honestly: new in kind, or a new form?)
   - EXACT or FAST
   - the gain (upper bound, then halved)
   - the cheapest test: a CPU replay tonight if the data exists, otherwise a 10-minute screen with a pre-registered line
4. **Cross-domain:** at least three ideas carried over from fields that have solved a similar problem, mapped onto our walls (the SSD stack, kernel boundaries, kept per check). Examples: video codecs (motion compensation, P-frames), CPUs (value prediction, branch prediction, prefetchers), databases (query planning, materialized views), compilers, network caching. NewIdeas' P-frame MoE output reuse came from codecs.
5. **The bet:** your top 3, plus one moonshot that would be revolutionary if it held, even at 20 % odds.

**Rules**
- CPU only; writes only in ~/FableGod; no downloads; web reading for prior art is fine.
- Check ~/TRIED/TRIED.md and NOVELTY.md first, and say why each idea differs.
- Both tracks are first-class. The goal stays the full 125B (all 512 experts, no pruning, no smaller model).
- Halve gains. Mark every number measured / arithmetic / judgement. Breadth first: one test per idea, then move on.
