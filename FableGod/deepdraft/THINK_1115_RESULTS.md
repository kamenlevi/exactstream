# THINK_1115 results: five desk-only thinkers, Sun 15 Nov 22:0x-22:2x

Files:
- ~/FableGod/zero/IDEAS_1115_branch.md
- ~/FableGod/fableb/IDEAS_1115_cheaptrees.md
- ~/FableGod/far/IDEAS_1115_info.md
- ~/FableGod/fablec/IDEAS_1115_nearties.md
- ~/FableGod/lunatic/IDEAS_1115_limits.md

Context and targets: THINK_1115.md (450 exact / 900 lossy on code).

## The convergence: three thinkers found the same free signal independently
- ZERO "TIE-RUN", Lunatic "let the miss pick the next check's shape", and Fable C idea 1 "margin-keyed restart" all say
  the same thing.
- The last check already holds the 125B's EXACT margin at every verified row, for free, before the next draft.
- Near-ties cluster. After a miss the median margin is 3.87 vs 11.87 nats, and near-ties run 12.2 vs 1.6 %.
- So: after an easy check (full accept, depth 0 right 99-100 %), use a chain. After a hard one (the bonus row's margin
  < 2 nats), use a tree over bands 0-6.
- A sibling row costs 0.84-0.91 of a chain row in new experts (M-10), so the switch is near cost-neutral.
- Exact. Tagged new as a margin-keyed shape rule; the nearest closed items are CALIB-TREE and TREE-SHAPE.
- Fable C's numbers (code; 20 s CPU on the map's windows):
  - P(tie at t+1 | tie at t) is 35 % vs 4.8 %. Lags 2-6 are still 24-31 % against a 6.9 % base.
  - After a near-tie miss, the next band-0 row is a tie 35.5 % of the time and a miss 16.9 %. After a clear miss:
    6.9 % / 2.9 % (5.8x).
  - Chat clusters weakly.
  - Estimated gain +2-6 % kept/32 over tree-32 (+1-3 % at half). Modest, but free and exact.
- Fable C's premise audit: the check's own distribution can't RESOLVE a post-check tie, since its second-best is the
  token the 125B didn't choose. Only forks resolve ties on the exact line. And the true-state ceiling at bands 0-6 is
  clustered coin flips, not drift.

## Ranked Monday screens (<= 15 min each, CPU first; owners proposed, NewIdeas orders)
1. **MARGIN-KEYED SHAPE** (ZERO 1 + Lunatic + Fable C 1).
   - P(margin < 1 at t+k | low margins at t-2..t), k = 1-6, from the cells. Then a desk join on S11's 32-node dumps:
     a margin-keyed tree-32 vs EAGLE-2 tree-32 at equal rows.
   - PASS: a lift >= 2x at k <= 3 AND kept/32 >= +3 %.
2. **SIBLING-STATE DRAFTING** (FAR 1).
   - The check already computes the 125B's exact state at every rejected row. After a miss at a near-tie, those are
     states of a sibling prefix.
   - Feed them to the head as its depth-d input in place of its own drifting carried state. 0 bytes, exact.
   - Lab test on the true-state cells. PASS: half of the own -> true gap closed at depths 1-8 (bands 1-2 / 3-6:
     91.5 / 92.4 toward 95.3 / 96.4).
3. **TIE-STUBS** (ZERO 2): a placed #2 branch gets a reserved #1 continuation of 2-4 nodes; EAGLE-2 fills the rest.
   - Oracle vs rule placement on the 127-node dev dumps (Tuesday's dry run).
   - KILL if <= EAGLE-2 + 2 %.
4. **LOOKUP AS A CHARGED NODE** (NewIdeas): prompt lookup + an n-gram table of the 125B's own answers, inserted in
   tree-32's best-first order.
   - Upper bound kept +8.7 % HE+. PASS >= +3 % kept at equal rows.
5. **EARLY DEATH probe** (Fable B, the only exact route to trees past 32):
   - a layer-8 "child survives" probe on the 125B's own parent state; dead subtrees leave the batch; exact repair if
     wrong.
   - It needs captured layer-8 states: Tuesday's dry run could add them.
   - PASS >= 80 % of off-spine nodes droppable at <= 5 % false drops; KILL < 50 %. Even a PASS is x1.08-1.16 at best.
6. **NEAR-TIE FORGIVENESS CASCADE** (Fable C 3, lossy only): at a close-call miss the check continues instead of
   restarting. +4-5 % at eps 1. It belongs to the lossy 900 line (EvalSuite-gated).
7. **SLOT PRIOR** (ZERO 3): P(near-tie | token class of the prefix). Ties are 1.6-2x denser in comments and strings. A
   lift table is the whole test.

## Settled tonight (do not reopen this week)
- Bigger trees: K* = 32 (HE+) / 16 (MBPP).
  - The one-pass byte floors on measured unions are 39 / 52 / 62 / 70 / 81 / 90 ms at K16-128, and kept per ms peaks
    at K32.
  - The marginal row past 32 is compute (~0.7 ms), not bytes.
  - Sibling reuse, union-aware choice and staged checks can't pay (Fable B).
- The 400-exact wall is real and sits in bands 0-6 after a miss. Even the true state caps a chain at 93-96 % there;
  only trees get past it (Lunatic).
- DFS-GDN is already Leap's TREE16 L1 (66b4af8).
- Tie-tolerant exactness (+1-3 %) is Kamen's definition call, not for Saturday (Lunatic).
