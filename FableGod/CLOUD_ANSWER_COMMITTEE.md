# Cloud Fable 5.1, task 2 (CLOUD_TASK_COMMITTEE.md): its summary as pasted by Kamen, Mon 2 Nov 2026 ~17:2x
(The full ANSWER.md with 46 lines of selector pseudocode stays in the cloud session; 6 of 8 Python runs used, no web, 1,114 words.)

- Your union is capped by overlap, not by accuracy. From your own numbers, when DFlash2 is wrong, the twin variant is wrong 64 % of the time. If errors were independent within class, the union would be 97.7 %, not 94.2 %. So member 2 should be trained to be wrong elsewhere, not to be better.
- Pick: member 2 = DFlash2 trained with boosting weights on member 1's misses and training blocks that start at those misses (that's where deployment blocks actually start, which is why depth 1-3 is your weakest region), reading different taps, plus a small diversity term. Member 3 = the match model, free. MTP head only as a depth-1/2 specialist and only if its MoE experts fit the budget, which I doubt.
- Selector: PAQ-style logistic mixing with context-selected weights and an APM, learned online from the 16 free labels per check. The new signal is the cross-probabilities (A's probability of B's token and vice versa), which today's confidence pick ignores.
- Numbers: code goes 92.7 -> ~94.0 (committee) -> ~95.3 (plus match), i.e. roughly 11.5 -> 12.3 -> 13.2 kept per 16 and 16 -> 18 -> 20 per 32. Those are judgement calls dressed in arithmetic, and I marked them as such. The /32 column is optimistic: nobody has measured a 32-block drafter past depth 16.
- Measure first: the overlap q for each member-2 candidate, the mixer's disagreement-resolution rate on held-out problems, and match precision on the pair's residual errors. If q stays at 0.64 after a night of training, the whole plan is dead and you keep the confidence pick.
- Two pushes: the ~50 % overlap assumed for the boosted member is the single number the plan rests on; and "one drafter trained only on unsure positions" being a dead end is a warning that aggressive reweighting backfires, so the x4 weight is deliberately moderate.

## Orchestrator's check (17:2x)
- Overlap verified on drafter_ensemble.out: P(B wrong | A wrong) = 6.0 / 9.1 = 0.66 (hpdH) and 5.6 / 9.2 = 0.61 (hpdU), mean ~0.64. The within-class independence union (sure 4.5 %, mid 34 %, tie 70 % misses, class shares 86.5 / 12.0 / 1.5 %) gives 97.7 %: the arithmetic holds.
- Warning in the data: hpdU (an HPD-trained variant, weighted toward unsure positions) still overlaps 0.61, so reaching ~0.50 is the open question.
- Its kept numbers use the same per-depth ratio arithmetic that overshot ~2x in MATCH (x1.038 real vs x1.071 estimated); treat 12.3 / 13.2 as upper bounds.
