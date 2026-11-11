Sweep (d), Apple hardware: summary
- SLC ≈ 24 MB on M5 Pro (spec site, unconfirmed): useless for GBs of weights.
- NAX for decode: ~+4% (Creative Strategies), bandwidth-bound.
- Draft model on ANE/SME in parallel: Mirror-SD (2510.13161) proposes it; no measured Mac run. Our cap: drafting is only 6–9% of spec time (SD §10), so ≤ 6–9%.
- Metal 4 residency sets (llama.cpp PR 11427): fixes warm-up stalls, no steady gain. M5 GPU compression covers texture writes only.
- NAX for 8–16-row verify: our skinny-M run already shows 16 rows at 1.37x vs 12 rows at 2.69x (the padding finding).
