# MoonShots PLAN (2026-10-21 00:10; account-1; reviewed by the orchestrator; goals in ~/.local/share/pacer/STATE.md)
Tonight (silent until 08:45: CPU 1-2 cores, GPU tests <= 1-2 min, no timings):
1. DONE: out-of-order experts bitwise + integration proposal (patches/planes_engine/OOO_PROPOSAL.md: post_ids one-line
   dep change, split post/spin, tier_call sketch) -> hand to BigMoE with bench_split numbers Wednesday.
2. DONE (FINDINGS.md): first-principles pass model (scripts/pass_model125.py, calibrated 21.9 vs 23.7 tok/s at 17 GB): kernel
   levers give +30 %; 100 tok/s at 12 GB needs compute at the read ceiling + thin inter-pass + all SSD hidden + OPT-class
   caching; a 3 GB/s mini tops out ~9-14 tok/s at 4-bit-class bytes. Write it up for the orchestrator (one big-result note).
3. records4 (2+1+1): the main-track path to ~80 tok/s at 24 GB IF a 2-bit-heavy plan passes quality (FINDINGS 11).
   Prototyped Wed 11:21: make_records4.py (layer 0 verified) + tierglu4.py (tiers 2/3/4 bitwise; v3 preserved). Waits on
   GSM8K/Eval of 4222222222 / 4422222222 and BigMoE's 3-pool engine side. Next on mine: tierglu4 speed vs tierglu.
Wednesday (GPU from 08:45, behind BigMoE) -- 09:05: day_runner14 DONE (exact GDN bitwise at model level, OoO cost,
sibling union, 3c closed); day_runner14b queued (chain under ref flags, GDN speed, verify-graph build time):
4. 3c load tuning toward ~310 GB/s (prefetch-up-front lost; next: more rows/threadgroup, simd-shuffle code sharing), bitwise.
5. bench_split.py (cost of out-of-order dispatches) -> numbers to BigMoE. (2+1+1 converter only if item 3 says so.)
6. Dependent-kernel chain (round 32k: ~55 kernels/layer on the critical path, 55 x 7.6 us = the per-layer excess over
   bytes): re-measure under the reference flags (day_runner14 job 5; HC already fused by HC_VREAD), then decide an exact
   fused GDN-attention chain (~20 -> ~5 kernels, MLX arithmetic replicated as in 3c). Ask BigMoE for the reference env.
7. Decisive research tests (all in scripts/day_runner14.sh): sibling union (tree width cost; night attempt stopped: lock
   lapsed); dense-compute attribution per kernel (46 ms at
   ~130 GB/s); prefill-routing as long-horizon residency signal (the 30 % of OPT that needs 25-50 tokens of foresight).
Will NOT: take quiet locks for CPU sims (share lock --cpu 100); edit ~/BigMoE; change exact modes' outputs; run heavy GPU or all-core CPU before 08:45; start the 2+1+1 conversion
(~75 GB) without BigMoE's OK and the disk floor; send the orchestrator anything but big results.
