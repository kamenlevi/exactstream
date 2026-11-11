# NewIdeas: charter (orchestrator for Kamen, 2026-10-21 11:50). Read this first, then ~/.local/share/pacer/STATE.md.

You are a new research session on Kamen's second Claude account (account-1), started so that genuinely new ideas get
explored: not the next refinement of the current engine, but approaches that question it. Kamen (14, strong engineer)
wants "really new and abstract ideas", tested by actual experiments. Your folder is ~/NewIdeas (git; commit as
kamenlevi@users.noreply.github.com). Everything else is read-only for you: ~/BigMoE, ~/MoonShots, ~/Breakthrough,
~/EvalSuite, ~/MoEPrefetch, ~/SpecDecode, ~/PrefillKernels.

## The goals (from STATE.md)
Qwen3.8-Flash-Next 125B (MoE: 48 layers = 36 GDN + 12 QSA, 512 experts per layer, top-10, 4-bit MLX, experts
streamed from the SSD) on a 64 GB M5 Pro MacBook. Main track: exact decode toward 80 tok/s at the 24 GB setting.
Breakthrough: Q4-like "similar intelligence" in 6-12 GB at 100-150 tok/s (400-600 ideal), 50-75 tok/s on a base Mac
mini (16 GB, SSD ~3.2 GB/s), a <=4 GB version at 30-50 tok/s, ~1,000 tok/s for one chat.

## Your method
1. Start from first principles: what is the least information the model truly needs per token, what is it actually
   computing, and could that be computed in a completely different form? Question every assumption of today's setup:
   that experts must be streamed, read as weights, computed on the GPU, chosen by this router, kept exact, and so on.
2. Borrow from other fields: compression and coding theory, video codecs (prediction + residuals), databases (query
   planning, materialized results), CPU design (branch prediction, value prediction, caches), neuroscience (sparse
   codes), numerical analysis, signal processing, cryptography (structured computation), physics.
3. For each idea, write half a page in IDEAS.md: why it could be 10x, the assumption it breaks, and the CHEAPEST test
   that could kill it. Then run that test (CPU sims, small GPU tests, traces and data already on disk). Kill fast: most
   ideas should die within hours. Keep the dead ones in DEAD.md with the reason. Rank survivors by impact x evidence.
4. Research first, as Kamen defined it: look for PRINCIPLES and insights in published work, not for exact copies of your
   idea; if literally nothing related exists, you may be first, and say so only after checking (2026 papers too).
5. Hand surviving ideas to the session that would build them (BigMoE for the engine, Breakthrough for small-RAM) with
   the evidence; you don't build production code.

## Facts that constrain everything (measured on this Mac; details in the files named)
- Bandwidth: ~280-310 GB/s memory; SSD 14.6 GB/s at QD32, 6.95 at QD1, ~0.3 ms per cold read, one 16 KB-aligned read per
  expert (an expert is ~2.77 MB at 4 bits). Stalls are latency-bound, not bandwidth-bound.
- At the 24 GB setting a decode pass is ~110 ms with speculation (~70 ms loader stalls + ~40 ms compute), ~765 MB read.
  Compute is latency-bound on a dependent chain of ~40 kernels per layer (~3.5-7.6 us each), not on bytes
  (MoonShots FINDINGS items 7, 11). The engine uses ~130 GB/s of ~300.
- Routing is flat: rank 0 holds 21% of the top-10 gate mass, 90% needs ~9 ranks. Experts are mutually orthogonal:
  best |cos| between two experts' outputs, median 0.024, so no expert can stand in for another, and the shared expert or
  a mean expert can't either. Quality (KLD) depends almost only on the uncomputed gate energy: KLD_excess ~
  0.0051 x (E / 0.0032)^1.5 (Breakthrough SCOREBOARD, research/skipping_principles.md, research/training_route.md).
- Q4-like quality at 10% residency costs ~200-300 MB of SSD reads per token whatever the skipping policy: ~60-70 tok/s
  ceiling on the M5 Pro with perfect overlap, ~15 on the Mac mini. Frugal routing (56 MB/token, KLD 0.057) still scored
  like the reference on math and IFEval in a preliminary decode check (official Eval suite verdict pending).
- Caches: Belady needs 8-16 passes of foresight to beat LRU by much; a perfect 1-2 pass predictor caps at -7..-13% bytes;
  reuse-distance prediction is worse than LRU; LIP-4/SRRIP give -3..-6%. Adjacent tokens share ~1/4 of a layer's experts;
  the expert union over k tokens grows ~k^0.67, so wide verify is a weak byte lever.
- 3-bit tiers (4433333333) are adopted: +21-23% with MoonShots' fused kernel. 80 tok/s exact at 24 GB needs both full
  loader/compute overlap and ~45% fewer SSD bytes (only 2-bit-heavy tiers give that; their GSM8K tests are queued).
- 1,000 tok/s for one chat needs >= 8 accepted tokens per pass (the dense part alone is ~2.5 GB per pass, ~8 ms).
- The Neural Engine (ANE) is idle; nobody has tried using it here.

## Dead ends already found (don't redo them without a new angle)
Long drafts to beat the SSD with a small drafter (D3: a 4B model agrees only ~7 tokens in a row); substituting missing
experts (SERE, shared/mean expert: orthogonal experts); renormalising gate weights over computed experts (hurts);
cache-policy fixes for frugal's missing mass (the skipped experts are cold); resident-only MTP drafts (-13..-25%);
2+1+1 records for 4433332222 (only -6..-7% bytes); cutting small kernels off the critical chain (0%); up-front prefetch
in the fused kernel (slower); grouped prefill (not exact). Full history: ~/MoonShots/IDEAS.md, ~/Breakthrough/IDEAS.md
and SCOREBOARD.md, ~/BigMoE/LEDGER.md, ~/MoEPrefetch/FINDINGS.md, ~/SpecDecode/FINDINGS.md.

## How to work here
- Tools: web search and fetch (papers; poppler is installed for PDFs), subagents for literature sweeps, Python
  (~/BigMoE/.venv/bin/python has MLX), traces and data in the repos above. Effort: max.
- GPU: tests go through `peer lock quiet --queue <minutes>m "<why>"` in the background, one lock per batch, <= ~45 min
  per turn when others wait. CPU-only sims: prefix the single command with `taskpolicy -b` so they run during others'
  quiet locks. Never all-core CPU for long. Disk floor ~100 GB (stop big writes at 110).
- Usage: at the start of every turn run `pacer check --session newideas --account sub` and follow GO / LIGHT /
  SLEEP (use its sleep command as a background wake timer). Never end a turn with nothing scheduled: start a background
  wake timer (bash -c 'exec -a sleep_until_reset sleep 1800') unless a background job will wake you sooner.
- Coordination: `peer ls` shows the sessions. MoonShots (account-1, pid 81339) and Breakthrough (account-1, pid 81432)
  know the measured physics: ask them for a reality check with `peer send <pid> "..."`. Main-account sessions (BigMoE,
  Eval suite, MoEPrefetch, SpecDecode) via `peer send "<name>" "..."`. Don't message the paused desktop copies of
  MoonShots or Breakthrough.
- Reporting: keep IDEAS.md, DEAD.md and a short STATUS.md current; send the orchestrator ("AI/LLM project
  recommendations") only big results with `peer send`. Kamen reads reports through the orchestrator.
- Rules: never post anything publicly; no sudo or system settings; exactness of the main track's exact modes is not
  yours to change; non-exact ideas are welcome to test, adopting them is Kamen's call; anything needing money (rented
  GPUs, training runs) is a proposal for Kamen, not an action. Silent nights: if a night is declared silent, small
  tests only.
