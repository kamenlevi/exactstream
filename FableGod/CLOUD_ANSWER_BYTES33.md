# CLOUD ANSWER, task 18: the 33 MB problem (cloud Fable 1, pasted by Kamen Mon 2 Nov 2026 ~23:2x)

(The full ANSWER.md is at /home/user/ANSWER.md in the cloud container. This is its final message as Kamen pasted it, followed by the orchestrator's check.)

- **Accepts the correction:** on code the SSD is bandwidth-bound. Its task-15 "SSD ~2 ms" was wrong. It marks the latency ideas (pre-route on the partial sum, the look-ahead) as zero or negative on code.
- **Lower bound:** Belady gives 27.2 MB per kept token at C 320 and ~21 at C 350, so EXACT 350 isn't forbidden by bytes at night. But Belady needs 5-20 checks of foresight, and a perfect next-check predictor buys only 15 %.
  - The realizable EXACT floor from caching is ~32 at C 320 and ~25 at C 350.
  - By day at C 256 even Belady is 49, which caps EXACT near 245 tok/s on bytes.
- **Mechanisms, ranked:**
  - Q1: -13.6 (measured).
  - The 1-bit sidecar: a fixed 112-slot pool that misses 15.8 per kept token at every C, a third of all bytes. If the 3-bit arm keeps scores, dropping it saves 12.2 MB and frees 4.1 GB for +32 slots.
  - C 350: -8.2.
  - Small ones: entropy coding decoded by the idle CPU -2.8; a CPU-decoded compressed warm tier -2.7 (likely new); a bigger sidecar pool -2.5; kept -1.5.
- **Path:**
  - FAST reaches 33 at C 320 with Q1 alone (24), and at C 350 (19).
  - By day at C 256 it needs Q1 plus the sidecar drop: 28, then ~17 with the freed RAM.
  - EXACT's best is ~29 at C 320 with coding and a 2x sidecar pool, but that spends all 55 GB.
- **Radical option (~60 %):** make the foresight instead of predicting it. Run the answer once in FAST at ~300 tok/s, keep its expert timeline, and replay the EXACT pass with that timeline as a Belady schedule. FAST tokens stream first; EXACT confirms behind and regenerates only from the first divergence.
- **Pushbacks:**
  - The look-ahead prefetch on code adds ~31 % wasted reads at precision 0.68: a loss on a bandwidth-bound SSD.
  - 32-row checks raise bytes per kept token (+17 %), because the union grows faster than kept.

## Orchestrator's check (Mon 2 Nov 23:2x)
- **Consistent with Fable A's SSD_WALL.md and Fable C's AUDIT_350.md on bytes:**
  - Q1 at C 320 ~24 MB (Fable C's replay: x0.79-0.88 at C 320, because the sidecar is untouched)
  - the sidecar is a third of the bytes
  - Belady 27.2 at C 320
- **But bytes aren't the binding term once the GPU is priced right.** Fable C's audit: EXACT ~150 (expert kernel + dense), FAST ~245 at 45 GB and ~270-280 at C 320.
- **The radical option doesn't make EXACT faster.** The GPU pays for both passes, and EXACT is GPU-bound at ~150. Using the FAST text as EXACT's drafter (kept ~15) still nets ~100 tok/s once the FAST pass's GPU time is counted. Its value is UX (stream FAST text, certify it exact behind), not throughput.
- **Routed:**
  - per-layer sidecar sizes and 1.5x / 2x slots -> MoonShots M-25 replay
  - sidecar drop on FAST -> Breakthrough's 3-bit ranks 0-1 arm tonight, then EvalSuite
  - look-ahead off on code -> BigMoE before Tuesday's build
