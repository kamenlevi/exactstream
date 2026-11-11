# MISSION B6 (Fable B): LONG CONTEXT. What 2-10K tokens of history do to a check, and the fixes.

(Draft by the orchestrator for Kamen, Mon 2 Nov 2026 ~23:5x.)

**Why.** Every speed number in the project is at ~200 keys of context. Leap's table (TABLE_350 around line 4083) shows the 12 attention layers alone take 12.6 ms per check at 2K keys and 24.3 at 10K. Real chats and agent sessions run that long. The cloud auditor flagged it: the public headline only holds for short prompts. GDN layers carry fixed state, so attention is the term that grows.

**Read first:** TABLE_350's attention-vs-keys rows, your GDN_PASS.md and BOUNDARIES.md, the engine's attention path in ~/BigMoE (KV layout, bits, the 16-row verify kernel), and TRIED's KV / attention lines.

**Deliver**
1. **The check's cost against context length** (200 / 2K / 4K / 8K / 16K keys): attention ms, KV bytes read per check, and tok/s at today's kept, for code (16-row checks) and chat (4-8 rows). Say where attention overtakes the experts.
2. **EXACT fixes:** a key-split (flash-decoding style) kernel for 16-row verify, the KV layout and bits as stored, and anything that keeps the bits. Price each.
3. **FAST fixes:** KV at 8 / 4 bits, sliding or chunked windows on some of the 12 layers, KV eviction or compression. Give the prior art and the quality risk of each, priced.
4. **One pre-registered 10-minute GPU screen** for Leap.

**Rules**
- CPU only. Writes only in ~/FableGod/fableb/. No downloads. Both tracks first-class. Halve gains. Mark every number [K] / [P] / [G].
- Within ~40 minutes, write a complete first draft to LONG_CONTEXT.md, then refine it, and stop.
