# MISSION 32: THE SSD WALL, FROM THE TRACE. Halve the SSD bytes per kept token.

(Draft by the orchestrator for Kamen; approved by Kamen Mon 2 Nov ~22:47; refocused after Fable C's audit. **Two phases.** Phase 1 starts NOW (Mon ~23:10) from the data on disk: deliverables 2-4. Fable C's audit already established the bytes floor. Phase 2 comes Tue ~06:00, when the orchestrator sends stall_report.txt and pr_256_preroute.txt: deliverable 1, and correct anything the trace changes. Stop after phase 1. Don't wait or poll for the trace.)

**Why.** The SSD is the largest piece of a day check: of ~166 ms, its exposed time is 39-65 ms. Fable C's audit (~/FableGod/fablec/AUDIT.md, section c) showed the wall is **bytes, not hiding**:
- On code the demand bursts run at the drive's bandwidth: HumanEval/68 moves 1,452 MB in 124 ms, 11.7 GB/s.
- Today's default (C 256) needs 62.4 MB per kept token. At 11.7 GB/s that caps exact at ~187 tok/s on code even if every other piece were free.
- C 320 + M-17 still needs 37.9 MB per kept token: ~40 ms per check at kept 12.6, the ladder's whole cycle. "SSD 2 ms at target" was a byte floor, not a hiding target.
- **350 needs <= ~33 MB per kept token** on code, at ~13 GB/s with zero margin.

Tonight BigMoE records:
- measurement 2: per layer misses, service time, issue delay and the GPU's stall, now with a bytes-vs-wake split
- the pre-route capture: cloud's #1, the same idea as Fable C's C1-1

**Read first:**
- ~/BigMoE/data/r57/stall_report.txt and pr_256_preroute.txt (tonight); BigMoE HANDOVER.md, the R-57 plan around lines 676-730
- Fable C's AUDIT.md, INVENT.md and replay_c1.out
- ~/MoonShots/data/cachesize45.txt
- CACHE.md
- CLOUD_ANSWER_SSD45.md and CLOUD_ANSWER_ROAD350.md
- MoonShots' M-20 SSDWAKE
- ~/TRIED/TRIED.md's SSD section (192 lines), especially line 721 (the 350 byte budget)

**Deliver**
1. **Confirm the wall from the trace:** the exposed SSD ms per check split into bytes at bandwidth, wake penalty, late issue, and queue and service time. Give it on code and chat, at C 256 (and C 300 / 320 if they ran).
2. **The bytes ladder from 62.4 to <= 33 MB per kept token, on both tracks.** Price each lever on the union per 16-row block, with its RAM cost and its gain halved:
   - cache size (C 300-350 and the RAM each needs)
   - per-layer sizes, sidecar 1.5x slots, residency policy (Belady headroom 56 %)
   - Q1 and Fable C's Q1-on-miss
   - tier precision of cold ranks and expert compression
   - kept per check, since bytes are per check and tokens per check multiply
   - anything new
   Say which combination reaches 33 MB at 45 GB, and at what RAM it does so exactly.
3. **The floor:** tok/s that SSD bytes allow at the 45 GB and 24 GB lines, by day and by night, on both tracks, with the other pieces at their measured floors.
4. **One new idea** the trace or the ladder suggests, with prior art, and a pre-registered 10-minute screen BigMoE can run.

**Rules**
- CPU only. Writes only in ~/FableGod. No downloads.
- The full 125B (all 512 experts). Both tracks are first-class.
- Halve gains. Mark every number [K] measured / [P] arithmetic / [G] judgement.
- **Price bytes and tiles on the union per 16-row block** (an expert read serves every row that uses it; Breakthrough's FLOOR method). Never per row: two of mission 31's numbers failed on exactly this.
- Check ~/TRIED/TRIED.md first, and say why each idea differs from what was tried.
- Within ~40 minutes, write a complete first draft of all four sections to SSD_WALL.md, then refine it.
