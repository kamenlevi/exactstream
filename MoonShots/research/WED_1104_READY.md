# MoonShots: Wednesday 4 Nov research day (13:45-23:30), ready list

Written Tue 3 Nov 23:0x EET after the move to main. Every number below is measured; numbers already went to BigMoE (the
wall 1 lead) and the orchestrator. Quiet tickets are on hold until BigMoE's night chain ends (~04:00-04:30), and none
will run during FLOOR (to Wed 13:00).

## 1. Results already in (Tue night, all exact)

| Item | Verdict | What it decided |
|---|---|---|
| M-28 SSDMICRO (18:59) | L2 split | The 0.28 ms per loader call = ~0.19 I/O (first byte 119 us, idle wake 44, GCD 8) + ~0.10 outside the I/O path. Pieces and spinning threads: KILL. One call instead of two (JOINT): -120..-155 us per layer. |
| M-29 SSDMICRO2 (22:33) | Z PASS, P PASS | Zero-copy into 4 KB-aligned slots: -173 us per 2-record call; 4 KB alignment as good as 16 KB; Metal memory as good as malloc. Copy tail 120 us per base record. Adopted: the conversion format is now native+p11+pad, and the A/B night adds a zero-copy arm. Wake-ahead ping (one 4 KB read 200 us before a burst) removes 97.5 % of the wake; tagged ADAPTED. Keepalive spacing up to ~340 us still works. |
| M-29b (22:46 / 22:47) | BETWEEN-PASS | At the 24 GB line's 4-15 records per call, zero-copy saves 8-158 us per call (the copy tail swings run to run). One aligned read per piece beats one contiguous read per record at high queue depth. |
| M-30 QUEUEMIX (22:39) | BETWEEN / BETWEEN | Lever 2: a busy drive saves a demand burst only the wake, never the first byte. Forecasts mixed into a demand burst cost it ~165 us per record; chained after it they cost nothing. IOPOL_THROTTLE shields demand reads but delays forecasts 1.6-2.5 ms. |
| M-31 HEADPING (22:45) | KILL | A 4 KB ping of a record's head never shortens its later full read: the first byte has no warmable cache. |
| M-33 COPYROUTE (23:02) | KILL | Copied spans do not reuse their source's experts past layer ~6 (recall@10 0.88 at layer 0, ~0.2 at layers 36-42). |
| M-23b (MTP Head ran it) | KILL | The MTP head as a stop signal loses with its 2.2 ms per step charged; with that cost hidden, +1.3 % code and +3.4-5.9 % chat. |

What is left of lever 1 after JOINT + keepalive + zero-copy: the drive's first byte (~119 us per burst; 46 bursts per
check at C 256 = ~5.5 ms) plus the release / pickup / plan (BigMoE's S0 instrumentation on Wed). Lever 2 is the only way
under that floor, through forecast hits, not through queueing.

## 2. Ready to run on Wednesday (mine)

1. **M-32 FCASTSIZE (CPU, ~15 min, pre-registered):** re-fit the per-layer pool sizes to the forecast (slots go to the
   layers the forecast misses). It runs as soon as BigMoE's pre-route capture (pr_224 / pr_160) gives per-layer recall, and
   only if lever 2 goes ahead. Line: >= 1.0 ms less exposed per check at equal RAM.
2. **One batched SSD ticket (<= 3 min, the first quiet window after 13:45):** an M-29b rerun with 300 rounds to settle
   the 24 GB zero-copy number, plus a wake-ahead ping in a burst train at the engine's real period (1-2 ms between bursts)
   to give BigMoE the lead time to wire. Harnesses are built: scripts/ssdmicro2.m, ssdmicro.c.
3. **Help BigMoE read its S0 trace** (the ~96 us outside the I/O path: release, pickup, plan) against M-28 / M-29.

## 3. Papers sweep on the two walls (research/sweep_1103/WALL1_SSD.md, WALL2_KEPT.md)

About 100 papers from 2025-26, each checked against TRIED, PRIOR_ART, DEAD, RANKED_45GB and QUEUE. None is a +20 % find.
Exact verification rules gain nothing at temperature 0, and the SSD literature reaches no floor we have not reached.
The items new to us go to the session that owns each kind of work:

| Item | Wall | Exact? | Est. | Owner |
|---|---|---|---|---|
| Test-Time Speculation (arXiv 2605.09329): one LoRA step on the drafter per check, from the target's outputs | 2 | yes | +0-5 % on our short suites, +5-15 % on long or agent turns | MTP Head (drafter training). On our captures, code's kept per check falls 12.7 -> 6.3 -> 5.7 across answer positions 0-150 / 150-300 / 300+; chat stays flat. |
| Causal-tail check (arXiv 2610.24993: DSpark / Domino +19 % over plain DFlash at temperature 0): DFlash2's selector on vs off | 2 | yes | screen only | MTP Head / Fable B |
| ToolSpec (2604.13519): state machine for tool-call schemas | 2 | yes | agent +3-8 % | MTP Head |
| FLy (2511.22972): lossy window acceptance | 2 | no | code +3-8 %, chat +5-12 % | FAST track |
| Trained same-layer (pre-attention) forecast (2511.10676) | 1 | yes (timing) | lever 2's fallback if pr_160 misses its line | MTP Head |
| Layer-wise budgets for unsure rows / the LSB in layers 12-35 (LExI, ASET) | 1 | no | -9..-10 % MB per kept token (est.) | Breakthrough / FAST |
| Logical cache for QOM's reproducibility (PSR 2610.34721) | 1 | no | unblocks QOM | Breakthrough |
| Forecast-aware pool sizes (Fate) | 1 | yes | -0.5..-1.5 ms exposed | mine: M-32 |
| Copy-span routing replay (no prior art) | 1 | yes | KILLED tonight (M-33) | done |

Taken (Tue 23:1x): MTP Head runs Test-Time Speculation step 1 (labels = verified rows only, LoRA r16, reset per window, read
by position bins < 150 / 150-300 / >= 300), the selector-gap check and ToolSpec (a), all pre-registered in
~/TrainLab/REPORT.md and running Wed from 13:45. The trained same-layer forecast: tonight's pr_160 'full' variant is its
untrained form; if it misses, MTP Head proposes a trained map (it needs a new BigMoE capture) through the orchestrator.
Kamen 23:1x: a FAST mode may depend on the cache state (QOM goes ahead; EvalSuite certifies under >= 2 prompt orders;
EXACT stays the default), so the logical-cache lead matters for certification, not for permission.

Outcomes (Wed): TTS step 1 KILL (MTP Head, -0.1..-3 % kept); SAMELAYER-T BETWEEN (recall 0.823 code / 0.851 chat;
priced by MoonShots at -40 % forecast-missed code demand, ~-2.4 ms exposed per check as a bound); SELGAP open, not
queued; ToolSpec (a) KILL; M-32 KILL; M-34's 2-bit quality screen BETWEEN / KILL / KILL.

Novelty watch for the papers (NewIdeas):
- PSR section 7 weights verifier expert demand by the drafter's confidence, which touches Q1's NOVELTY wording.
- Memory-Sovereign Inference (arXiv 2608.23805) streams Qwen3-Next-80B from NVMe with byte-identical logits. PRIOR_ART's
  "whole stack not found" line should cite it.
