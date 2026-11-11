# Per-request cold-expert cut: COLD-CUT with the prompt's own footprint (Breakthrough, Tue 3 Nov 2026, 20:3x EET)

Written for NewIdeas' ~/TRIED/NOVELTY.md (the orchestrator 20:3x: "paper-section material"). Evidence: SCOREBOARD
0a565a6, 2d0327e, 4236037, be67ff0; code ~/FableGod/decision_map/dm_cold_masks.py, dm_cold_lru.py, dm_cold_read.py;
read-out out/COLD_read.txt, out/COLD_lru.json. Marks: [K] measured, [P] arithmetic, [G] judgement.

## Rows for NOVELTY.md's prior-art table
| candidate | verdict | closest prior art |
|---|---|---|
| per-request COLD table: on drafter-unsure rows, drop a rank >= 4 expert unless a corpus table OR the request's own prefill routing knows it (+ the check's union) | adapted: the footprint as a CUT is not found (one search) | prefill footprint used to place requests (ELDR, arXiv 2607.00466), to cluster them (arXiv 2604.23150), to pick a routing table for offloading (DoMoE, IJCAI 2026); cache- / sensitivity-aware skipping with a fixed table (SwapMoE 2308.15030, AdapMoE 2408.10284); the union step (OEA 2511.02237); Fable C's COLD-CUT (a corpus table only) |
| the prompt footprint alone as the code path's COLD table | adapted (same prior art) | as above; on code it beats the corpus table |

## The method
- **Setting.** MoE decoding with speculative verification. A drafter (DFlash2) proposes 16 tokens with a confidence each,
  and the 125B checks them in one pass. Checks are cut at live lengths (running confidence 0.3). Expert weights stream
  from the SSD into a 45 GB cache, and bytes are priced on each check's union.
- **Sure rows** (the drafter's confidence >= 0.9) keep their top-2 experts, at 3 bits, plus any lower-rank expert that is
  already in the check's union. This is Q1-UNION + t3, unchanged.
- **Unsure rows** keep ranks 0-3, plus any rank 4-9 expert that is "warm" for this request, plus the check's union members.
  An expert is warm if either holds:
  - the corpus table marks it popular (share of checks >= 0.05);
  - the request's own prompt routed to it at least once during prefill, at any rank.
- **Token-repeatable.** The prompt's routing is fixed before decoding, so the output depends only on the prompt and the
  model, not on the cache state. That is unlike miss-driven skipping. It needs no domain switch and no per-domain corpus.

## Results [K]
Data: the decision map, 36 teacher-forced windows (12 code, 12 IFEval, 12 multilingual). The metric is all-row top-1
flips against the exact baseline. Every arm is compared with the same control (u8 + t3 at the same live checks) by a
paired McNemar test. "Bytes" means SSD bytes against EXACT, from a 45 GB plain-LRU replay on the same checks.

| table | code flips (z) | IFEval flips (z) | multi flips (z) | bytes code / IFEval / multi | verdict |
|---|---|---|---|---|---|
| control: u8 + t3 | 1.37 % | 7.74 % | 5.54 % | x0.295 / 0.597 / 0.641 | |
| corpus only (Fable C's cap45: code + math prompts) | 1.25 % (-0.78) | 10.60 % (+6.07) | 11.37 % (+11.66) | x0.254 / 0.498 / 0.532 | code only |
| in-domain, leave one window out | 1.37 % (0.00) | 7.68 % (-0.14) | 6.67 % (+3.02) | x0.267 / 0.363 / 0.372 | cost on multi |
| **prompt footprint OR corpus** | **1.13 % (-1.79)** | **8.42 % (+1.63)** | **5.77 % (+0.62)** | **x0.285 / 0.520 / 0.539** | **GENERAL** |
| prompt footprint alone | 1.25 % (-0.78) | 9.16 % (+3.33) | 6.92 % (+3.56) | x0.199 / 0.329 / 0.343 | code only |

What the table shows:
- **General (prompt footprint OR corpus).** It shows no measurable cost on any domain, at -3 / -13 / -16 % bytes against
  the control (halved for decisions: about -1.5 / -6.5 / -8 %).
- **Code path (prompt footprint alone).** On code it has no cost at -33 % bytes against u8 + t3, and -22 % against the
  corpus-only table.

## Why it works [K + G]
- **Cold means missed.** On code, the corpus-only cut keeps MORE experts per unsure row than u8 (9.3 vs 8.5 of 10), yet
  it reads fewer cached bytes. The experts it drops are the ones that would have missed the cache.
- **A fixed table mislabels specialists.** A code + math table marks chat and multilingual specialists as cold, so chat
  fails (z +6 / +12). An in-domain table rescues IFEval, but multilingual still fails (z +3.0).
- **The prompt's own routing names the specialists.** The prompt is in the same language and domain as the answer, so its
  prefill routing lists the specialists this request needs, at no cost (they ran in prefill).
- **Short prompts need the corpus table too.** Chat prompts here are 27-52 rows (88-145 experts per layer seen), so the
  footprint alone misses experts the answer needs. Adding the corpus table covers them. Code prompts are 181 rows (about
  249 experts seen), so the footprint alone suffices there.

## Caveats
- **The byte win depends on the cache size.** At 12 GB's cache (51 experts per layer) every COLD table reads MORE than
  u8 + t3 (+11 to +18 %; out/COLD_lru_c51.txt). Cold means missed only when the cache holds the warm set (~250-300 per
  layer: 45 GB's ~295 does), so the method is general across domains for quality but a large-cache lever for bytes.
- The map measures teacher-forced flips, not task scores. The 12 GB line is judged on task results (EvalSuite).
- Bytes come from a plain LRU replay with no popularity prior. Tok/s is not measured.
- IFEval sits near the line (z 1.63). A 78-window re-test (FLOOR night 1, step C, Wed) firms it up.
- Only one setting was tested: threshold 0.05, r = 4, a prompt seen at least once. Prompts were short on chat.
- One prior-art search, so the "not found" claim is narrow.
