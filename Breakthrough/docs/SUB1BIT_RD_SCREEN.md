# Sub-1-bit expert copies in the rate-distortion sim (Kamen's idea; screen, Wed 11 Nov 2026)

**Idea** (Kamen, 17:5x, via the orchestrator):
- Store each expert at many precisions and pick one per token by importance.
- Run low ranks at very low bits ("0.5 bit just so they go through") instead of dropping them: dropping tested worse.
- Spend SSD space on extra low-bit copies: "needing ssd is much cheaper than needing ram".

**Prior work** (the ledger, as the orchestrator listed it):
- Ranks 0-1 at 4 bit with ranks 2-9 at MSB-3: ADOPTED (-24 % bytes).
- POM (cold ranks >= 6 at two planes): PASS (-12.4 %).
- RTN 2-bit beyond the top 4: BETWEEN (+1.5 % ppl).
- Uniform sub-2-bit experts: KILL. Our MSB-1 "1.5-bit" truncation was no better than dropping.
- The closest work is my RD-optimal sim of 21 Oct (`scripts/rd_sim.py`, SCOREBOARD.md:308-318,
  `data/gmass/rd_sim_c51*.txt`). Its options were drop, MSB-2, MSB-3, 4-bit, and trellis 2- and 1-bit copies. At the
  Q4-like line it needed ~245 MB/token with the copies and ~280-310 without. It was killed only against the 150+ line.
  Today the 12 GB goal is 90 tok/s, i.e. ~150 MB per kept token at the SSD's ~12-14 GB/s. 12 GB chat today reads
  449-491 MB per kept token.

## Pre-registration (written before any run)

**Model: the 21 Oct sim, unchanged.**
- Traces: 125B router traces, `data/expA/traces125`: bulgarian, chat, code and math, 2,926 decode tokens in all.
- Cache: an LRU of C = 51 whole 4-bit experts per layer (the 12 GB line's pool, ~6.8 GB), one token per step.
- Choices: ranks 0-1 are always computed at 4 bits; a missing one is read and enters the cache. Each missing rank r >= 2
  takes the option that minimises `err * w^2 + lam * MB`, where w is the top-10 renormalised router weight.
- Energy law: E is the mean over token-layers of the summed err, and KLD ~ 0.0043 + 0.0051 (E / 0.0032)^1.5. It was fit
  on the hard subset and validated on m85_nr_t2 (predicted 0.0152, measured 0.0151).

**Options** (err = output NMSE; MB per expert read):

| option | err | MB | extra SSD |
|---|---|---|---|
| drop | 1 | 0 | 0 |
| MSB-2 / MSB-3 / 4-bit (4-bit enters the cache) | 0.10 / 0.02 / 0 | 1.536 / 2.150 / 2.765 | 0: bit-plane prefixes of the 4-bit record |
| tr2, tr1 = the 1.0-bit tier (the 21 Oct trellis copies) | 0.071 / 0.27 | 1.30 / 0.70 | 31.9 GB / 17.2 GB |
| **NEW q05: a dedicated 0.5-bit copy** | bracketed below | 0.39 (0.31 best case) | 9.6 GB (7.6 GB) |

- q05's size is 0.5 bit x 4.92 M parameters plus tr1's 0.086 MB of scales; the best case has no scales.
- Each copy tier is stored for all 24,576 experts.

**q05's output error is bracketed** (NMSE), from "like drop" down to the prior-art figures:

| NMSE | what it represents |
|---|---|
| 1.00 | like drop, as our MSB-1 1.5-bit truncation measured |
| 0.80, 0.70 | intermediate points |
| 0.61 | entropy-coded ternary at 0.5 bit (QMoE's format family, ~0.8 bit/param there) on Gaussian weights |
| 0.54 | a trellis code at tr1's distance from the bound (1.08x) |
| 0.50 | the Shannon bound for Gaussian weights at 0.5 bit |
| 0.40, 0.30 | beyond the bound: they need weight structure we have no evidence for |

Plausible = NMSE >= 0.50.

**Mixes:**
- A: the base options (drop, MSB-2, MSB-3, 4-bit).
- B: A + trellis. This is the 245 baseline.
- C: A + q05.
- D: B + q05, the full ladder.
- E: A + q05 + the 1.0-bit tier, with one quality factor k for both: NMSE = min(1, k 2^(-2R)) for k in
  {1.0, 1.08, 1.22, 1.6, 2.0}.
- F: D with the copies also open to ranks 0-1 ("any rank").

**The line (Q4-like):** predicted KLD 0.0096, ar4's accepted KLD (the 21 Oct convention), so E* = 0.00328. lam is found
by bisection so that E = E* (within 0.5 %), and MB/token is reported there.

**Self-check** (else the script is wrong):
- A and B at lam 1e-3 and 2e-3 reproduce the rows of `rd_sim_c51*.txt` exactly (MB/token to 0.1, E to 0.0001).
- B at the line lands within 245 +- 5 %.

**Verdict lines** (the orchestrator's):
- **PASS:** some mix reaches <= 150 MB/token at the line with q05 NMSE >= 0.50. Next step: real low-bit copies of a few
  layers, checked on the map.
- **KILL:** no mix at NMSE >= 0.50 reaches <= 196 MB (245 - 20 %).
- **BETWEEN:** otherwise.
- Also reported: the q05 NMSE at which the best mix would reach 196 and 150 MB, i.e. how good a 0.5-bit copy must be to
  pay.

**Basis and limits [P]:**
- One token per step, as on 21 Oct. The drafted engine prices the union of a check's rows, so its bytes per kept token
  are lower, for every mix alike.
- Copies are not cached, as before.
- The energy law is extrapolated to many small errors.
- The line is the 21 Oct KLD convention, not a decode test.

**Cost:** CPU only, one process, <= 4 GB, ~10-15 min (`scripts/m1/rd_sim_sub1.py`, output `data/m1/rd_sim_sub1.out`).

## Result (Wed 11 Nov 18:1x): KILL

The full sim took 51 s on the CPU (413 passes); the scripts and outputs are listed at the end of this section.

**Self-check: PASS.**
- A and B reproduce the four rows of 21 Oct exactly at lam 1e-3 and 2e-3.
- At the line, A needs 305.0 MB/token (21 Oct: "~280-310") and B needs 235.6 (245 - 3.8 %, within the pre-registered 5 %).

**At the Q4-like line** (KLD 0.0096; MB per token, one token per step):

| q05 NMSE | C: A + q05 | D: B + q05 (full ladder) |
|---|---|---|
| 1.00 (like drop) | 305.0 (q05 unused) | 235.6 (unused) |
| 0.61 (ternary, QMoE family) | 298.9 | 235.6 (unused) |
| 0.54 (trellis-like) | 293.2 | 235.4 |
| **0.50 (Shannon bound)** | 289.1 | **235.0** |
| 0.40 (beyond the bound) | 274.0 | 230.8 |
| 0.30 | 245.6 | 212.9 |
| 0.20 | 184.7 | 174.7 |
| 0.10 / 0.00 (hypothetical) | 118.6 / 105.8 | 118.6 / 105.8 |

- **Best plausible mix:** D with q05 at NMSE 0.50 needs 235.0 MB/token. That is -0.2 % vs B and -4.1 % vs the 245 of 21 Oct.
  - It stores 58.7 GB of extra copies: tr2 31.9, tr1 17.2 and q05 9.6.
  - The RD uses q05 for only 13.5 reads per token, on the lowest-weight experts.
  - Without scales (0.31 MB), the same mix needs 233.4.
- **How good a 0.5-bit copy must be to pay:**
  - For 196 MB it needs NMSE <= 0.26 (D) or 0.22 (C); for 150 MB, <= 0.16.
  - The Shannon bound for Gaussian weights is 0.50 at 0.5 bit; 0.16 is the bound at ~1.3 bits.
  - So the copy would have to carry about 2.6 times the information of 0.5 bit. We have no evidence of weight structure
    that would allow that: our 1.5-bit MSB-1 truncation behaved like a drop.
- **E, one quality factor for 0.5 and 1.0 bit (no 2-bit copy):** 260.3 MB at k 1.00 and 268.0 at k 1.08. Both are worse
  than B, because B's 2-bit trellis copy (NMSE 0.071) is the workhorse at the line: ~90 reads per token.
- **F, copies open to ranks 0-1 too: 366 MB, worse.** The per-read choice serves ranks 0-1 from copies and loses their
  cache fills, so more misses follow. Ranks 0-1 should stay at 4 bits, as adopted.

**Why it does not pay** [P]:
- The line allows a missed energy of E* = 0.00328 per token-layer, which forces near-2-bit fidelity on most of the tail.
  At the line the bytes are 4-bit reads (19.7 x 2.765 = 54 MB), 2-bit copies (~92 x 1.30 = 120 MB) and 1-bit copies
  (~79 x 0.70 = 55 MB).
- A 0.5-bit copy removes at most half of a missing expert's error, so it can replace only drops and a few 1-bit reads.
  Its bytes at the line are ~5 MB per token.
- SSD space is not the constraint; read bytes per token are.

**Verdict (pre-registered lines): KILL.** No mix with q05 NMSE >= 0.50 reaches <= 196 MB/token; the best is 235.0.

**POST-HOC, information only** (`rd_sim_sub1_at150.py`, `data/m1/rd_sim_sub1_at150.out`): at a fixed 150 MB/token, the
predicted KLD is:

| mix | predicted KLD |
|---|---|
| A | 0.0419 |
| B (trellis) | 0.0260 |
| C (q05 at 0.50) | 0.0304 |
| D (q05 at 0.50) | 0.0250 (-4 % vs B) |
| D (q05 at 0.50, no scales) | 0.0231 |

All of them are ~2.6 times the Q4-like line. Even on a looser, benchmark-gated route, the 0.5-bit copy moves the predicted
quality only a few percent.

**Limits** [P]:
- One token per step. The drafted engine prices a check's union, which helps every mix alike.
- The energy law is extrapolated to many small errors.
- Copies are not cached. Caching them could only cut q05's own ~5 MB at the line, so it cannot change the verdict.
- The line is a KLD proxy, not the decode gate.

Files: `scripts/m1/rd_sim_sub1.py` and `data/m1/rd_sim_sub1.out` (pre-registered), `scripts/m1/rd_sim_sub1_at150.py`
and `data/m1/rd_sim_sub1_at150.out` (post-hoc).
