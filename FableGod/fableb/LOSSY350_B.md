# LOSSY350-B: the dense pass and the fixed costs per check, under the lossy track (Fable B, Thu 5 Nov 2026, first version 16:3x EET)

Brief: the orchestrator's MISSION LOSSY350-B (15:5x): 125B at 350 tok/s LOSSY (FAST) on code at 45 GB; my wall = the dense
pass and the fixed costs per check (attention, shared dense layers, norms, the drafter, commit, dispatch); never-tried ideas
or a broken assumption only; CPU replays on disk data; every idea a number or an honest [G].
Read: `fablec/AUDIT_1105.md` 0-1, `full/FULL.md` 0-1, DAY9 line 93 (C0) and 875, `HC_READ.md`, `CLOUD_ANSWER_DENSE.md`,
Leap's TABLE_350 GDNB-2 (7370-7391), `~/TRIED/TRIED.md` (grepped for every idea below; line numbers given).
Three CPU replays of my own (numpy, python -I, seconds each, < 1 GB, no GPU; scripts and outputs beside this file):
`l350_checks.py` -> `.out` (who pays the fixed cost; all-sure checks; pass skipping), `l350_grid.py` -> `.out` (the 350
identity on the real check distribution; pass skipping with the drafter's decay priced), `l350_head.py` -> `.out` (what a
screened verify head would flip). Data: Breakthrough's DFlash2 lattices (664 HumanEval+ checks, 6,134 chat checks), MTP
Head's exact walks and 32-row walk (`~/TrainLab/data/d02b_walk*`), Leap's FR-Spec list (`~/Leap/data/frspec_rank.npy`).
Marks: [K] measured (source named), [P] arithmetic or replay on [K], [G] judgement. Gains halved where said.
Cost model [K FULL.md table, "tonight + Thu", code FAST at 45 GB]: fixed per check F = dense 28 + draft 6 + commit 5 = 39 ms,
flat in rows [K C0: every dense piece flat 1-16 rows]; per verified row c = (experts 22 + exposed SSD 12) / 9.84 = 3.46 ms.
On the lattice's checks this gives 77.1 ms and 122 tok/s at kept 9.38 (the live default keeps ~10.5 and runs 110-117 [K]).

## Corrections after the build (Thu 5 Nov 19:0x; read from the code while writing build_1105, which supersedes the rows below)

- **Row 1 is att_single alone, -2.4 ms.** `gdn_inproj` cannot run in the FAST all-in arm: ref2's `_InProjPad` concatenates
  the four GDN projections' packs as stored, and DENSE-N-RAM (R57_DNR=1, in that arm) has already replaced two of them by
  4-D native copies. K-83's exact `gdn_inproj_x`, on in the default, already holds most of R-68's saving. So "-2.4 to -2.9"
  reads -2.4, and the sums "-4 to -6 ms" read about -4 to -5 [K ref2.py 1009-1045, 288-312; r57_screen.py 706-712, 729-731].
- **att_single does not drop the AR classes or the key buckets.** It replaces MTPLX's 12 head-chunk calls by one fused
  SDPA per attention call; the class re-runs and bucket pads stay, and below 512 keys its bits equal REF2-T's
  [K ref2.py 1229-1285]. A FAST attention switch that does drop them exists too (F1, `fatt_verify`, fs4: TRIED 484), built
  for long contexts and in no BigMoE arm.
- **Row 5 is not untried.** The screened verify head is built: SVH, ref2 `screened_head`, FR-Spec top-98,304 + 4,096 session
  ids, -0.79 ms per check, PASS on wiring and speed (TRIED 1215), in no BigMoE arm. My replay only adds the outside-the-list
  rates of smaller lists.
- **Row 2, lazy commit, from the code:** 35 pure-GDN layers x 3.15 MB of fp32 state = 110 MB. Per 16-row check and layer
  today's protocol makes 1.75 state reads + 1.75 writes + 0.76 replay kernels, the lazy one 1.00 + 1.00 + 0, for 3.9 more
  recurrence steps [P fuzz at 24 % full accepts]. Expected -1.4 ms per code check (-0.8 to -2.0) [P/G]. The design changed:
  the pending rows travel as a third state leaf, because MTPLX's snapshots are views and its restores are clones.
- Build, checks and arms: `build_1105/INSTALL.md`.

## 0. Verdict

1. **No lossy lever on this wall reaches x1.3 on the code line. I looked for one and killed each candidate with a number.**
   The best items are worth +1 to +7 % each (section 3). Nothing here is a next-caliber lead.
2. **Why (three laws, section 2):** (a) the dense pass is flat in rows, so every per-ROW approximation (the brief's "skip
   layers for sure rows", "cheaper attention for drafted rows") saves 0 ms of dense time [K C0, D32 x1.24]; (b) per-CHECK
   approximations are capped by how many checks are all-sure: 16.9 % of code checks at cf >= 0.9, 11.4 % at 0.95, 0.1 % on
   chat [P replay]; (c) the dense floor is dependent stages (~400 x 20 us), not bytes, and the approximations that remove
   dependencies are exactly the ones the decision map kills (HC static 18.3 %, P-frame attention 29 %, GDN 88.5 %, early
   exit agreement 0.057 [K]).
3. **The lossy dense floor is ~14.3 ms against the exact 17.5 [P]:** lossy removes bytes (HC at 8 bits -0.64 GB, a screened
   verify head -0.30 GB), not stages. Dense at 3 bits is dead (+12.5 % ppl, 17-18 % rel rms [K TRIED 953, 1208]).
4. **The broken assumption is about priority:** today the fixed cost is 50.6 % of code time (75-77 % of chat time) [P]. Once
   the split form lands and the SSD is hidden (c 3.46 -> 1.05) it is 77 % of code time. So F 39 -> 20.3 is worth x1.31 at
   today's rows (122 -> 160) but **x1.59 at split rows (185 -> 294)** [P grid]. After the split arms this wall is the code
   wall, and it is engineering to the floor, not approximation.
5. **350 on code needs all three at once [P grid]:** F <= ~20 ms, c <= ~1 ms per row, kept >= ~11-12. At the FAST floors
   with split rows the lattice gives 294 tok/s at kept 9.38 (336 at the group floor), i.e. ~330-375 at the live kept 10.5.
   Even a fully kept 16-row check costs 5.9 ms per token tonight, twice the 2.86 ms that 350 allows.
6. **The single best bet (section 4): stop paying the exact track's dense taxes in the FAST preset.** Two built, measured
   switches are still off in every FAST arm (att_single -2.4 ms, gdn_inproj up to -0.5 [K]; SMALL-N is already in) = -2.4 to
   -2.9 ms of F with no new code; with the two untried kernels below (lazy commit, chunk-form GDN) -4 to -6 ms. That is +3
   to +9 % tonight and +5 to +14 % at split rows, halved +2-4 % / +3-7 %. It is the largest certain item on this wall, not
   a breakthrough.
7. **One number error to carry upstream:** FULL.md's "realistic ~52 ms" and AUDIT_1105 row 9 book commit 5 -> 3 ms to
   "Mission 29 design D". TRIED 831 records that design (the draft on a second stream beside the commit replay) as KILL at
   0.06 ms per cycle [K]. The commit is still ~4.7 ms [K TRIED 677]; that -2 ms is not owned by anything alive.

## 1. Who pays the fixed cost [P `l350_checks.out`]

| checks that keep | share of checks | share of tokens | share of time | ms per token |
|---|---|---|---|---|
| code 1-2 | 14.5 % | 2.5 % | 9.9 % | 33.2 |
| code 3-4 | 13.1 % | 4.7 % | 9.9 % | 17.3 |
| code 5-8 | 17.5 % | 12.0 % | 15.9 % | 10.9 |
| code 9-15 | 30.9 % | 39.7 % | 34.9 % | 7.2 |
| code 16 (the block cap) | 24.1 % | 41.1 % | 29.5 % | 5.9 |
| IFEval 1-2 | 68.2 % | 42.1 % | 65.0 % | 33.7 |
| multi 1-2 | 76.9 % | 52.5 % | 74.3 % | 35.9 |

- Code: fixed cost = 50.6 % of all time. The 27.6 % of checks that keep <= 4 tokens give 7.2 % of the tokens for 19.8 % of
  the time (22.7 ms per token). Chat (DFlash2 lattice; the live chat route is the MTP chain, kept 2.53): fixed = 75-77 %.
- The number of passes per token is set by misses on the drafter's unsure rows. Sure rows already ride for ~0 ms of dense
  time, so "not verifying" them saves no pass. A pass can only be skipped when a whole block has no unsure row.
- All-sure checks (all 15 drafted rows at cf >= tau): 16.9 % (tau 0.9; all 15 right in 90.2 %), 11.4 % (0.95; 96.1 %), 4.1 %
  (0.99) of code checks; 0.1 % on chat. Sure up to the live cut: 21.7 % / 15.4 % / 6.6 %.

## 2. Why lossy does not unlock this wall

**Law 1, flat in rows [K].** C0: check = 36-38 ms + 2.9-3.2 ms per row, the row term all experts. D32: 32 rows cost x1.24 of
16 on dense (TRIED 478). So a sure row's dense cost is 0 to 0.26 ms. Removing layers or attention work for some rows of a
check changes nothing on this wall; what it saves is expert pairs (Fable C's wall; DM chained skip of sure rows is KILL on
code there, TRIED 85).

**Law 2, per-check gating is capped at ~17 % of code checks [P].** Any "sure-check mode" (thinner dense pass, no pass at
all) applies to 11-17 % of code checks and ~0 % of chat checks at flips <= 0.05-0.4 %. Halving the dense pass on those
checks would be -3 % time. An oracle that skips every fully accepted block (26.4 %) is x1.12 (row 3 below).

**Law 3, the floor is dependencies, and removing dependencies is what the map punishes [K].** Dense floor = 2.7 GB of bytes
(9.5 ms) + ~400 true dependent stages x 20 us [K FULL / cloud]. Tonight's 28 ms is ~18 ms of stage chain. Every measured
approximation that cuts a dependency fails on the map: early exit after layer 40 agrees on 0.057 of code positions, after
47 on 0.925-0.950 (TRIED 644); reusing the previous position's attention / GDN output flips 29.0 % / 88.5 % of code-sure
rows (609); stale MoE reuse 2.46 % at lag 2 (166); the middle-band MoE reuse flips 30-40 % of mid rows (84); static HC
mixing flips 18.3 % of rows, 3.7 % of sure rows (HC_READ 0.1). Layer-pair parallelism (layer l+1 reading layer l's input)
is the same family: layer L+1's router on L's input names only 50.5 % of the misses [K ROUTEPRED], so I price it KILL [G]
without a run. What lossy does buy is bytes: FAST dense floor = 2.7 - 0.64 (HC8) - 0.30 (screened verify head) = 1.76 GB =
6.2 ms + 8.0 ms of stages = **~14.3 ms** [P], 3.2 ms under the exact floor.

## 3. Ranked ideas (code FAST at 45 GB; ms per check at tonight's engine; tried-status from TRIED)

| # | idea | gain [P unless marked] | halved | status | quality |
|---|---|---|---|---|---|
| 1 | **FAST without the exact track's dense taxes**: att_single (one fused SDPA per attention call: no AR classes, key buckets, pads, straddle recomputes) and gdn_inproj in the FAST preset | attention is ~0.6 ms per call, 0.45 of it REF2's exact verify SDPA at 81-87 kernels [K GDNB-2]; att_single -0.2 ms per call = -2.4 ms [K]; gdn_inproj up to -0.5 [K 548; part may already be in the exact gdn_inproj_x]: **-2.4 to -2.9 ms**, +3.2-3.9 % tonight, +5-6 % at split rows | +1.6-2 % / +2.5-3 % | built and measured, but OFF: `att_single` and `gdn_inproj` are False in ref2.py's CFG (lines 177-178) and appear in no BigMoE arm; SMALL-N (-1.14) is already in the FAST all-in arms (c224_thu.sh, fri_night.sh) [K, read from the code]. Not new work: an adoption the FAST line allows and the exact line blocked | REF2c-T (att_single + gdn_inproj): IFX STOP PASS +0.9 pt, core IFEval -5.0 [K DAY9 639] |
| 2 | **Lazy commit** (untried): the next check's GDN kernel first replays the previous check's accepted rows in its own pass over the state and writes it back; no separate commit replay | the state is 113 MB [K 677: 16 rows' states = 1.81 GB]; the replay moves 220 MB [K 681]. Saves the replay's read (0.6 ms), its 35 kernels and its slot in the serial tail, costs ~0.3 ms of register steps: **-1.0 ms (-0.6 to -2.4)**; the bracket is whether today's replay is R-21's bandwidth-bound 1.2 ms or R-12's 3.3 ms | +0.7 % tonight, +1.5 % at floors | UNTRIED (COMMIT-SELECT 677, GDNB batch 681, design D 831 are KILL; none folds the replay into the next check) | exact: the same steps in the same order |
| 3 | **Pass skipping on all-sure blocks** (emit the block with no pass; its rows ride the next pass at x1.24 dense, sure rows at half the row price) | ceiling x1.078-1.124 with no drafter decay; **x1.045-1.075 with the measured decay** (the drafter's per-row rate without the 125B's features is 0.78-0.83 at depths 16-20 vs 0.87-0.95 [K base32 walk]); x1.01 if the deferred rows cost full price; oracle x1.12; x1.036-1.059 at the floors; chat 0 | +2-4 % | the exact cousin is KILL: 32-row + Q1 with the 0.3 cut -1.3 %, "deep rows assumed sure" +3.8-4.3 % (TRIED 192, 444, 872) | flips <= 0.05 % (tau 0.95), 0.18-0.39 % (0.9); prior art: token-level routing / speculative cascades (adapted) |
| 4 | **Chunk (WY) form of the GDN verify in FAST** (one chunk solve for 16 rows; with row 2 the commit becomes a rank-a fold) | body 2.9 ms [K GDNB-2] -> ~2: -1 ms [G] | +0.6 % | KILL on paper for exact modes only (TRIED 1186: "changes the arithmetic") | arithmetic noise only |
| 5 | **Screened verify head** (the 125B's lm_head over FR-Spec + session ids; exists for the MTP chain's head, TRIED 751) | head bytes 0.356 -> 0.05-0.095 GB: -0.9 to -1.1 ms, +1.3 % | +0.6 % | UNTRIED on the target's head (exact screens KILL / PARKED: 370, 549, 567) | **measured [P `l350_head.out`]: the 125B's token is outside the list on 0.235 % of held-out code tokens (32K + session), 0.094 % (64K + session); 26.9 % on the agent set.** A poor trade: 0.1-0.24 points of code's 1.34 for +1.3 %; KILL on tool-call text. (The 0.00 % on IFEval / multi / math is in-sample: the list was ranked on those captures.) |
| 6 | HC gate lag (the MoE-side read's two GEMMs run on the pre-mixer streams, overlapping the mixer) | -2.4 ms of chain [G] | +1.5 % x odds 0.25 | UNTRIED | Law 3's family; I expect BETWEEN / KILL [G] |
| 7 | Lower KV precision / windowed keys for drafted rows | key bytes are +1.37 ms per 1K keys [K LONG_CONTEXT]: <= -0.7 to -1.4 ms at the line's 1-2K context; -5 ms at 8K | ~+0.5 % | KV8 BETWEEN, stopped (537); KILL at 12 GB (1069) | a long-context lever, not a 350 lever |
| 8 | Activation sparsity on narrow (<= 4-row) checks | bytes x0.6 on the 4-bit projections of short checks: ~-1.4 ms on 27.6 % of code checks, most chat checks [G] | ~+0.5 % code, +2 % chat | TEAL PARKED (364), never priced per check width | lossy; untested |
| 9 | Skip layers for sure rows | 0 ms dense (Law 1) | 0 | expert side KILL on code (85) | |
| 10 | Early exit / layer-pair parallel / static HC / P-frame mixers / dense at 3 bits | n/a | n/a | KILL [K 644, 609, 953, 1208, HC_READ] | |

Rows 1 + 2 + 4 together: -4 to -6 ms of F [P/G]: 77 -> 71-73 ms tonight (+5-9 %), 50.6 -> 44.3-46.6 ms at split rows
(+9-14 %). The FAST dense pass goes 28 -> ~24-25 and commit 5 -> ~4 without waiting for any exact certification.

Not listed: a lower-trunk-only pass for sure blocks. The drafter's taps span the 48 layers (EARLYTAP KILL, TRIED 845), so
a half pass gives it nothing.

The brief's four options, answered: cheaper attention for drafted rows = row 1 (the saving is the exactness machinery, not
the attention math); skipping layers for sure rows = row 9 (0 ms here); lower KV precision = row 7 (long context only);
merging passes = rows 2 and 3 (commit into the next check; block into the next pass), and the draft into the check is
DRAFT-AHEAD (hit 35-41 % [K TRIED 1207], owned).

## 4. The single best bet: the FAST preset stops paying the exact track's dense taxes

The assumption it breaks: "FAST = the exact dense pass + lossy experts". The exact pass carries machinery whose only job is
spec == plain decoding bit for bit: AR-class SDPA calls, key buckets and pads, recomputes when rows straddle a class, 16-row
padding of small calls, bitwise replicas of MLX's reduction order, the commit's separate replay. FAST is judged by task
scores, so none of it is owed. Two pieces are built, measured and still off (row 1); two are new kernels (rows 2, 4).

Test 1 (BigMoE, <= 10 min GPU after Fri 00:15): 12 HumanEval at C 224, A-B-A-B, the FAST all-in arm vs the same arm with
ref2's `att_single` and `gdn_inproj` on, with the decision-map arm for the same two switches in the same window.
- PASS: median check time -2.0 ms or better AND the map's code flips rise by <= 0.3 points (total within 1.34).
- KILL: saving < 1.0 ms, or flips rise by > 0.6 points. Between: keep att_single alone if it carries the saving.

Test 2 (Leap, <= 10 min GPU, the gdnb_wire_check harness on real layers): lazy commit. The merged recurrence kernel with
keep = 2, 9, 16 accepted rows vs today's check body + commit replay.
- PASS: outputs, conv state and recurrent state bitwise equal at all three keeps AND >= 0.8 ms saved per cycle over the
  36 GDN layers.
- KILL: < 0.4 ms. If it is fast but not bitwise, it moves to the FAST preset with row 4's chunk form.

Why lazy commit works [P]: today each cycle walks the 113 MB of fp32 GDN state three times: the check reads it
(speculative, no write), the commit replay reads it again for the accepted rows and writes it back. Keep the state as it
was before check N plus the accepted rows' small inputs (k, v, gates: kilobytes). Check N+1's kernel, per state row, runs
those accepted steps in registers, writes the committed row, then runs its own 16 speculative steps: one read and one
write per cycle. The drafter reads the 125B's layer taps, not the GDN state, so nothing waits on the fold.

## 5. What 350 needs from this wall [P `l350_grid.out`; tok/s on the 664 real checks, kept 9.38]

| fixed per check F | c 3.46 (tonight) | c 2.24 (SSD hidden) | c 1.05 (split form) | c 0.69 (split at the group floor) |
|---|---|---|---|---|
| 39.0 tonight (28 + 6 + 5) | 122 | 147 | 185 | 201 |
| 29.5 realistic (22 + 4.5 + 3) | 139 | 173 | 228 | 253 |
| 23.0 exact floors (17 + 3.5 + 2.5) | 153 | 197 | 271 | 306 |
| 20.3 FAST floors (14.3 + 3.5 + 2.5) | 160 | 208 | 294 | 336 |
| 10.0 (no mechanism) | 195 | 270 | 434 | 532 |

Multiply by ~1.12 for the live kept 10.5 and by a further ~1.14 for RELAX's kept 12 [K FULL]. So 350 = the FAST floors on
this wall (F ~20) x the split form with the SSD hidden x kept 10.5-12. No row of section 3 moves F below ~20; the gap from
29.5 to 20.3 is Leap's stage fusion (900 -> 400 stages), which the FAST track can take without bitwise replicas.

## 6. Reproducibility and limits

`nice -n 15 ~/BigMoE/.venv/bin/python -I l350_checks.py > l350_checks.out`, and the same for `l350_grid.py` and
`l350_head.py` (seconds each; free memory 23 GB before, swap unchanged at 8.95 GB). The lattice walk keeps 9.38 per check
with the live cut (the engine's default adds COPYGATE and draft trims: ~10.5), so absolute tok/s are a model; ratios are
the result. The flips quoted for pass skipping count the drafter's wrong tokens in trusted blocks, not a map run. The pass
skipping replay was run without a pre-registered line; I read it as BETWEEN-weak. Rows 4, 6 and 8 are [G] sizes with no
replay behind them. Row 1's "off" status is read from ref2.py's CFG defaults and a grep of BigMoE's arm scripts, not from a
run. The brief names HC8_READ.md; no file of that name exists within four levels of the home folder, so the HC8 numbers here
come from HC_READ.md and TRIED 492, 493, 533. Nothing was run on the GPU; no other session's files were touched; nothing was committed (this folder is not a git
repository) and no findings note was written, since nothing here passed a line.

## 7. Lines in TRIED's shape (for whoever merges; I did not write into ~/TRIED)

- LOSSY350-B GATECAP: how many checks are all-sure (every drafted row's cf >= tau), the cap on any per-check lossy mode | wall: fixed-cost | line: 45 | exact: n/a (measurement) | verdict: PASS (measured) | code 16.9 % (tau 0.9), 11.4 % (0.95), 4.1 % (0.99); chat 0.1 %; fixed cost = 50.6 % of code time, 75-77 % of chat time; checks keeping <= 4 = 27.6 % of checks, 7.2 % of tokens, 19.8 % of time | 2026-11-05 | ~/FableGod/fableb/l350_checks.out
- LOSSY350-B PASS-SKIP: emit an all-sure block with no pass, its rows ride the next pass (x1.24 dense, sure rows at half the row price) | wall: fixed-cost | line: 45 | exact: no | verdict: BETWEEN-weak (CPU model, no pre-registered line) | x1.045 (tau 0.95) / x1.075 (0.9) with the drafter's feature-free decay priced (0.80 vs 0.92 per row), x1.01 at full row price, oracle x1.12, chat 0; flips <= 0.05 % / 0.18-0.39 %; the exact cousin is M-18c | 2026-11-05 | ~/FableGod/fableb/l350_grid.out
- LOSSY350-B VHEAD: the 125B's verify lm_head screened to FR-Spec + session ids (FAST) | wall: fixed-cost | line: 45 | exact: no | verdict: BETWEEN on code, KILL on tool-call text (CPU replay of the outside-the-list rate) | held-out code 0.235 % of tokens outside 32K + session, 0.094 % at 64K + session; agent 26.9 %; saves 0.9-1.1 ms per check | 2026-11-05 | ~/FableGod/fableb/l350_head.out
- LOSSY350-B LAZY-COMMIT: the next check's GDN recurrence kernel replays the previous check's accepted rows in its own pass over the state and writes it back; no separate commit replay | wall: fixed-cost | line: all | exact: yes (same steps, same order) | verdict: PROPOSED (priced, not built) | -1.0 ms per cycle (-0.6 to -2.4: R-21's 1.2 ms replay vs R-12's 3.3 ms) | 2026-11-05 | ~/FableGod/fableb/LOSSY350_B.md section 4
- LOSSY350-B FAST-TAX: att_single + gdn_inproj switched on in the FAST preset (the exact track's spec == plain machinery is not owed by FAST; both are False in ref2's CFG and in no BigMoE arm, SMALL-N is already in) | wall: fixed-cost | line: 45 | exact: no | verdict: PROPOSED (an adoption; pieces measured) | -2.4 to -2.9 ms per check from measured pieces (-2.4 / up to -0.5); +3.2-3.9 % tonight, +5-6 % at split rows | 2026-11-05 | ~/FableGod/fableb/LOSSY350_B.md section 4
- LOSSY350-B PAIRPAR: layer l+1 reads layer l's input (layer-pair parallel dense pass) | wall: fixed-cost | line: 45 | exact: no | verdict: KILL (on paper, [G]) | the map's neighbours: P-frame attention 29.0 %, GDN 88.5 %, middle-band MoE reuse 30-40 % of mid rows; L+1's router on L's input names 50.5 % of misses | 2026-11-05 | ~/FableGod/fableb/LOSSY350_B.md section 2
