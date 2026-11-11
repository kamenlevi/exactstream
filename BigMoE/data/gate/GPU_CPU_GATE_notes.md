# Notes for GPU_CPU_GATE_draft.md (Mon 9 Nov 2026)

Scope: draft written only from the two inputs plus re-reads of the raw files they name. Nothing was run on the GPU, nothing
outside the scratchpad was edited, no web search, no commit. Local read-only checks run for section 7: `sw_vers`,
`system_profiler SPHardwareDataType / SPDisplaysDataType`, `softwareupdate --history`, `ioreg` (plugin name only),
`cmp` of the gate files against the release package, `git show 7c9f049:data/test_gate.txt` (read-only).
Draft length: 458 lines, of which ~44 are references and ~70 blank; tables are one line per row.

## A. Every number used, with its source (re-read on 9 Nov unless marked)

Paths: BM = ~/BigMoE, LP = ~/Leap, MS = ~/MoonShots.

### Abstract and section 1
| number | source | note |
|---|---|---|
| mx.eval +157.4 / 160.6 / 161.5 us per layer (48 x 20 MB synthetic) | BM/data/gpuwait.txt lines 4, 14, 24 | re-read |
| "about 230 us" mx.eval on the 35B | BM/FINDINGS.md 'The gate' table (l.84) | run files gone (MoEPrefetch repo deleted); the 107.8 -> 54.5 tok/s table survives only in a session transcript, so NOT quoted (STYLE: never quote chat) |
| MTLSharedEvent handshake +147.9 / +166.2 (eventwait), +131.2 (v2), +132.7 (v3), +139.6 (v4) -> 131-166 us | BM/data/eventwait.txt l.4,10; eventwait_v2..v4.txt l.4 | re-read; "busy-poll" variants 132.4 / 142.5 / 147.8 inside the range |
| GPU-only signal->wait chain +127.1 / +142.3 / +142.3 | eventwait_v2/v3/v4.txt l.5 | re-read |
| mem-flag + CPU-signalled wait: -0.8 (v4 l.8); v5: +0.8, +1.0, -0.7, +0.7 -> "-0.8 to +1.0 us" | eventwait_v4.txt l.8; eventwait_v5.txt | the files print "+-0.8" = a negative value |
| command buffer per layer -3.3 to +0.5 us | eventwait.txt l.3,9; v2-v4 l.3 (-1.2, -0.9, -3.3) | re-read |
| 125B bare sync 0.256 ms; 97 sync latencies per check; GPU idle ~59.7 of 185.1 ms | LP/data/r50c_screen.txt l.2-5 | re-read (HO not used) |
| 0.155 ms x 48 = 7.4 ms per pass | BM/FINDINGS.md 'Phase 1 verdict' | modelled [P] |
| forecast recall 0.634 (R-57), 0.794 (full) | BM/data/r57/pr2_160_preroute.txt ('all' rows) | re-read |
| model facts 48 layers / 512 experts / top-10; 35B 40 / 256 / top-8 | [MAIN] 4.1; BM/FINDINGS.md section 3 l.179 | |
| pickup 2.9 us (coherent, median) | LP/data/w1_pickup.txt arm B rows w 50-2000 | B1 gives 2.8 |
| 70-80 us per gated layer (35B) | 79 us derived from BM/data/series35.txt l.23 (77.2 tok/s); 69 us from FINDINGS l.87 (not saved) | |
| 146-181 us event gate | derived from BM/data/diag35.txt l.3,6,7,8 (58.7 / 62.8 / 64.0 / 62.5 tok/s vs C=256 102.1) -> 181 / 153 / 146 / 155 us | formula (1/tps - 1/102.1)/40, 4,036 gated calls per ~100 tokens |
| 232.6 -> 117.3 ms; 213.7 -> 106.2 ms; logits equal in 4 passes | BM/data/r55/r55_screen.txt l.6, 11 | re-read; verdict line: gate 111.8 vs slot 223.1 ms median over prompts (not used) |
| MLX fast fence: PR #1773, merged 18 Jan 2025, 15-20 vs 150-200 us | BM/data/gate/gate_prior_art.md A1 | author "Awni Hannun" from the task brief / BM/HANDOVER.md 17:2x line ("A. Hannun ... verified on GitHub"); the prior-art file itself does not name him |
| fence_wait pickup 1.2 us (w >= 50 us), 1.8-2.0 us (w <= 0) | LP/data/w1_pickup.txt arm M | re-read |
| 21.7 us per layer all-hit gate | BM/data/test_gate.txt l.2 | "all hits" with 16 misses in 720 calls |

### Section 2 table and notes
| number | source |
|---|---|
| hit path +5.1 / 3.6 / 3.2 us | gpuwait.txt l.3,13,23 |
| stalled layer 70,537-196,042 us (no I/O) | gpuwait.txt l.5,15,25 |
| 47 of 48 slot checksum mismatches (3 variants) | gpuwait.txt l.9,19,29 |
| 107 / 112 us buffer start after the CPU signal (medians) | BM/data/trace35.txt l.1,3 |
| Swift spin +2.9 us/layer, 0 timeouts; +59.9 at 100 us delay; variants +46.3 to +58.5 | BM/data/spinwait.txt; spinwait_variants.txt |
| 2,303 of 4,037 fast-path (57 %), C=153 with prefetch | series35.txt l.40 (and l.49) |
| JIT tables +6 % (37.1-37.3 -> 39.0-39.9 tok/s, spec k=3), identical | BM/LEDGER.md row PS |
| slow path: 62.4 tok/s exact, 14.4 slow calls at 305 us CPU; never-wait 91.2 tok/s (wrong output) -> 351 us wait, ~46 us round trip | series35.txt l.80-82; derived [P] |
| chunk_bench 0.392 / 0.387 / 0.460 / 0.477 / 0.531 ms | BM/data/chunk_bench.txt |
| row widths 40 -> 61 -> 92 -> 185, default 160 | fact sheet steps 7a/7h; commits 945cd21, 46db7d3, 4be3b31, 8e2b22e (git log) |
| R-55L 21.9 -> 31.0 (+41.5 %) whole run; steady 23.7 -> 32.0 (+35.0 %) | BM/data/r57/r55l_summary.txt l.10 |
| R-55 misses per warm check 64.1 vs 32.2 (HE/68), 26.9 vs 38.4 (prealgebra) | r55_screen.txt l.6, 11 |
| OOO +4-6 % (34.2 -> 36.0, 24.8 -> 26.3, 33.4 -> 34.8) | BM/LEDGER.md row OOO1 |
| OOO +13 to +18.7 % at 45 GB | [MAIN] 4.4 lever table (T14, T490) |
| stamps: 5 + 4 = 9 torn calls, 1,077 / 906 re-reads, 0.03 / 0.02 ms; 7 torn calls in 6 loaded runs before | BM/LEDGER.md row GS1 |
| harness release 6.38-7.92 -> 0.17-0.21 us; pickup 15.7-73.5 -> 2.8-2.9 us; PS 7.4 us | LP/data/w1_pickup.txt; w1_pickup_P.txt |
| engine +1.9 % (coherent spin), -0.6 % (system post), traced single passes, geo steady | BM/docs/WED_TESTS.md item 2 (l.87-91) |
| CHS4: 3/12 == AR (ycoc, ycoc2); 12/12 (ycoh, ycoh2); 37.7 / 38.0 vs 37.8 / 37.9 tok/s | BM/data/chs/chs4_report.txt (re-read) |
| 465 misses in 4,036 gated calls; NOWAIT 103.6 tok/s | diag35.txt l.5-8; series35.txt l.23 |
| ~5 %: 93.4 vs 102 (-8.5 %) | FINDINGS l.88, not saved |

### Section 3
| number | source |
|---|---|
| MAXK 160 default | BM/bigmoe/gated.py l.22 |
| 96 gate rows | BM/bigmoe/stream.py l.493 (nl + nmtp + nl, tiers on, MTP stream off) |
| row 1 KB at 160 ids (96 + 32 + 128 ints), 185 ids max | gated.py l.25-31 |
| 128 regions x 128 entries x 192 ints = 96 KB / region, 12.6 MB | gated.py l.106-111; ext l.1736 |
| 1,024 tables x 17 x 32 ints = 2.2 MB, 527 entries | gated.py l.91, 114-115 |
| slot ring 16 MB, status 16 KB, commit input 64 MB | gated.py l.105, 112; ext l.2179 |
| stamp bits 22-31 / 0-21 | ext l.592-601 |
| 64-192 post threads | ext PostIds (nt = 64 / 128 / 192) |
| MLX limits 40 / 50 MB (and 20-50 ops) by class | BM/ref/mlx_device.cpp l.604-626; commit rule l.511-514 |
| back-off d = 32, x1.1, cap 400,000; 128 polls | ext l.1123-1165 (SPIN_G1/G2 defaults 1.1) |
| entry index vs time: p6 22 us, p41 236, p65 2,074 | w1_pickup.txt l.43 |
| ~0.2 s fresh-line timeout | [P] extrapolation; LP/patches/w1_pickup/README.md says "~0.2 s"; my own check: d caps at poll ~99, ~15.6 M FP steps at ~13 ns each |
| fresh-line pickup 15.7-17.4 (w <= 50), 28.9 (200), 73.5 (1 ms), 67.2 (2 ms) | w1_pickup.txt arm A |
| 0.25 us per poll (3,958 polls at w = 1 ms; 7,938 at 2 ms) | w1_pickup.txt arm B |
| 1,000,000 poll cap (~0.25 s) | ext l.1000-1001 comment; [P] |
| region reuse 1.33 passes, table 10.7 | [P] 128/96, 1024/96 |
| 45 us per layer (status as output) | ext l.1847-1849 comment, not saved |

### Section 4
| number | source |
|---|---|
| 6 us row cadence (Swift, encoder per kernel) | spinwait.txt l.1 |
| 285 / 293 of 300 not seen in 5 ms | w1_pickup.txt l.39; w1_pickup_P.txt l.3 |
| 20.0-27.4 us after 8 / 64 MB dispatch (P2 20.0; P1 23.0 and 27.4) | w1_pickup_P.txt l.4-5; w1_pickup.txt l.40 |
| 0.5-999 ms visibility (V0-V2) | BM/data/coherence_v012.txt (0.489 to 998.7 ms) |
| 0.3-0.9 ms with 64 MB sweep (V4) | BM/data/coherence.txt l.11-12 (0.323-0.898 ms; first trials 28-36 ms). NOTE: the fact sheet says 0.6-0.9; one trial is 0.323 |
| 119 / 920 / 1,160 / 995 timeouts; fifth arm 0 timeouts, changed shas, 3.56 tok/s | MS/data/day12/summary.txt; cb_mbopsbfs_256.txt |
| 0 / 2,400 mismatches | test_gate.txt l.5 |
| 12,000 / 12,000; 4 x 4,800 / 4,800 | series35.txt l.21, 39, 48, 58, 69 |
| stale-read probe: B0 0/20; B1 19-20; B2 17-20; B3 16-20; B4 0; B5 0; B6/B7 0 | LP/data/s7b_probe{,2,3,4}.txt |
| coherent/plain bandwidth 1.09, 1.01, 1.04, 1.24 | s7b_probe*.txt line A |
| stale-slot check: 3 configs x 3 prompts x 128 tokens, 127 hashes | BM/data/sc/sc.log |
| +0.7 / +1.5 / +26.3 % coherent cost | [KERNELS] 5.7 (R13) |
| 10-bit stamp, 1,024 passes | ext l.596-597 |

### Sections 5-10
| number | source |
|---|---|
| 2,397 of 2,400 (event gate commit) | `git show 7c9f049:data/test_gate.txt` (re-read) |
| 0 zero-hit calls of 10,224 (8,540 + 1,684) | BM/data/r57/s0_report.txt |
| 14.0 ms gate service; -2.8 % | BM/docs/WED_TESTS.md l.415-421; [MAIN] 10.1 |
| M5 Pro facts (Mac17,9, 18 cores 6 + 12, 20-core GPU, Metal 4, 64 GB) | system_profiler 9 Nov |
| macOS 26.5 (25F71); CLT 26.6 06.07.2026 | sw_vers; softwareupdate --history |
| MLX 0.32.2, mlx-lm 0.31.3, Python 3.12 | BM/.venv site-packages listing |
| GH200 0.75-0.83 us; MI300A 16-18 us; UVM 30-45 us; Anukari < 50 us (not used); flash-moe -23 % | gate_prior_art.md C, B6, A5, A6 (all self-reported) |
| frame budgets 16.7 / 8.3 ms; 0.06-0.5 % and 0.13-1.0 % | [P]: 10.5 and 80 us over 16,667 / 8,333 us |
| 10.5 us harness sum | [P]: 7.4 + 0.17-0.21 + 2.8-2.9 |
| ~31 MB rings | [P]: 16 + 12.6 + 2.2 MB + small |
| 24 GB: 19.2 of 34.1 ms (56 %) | BM/data/b40/split.txt l.10 |
| 45 GB seen->release 0.962-1.012 ms; pread 0.709-0.768 ms | s0_report.txt ('hits' class = calls with misses) |
| 81 % busy, >= 19 % idle | [KERNELS] section 9 (R22) |
| fixed part per check 19.2 -> 19.5 ms (line <= 10) | WED_TESTS.md item 2 |
| ~96 KB with one entry per region | LP/patches/w1_pickup/README.md l.33-34 |
| 1.6-1.9 us MLX fence advantage; 0.15-0.2 ms per 96-row pass | [P] from w1_pickup.txt arms M vs B/B1 |
| 21 months (Jan 2025 -> Oct 2026) | [P] |
| d45 +6.4 % lever-1 bundle | NOT used in the draft (bundle includes non-gate levers; steady metric) |

## B. Could not source / deliberately not used

- The 35B mx.eval "~230 us" and its 107.8 -> 54.5 tok/s table: raw repo gone; only a session transcript holds the table.
  Quoted as "about 230 us, run not saved"; the saved synthetic 157-162 us leads.
- FINDINGS' "~69 us per gated layer" and "~5 % total on hits" (93.4 vs 102): no saved runs. Shown with that caveat.
- The "inside MLX, GPU writes reach the CPU only at command-buffer end" claim: BM/scripts/test_vis.py exists, output not
  in data/. Shown as notes-only.
- The under-4 KB constant binding: failing run not saved; MLX custom-kernel source not checked (not on disk).
- Failing counts for the three 17 Oct event-gate bugs: not saved (only the final 12,000 / 12,000).
- The "status as output ~45 us/layer" figure: code comment only.
- test_gate_wide.py (300 passes at MAXK 40 / 80 / 160): output only in HANDOVER, not used.
- 26 Oct thermal emergency sleep timeouts: HANDOVER only, not used.
- OOO at 45 GB from HANDOVER (+18.7 % steady / +15.9 % whole): used [MAIN] 4.4's "+13 to +18.7 %" instead.
- The ~5 s GPU watchdog said to be in MLX #3830 (fact sheet): not verifiable offline; not used.
- Which MLX GPU class letter the M5 Pro reports (40 vs 50 MB): not determinable without running Metal code (ioreg shows
  only "AGXMetalG17X"). Draft says "40 or 50 MB"; 64 MB exceeds both, so the conclusion stands.
- Author names / titles for several references (Allen and Ge, Kim et al., Fusco et al., Wahlgren et al., GPUfs, Genesys,
  NCCL papers): the inputs give only "X et al." + venue/arXiv id; I did not add titles from memory. The reference list
  must be regenerated by ~/MoonShots/papers/make_refs.py (STYLE section 9) before release.

## C. Things the editor should know (possible conflicts with MAIN or the fact sheet)

1. MAIN F.5 (l.~1189) and TRIED T76 say device-scope stores reach the CPU "about 20-23 us after the next dispatch". Raw
   LP/data/w1_pickup.txt l.40 has a 27.4 us median (run 1, P1, 64 MB); w1_pickup_P.txt has 23.0 (P1) and 20.0 (P2). The
   draft says 20.0-27.4 us. Suggest MAIN say "about 20-27 us".
2. MAIN F.5 quotes "about 69 us ... against about 230 us, from runs that were not saved". The draft prefers saved data:
   79 us (one saved run, series35 l.23), so "about 70-80 us", against 157-162 us synthetic mx.eval (gpuwait.txt).
3. MAIN F.5's "visibility is guaranteed only at command-buffer boundaries [K, BM:FINDINGS.md section 4]": that is a
   reading [G] (consistent with MLX maintainers in #3142), and FINDINGS section 4's "in-kernel signalling does not work on
   Metal today" is superseded by the 3 Nov coherent(system) result. The draft states both.
4. MAIN's lever table gives the gate "+35 %" (R-55L steady, a metric STYLE marks information only). The same summary's
   whole-run figure is +41.5 % (21.9 -> 31.0, mean of six prompts, not pooled). The draft leads with the whole-run figure.
5. NEW pitfall not in the fact sheet: MS/IDEAS.md 'Round 30' + MS/data/day12: MLX_MAX_MB_PER_BUFFER=4096 broke the gate
   (119-1,160 timeouts per run in four arms; the IDEAS note says "920-1160" and omits the 119 arm). A fifth arm with
   MLX_BFS_MAX_WIDTH=80 had NO timeout (gate.check() passed, flag_ab125.py l.65) but a different logits hash and tokens at
   3.56 tok/s. Cause not isolated: possibly a silent exactness failure the timeout flag misses, possibly MLX's traversal.
   Worth a look before the paper claims "never silent" too strongly (the draft says timeouts are never silent; it does not
   claim all failures are caught).
6. The fact sheet's coherence V4 range "0.6-0.9 ms" omits a 0.323 ms trial; the draft says 0.3-0.9 ms.
7. The novelty ledger tags the gate "adapted (anemll-flash-mlx / SlotStream sync-free slot gate)" (TRIED l.488, 501, 507;
   NOV). The 9 Nov prior-art search did not cover those two projects; NewIdeas/PRIOR_ART.md describes them as slot banks
   with expert ids as data / a fixed slot pool, with no mention of a GPU-side wait. The draft credits them as the designs
   the records name and makes no claim about how they wait. A read of their code would settle it.
8. The DFlash2 credit was added (STYLE section 5) because the R-55L live loop used the block drafter.
9. STYLE's six-paper list has no slot for this paper; companions cite it as e.g. "[GATE]" if the editor agrees.
10. KERNELS uses "D:" keys for Leap data; the draft uses MAIN's "LP:data/..." keys.
11. Design constants (sizes, limits) carry code keys but no [K]/[P]/[G] mark; the header says so. If the editor wants
    every number marked, they could take "[code]" or [P].

## D. Open questions (also in the draft's 10.3)

- Is the forced commit still needed with system-scope posts (BIGMOE_NOPOSTCOMMIT exists; no recorded run)?
- Which cache level holds stale lines; any Apple statement of the never-read behaviour?
- The under-4 KB binding rule of mx.fast.metal_kernel.
- Why the event wait inside MLX started 107-112 us after the signal (trace35) when Swift showed no delay.
- The 19 Oct BFS-width arm (C.5).
- An engine-side pickup histogram under the coherent poll (kernel has the poll counter `it`).
- MLX's own round trip on this M5 Pro (synchronize_bench.py with MLX_METAL_FAST_SYNCH=1), suggested by the prior-art file.
- Other chips (the 36 GB M5 Max named in HANDOVER is unmeasured) and OS versions; energy of the busy-polling loader.
