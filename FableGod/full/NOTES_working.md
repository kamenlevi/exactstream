# Working notes (Fable FULL, Wed 4 Nov 2026, 20:2x-) — key numbers gathered while reading; sources in brackets

## Machine / model physics [K]
- M5 Pro 64 GB, 20 GPU cores. DRAM stream 258-294 GB/s (bw_probe 285-286 at 4 GB, best 292). SSD ~12-14 GB/s random 1.77-2.2 MB reads at QD>=4; 5.7 GB/s QD1.
- Matrix unit: bf16xuint4 from TG memory 12.4 T MAC/s; bf16 regs 14.0; int8xint4 19.2; int8 peak 25.6. ALU ~4.1 T lane-ops/s [G].
- Model: 125B total, ~5B active, 512 experts/layer, top-10 routing, 48 layers, trunk (dense) ~2.7 GB per pass at HC8 (3.3 GB bf16?), expert record 2.16 MB (MSB-3) + 0.77 MB LSB sidecar (ranks 0-1 only); 11-bit pack 2.064/0.721 MB. 4-bit record 2.79-2.93 MB.
- Per 16-row exact check: experts 7.30-7.38 GB (3,680 groups), 8.0 GB exact incl sidecar; dense 3.3 GB. Union per block code: 10/17/29/47/75/111/158 experts per layer at 1/2/4/8/16/32/64 rows.

## Today's engine (code, 45 GB, BigMoE 5b, C 224) [K BUILD_DAY]
- EXACT (== REF2-T) 83.1 tok/s; FAST U3C 110.1 tok/s at 15.0 MB/kept. FAST is GPU-bound at 45 GB (U3->U3C cut SSD bytes 27 %, +3.7 % only).
- Kept per 16-row check: DFlash2 L1-20 9.57 held-out code (exact walk), ~10.5 in engine. Perfect pick among its 16 candidates 12.37 (RERANK-0). Data slope +0.13 kept per doubling. P-CAP probe (10-layer 2x drafter) running tonight.
- Per check today (RELAX_350, Mon): draft 12.0 -> D7b 6.3 ms; dense 36 (C0) -> Leap's builds EXACT -3.0, FAST -7 ms (HC8-1P); rows: NAX6 43.2 ms (C2) at 16 rows, NAX6-N 35.7 (33.3 quiet-rebased), NAX6-NH 33.2 (30.9); byte floor 25.7 at 284; 3c 146 GB/s (exact REF2-T kernel; 3c-N 4-7 % slower). SSD exposed ~65 ms (Mon, C 256).
- Floor engine (FINDINGS M32): GPU floor code ~55 ms/check EXACT (dense 19-21 + experts 30 + draft 3-6 + 2.5) = 229 tok/s at kept 12.6; FAST ~42 ms = 300. "300 FAST = every GPU part at its floor".
- Mission 14 "wall equation": bytes/token = (trunk + union)/kept. 32-row check KILL (Tue night: tokens differ, code -62 %); k-head KILL.

## Kernel (KERNEL90) [K]
- NAX6-N exact: 219 GB/s = 74 % of R(296) = 77 % of machine; NAX6-NH FAST 79-81 %; 12 bitwise arms all KILL. Verdict: stop kernel nights; gains are in bytes & pairs (gate union cut pairs x0.48, bytes x0.58).
- Ratio: 16-row padded tile needs 39.7 MACs/byte vs 43-54 available -> op set 0.72-0.91 of stream time. 2.1 real rows per expert group at S 16; 57 % of groups are singletons.
- Per-group: STREAM 6.8 us, NAX6-N 9.2 us, NAX6-NH 8.4-8.6 us at S 16 quiet.

## Drafter (DRAFTER_PLAN_1104)
- 300 needs ~14 kept at ~50 ms cycle, or 10.5 at ~35 ms. Realistic drafter 11.5-12.5 at 16 rows; 14 only at 32 rows. MTP chain step 1.35 ms (screened head); as drafter keeps 9.97@16, 12.55@32. MTP-BLOCK untested idea.
- R57_D7B=code 4-bit drafter 6.30 ms; R57_D17 (17th draft) free; R57_DA draft-ahead exact.

## Bytes (SWEEP / TEN_X_LEDGER / BYTES10) [P]
- 24 GB FAST today's stack (MSB-3 11-bit + POOLSPLIT + DRAFT17): 19.5 / 14.3 / 10.6 MB per kept (code/IFEval/multi); prefill 117.9 / 43.1 / 17.0 on top (!).
- TRL-2 FAST store (276 slots): 4.0 / 3.0 / 2.7; the 2.40 b mix 7.2 / 5.0 / 4.1. Quality est code PASS-ish (credit 1.19 %, strict 1.40 % vs line 1.34). s7_trl waits for Fit12's store (encoding tonight).
- EXACT at 24 GB has no path to 10 (OPT 117/94/84). Chat at 24 GB: gate threshold rule (QOM cf 0.5) ~88 MB PASS.
- 45 GB FAST: QOM cf 0 + t3 + gated bonus row 5.7-7.7 MB per kept (engine 15.0 measured U3C).

## Relaxed acceptance (RELAX_350) [K re-read]
- Code: R1(0.5 nat) +11 % kept at 1.19 % deviation; R2(top-2, 0.25 p1) +18 % at 1.9 %; R2(top-5,0.1) +30 % at 3.1 %. Comment-only R2: x1.06-1.08 real with code tokens exact.
- 350 on code only with full stack + 3 % rule; twin-level ~290-295; exact ~255-260 (floor engine + kept-13 drafter, night C 320).

## Prior art (RESULTS_FIRST)
- Kapitsa 61.3 / Splash 56.5 tok/s on same model at 2.4-bit, 16 GB expert store, M5 Pro 48 GB. Ours 24 GB exact 33.2; 12 GB exact 40.8 (likely a first); 45 GB exact 83.1 code, FAST 110.
