# One in-place records3 conversion: native layout + 11-bit packed scales (plan, for Kamen's single decision)

BigMoE, Tue 3 Nov 2026 17:2x, on the orchestrator's ask ("plan ONE in-place records3 conversion that does both"). Nothing is converted; this is the plan and its price. Inputs:
- Leap's native layout: NAX6-N +17.3 % on the expert call (NAX6-NH +23.3 %), bitwise == NAX6. Leap's `native_3c` runs REF2-T's 3c kernels on native records, == 3c bitwise, 1-7 % slower. The tool is `~/Leap/scripts/convert_records_native.py`: per layer temp + verify + atomic rename, index.json "layout", `--inverse`. See TABLE_350 DISK-PLAN / NATIVE-3C.
- MoonShots' M-27 (`~/MoonShots/data/lossless45.txt`): s3 / b3 / b at a fixed 11 bits are lossless; all 221,184 pieces round-trip.
  - Parts shrink: MSB 2,162,688 → 2,064,384 B (-4.5 %), LSB 770,048 → 720,896 B (-6.4 %).
  - Held packed in RAM (11-bit dequant inside the kernels), the same pool RAM holds ~4.6 % more parts. With joint-fit sizes, code SSD MB per kept token drops -17.6 % at C 256 and -24.5 % at C 320; implied C50 steady 63.6 → ~66.9 tok/s.

## 1. What changes on disk
- The files: records3/L00..L47.bin. Each is 512 records × 2,932,736 B = 1.50 GB; 70.4 GB in all.
- Each record holds an MSB-3 part (9 pieces: w3 614,400 + s3 51,200 + b3 51,200 per projection) and an LSB-1 part (6 pieces: l1 204,800 + b 51,200 per projection). Both are 16 KB aligned.
- After the conversion:
  - every piece is in Leap's native order: native[((t*KG + g)*32 + r)*per + w] = rowmajor[((t*32 + r)*KG + g)*per + w]. gate / up have O 640, KG 40; down has O 2560, KG 10;
  - every s3 / b3 / b piece is packed to 11 bits (25,600 values → 35,200 B);
  - the parts are re-aligned to 16 KB: MSB 2,064,384, LSB 720,896, record 2,785,280 B;
  - index.json gets per layer `"layout": {"order": "native", "scales": "p11"}` plus the new recsz / msb_sz / lsb_sz / piece offsets and sizes.
- records3m (Breakthrough's 12 GB line) and every other store stay as they are.

## 2. Disk plan: in place, never near the 110 GB stop line
- Per layer:
  1. read L.bin (1.50 GB);
  2. write the converted L.tmp (1.43 GB);
  3. verify: inverse-transform L.tmp in memory and compare bitwise with L.bin, record by record;
  4. atomic rename over L.bin;
  5. update that layer's index.json entry (atomic rewrite).
- Peak extra disk is one layer's temp, ~1.4 GB. Free today: 169 GB. The store shrinks by ~3.5 GB overall.
- Reversibility: md5 of all 48 original files is taken before the run (~1-2 min) and kept in data/records3_md5_rowmajor.txt. `--inverse` restores row-major + bf16 scales, and the md5s must then match.
- Time: ~2-3 min (Leap's tool on the native part: ~2 min for 48 layers) plus the md5 pass.
- No engine process may hold records3 open during the run: no runner, no EvalSuite chunk, no Breakthrough map (lsof check first).
- A failure mid-run leaves a mixed store, which is safe: index.json is per layer, the guards read it per layer, and a re-run resumes from the first unconverted layer.

## 3. The engine (BigMoE) side, all behind the index's per-layer layout. ~5 h CPU + tests
1. **Pools** (bigmoe/tier.py): piece shapes come from index.json's piece sizes, so p11 layers allocate packed s3 / b3 / b pieces.
   - The ext needs nothing: pieces carry their own `per`; demand / prefetch / JOINT loads copy part bytes straight into slots, and a native file gives native pools. The N7-15 dirty flags stay 0.
   - BIGMOE_VTIER copies whole slots, so it is layout-agnostic.
2. **read_rebuilt** (ext; the prefill's records → 4-bit MLX rows on the CPU for the stock MLX switch): the inverse native permutation per piece and the 11-bit unpack to bf16 before today's rebuild. That makes the prefill scratch identical to today's bit for bit. (_prefill_grouped's R-52 group buffers use the same routine.)
3. **LP1**: a kernel-level test inside tierglu sources, so it carries over to Leap's tierglu_3cn / NAX6-N (Leap: re-derived from tierglu's current sources). No action.
4. **Guards** at load: a layer whose layout is native / p11 refuses every row-major or bf16 reader:
   - stock 3c without native_3c;
   - TIER_GLU 3b;
   - NAX6 without nax_native;
   - tier_nax4;
   - read_rebuilt without the inverse.
   And the reverse: native_3c / nax_native on a row-major layer refuses. The assert is Leap's: (BIGMOE_NAX_NATIVE == 1) == (CFG nax_native and tier_nax_ooo), plus TIER_GLU 3c. A mismatch gives wrong outputs, not a crash, so this guard is mandatory.
5. **Direct readers**: ~238 lines in `~/Leap/data/records3_readers.txt` (old runners, replays, converters) read records3 without the index. bigmoe/tier.load_index raises on a native / p11 layer unless the caller passes `layout_ok=True`. The scripts still in use (r57_screen, c45_runner, EvalSuite's model.load, Breakthrough's map, MoonShots' replays) get checked one by one; the rest are marked row-major only.
6. **Tests before any timing** (CPU, on a converted copy of 2-3 layers in a scratch store, never on records3):
   - the ext's slot bytes for a set of experts == Leap's repack + pack of the row-major record;
   - read_rebuilt's 4-bit rows == today's, bitwise;
   - the guards refuse each mismatched config.

## 4. The exact A/B line (pre-registered with the plan)
- **Before:** the default (R57_PRESET=45 at C 256, M-17 sizes) on records3 row-major. Tokens on chat23's 30 prompts + S45's six, plus SSD MB per kept token and steady tok/s; three passes A-B-A under a quiet lock.
- **After:** the same default with native_3c + Leap's p11 dequant on (both bitwise by Leap's self-tests), on the converted store, same C.
  - The pool RAM is the same, so ~4.6 % more slots: the joint-fit sizes for the packed part sizes (MoonShots' M-27 machinery) as a second arm.
- **PASS:** every token equal to the "before" run on all 36 prompts, AND code SSD MB per kept token <= -10 % at C 256 (the joint-fit arm; M-27 predicts -17.6 %), AND steady tok/s not down.
- **KILL:** any token differs: run `--inverse` and check the md5s. Or MB per kept token > -3 %.
- **BETWEEN** otherwise.
- The NAX preset (FAST: NAX6-N / NAX6-NH on native pools, no repack) is measured beside it. It is not part of the exact line.

## 5. Order of work once Kamen says yes
1. Leap: the p11 dequant inside tierglu_3cn / NAX6-N (bitwise, default off) and the combined tool (native + p11 + inverse + verify).
2. BigMoE: §3 1-5 and the §3 6 tests on a scratch copy.
3. In a quiet window with nothing holding records3: md5 → convert → verify → the engine's residency check + a short REF2-T decode against the "before" tokens.
4. The §4 A/B at night. If KILL: `--inverse` + md5.
