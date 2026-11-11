# Gate handoff fix: seq stamps on every handoff word (MoonShots, Sun 25 Oct 2026)

## The bug (BigMoE-family gate: v3, v3f, v3g and forks)
The routing row (GPU writes, loader reads) and the release regions / residency tables (loader writes, GPU reads) are
128-byte lines with the seq tag only in word 0. Under another process's GPU load the loader sometimes saw a fresh tag next
to a stale 64-byte sector: it read the previous post's ids from words 16+ of a routing-row line.
- Measured on v3g, B12LL MTP decode, 6 runs under 50 % GPU load: 7 torn calls, at positions 45-52+ and 76-79 (always the
  second sector of a line). Quiet runs: 0.
- When the torn half holds ranks the loader must read, it loads and publishes the wrong experts. The MoE then computes with
  another expert's bytes: silent, intermittent wrong outputs whenever something else uses the GPU.
- Evidence: ~/MoonShots/data/b8mr/handoff_*_cs_gpu*.json, csum.txt, csum_try2.txt.

## The fix
Bits 22-31 of every handoff word carry seq & 0x3FF. Payloads fit in bits 0-21: expert ids < 2^20 plus flag bits 20-21,
slots, positions; -1 travels as 0x3FFFFF.
- **post_ids kernel:** stamps every id and pred word. It checks each residency-table entry's stamp; a stale entry counts as
  unknown, so must-ids fall back to the slow path and partial hits read -1.
- **gated.py _copy (event mode):** stamps too.
- **loader_main:** re-reads the row until every id word carries the post's stamp, then strips it. A 250 ms guard prints an
  error. Counters: stamp_calls / stamp_rereads / stamp_wait_s in ext.stats().
- **joint_side (v3g):** declines to pair on any stale word (stamp_joint); the row's own service handles it.
- **Pred reads:** stamp_dec. A stale pred only misdirects prefetch.
- **write_region:** stamps every word. The plain spin-mode branch now calls write_region (same layout).
- **Residency tables (both publications):** stamped with their tag's seq.
- **Spin kernel:** a pair whose line tags match counts only if every word it reads is stamped; otherwise it polls the next
  pair (gate.status[17] counts these). The spin decodes as it checks.
- **Wrap:** 10 bits = 1024 gate passes. A stale word comes from an earlier post / publication of the same row, region or
  table a few passes back (MTP draft steps advance the pass too), far below 1024.

## Files
- v3g_ext.diff, v3g_gated.diff: ~/Breakthrough/v3g -> ~/MoonShots/defer/v3h (built: defer/v3h/ext/build).
- v3d_ext.diff, v3d_gated.diff: the same patch on my v3f-based fork (~/MoonShots/defer/v3d -> v3d2). This shows how it maps
  onto BigMoE's v3 / v3f code (no joint_side there).

## Verification (Sun 05:49-05:55; data/b8mr/v3h.txt)
- Handoff test, 6 runs under 50 % GPU load (B12LL MTP decode, code + prose x 3, MR_CSUM + MR_HANDOFF): 0 torn calls
  (before: 7 in 6). The stamps caught 7 tears and waited them out, at most 0.01 ms per run. Every loaded run's output is
  bitwise equal to the quiet run (before: 4 of 6 differed).
- Quiet bitwise:
  - B12LL v3h == v3g, code 7c35e4a319, prose 70bbe4feac, both interleaved rounds;
  - v3 exact v3d2 == v3d, code 127b6932ba, prose d4ef51baea.
- Speed, quiet, interleaved:
  - B12LL loop ms/pass: code 154.1 / 154.3 (v3g) vs 154.5 / 154.3 (v3h); prose 154.2 / 153.9 vs 153.9 / 153.9.
  - v3 exact prose verify 53.8 vs 53.9 ms/pass (same pass structure).
  - Stamp waits when quiet: 0. Cost <= 0.2 %, inside noise; kill line was > 1 %.

## Test commands (from ~/MoonShots, holding a quiet lock)
- `nohup sh b8mr/run_v3h.sh &` runs everything above: builds both exts, quiet bitwise + speed, handoff under load.
- Handoff test alone on an engine copy at path P (bt_b12's B12_ENGINE is relative to ~/Breakthrough):
  `B12_ENGINE=P MR_FRAC=0.10 B12_LATE3=2,10 MR_CSUM=1 MR_HANDOFF=1 MR_ARM=wait MR_TAG=_x python b8mr/run.py gen code 192`,
  with b8mr/gpuload.py 900 0.5 running for the loaded case. It prints [handoff] (torn calls) and [stamps] (catches).
