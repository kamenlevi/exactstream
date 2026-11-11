# TRL full store: runbook (Leap, Wed 2026-11-04 21:2x; TABLE_350 'TRELLIS FULL-STORE PREP')

**What:** Fit12's TRL codes (the 2.40 b mix: gate / up k 2, down k 3; or TRL-3: k 3 everywhere) for all 24,576
experts, as a trl-v1 store that the TRL-NAX kernel serves.

> (a quote from the project owner, kept private)

**Status (Wed 23:3x): NOT BUILT.**
- Screen 7 KILLed the 2.40 b mix (NewIdeas S7-TRL: in-frame code / IFEval / multi flips 1.45 / 9.54 / 10.44 %).
- The trellis family stopped at 22:4x (NewIdeas TRL-JUDGE-MAP).
- This runbook stays for a reopened family (e.g. a down-only TRL-3 arm).

## 0. Status of the parts (all verified Wed 20:1x-21:2x)
| part | file | check |
|---|---|---|
| Viterbi on the GPU | scripts/trl_metal.py | == Fit12's C bit for bit (k 2 / 3, ties included) |
| encoder (batched) | scripts/trl_encode.py | layer 18: 1,536 / 1,536 views == Fit12's store; L30 (H_x computed) 48 / 48; TRL-3 == Fit12's C path 6 / 6 |
| converted-store reader | trl_encode.read_record | 8 / 8 experts of BigMoE's native+p11+pad copy == fit_q.read |
| store format trl-v1 | scripts/trl_store.py | real codes -> store -> pools -> kernel == dense W = Q R^T within 6e-7 |
| --store writer | trl_encode.py --store | records == the run's arrays, 4 / 4 |
| serving kernel | tierglu_trl.tier_glu_trl_nax | ~5e-4 vs float64; 0.87x NAX6-N per S 16 call (quiet) |

## 1. Disk (the blocker)
- **Free:** 124 GB (df 21:2x). Thursday's records3 conversion needs >= 112 GB free at ~10:55.
- **Needs:**
  - mix store 37.5 GB (TRL-3: 47.5 GB);
  - temporary capture 8.05 GB (R0..R7 x 48 layers), deleted after the encode;
  - per process, one layer's x (fp64 + fp32, ~1 GB RAM, not disk).
- **Before the conversion:** free space must stay >= 112 + the store. So ~26 GB (mix) or ~36 GB (TRL-3) must be freed
  first, + 8 GB while the capture exists.
- **After the conversion:** the converted records3 is ~3.5 GB smaller, but the 112 GB floor stays for everything else.
- **Candidates, the owners' calls:**
  - Fit12's 4-layer 8-bit views ~/Fit12/data/trl, 8.3 GB (after screen 7);
  - BigMoE's scratch layout copies (r3p11 / r3p11pad 4.0 GB each, r3nat 1.4 GB) after the conversion;
  - stale stores and logs per session.

## 2. Capture (Fit12's tool, Fit12 runs it; ~2-5 min under a quiet lock)
- Fit12's b10r.sh capture step with F12_CAP_LAYERS = 0..47 and F12_CAP_TEXTS = R0..R7, into a new dir (e.g.
  ~/Fit12/data/cap48). It keeps b10r.sh's guards: SSD <= 66 C to start, stop at 73 C, swap +2 GB, keepalive.
  - fit_b12.py's engine reads Fit12's own v3k store, not records3, so the conversion does not affect it.
- Privacy: the "agent" domain (1 of 7) is Kamen's own Claude Code sessions. It matters only if the store is ever
  published (his call).

## 3. Encode (~1.1-1.5 h: 4 processes x 4 BLAS threads, GPU light)
    D=~/Leap/data/trl_store_mix; CAP=<capture dir>
    # process i in 0..3 takes layers i, i+4, ..., i+44 (12 each):
    for L in $LAYERS; do env VECLIB_MAXIMUM_THREADS=4 OMP_NUM_THREADS=4 B3_THREADS=4 \
      ~/BigMoE/.venv/bin/python -B ~/Leap/scripts/trl_encode.py $L 0-511 --batch 16 --cap $CAP --store $D; done
- **Rate:** 0.64 s per expert per process (5.5 min per layer) -> 12 layers ~66 min, if 16 cores are free (a night slot).
- **Before the conversion:** fit_q.read is used. **After it:** ext.record_rowmajor (automatic, per layer).
- **Writes:**
  - each layer goes to L{L}.trl.part, a sparse file filled in place;
  - when experts 0-511 are all in, it is renamed (os.replace) to L{L}.trl and its entry merged into index.json;
  - an interrupted layer restarts from expert 0. Wrapper TODO: skip layers already in index.json.
- **TRL-3:** add `--form trl3` and use another dir.
- **Guards to add in the wrapper:**
  - disk >= 112 GB + the remaining layers' size before each layer;
  - swap growth < 2 GB;
  - PAUSE / HOLD checks between layers;
  - a CPU share lock per process.

## 4. Checks after the encode
- Per layer, md5 of L{L}.trl into index.json.
- Spot-check 2 experts per layer against Fit12's per-expert pipeline with the C Viterbi (trl_l18_check.py with
  TRL_VS_C=1 generalised to any layer; ~12 s per expert of CPU, ~20 min for 96).
- Kernel smoke per layer: tierglu_trl.tier_glu_trl_nax on load_pools(D, L, a few experts) vs ref_forward (fp32,
  ~5e-4).

## 5. Serving (engine wiring: BigMoE's engine, not built)
- **Pools per slot:** wg / wu [C, 40, 160, 17], wd [C, 160, 40, 25] u32, bf16 scales, down signs. The record holds
  everything a slot needs; the x signs are per layer (seed L).
- **Kernel:** tier_glu_trl_nax(x, P, slot) with slot = pool slots per pair (as NAX6-N's msb_slot). The x' / h'
  rotations are inside; the combine is unchanged.
- **BigMoE's side:**
  - a TRL tier: the loader reads whole trl-v1 records into slots, like records3 parts;
  - a CFG switch (default OFF) in bigmoe.tier / ref2 routes a TRL layer's calls to the kernel;
  - QOM / OOO phases as today, with TRL slots.
- **Rollback:** the store is its own directory; records3 is untouched; deleting the directory reverts.
