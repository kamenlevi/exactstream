# REF2 hand-off: the new exact reference for the 45 GB line (Leap -> EvalSuite, Wed 2026-10-28 16:29)

Kamen approved decision 4 (Tue 22:40): today's exact reference may be replaced by REF2 once the Eval suite shows both
score the same on math, code and instructions. This note is EvalSuite's part: how to decode with REF2.

## Update Wed 16:52: the full model runs streamed (EvalSuite's catch)
- On BigMoE's streamed engine (load_streamed; separate gate / up / down pools), leave MTPLX_FUSED_GATE_UP unset. ref2
  runs the experts as two qmmg calls on the pools (experts_split: equal to the fused kernel bit for bit), patched into
  the base SwitchGLU.__call__ that BigMoE's streamed subclass calls with slot ids. BIGMOE_TIER is not supported.
- Proven in EvalSuite's exact REF harness (evalsuite.model.load('ref', C = 192)): all 145 records x 48 layers and all
  16 logits rows bit-identical between ref2.ar_step tokens and one 16-row verify (scripts/ref2_stream_check.py).
- Counters: `ref2.counts()` / `ref2.reset_counts()` (qmm16c, mlx16, linear, hc_read, hc_eager16, attention, experts).
  Per AR token expect experts 48, attention 12, hc_read 96, hc_eager16 1. Call reset_counts() after self_test().

## How to apply (greedy AR on the standard sets)
```python
import os, sys
# resident model only: os.environ["MTPLX_FUSED_GATE_UP"] = "1" before load; streamed (BigMoE): leave it unset
sys.path.insert(0, os.path.expanduser("~/Leap/scripts"))
model = ...                                        # load the 125B the usual MTPLX way (45 GB / resident line)
import ref2
ref2.apply()
assert all(ref2.self_test().values())              # batch invariance of the matmul kernels on random data
new_ids = ref2.generate(model, prompt_ids, max_new, stop=eos_ids)
```
- `ref2.generate` = plain prefill, then each new token as a 1-row step through the engine's verify scopes with commit
  keep 1 (`ref2.ar_step`). That is exactly the AR that INV1 / R-40 proved bit-identical to the 16-row spec checks.
- Compare against today's reference: same harness, without `ref2.apply()` (and with MTPLX_FUSED_GATE_UP unset for
  the shipped default).

## What it is
- Every matmul / HC read / attention call with <= 16 rows runs at 16 padded rows through fixed kernels:
  - qmm16c for the big 4-bit dense shapes;
  - MLX's own 16-row call for the rest;
  - the S4 HC read;
  - experts_nax for the routed experts;
  - QSA attention padded to 16 rows with a cache trim.
- Each row's bits are therefore the same alone or inside a 16-row check, so spec decoding against REF2's own AR output
  is exact by construction.

## Evidence (CA's resident 12-layer slice of the 125B; TABLE_350 'R-40')
- INV1 / R-40 Part 1: 16 one-token AR steps vs one 16-row verify. The logits of all 16 rows and every layer record
  (GDN, PLE, attention, MoE, layer outputs) are bit-identical.
- R-40 Part 2: per 12-layer check, today 39.60 -> REF2 + SB 26.78 ms (1K), 41.88 -> 27.36 (8K). x4 = -51 / -60 ms per
  48-layer check (SB is the engine-loop scheduling lever, bit-identical).

## What to expect
- Bits differ from today's reference by rounding: qmm16c / experts_nax dequantize exactly in fp32, more accurately
  than MLX's paths (L1-23, L1-25). Scores should match within noise; that is the verdict asked of you.
- REF2's AR is slower than today's AR: every token runs attention and the matmuls at 16 padded rows. The 16-row spec
  checks are the fast path.

## Conditions
- MLX 0.32.2 on the M5 (gen 17, non-'d'); the 125B with QSA attention and GDN as MTPLX ships it.
- `ref2.self_test()` must pass before use. After any MLX upgrade, re-run scripts/r40_screen.py (bitwise part).
- The prefill (rows > 16) runs today's kernels in both AR and spec. A spec loop must prefill with the same function
  (`ref2.prefill`) to stay bit-identical to `ref2.generate`.
- Code: ~/Leap/scripts/ref2.py, leapk.py (S4), qmm16.py (qmm16c), qmmg25.py + gmr24.py (experts_nax). Tests:
  scripts/inv1_screen.py, r40_screen.py.
