# S22 ENGINE: the r57 per-level loop's head in S22's format (MoonShots, Tue 17 Nov 2026; QUEUE M-47)

## What it is
- S22 is MTP Head's QAT head trained under path A's cut C2 (TrainLab REPORT "WEEK98 S22 QATC2"; the trainer reads the
  bits from p3_dd_tree.CUTS['C2'] through apply_cut's own mapping).
- The engine form is the same cut as REAL packed modules (`week98/draft/s22_format.py`, FORMATS['c2']):
  - P's 9 Linears (q / o / k / v, se_gate / se_up / se_down, fc_hidden / fc_embedding): 4-bit g64;
  - the router: 8-bit g64;
  - the head's 6 HC mixes (attn_hc / mlp_hc / mixer, down + up): 4-bit g64;
  - top-4 routed experts (head.layers[0].mlp.top_k = 4);
  - Leap's ScreenedHead over FR-Spec's top 8,192 (or 32,768) + the session extras (cap 4,096).
- As stored: se_gatew, the norms, the routed experts (4-bit g32), the lm_head rows.
- Each module is `mx.quantize(w.astype(bfloat16), group_size=64, bits)` in QuantizedLinear layout. That is p3's `fq`
  quantization exactly, so each packed module dequantizes to M3's fake-quant weight bit for bit (checked at load: 16 / 16).
- Bytes per level: 74.7 MB at the 8K screen (dense 50.6 + 4 experts 12.3 + screened head 11.8 with the window's extras).
  Q8H32 is ~163-178 MB.

## For BigMoE's r57 loop (the drop-in, same boundary as the Q8H32 format)
```python
sys.path.insert(0, os.path.expanduser("~/MoonShots/week98/draft"))   # read-only import, pinned commit
import s22_format as SF22
P.load_weights(R57_T30B_WEIGHTS or dd_t30b.safetensors)              # S22: its dd checkpoint, when TrainLab lands it
done, orig = SF22.apply_real(mx, nn, P, head, "c2")                  # replaces the Q8 loop; sets top_k = 4
ok, n, bad = SF22.check_vs_fakequant(mx, P, head, orig, P3.fq)       # optional load-time gate: ok == n == 16
sh = ScreenedHead(lm.lm_head, RANK[:SF22.FORMATS["c2"]["screen"]], extra_cap=4096)   # then tree_sf as today
```
- Flags, matching the 10:5x set: `R57_T30B_FMT=c2` plus a weights path.
- Everything after the load (tree_sf + GPUFrontier + topk_gpu, tp on the GPU) is unchanged.

## Bit test (pre-registered M-47; `s22_bittest.sh`, results in `s22_bit_*.json`)
1. FORMAT: 16 / 16 packed modules == M3's fake quant, bitwise.
2. LOOP: SF == HS bitwise on every node of the 120 real trees (HE+ t16 48 + t32 24, MBPP t16 48), at the 8K screen.
3. The 32K screen: HE+ t16, 48 trees.

On S22's checkpoint, re-run in ~2 min:
```
W=<s22 dd checkpoint> sh s22_bittest.sh he16
```
