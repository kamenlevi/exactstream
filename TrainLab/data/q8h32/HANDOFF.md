# t30b Q8H32 engine artifact: hand-off to BigMoE's r57 path (MTP Head, Tue 17 Nov; orchestrator 11:0x)
- File: ~/datasets/deepdraft/q8h32/t30b_q8h32.safetensors (94.4 MB),
  sha256 b5388cc460006cccf0c77b5a502b67e6145af5d245a825706adbeeb1b459e9c4.
  - Built by TrainLab sb98/q8h32_artifact.py (94795c9); recipe and verification in t30b_q8h32.json (this folder).
- Source: dd_t30b.safetensors sha256 790c42db...d502 (the pinned CERTIFY98 candidate). t30b trains only the 10 Linears
  of mtp125.Trainable(head); everything else is the 125B's stored MTP head.
- CONTENT, each as mx.quantize(w.astype(bfloat16), group_size=64, bits=8), keys <module>.weight (uint32 packed) /
  .scales / .biases (QuantizedLinear layout):
  - P.q_proj, P.o_proj, P.k_proj, P.v_proj, P.se_gate, P.se_up, P.se_down, P.fc_hidden, P.fc_embedding, P.router
    (t30b's trained Linears; the router also at 8 bits);
  - head.attn_hc / head.mlp_hc / head.mixer .input_mix_weight_down / _up (the head's 6 bf16 HC mixes; head =
    lm.mtp, attn_hc = layers[0].attn_hyper_connection, mlp_hc = layers[0].mlp_hyper_connection, mixer =
    hyper_connection_mixer).
- NOT in the file, use as stored: se_gatew, the norms, the routed experts, the lm_head, top-k = 10.
- SCREEN at run time: Leap's ~/Leap/data/frspec_rank.npy (sha256 f7b9dc9b...)[:32768] + the session's own tokens
  (prompt + committed answer up to the anchor, first appearance, cap 4096); logits outside masked to -inf
  (MoonShots' screen_masks / Leap's ScreenedHead).
- VERIFIED: every module, dequantized back to its stored dtype, equals MoonShots' apply_cut(CUTS['Q8']) on the same
  model BITWISE (16 / 16). So the dequantized form IS the form M3 measured as Q8H32.
- EXPECTED KEPT (MoonShots q8h32early_scores.txt; TRUE-STATE, real best-first trees, vs the full head):
  - HE+: tree-16 x0.9775 [0.9595, 0.9939]; tree-32 x0.9751 [0.9568, 0.9906].
  - MBPP: tree-16 x0.9981 [0.9910, 1.0050]; tree-32 x1.0046 [0.9978, 1.0116].
  - The HE+ loss is the 32K screen. Q8 alone is x0.9947 / x0.9975.
- An engine using a real quantized matmul on the packed form is numerically close to the measured dequantized form,
  but not bitwise. EvalSuite's engine check (tokens per check within +-3 % of kept_walk) covers that.
- In-process alternative (no file): apply exactly the quantize call above to the same 16 modules at load.
  - Dequantize back for the measured form, or keep them packed for quantized matmuls.
  - This is p3_dd_tree.py's apply_cut with CUTS['Q8'] plus ScreenedHead(RANK[:32768]).
- ENGINE NOTE (MoonShots 20:12, measured on r57's live head): se_gatew is fp32 ("as stored in t30b"). Installed as a
  plain Linear, its fp32 output promotes the MoE output and the residual stream to fp32. The stored experts'
  bf16-scale gather_qmm then costs ~1.25 ms each: the MTP MoE takes 6.0 ms per pop instead of 0.55.
  - Cast se_gatew's output to the stream's dtype, as TrainLab mtp125's moe_train does (.astype(hyper.dtype)).
  - BigMoE has the fix. The artifacts themselves don't change.

# C2-QAT head pin (S22 qa; the orchestrator 19:4x: ONE A/B vs Q8H32 in Thursday night's EXACT timing, never lossy)
- File: ~/datasets/deepdraft/ckpt/dd_t30qa.safetensors (276.6 MB),
  sha256 353965c596dc363001913fc9ccf22bf1a72bd2f9d6ecbe562d7acd671d346ad7.
  - Made read-only (chmod a-w) so nothing overwrites it until Saturday; the same for its .json / _opt.
- LATENT fp32 weights of mtp125.Trainable(head)'s 10 Linears (t30b's keys). THE ENGINE APPLIES C2's CUT: MoonShots'
  p3_dd_tree.CUTS['C2'] =
  - 4-bit Linears (bf16 cast -> mx.quantize g64 -> dequantize), router 8-bit;
  - the head's 6 HC mixes at 4 bits;
  - routed experts top-k 4; the 8K screen (frspec_rank[:8192] + session tokens).
  - This is exactly the form it was trained under (sb98/dd_qat.py, STE) and scored in (sb98/m3_tree.py --cut C2).
- The same se_gatew cast applies.
- Measured kept (125B, M3 TRUE-STATE, vs the full t30b): HE+ t16 x0.9513 / t32 x0.8849; MBPP t16 x0.9393 / t32
  x0.9097 (REPORT 'S22 QATC2 RESULT', 0669fe3).
