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
