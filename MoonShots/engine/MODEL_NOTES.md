# 125B (qwen4_exp, "Qwen3.8-Flash-Next") block math for the roofline engine (Phase 0 study, 2026-10-21 16:3x)
Sources: reference math HF = .venv/.../transformers/models/qwen4_exp/modeling_qwen4_exp.py; what runs MX = .venv/.../mtplx/
models/qwen4_exp.py (GDN core mlx_lm/models/gated_delta.py, MoE mlx_lm/models/qwen3_next.py:308-354); config
~/BigMoE/models/bare/config.json. All RMSNorms: x * rsqrt(mean(x^2) + 1e-6) * w in fp32; the bare pack already holds 1+w for
trunk norms (multiply by w); GDN linear_attn.norm is one-centred (use as is); MTP pre_fc_norm_* are zero-centred (use 1+w).

## Layer (no input/post layernorms, no final model.norm; residual H [T, 10240] = 4 streams x 2560, H0 = tile(embed, 4))
- HC read (HF:941-969, MX:1294-1307): N = grouped_rmsnorm(H, 2560) * w_hc[10240]; m = silu((N Wdown[320,10240]^T)/4);
  u = sigmoid(m Wup[10240,320]^T); x = mean_s(u_s * N_s) [2560]; inj = 2 sigmoid((N Winj[4,10240]^T)/4) [4].
- HC write (HF:1236-1243): H_s += inj_s * y (H = un-normed input). Order: layer 1 (0-based) only: H += PLE(H, ids);
  x, inj = HCread_attn(H); H = write(H, attn_or_gdn(x), inj); x, inj = HCread_mlp(H); H = write(H, moe(x), inj).
- Final: logits = lm_head(HCread_final(H)) (final mixer has no inject); lm_head untied 4-bit g64.

## GDN (layers with (i+1) % 4 != 0; HF:403-563, MX:754-984)
qkv = in_proj_qkv(x) [10240 = q 16x128 | k 16x128 | v 48x128, head-major]; z = in_proj_z(x) [48x128]; b = in_proj_b(x) [48];
a = in_proj_a(x) [48]. c = silu(causal depthwise conv4(qkv; W[10240,4,1]), tap 3 = current token); q = l2n(q)/sqrt(128),
k = l2n(k), l2n(x) = x * rsqrt(sum(x^2) + 1e-6) (SUM-based). beta = sigmoid(b); g = -exp(A_log) * softplus(a + dt_bias).
Per v-head h (k-head h // 3), fp32: S <- S e^g; d = beta (v - S^T k); S <- S + k d^T; o = S^T q.
y = rmsnorm(o_h; w[128]) * SIGMOID(z_h) (output_gate_type "sigmoid"); out = out_proj(y [6144]) -> 2560.
State: conv tail 3 x 10240 bf16; SSM fp32 [48,128,128] stored [Hv, Dv, Dk] in MX (GD:48).

## QSA (layers 3, 7, ..., 47 and the MTP layer; HF:611-839, MX:3279-5137)
Indexer (no per-head weights): qi[4,128] | ki[128] = index_qk_proj(x) (8-bit, 640 out); qi = rope(rmsnorm(qi)); raw keys
cached unnormed; block b = tokens 4b..4b+3: pb = rope(k_layernorm(bf16(mean_f32(raw[4b:4b+4]))), pos 4b);
score[t,b] = sum_h relu(qi_h . pb)/sqrt(128) for b < (t+1)//4; sel = top-512 blocks (ties to lower id) U tail block.
T <= 2051: selection = full causal (MX skips it). Attention: qg = q_proj(x).view(24, 512): q = qg[:, :256], gate =
qg[:, 256:] (interleaved per head); q = rope(q_norm(q)); k = rope(k_norm(k_proj(x).view(2,256))); v = v_proj(x).view(2,256);
o = softmax(q k^T / 16 + mask) v (GQA 12:1); out = o_proj(o * sigmoid(gate)). RoPE: theta 1e7, first 64 dims (of 256 attn,
of 128 indexer), NeoX rotate-half, pairs (i, i+32); mrope sections are no-ops for text. KV cache bf16 [1,2,T,256] (post
norm+rope), raw index keys [T,128], pooled block keys [T/4,128].

## MoE (every layer; HF:842-938)
p = softmax_f32(router(x)) over 512 (router 8-bit g64 [512, 640] u32); idx = top10(p); w = p[idx]/sum (norm_topk_prob);
y = sum_k w_k down_e(silu(gate_e x) * up_e x) (2560 -> 640 -> 2560); y += sigmoid(shared_expert_gate(x)) *
down_s(silu(gate_s x) * up_s x) (shared 640 wide, 4-bit; shared_expert_gate 4-bit [1, 320] u32).
Experts 4-bit g64: gate/up [512,640,320] u32 + [512,640,40] scales/biases; down [512,2560,80] u32 + [512,2560,10].

## Embedding, PLE (0-based layer 1 only), MTP
- embed_tokens 4-bit g64 [248320, 2560], no scaling. N-gram ids (HF:1018-1114): history [prev2 | ids] (start EOS 248044 x2);
  bigram h = (t0 m0) ^ (t1 m1), trigram ^ (t2 m2), m = [23703573157769, 20109073645365, 8052911324071]; 8 heads per order,
  row j = h mod P_j + off_j (16 primes 20,000,003..20,000,171; 320,001,536 rows padded); gather 16 rows x 160 = 2560 dims.
  Table: ngram-table.safetensors U32 [320001536, 20] + bf16 scales/biases [.., 5], 4-bit g32 (32 GB).
- PLE (HF:1117-1189): K = grouped_rmsnorm(key_proj(emb)).view(4,2560); V = value_proj(emb); Q = grouped_rmsnorm(H).view(4,2560);
  g_s = <K_s, Q_s>/sqrt(2560); g_s = sign(g_s) sqrt(max(|g_s|, 1e-6)); G_s = sigmoid(g_s) V; H += G + silu(dwconv(
  grouped_rmsnorm(G); W[10240,4,1], dilation 3)) (taps t-9, t-6, t-3, t). State: 9 conv rows + last 2 token ids.
- MTP (mtp.safetensors, MX:6866-6983, wiring reconstructed by mtplx): hn = rmsnorm_full10240(H_t) (1+w_hidden);
  e = fc_embedding(rmsnorm(embed(tok_{t+1})) (1+w_emb)); H0_s = fc_hidden(hn_s) + e; H' = one QSA+MoE layer (own HC, cache);
  draft = head(mtp.hyper_connection_mixer(H')); deeper drafts feed H' back (depth <= 3). FR-Spec draft head (BIGMOE_FRSPEC,
  ~/Scout/patches/frspec4.py): rows of the 4-bit lm_head for 65,536 ranked ids (+ prompt ids), scattered into -1e30.
- Dense pack: shards 00016 (+00017) hold dense + embed + head; 00001-00015 experts. bf16: HC mixers, inject, PLE key/value
  proj, norms, conv weights, A_log, dt_bias, MTP fc_*. 8-bit: router, indexer. Everything else 4-bit g64.
- Gotchas: no layer norms (hc_norm is the pre-norm); sigmoid gates in the GDN gated norm and the per-head attention gate;
  fused vs standalone bf16 sigmoid differ in MLX (mtplx/attention_math.py) - irrelevant for the non-exact engine.
