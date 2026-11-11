"""PORT (MTP Head, Wed 11-04 20:1x; the orchestrator 20:0x): the DFlash2 drafter in plain PyTorch, for the Kaggle
2 x T4 trainer (drafters <= ~1B). Numerically the MLX trainer's own forward and loss (sticky/d09_train.py fwd / heads /
loss_fn), not the generation path:
- frozen and precomputed: fc + hidden_norm (the windows hold h = hidden_norm(fc(taps)) as bf16 bits);
- frozen: the 125B's 4-bit affine (group 64) input embedding and LM head (model-00016-of-00017.safetensors). They are kept
  packed. The head mirrors MLX's GPU QuantizedLinear on bf16 input: W = s * q + b exactly, logits = bf16(bf16(hid) @ W^T)
  with fp32 accumulation, and in backward d_hid = bf16(bf16(d_logits) @ W) (vs float64 truth: 99.98 % bitwise; MLX
  0.32.2's CPU kernel is far off, 4 %). The embedding is bf16(s * q + b), one rounding (see deq_rows_bf16);
- trained (fp32 master weights): the layers, the final norm, the candidate selector.
Module and parameter names equal the MLX ones, so checkpoints map 1:1 (the HF DFlash2 file, MLX-trained safetensors).
Attention: "explicit" = the MLX trainer's grouped layout (context rows shared per group, exact same math); "sdpa" =
F.scaled_dot_product_attention over [context; all blocks] with a block-diagonal mask (the same softmax; memory-efficient
kernels on CUDA). Head + loss run in row chunks under torch.utils.checkpoint (logits are recomputed in backward), so the
step-size logits (rows x 248320) never exist at once."""
import json, math, os, struct
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

DRAFT_DIR = os.path.expanduser(os.environ.get("PORT_DRAFT_DIR", "~/.cache/huggingface/hub/models--incoai--Qwen3.8-Flash-"
                                              "Next-DFlash2/snapshots/a22a1744e85c17364dcc54a6a420af7135d4c7f6"))
# the 125B's 4-bit embedding + LM head: the model shard, or the 6-tensor file port/kaggle/make_upload.py writes
TARGET_SHARD = os.path.expanduser(os.environ.get("PORT_TARGET", "~/BigMoE/models/bare/model-00016-of-00017.safetensors"))
SLIDE = 2048                     # the drafter's sliding window (config sliding_window)


# ---------------------------------------------------------------- safetensors without a framework (uint32 / bf16 aware)
_DT = {"BF16": np.uint16, "F16": np.float16, "F32": np.float32, "U32": np.uint32, "I32": np.int32, "U8": np.uint8,
       "I64": np.int64}


def st_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
    h.pop("__metadata__", None)
    return h, 8 + n


def st_load(path, keys=None):
    """{name: (numpy array, dtype tag)}; BF16 comes back as uint16 bits (exact)"""
    h, off = st_header(path)
    out = {}
    mm = np.memmap(path, dtype=np.uint8, mode="r")
    for k, m in h.items():
        if keys is not None and k not in keys:
            continue
        a, b = m["data_offsets"]
        out[k] = (np.frombuffer(mm[off + a:off + b], dtype=_DT[m["dtype"]]).reshape(m["shape"]), m["dtype"])
    return out


def st_save_raw(path, tensors):
    """{name: (numpy array, dtype tag)} -> safetensors, tags kept (U32 packed weights, BF16 as uint16 bits)"""
    hdr, off, blobs = {}, 0, []
    for k, (a, tag) in tensors.items():
        b = np.ascontiguousarray(a).tobytes()
        hdr[k] = dict(dtype=tag, shape=list(a.shape), data_offsets=[off, off + len(b)])
        off += len(b); blobs.append(b)
    h = json.dumps(hdr).encode()
    h += b" " * (-len(h) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(h))); f.write(h)
        for b in blobs:
            f.write(b)


def bf16_bits_to_f32(u16):
    return (np.asarray(u16).astype(np.uint32) << 16).view(np.float32)


def to_torch(a, tag):
    if tag == "BF16":
        return torch.from_numpy(bf16_bits_to_f32(a).copy())
    return torch.from_numpy(np.array(a))


def round_bf16(x):
    return x.to(torch.bfloat16).to(torch.float32)


# ---------------------------------------------------------------- the 125B's frozen 4-bit matrices
class Quant4(nn.Module):
    """MLX affine 4-bit, group 64: packed (R, C/8) uint32 (value i at bits 4i..4i+3), scales / biases (R, C/64) bf16.
    Buffers only (frozen)."""
    def __init__(self, packed_u32, scales_bf16_bits, biases_bf16_bits, group=64):
        super().__init__()
        self.register_buffer("packed", torch.from_numpy(np.ascontiguousarray(packed_u32).view(np.int32)), persistent=False)
        self.register_buffer("scales", torch.from_numpy(bf16_bits_to_f32(scales_bf16_bits)), persistent=False)
        self.register_buffer("biases", torch.from_numpy(bf16_bits_to_f32(biases_bf16_bits)), persistent=False)
        self.group = group
        self.rows_n, self.cols = self.packed.shape[0], self.packed.shape[1] * 8
        self.cache = None                                # optional full dequantized copy (GPU: fp16)
        self.q8 = None                                   # optional unpacked 4-bit values as uint8 (CPU head: 636 MB)

    def unpack_u8(self, chunk=16384):
        q8 = torch.empty((self.rows_n, self.cols), dtype=torch.uint8, device=self.packed.device)
        sh = torch.arange(0, 32, 4, dtype=torch.int32, device=self.packed.device)
        for r0 in range(0, self.rows_n, chunk):
            p = self.packed[r0:r0 + chunk]
            q8[r0:r0 + chunk] = ((p.unsqueeze(-1) >> sh) & 0xF).reshape(p.shape[0], self.cols).to(torch.uint8)
        self.q8 = q8

    def deq(self, idx=None, r0=None, r1=None):
        """fp32 dequantized rows (exact s * q + b): by index tensor, or the range r0:r1"""
        s = self.scales[idx] if idx is not None else self.scales[r0:r1]
        b = self.biases[idx] if idx is not None else self.biases[r0:r1]
        if self.q8 is not None:
            q = self.q8[idx] if idx is not None else self.q8[r0:r1]
        else:
            p = self.packed[idx] if idx is not None else self.packed[r0:r1]
            sh = torch.arange(0, 32, 4, dtype=torch.int32, device=p.device)
            q = ((p.unsqueeze(-1) >> sh) & 0xF).reshape(*p.shape[:-1], self.cols)
        q = q.reshape(*q.shape[:-1], self.cols // self.group, self.group)
        return (q.to(torch.float32) * s.unsqueeze(-1) + b.unsqueeze(-1)).reshape(*q.shape[:-2], self.cols)

    def deq_rows_bf16(self, idx, rounding="once"):
        """rows as MLX's QuantizedEmbedding returns them (bf16). rounding="once": bf16(s * q + b), one rounding (an fma
        in float, as the GPU kernel evidently does: L1-20's GPU step-1 loss 5.344 vs the port's 5.3434 with this, 5.3474
        with "twice"); "twice": bf16(bf16(s * q) + b), MLX 0.32.2's CPU arithmetic (bitwise vs MLX-CPU, test_mlx_units)"""
        p, s, b = self.packed[idx], self.scales[idx], self.biases[idx]
        sh = torch.arange(0, 32, 4, dtype=torch.int32, device=p.device)
        q = ((p.unsqueeze(-1) >> sh) & 0xF).reshape(*p.shape[:-1], self.cols // self.group, self.group).to(torch.float32)
        sq = q * s.unsqueeze(-1)
        out = round_bf16(round_bf16(sq) + b.unsqueeze(-1)) if rounding == "twice" else round_bf16(sq + b.unsqueeze(-1))
        return out.reshape(*p.shape[:-1], self.cols)

    def build_cache(self, dtype, chunk=16384):
        out = torch.empty((self.rows_n, self.cols), dtype=dtype, device=self.packed.device)
        for r0 in range(0, self.rows_n, chunk):
            out[r0:r0 + chunk] = self.deq(r0=r0, r1=r0 + chunk).to(dtype)
        self.cache = out


class _Bf16Head(torch.autograd.Function):
    """logits = bf16(bf16(x) @ W^T) (fp32 accumulation over the hidden dim); d_x = bf16(bf16(d_logits) @ W). W stays
    frozen and is dequantized per vocab chunk (fp32) unless the module holds a full cache. This is MLX's
    QuantizedLinear on a bf16 input, as the MLX trainer runs it (astype(bf16) -> qmm -> astype(f32))."""
    @staticmethod
    def forward(ctx, x, q4, chunk, mode):
        ctx.q4, ctx.chunk, ctx.mode = q4, chunk, mode
        xb = round_bf16(x) if mode == "bf16" else x
        # one output buffer, filled chunk by chunk (bf16 in "bf16" mode: the values are bf16 anyway): no list + cat +
        # rounding copies (the walk's 512-row calls peaked ~1.8 GB of transients that way)
        y = torch.empty((*x.shape[:-1], q4.rows_n), dtype=torch.bfloat16 if mode == "bf16" else torch.float32,
                        device=x.device)
        for v0 in range(0, q4.rows_n, chunk):
            Wc = q4.cache[v0:v0 + chunk].to(torch.float32) if q4.cache is not None else q4.deq(r0=v0, r1=v0 + chunk)
            y[..., v0:v0 + chunk] = (xb @ Wc.T).to(y.dtype)
        return y.to(torch.float32)

    @staticmethod
    def backward(ctx, dy):
        q4, chunk, mode = ctx.q4, ctx.chunk, ctx.mode
        dyb = round_bf16(dy) if mode == "bf16" else dy
        dx = None
        for v0 in range(0, q4.rows_n, chunk):
            Wc = q4.cache[v0:v0 + chunk].to(torch.float32) if q4.cache is not None else q4.deq(r0=v0, r1=v0 + chunk)
            part = dyb[..., v0:v0 + chunk] @ Wc
            dx = part if dx is None else dx + part
        return (round_bf16(dx) if mode == "bf16" else dx), None, None, None


class Head(nn.Module):
    """the frozen LM head; mode "bf16" = MLX-exact emulation in fp32 math (CPU parity), "native" = one matmul in the
    cache's dtype under autocast (GPU: fp16 on T4)"""
    def __init__(self, q4, mode="bf16", chunk=32768):
        super().__init__()
        self.q4, self.mode, self.chunk = q4, mode, chunk

    def forward(self, x):
        if self.mode == "native":
            return (x.to(self.q4.cache.dtype) @ self.q4.cache.T).to(torch.float32)
        return _Bf16Head.apply(x, self.q4, self.chunk, "bf16")


# ---------------------------------------------------------------- DFlash2 modules (MLX names)
class RMSNorm(nn.Module):
    def __init__(self, d, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class MLP(nn.Module):
    def __init__(self, d, m):
        super().__init__()
        self.gate_proj = nn.Linear(d, m, bias=False)
        self.up_proj = nn.Linear(d, m, bias=False)
        self.down_proj = nn.Linear(m, d, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


def grouped_dynamic_convolve(hidden, dynamic, base, group_size):
    """dflash_model_mlx._grouped_dynamic_convolve, same order of additions"""
    B, L, D = hidden.shape
    G = D // group_size
    blocks = hidden.reshape(B, L, G, group_size)
    dynamic = dynamic.reshape(B, L, base.shape[0], G, 1)
    out = torch.zeros_like(blocks)
    for off in range(base.shape[0]):
        vals = blocks if off == 0 else torch.cat((torch.zeros_like(blocks[:, :off]), blocks[:, :-off]), dim=1)
        kern = base[off].reshape(1, 1, G, group_size).to(hidden.dtype)
        out = out + kern * vals
        out = out + dynamic[:, :, off] * vals
    return out.reshape(B, L, D)


class GroupedDynamicCausalConv(nn.Module):
    def __init__(self, d, k, gs):
        super().__init__()
        self.kernel_size, self.group_size = k, gs
        self.base_kernel = nn.Parameter(torch.zeros(2, k, d))
        self.kernel_projection = nn.Linear(d, 2 * k * (d // gs), bias=False)

    def prepare(self, h):
        G = h.shape[-1] // self.group_size
        dyn = self.kernel_projection(h).reshape(*h.shape[:-1], 2, self.kernel_size, G)
        return grouped_dynamic_convolve(h, dyn[..., 0, :, :], self.base_kernel[0], self.group_size), dyn[..., 1, :, :]

    def finish(self, h, dyn):
        return grouped_dynamic_convolve(h, dyn, self.base_kernel[1], self.group_size)


class Attention(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.H, self.KV, self.HD = c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"]
        d = c["hidden_size"]
        self.q_proj = nn.Linear(d, self.H * self.HD, bias=False)
        self.k_proj = nn.Linear(d, self.KV * self.HD, bias=False)
        self.v_proj = nn.Linear(d, self.KV * self.HD, bias=False)
        self.o_proj = nn.Linear(self.H * self.HD, d, bias=False)
        self.q_norm = RMSNorm(self.HD, c["rms_norm_eps"])
        self.k_norm = RMSNorm(self.HD, c["rms_norm_eps"])
        self.scale = self.HD ** -0.5


class Layer(nn.Module):
    def __init__(self, c):
        super().__init__()
        d = c["hidden_size"]
        self.self_attn = Attention(c)
        self.mlp = MLP(d, c["intermediate_size"])
        self.input_layernorm = RMSNorm(d, c["rms_norm_eps"])
        self.post_attention_layernorm = RMSNorm(d, c["rms_norm_eps"])
        k, gs = c["dflash_config"]["conv_kernel_size"], c["dflash_config"]["conv_group_size"]
        self.attention_conv = GroupedDynamicCausalConv(d, k, gs)
        self.mlp_conv = GroupedDynamicCausalConv(d, k, gs)


class CandidateSelector(nn.Module):
    def __init__(self, c):
        super().__init__()
        r, V = c["dflash_config"]["selector_rank"], c["vocab_size"]
        self.predecessor_codebook = nn.Embedding(V, r)
        self.successor_codebook = nn.Embedding(V, r)
        self.hidden_projection = nn.Linear(c["hidden_size"], r, bias=False)

    def rows(self, which, idx):
        """codebook rows; under bf16_storage gather the bf16 rows first (no full-table upcast per access)"""
        e = getattr(self, which)
        if hasattr(e, "parametrizations"):
            return F.embedding(idx, e.parametrizations.weight.original).to(torch.float32)
        return e(idx)


class DFlash2(nn.Module):
    def __init__(self, c, embed_q4=None, head_q4=None, head_mode="bf16", attn_impl="explicit", ckpt_layers=False):
        super().__init__()
        self.c = c
        self.layers = nn.ModuleList([Layer(c) for _ in range(c["num_hidden_layers"])])
        self.norm = RMSNorm(c["hidden_size"], c["rms_norm_eps"])
        self.candidate_selector = CandidateSelector(c)
        self.embed_q4, self.head = embed_q4, (Head(head_q4, head_mode) if head_q4 is not None else None)
        self.MASK = int(c["dflash_config"]["mask_token_id"])
        self.OUTM = float(c["dflash_config"].get("output_multiplier", 1.0))
        self.embed_scale = float(c["dflash_config"].get("input_embedding_scale", 1.0))   # the Target has none: 1.0
        self.top_k = int(c["dflash_config"]["selector_top_k"])
        self.attn_impl, self.ckpt_layers = attn_impl, ckpt_layers
        self.attn_chunk = 0                              # explicit attention: anchors per score block (0 = all)
        self.embed_rounding = "once"                     # "twice" = MLX-CPU's dequantize (see Quant4.deq_rows_bf16)
        HD = c["head_dim"]
        base = float(c.get("rope_theta") or (c.get("rope_parameters") or {}).get("rope_theta", 10000.0))
        self.register_buffer("inv", torch.from_numpy((base ** (-np.arange(0, HD, 2) / HD)).astype(np.float32)),
                             persistent=False)

    # -------------------------------------------------- pieces
    def rope(self, x, pos):             # non-traditional RoPE (rotate halves), = the MLX trainer's rope()
        ang = pos[..., None].to(torch.float32) * self.inv
        c, s = torch.cos(ang), torch.sin(ang)
        h = x.shape[-1] // 2
        x1, x2 = x[..., :h], x[..., h:]
        return torch.cat([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1).to(x.dtype)

    def embed(self, tok):
        e = self.embed_q4.deq_rows_bf16(tok.reshape(-1), self.embed_rounding).reshape(*tok.shape, -1)
        return e * self.embed_scale

    def attn_explicit(self, sa, x, hc, kpos, qpos, neg, S, Aa, U, BS):
        B, H, KV, HD = S * Aa, sa.H, sa.KV, sa.HD
        q = self.rope(sa.q_norm(sa.q_proj(x).reshape(B, BS, H, HD)).transpose(1, 2), qpos) * sa.scale  # (B,H,BS,HD)
        kb = self.rope(sa.k_norm(sa.k_proj(x).reshape(B, BS, KV, HD)).transpose(1, 2), qpos)         # (B,KV,BS,HD)
        vb = sa.v_proj(x).reshape(B, BS, KV, HD).transpose(1, 2)
        kc = self.rope(sa.k_norm(sa.k_proj(hc).reshape(S, U, KV, HD)).transpose(1, 2), kpos)         # (S,KV,U,HD)
        vc = sa.v_proj(hc).reshape(S, U, KV, HD).transpose(1, 2)
        ac = self.attn_chunk or Aa                     # anchors per score block: the same math, less transient memory
        q6, kb5, vb5 = q.reshape(S, Aa, H, BS, HD), kb.reshape(S, Aa, KV, BS, HD), vb.reshape(S, Aa, KV, BS, HD)
        outs = []
        for a0 in range(0, Aa, ac):
            n = min(ac, Aa - a0)
            outs.append(self._attn_block(q6[:, a0:a0 + n], kb5[:, a0:a0 + n], vb5[:, a0:a0 + n], kc, vc,
                                         neg[:, a0:a0 + n], S, n, U, BS, H, KV, HD))
        o = torch.cat(outs, dim=1).reshape(B, H, BS, HD)
        return sa.o_proj(o.transpose(1, 2).reshape(B, BS, H * HD))

    @staticmethod
    def _attn_block(q, kb, vb, kc, vc, neg, S, Aa, U, BS, H, KV, HD):
        """the MLX trainer's attn() for Aa anchors: q (S,Aa,H,BS,HD), kb / vb (S,Aa,KV,BS,HD) -> (S,Aa,H,BS,HD)"""
        g, B = H // KV, S * Aa
        qg = q.reshape(S, Aa, KV, g, BS, HD).permute(0, 2, 1, 3, 4, 5).reshape(S, KV, Aa * g * BS, HD)
        sc = (qg @ kc.transpose(-1, -2)).reshape(S, KV, Aa, g, BS, U) + neg[:, None, :, None]
        sb = (q.reshape(B, KV, g * BS, HD) @ kb.reshape(B, KV, BS, HD).transpose(-1, -2)).reshape(S, Aa, KV, g, BS, BS)
        p = torch.softmax(torch.cat([sc, sb.permute(0, 2, 1, 3, 4, 5)], dim=-1), dim=-1)
        oc = (p[..., :U].reshape(S, KV, Aa * g * BS, U) @ vc).reshape(S, KV, Aa, g * BS, HD)
        oc = oc.permute(0, 2, 1, 3, 4).reshape(B, KV, g * BS, HD)
        pb = p[..., U:].permute(0, 2, 1, 3, 4, 5).reshape(B, KV, g * BS, BS)
        return (oc + pb @ vb.reshape(B, KV, BS, HD)).reshape(S, Aa, H, BS, HD)

    def attn_sdpa(self, sa, x, hc, kpos, qpos, neg, S, Aa, U, BS):
        """the same softmax over [own context rows; own block] as one SDPA call per group (other blocks masked)"""
        B, H, KV, HD = S * Aa, sa.H, sa.KV, sa.HD
        q = self.rope(sa.q_norm(sa.q_proj(x).reshape(B, BS, H, HD)).transpose(1, 2), qpos)               # unscaled
        kb = self.rope(sa.k_norm(sa.k_proj(x).reshape(B, BS, KV, HD)).transpose(1, 2), qpos)
        vb = sa.v_proj(x).reshape(B, BS, KV, HD).transpose(1, 2)
        kc = self.rope(sa.k_norm(sa.k_proj(hc).reshape(S, U, KV, HD)).transpose(1, 2), kpos)
        vc = sa.v_proj(hc).reshape(S, U, KV, HD).transpose(1, 2)
        qq = q.reshape(S, Aa, H, BS, HD).permute(0, 2, 1, 3, 4).reshape(S, H, Aa * BS, HD)
        kk = torch.cat([kc, kb.reshape(S, Aa, KV, BS, HD).permute(0, 2, 1, 3, 4).reshape(S, KV, Aa * BS, HD)], dim=2)
        vv = torch.cat([vc, vb.reshape(S, Aa, KV, BS, HD).permute(0, 2, 1, 3, 4).reshape(S, KV, Aa * BS, HD)], dim=2)
        ctx_ok = (neg == 0).reshape(S, Aa * BS, U)                                                        # (S,Aa*BS,U)
        blk = torch.arange(Aa, device=x.device).repeat_interleave(BS)
        own = (blk[:, None] == blk[None, :]).expand(S, -1, -1)                                            # (S,Aa*BS,Aa*BS)
        mask = torch.cat([ctx_ok, own], dim=-1)[:, None]                                                  # (S,1,Q,K)
        kk, vv = kk.repeat_interleave(H // KV, dim=1), vv.repeat_interleave(H // KV, dim=1)   # GQA (no enable_gqa:
        o = F.scaled_dot_product_attention(qq, kk, vv, attn_mask=mask, scale=sa.scale)      # mem-efficient on T4)
        o = o.reshape(S, H, Aa, BS, HD).permute(0, 2, 3, 1, 4).reshape(B, BS, H * HD)
        return sa.o_proj(o)

    def _layer(self, L, h, hc, kpos, qpos, neg, S, Aa, U, BS):
        att = self.attn_sdpa if self.attn_impl == "sdpa" else self.attn_explicit
        x, k = L.attention_conv.prepare(L.input_layernorm(h))
        h = h + L.attention_conv.finish(att(L.self_attn, x, hc, kpos, qpos, neg, S, Aa, U, BS), k)
        x, k = L.mlp_conv.prepare(L.post_attention_layernorm(h))
        return h + L.mlp_conv.finish(L.mlp(x), k)

    def fwd(self, hctx, upos, anc, tok):
        """hctx (S, U, D) fp32 context rows, upos (S, U) absolute positions (-1 = pad), anc (S, A) anchors, tok (S, A, BS)
        block tokens -> normed hidden (S*A, BS-1, D) at block positions 1..BS-1 (= the MLX trainer's fwd)"""
        S, Aa = anc.shape
        U, BS = hctx.shape[1], tok.shape[-1]
        B = S * Aa
        h = self.embed(tok.reshape(B, BS)).to(torch.float32)
        qpos = (anc.reshape(B, 1) + torch.arange(BS, device=anc.device)[None]).reshape(B, 1, BS)
        r, a, i = upos[:, None, None, :], anc[:, :, None, None], torch.arange(BS, device=anc.device)[None, None, :, None]
        neg = torch.where((r >= 0) & (r <= a - 1) & ((a + i) - r < SLIDE), 0.0, -1e9).to(torch.float32)
        hc = hctx.to(torch.float32)
        kpos = upos[:, None, :]
        for L in self.layers:
            if self.ckpt_layers and torch.is_grad_enabled():
                h = checkpoint(self._layer, L, h, hc, kpos, qpos, neg, S, Aa, U, BS, use_reentrant=False)
            else:
                h = self._layer(L, h, hc, kpos, qpos, neg, S, Aa, U, BS)
        return self.norm(h[:, 1:])

    # -------------------------------------------------- head + loss (row-chunked)
    def _chunk_terms(self, hid, pred, tkl, lpl, y):
        """per row: soft CE to the 125B's top-32 + the selector CE (the MLX loss_fn), and the eval picks"""
        logits = self.head(hid) * self.OUTM                                          # (R, V)
        cand = torch.topk(logits.detach(), self.top_k, dim=-1).indices                 # = argpartition's top-16 set
        sel = self.candidate_selector
        sc = torch.gather(logits, -1, cand) + (sel.rows("predecessor_codebook", pred)[:, None] * sel.hidden_projection(hid)[:, None]
                                               * sel.rows("successor_codebook", cand)).sum(-1)
        ce = torch.logsumexp(logits, dim=-1) - (torch.softmax(lpl, dim=-1) * torch.gather(logits, -1, tkl)).sum(-1)
        inc = cand == y[:, None]
        sce = (torch.logsumexp(sc, dim=-1) - (sc * inc).sum(-1)) * inc.any(-1)
        return ce + sce, logits.detach().argmax(-1), torch.gather(cand, -1, sc.detach().argmax(-1, keepdim=True))[:, 0]

    def loss_sum(self, hid, pred, tkl, lpl, y, w, rows=256, recompute=True):
        """sum over rows of w * (ce + sce); hid (B, KD, D) -> flattened rows, processed in chunks of `rows`"""
        D = hid.shape[-1]
        hid, pred, tkl, lpl, y, w = (hid.reshape(-1, D), pred.reshape(-1), tkl.reshape(-1, tkl.shape[-1]),
                                     lpl.reshape(-1, lpl.shape[-1]), y.reshape(-1), w.reshape(-1))
        keep = torch.nonzero(w > 0).reshape(-1)                  # rows with zero weight add exactly 0 (skipped)
        total = hid.new_zeros(())
        for r0 in range(0, keep.numel(), rows):
            ix = keep[r0:r0 + rows]
            fn = lambda h_, p_, t_, l_, y_: self._chunk_terms(h_, p_, t_, l_, y_)[0]
            if recompute and torch.is_grad_enabled():
                terms = checkpoint(fn, hid[ix], pred[ix], tkl[ix], lpl[ix], y[ix], use_reentrant=False)
            else:
                terms = fn(hid[ix], pred[ix], tkl[ix], lpl[ix], y[ix])
            total = total + (w[ix] * terms).sum()
        return total

    @torch.no_grad()
    def picks(self, hid, pred, rows=256):
        """argmax and the selector's teacher-forced pick at every row (for the on-path accuracy eval)"""
        D = hid.shape[-1]
        h2, p2 = hid.reshape(-1, D), pred.reshape(-1)
        dummy_t = torch.zeros((h2.shape[0], 1), dtype=torch.long, device=hid.device)
        dummy_l = torch.zeros((h2.shape[0], 1), dtype=torch.float32, device=hid.device)
        dummy_y = torch.zeros((h2.shape[0],), dtype=torch.long, device=hid.device)
        am, pk = [], []
        for r0 in range(0, h2.shape[0], rows):
            _, a, p = self._chunk_terms(h2[r0:r0 + rows], p2[r0:r0 + rows], dummy_t[r0:r0 + rows], dummy_l[r0:r0 + rows],
                                        dummy_y[r0:r0 + rows])
            am.append(a); pk.append(p)
        return torch.cat(am).reshape(hid.shape[:-1]), torch.cat(pk).reshape(hid.shape[:-1])


# ---------------------------------------------------------------- loading
def draft_config(n_layers=None, intermediate=None):
    c = json.load(open(os.path.join(DRAFT_DIR, "config.json")))
    if n_layers is not None:
        c["num_hidden_layers"] = n_layers
        c["layer_types"] = ["sliding_attention"] * n_layers
    if intermediate is not None:
        c["intermediate_size"] = intermediate
    return c


def load_target_q4(device="cpu"):
    t = st_load(TARGET_SHARD, {f"language_model.{p}.{s}" for p in ("model.embed_tokens", "lm_head")
                               for s in ("weight", "scales", "biases")})
    g = lambda p: Quant4(t[f"language_model.{p}.weight"][0], t[f"language_model.{p}.scales"][0],
                         t[f"language_model.{p}.biases"][0]).to(device)
    return g("model.embed_tokens"), g("lm_head")


def hf_draft_weights():
    """the HF DFlash2 checkpoint in MLX names (codebooks get '.weight', as load_draft does); fc / hidden_norm dropped"""
    w = st_load(os.path.join(DRAFT_DIR, "model.safetensors"))
    out = {}
    for k, (a, tag) in w.items():
        if k.startswith("fc.") or k.startswith("hidden_norm."):
            continue
        if k in ("candidate_selector.predecessor_codebook", "candidate_selector.successor_codebook"):
            k = k + ".weight"
        out[k] = to_torch(a, tag)
    return out


def build(n_layers=None, intermediate=None, init="hf", head_mode="bf16", attn_impl="explicit", ckpt_layers=False,
          device="cpu", expand=None, param_dtype=torch.float32):
    """init: 'hf' (incoai DFlash2), a path to MLX-trained / port safetensors (loaded over the HF weights, non-strict, as
    the MLX trainer's --init), or 'none'. expand='interleave' = P-CAP's block expansion (zero o_proj / down_proj copies)."""
    c = draft_config(None if expand else n_layers, intermediate)
    eq, hq = load_target_q4(device)
    dd = torch.get_default_dtype()
    torch.set_default_dtype(param_dtype)                 # bf16 for forward-only runs (bf16_storage adds fp32 views)
    try:
        m = DFlash2(c, eq, hq, head_mode, attn_impl, ckpt_layers)
    finally:
        torch.set_default_dtype(dd)
    def load_into(path, is_hf):           # one tensor at a time (no second full copy of the weights in memory)
        P = dict(m.named_parameters())
        seen = set()
        for k, (a, tag) in st_load(path).items():
            if is_hf and (k.startswith("fc.") or k.startswith("hidden_norm.")):
                continue
            if is_hf and k.startswith("layers.") and int(k.split(".")[1]) >= len(m.layers):
                continue                  # n_layers below the checkpoint's: its first n layers (small parity tests)
            if k in ("candidate_selector.predecessor_codebook", "candidate_selector.successor_codebook"):
                k = k + ".weight"
            assert k in P, f"unexpected weight {k}"
            with torch.no_grad():
                P[k].copy_(to_torch(a, tag).reshape(P[k].shape))
            seen.add(k)
        return seen, set(P)
    if init != "none":                    # the MLX order: HF weights, then the expansion, then the init file
        seen, allp = load_into(os.path.join(DRAFT_DIR, "model.safetensors"), True)
        assert seen == allp, f"missing weights: {sorted(allp - seen)[:5]}"
    if expand == "interleave":
        expand_interleave(m)
    if init not in ("none", "hf"):
        load_into(os.path.expanduser(init), False)
    return m.to(device)


@torch.no_grad()
def expand_interleave(m):
    """P-CAP (sticky/pcap_expand.py) in PyTorch: after each layer a copy whose o_proj and down_proj are zero, so the
    expanded drafter computes exactly the original function before training"""
    import copy
    new = []
    for L in m.layers:
        new.append(L)
        c = copy.deepcopy(L)
        c.self_attn.o_proj.weight.zero_()
        c.mlp.down_proj.weight.zero_()
        new.append(c)
    m.layers = nn.ModuleList(new)
    m.c["num_hidden_layers"] = len(new)
    m.c["layer_types"] = ["sliding_attention"] * len(new)
    return list(range(1, len(new), 2))


def build_synthetic(hidden=256, layers=2, vocab=4096, heads=4, kv=2, head_dim=64, inter=512, rank=32, seed=0,
                    head_mode="bf16", attn_impl="explicit", device="cpu"):
    """a small random DFlash2 with random 4-bit embedding / head (same code paths, ~MBs): for the checkpoint / resume
    and multi-process tests, not for parity with MLX"""
    g = np.random.default_rng(seed)
    c = dict(hidden_size=hidden, num_hidden_layers=layers, num_attention_heads=heads, num_key_value_heads=kv,
             head_dim=head_dim, intermediate_size=inter, vocab_size=vocab, rms_norm_eps=1e-6, rope_theta=10000,
             layer_types=["sliding_attention"] * layers, sliding_window=SLIDE,
             dflash_config=dict(conv_group_size=16, conv_kernel_size=2, mask_token_id=vocab - 1, selector_rank=rank,
                                selector_top_k=16))
    bits = lambda a: (np.asarray(a, np.float32).view(np.uint32) >> 16).astype(np.uint16)
    q4 = lambda: Quant4(g.integers(0, 2 ** 32, (vocab, hidden // 8), dtype=np.uint64).astype(np.uint32),
                        bits(g.uniform(0.01, 0.03, (vocab, hidden // 64))), bits(g.uniform(-0.15, -0.1, (vocab, hidden // 64))))
    torch.manual_seed(seed)
    m = DFlash2(c, q4(), q4(), head_mode, attn_impl)
    with torch.no_grad():
        for n_, p in m.named_parameters():
            if p.dim() > 1:
                p.normal_(0.0, 0.02)
            elif "base_kernel" in n_:
                p.normal_(0.0, 0.1)
    return m.to(device)


class _Upcast(nn.Module):
    def forward(self, w):
        return w.to(torch.float32)


def bf16_storage(m):
    """forward-only memory saver: keep every weight in bf16 storage and upcast it to fp32 on access (torch
    parametrize). Exact when the weights are bf16 values (the HF checkpoint, MLX-saved files): the same numbers in
    half the memory. Not for training (it would train the bf16 copy)."""
    import torch.nn.utils.parametrize as P
    todo = [(mod, name) for mod in list(m.modules()) for name, p in list(mod.named_parameters(recurse=False))
            if p.dim() >= 1]                                   # a snapshot: registering changes the module tree
    for mod, name in todo:
        p = getattr(mod, name)
        if p.dtype != torch.bfloat16:                          # built in fp32: convert this one tensor (exactness checked)
            with torch.no_grad():
                lo = p.detach().to(torch.bfloat16)
            assert torch.equal(lo.to(torch.float32), p.detach()), f"{name} is not bf16-exact"
            delattr(mod, name)
            mod.register_parameter(name, nn.Parameter(lo, requires_grad=False))
            del p, lo
        else:
            p.requires_grad_(False)
        P.register_parametrization(mod, name, _Upcast(), unsafe=True)   # bf16 storage, fp32 view
    return m
