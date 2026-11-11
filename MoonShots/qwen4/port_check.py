"""Qwen 4 port check (MoonShots, Sat 2026-11-07; the day-0 kit): what changed in a new model's config.json against today's
model (~/BigMoE/models/bare/config.json, qwen4_exp_text, Qwen3.8-Flash-Next), grouped by the engine part it touches, plus
the derived sizes the engine is built on. READ-ONLY: it reads a local config.json or fetches one config.json (a few KB of
public metadata, never weights) from a Hugging Face repo id.
Derived (records3's layout, ~/BigMoE/DESIGN_3bit.md; 4-bit affine group 64 split into an MSB-3 part and an LSB sidecar):
  per expert: 3 projections of moe_intermediate x hidden weights;
  MSB-3 body = 3 x (w x 3 / 8 + 2 x 2 x w / 64) bytes (3-bit codes + bf16 s3, b3 per group of 64);
  LSB body   = 3 x (w / 8 + 2 x w / 64) bytes (1-bit l + the original bf16 bias); records pad each part to 4 KB pages.
  Today: 2,150,400 -> 2,162,688 and 768,000 -> 770,048 B (the p11 layout packs the scales: 2,064,384 / 720,896).
  Slots per layer at a RAM line: (budget - dense - KV / state) / layers / (MSB + LSB share), a rough first sizing only.
usage: python3 port_check.py <config.json path | hf repo id, e.g. Qwen/Qwen4-...> [--json]"""
import json, math, os, sys, urllib.request

H = os.path.expanduser("~")
TODAY = f"{H}/BigMoE/models/bare/config.json"
GROUPS = {
    "experts (store, tiers, kernels, keep-set, budgets)": ["num_experts", "num_experts_per_tok", "moe_intermediate_size",
                                                           "shared_expert_intermediate_size", "norm_topk_prob", "decoder_sparse_step",
                                                           "router_aux_loss_coef"],
    "trunk shapes (every dense kernel, KV, hc mixers)": ["hidden_size", "num_hidden_layers", "intermediate_size", "rms_norm_eps",
                                                         "hidden_act", "tie_word_embeddings"],
    "full attention (ref2 attention, KV cache, RoPE)": ["num_attention_heads", "num_key_value_heads", "head_dim",
                                                        "partial_rotary_factor", "rope_parameters", "rope_theta",
                                                        "max_position_embeddings", "attention_bias", "output_gate_type"],
    "linear attention / GDN (state, conv, chunked kernels)": ["linear_num_value_heads", "linear_num_key_heads",
                                                              "linear_key_head_dim", "linear_value_head_dim",
                                                              "linear_conv_kernel_dim", "mamba_ssm_dtype"],
    "layer pattern (which layers are GDN / full)": ["full_attention_interval", "layer_types"],
    "sparse-attention indexer (QSA)": ["indexer_budget", "indexer_compress_ratio", "indexer_head_dim", "indexer_kv_heads",
                                      "indexer_n_heads"],
    "hyper-connections (hc mixers)": ["hc_count", "hc_lowrank"],
    "n-gram sidecar (off-accelerator tables)": ["ngram_sidecar", "ngram_size", "ngram_vocab_size_base", "heads_per_ngram",
                                                "split_ngram_parts", "make_ngram_vocab_size_divisible_by"],
    "per-layer embeddings (PLE)": ["ple_embed_dim", "ple_layer_ids", "ple_conv_kernel_size"],
    "MTP (the model's own drafter)": ["mtp", "mtp_num_hidden_layers", "mtp_use_dedicated_embeddings"],
    "vocab / tokens (tokenizer, chat template, drafter head, EOS)": ["vocab_size", "bos_token_id", "eos_token_id",
                                                                     "pad_token_id"],
}


def load(src):
    if os.path.exists(src):
        return json.load(open(src))
    req = urllib.request.Request(f"https://huggingface.co/{src}/resolve/main/config.json",
                                 headers={"User-Agent": "moonshots-qwen4-port-check/1.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read(2_000_000))


def text(c):
    return c.get("text_config", c)


def page(n):
    return int(math.ceil(n / 4096) * 4096)


def sizes(t):
    hd, mi = t.get("hidden_size"), t.get("moe_intermediate_size")
    if not hd or not mi:
        return None
    w = hd * mi
    msb, lsb = 3 * (w * 3 // 8 + 2 * 2 * w // 64), 3 * (w // 8 + 2 * w // 64)
    pad16 = lambda n: (n + 16383) // 16384 * 16384                    # make_records3.layout (ALIGN 16 KB, exact): today
    msb_p = pad16(msb)                                                 # 2,162,688 / 770,048 B (records3 index row-major)
    lsb_p = pad16(msb_p + lsb) - msb_p
    L, E = t.get("num_hidden_layers", 0), t.get("num_experts", 0)
    per_layer = E * (msb_p + lsb_p)
    return dict(w=w, msb=msb, lsb=lsb, msb_p=msb_p, lsb_p=lsb_p, store=L * per_layer, per_layer=per_layer)


def fmt(v):
    if isinstance(v, list) and len(v) > 8:
        return f"list[{len(v)}] " + json.dumps({x: v.count(x) for x in sorted(set(map(str, v)))})
    s = json.dumps(v)
    return s if len(s) < 90 else s[:87] + "..."


def hazards(new, old):
    """the port map's hard-coded spots this config trips (Sat 7 Nov scan of BigMoE / Leap / TrainLab / EvalSuite / FableGod /
    Breakthrough / mtplx; path:line evidence in QWEN4_READY.md section 2)"""
    tn, to = text(new), text(old)
    g = lambda k: tn.get(k)
    out = []
    E, k, I, Hd, L = g("num_experts"), g("num_experts_per_tok"), g("moe_intermediate_size"), g("hidden_size"), g("num_hidden_layers")
    if E and E > 527:
        out.append(f"SILENT: {E} experts > 527: bigmoe/gated.py TLINES=17 x 31 ids, ext li = e/31 unchecked -> wrong slots "
                   f"(and > 1024 overflows ext slot_of); raise TLINES = ceil(E/31) + add bounds checks, rebuild the ext")
    if k and 16 * k > 185:
        out.append(f"top-{k}: a 16-row check needs {16 * k} ids > the gate's 185 (gated.py:25-30, ext:587-591); screens pin MAXK 160")
    if k and k != 10:
        out.append(f"top-{k} != 10: tier plan '4433333333' + assert len(PLAN)==10 (stream.py:594), qom_keep K=10, relax_core "
                   f"k_top=10 (2 copies), EvalSuite KTOP / TOPK, the decision map's 10 ranks")
    if (I, Hd) != (to.get("moe_intermediate_size"), to.get("hidden_size")):
        out.append(f"expert shape {I}x{Hd} != 640x2560: make_records3 SHAPES, p11 PADW 9216 (pieces would OVERLAP), "
                   f"r3layout_ref 8804 words, convert_records3.sh recsz 2785280, kernel asserts N*40 / D*10 (3c-N, NAX6-N, "
                   f"OOO-1C), NAX6 split-K ks 4 / ks_b 2 (K/64 % 4, N/64 % 2)")
        if I and I % 512 == 0:
            out.append(f"moe_intermediate {I} is a multiple of 512: MLX may switch down_proj to qmv_fast while 3c copies qmv "
                       f"-> the 'bitwise == MLX' proof breaks; re-prove tierglu vs MLX")
    gdn = ("linear_num_value_heads", "linear_num_key_heads", "linear_key_head_dim", "linear_value_head_dim", "linear_conv_kernel_dim")
    if any(tn.get(x) != to.get(x) for x in gdn):
        out.append("SILENT: GDN dims changed: gdnbody.py / gdn_exact.py compile 10240 / 2048 / 128 / 6144 / 48 heads; ref2 guards "
                   "fall back to MTPLX's NON-exact path without an error -> the bitwise spec == AR guarantee is lost")
    if L and L != 48:
        out.append(f"{L} layers != 48: r57 tables (48, 512) / range(512) / L < 48, POP45 shape assert, md5-pinned laysz45 pools "
                   f"(applied silently), stream.py uniform past the list, the decision map's range(48)")
    if E and E != 512:
        out.append(f"{E} experts != 512: r57 zeros((48,512)) / range(512), pop_cap45 (48,512), router_fused E % 128, the map")
    if tn.get("vocab_size") != to.get("vocab_size") or tn.get("eos_token_id") != to.get("eos_token_id"):
        out.append("SILENT: vocab / EOS changed: EOS {248044, 248046} literal in r57 + 6 EvalSuite runners + TrainLab; vocab "
                   "248320 literal in dfl2c_train / d02b_walk; EvalSuite's pre-tokenized items + the AR reference are today's "
                   "tokenizer -> generation may not stop, comparisons go wrong")
    if (tn.get("mtp_num_hidden_layers") or 0) > 1:
        out.append(f"{tn.get('mtp_num_hidden_layers')} MTP layers: stream.py streams mtp.layers.0 only")
    if tn.get("hc_count") != to.get("hc_count") or tn.get("hc_lowrank") != to.get("hc_lowrank"):
        out.append("hc_count / hc_lowrank changed: the drafter tap reshape hard-codes hc 4 (r57:682, EvalSuite, TrainLab); "
                   "mtplx fast paths gate on hc 4 / lowrank 320")
    q = new.get("quantization") if isinstance(new.get("quantization"), dict) else {}
    if not q or q.get("bits") != 4 or q.get("group_size") != 64:
        out.append("not 4-bit affine g64 (e.g. bf16 upstream): records3 / tiers / p11 / qmm16 / ref2 assume 4-bit g64 -> "
                   "quantise first (mtplx recipe: base 4-bit g64, mlp.gate + indexer 8-bit, hc mixers + PLE bf16)")
    out.append("ALWAYS (per new model): retrain or replace the drafter (DFlash2 taps 5/15/25/35/45, hidden / vocab / 48 layers; "
               "the md5-pinned d22_b5 MTP head), regenerate pinned data (pools, popularity, priors, corrections), the AR "
               "reference, EvalSuite items, the decision map inputs; mtplx must support the model_type")
    return out


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    new, old = load(sys.argv[1]), json.load(open(TODAY))
    tn, to = text(new), text(old)
    print(f"== port check: {sys.argv[1]} vs today's {to.get('model_type')} ({old.get('mtplx_source_repo', '')})")
    print(f"   architectures {new.get('architectures')} (today {old.get('architectures')}); model_type {tn.get('model_type')} / "
          f"top {new.get('model_type')}; quantization {json.dumps(new.get('quantization', {}).get('bits') if isinstance(new.get('quantization'), dict) else new.get('quantization'))} "
          f"(today 4-bit g64 mtplx)")
    seen = set()
    changed = 0
    for g, keys in GROUPS.items():
        rows = []
        for k in keys:
            seen.add(k)
            a, b = to.get(k), tn.get(k)
            if a != b:
                rows.append(f"     {k}: {fmt(a)} -> {fmt(b)}")
        print(f"  [{'CHANGED' if rows else 'same   '}] {g}")
        for r in rows:
            print(r)
        changed += len(rows)
    extra = sorted(k for k in set(tn) | set(to) if k not in seen and tn.get(k) != to.get(k)
                   and not isinstance(tn.get(k, to.get(k)), dict))
    if extra:
        print("  [other] " + ", ".join(f"{k}: {fmt(to.get(k))} -> {fmt(tn.get(k))}" for k in extra[:20]))
    if "vision_config" in new or "vision_config" in old:
        print(f"  [vision] today {'present' if 'vision_config' in old else 'absent'}, new {'present' if 'vision_config' in new else 'absent'}")
    so, sn = sizes(to), sizes(tn)
    if so and sn:
        print(f"  derived (records3 row-major layout, make_records3's formula): expert MSB-3 record {sn['msb_p']:,} B (body {sn['msb']:,}; "
              f"today {so['msb_p']:,}), LSB record {sn['lsb_p']:,} "
              f"(today {so['lsb_p']:,}); expert store {sn['store'] / 1e9:.1f} GB (today {so['store'] / 1e9:.1f})")
        Ln, En, kn = tn.get("num_hidden_layers"), tn.get("num_experts"), tn.get("num_experts_per_tok")
        for gb in (45, 24, 12):
            slots = (gb * 1e9 * 0.60) / max(1, Ln) / (sn["msb_p"] + sn["lsb_p"] * 0.2)
            print(f"   ~{gb} GB line (rough: 60 % of RAM to experts, LSB pool 20 %): ~{min(En, int(slots))} of {En} experts per "
                  f"layer resident ({100 * min(En, slots) / En:.0f} %), top-{kn}")
    hz = hazards(new, old)
    print(f"  HAZARDS from the port map (QWEN4_READY.md section 2; SILENT = no error, wrong results): {len(hz)}")
    for h in hz:
        print(f"   - {h}")
    if not os.path.exists(sys.argv[1]):                                # an HF repo: its file list and sizes (metadata, no download)
        try:
            req = urllib.request.Request(f"https://huggingface.co/api/models/{sys.argv[1]}?blobs=true",
                                         headers={"User-Agent": "moonshots-qwen4-port-check/1.0"})
            with urllib.request.urlopen(req, timeout=30) as r:
                info = json.loads(r.read(5_000_000))
            sib = info.get("siblings", [])
            st = sum(x.get("size") or 0 for x in sib if x["rfilename"].endswith(".safetensors"))
            print(f"  download size (for Kamen's OK): {st / 1e9:.1f} GB of safetensors in {sum(1 for x in sib if x['rfilename'].endswith('.safetensors'))} "
                  f"files; {len(sib)} files in all; gated {info.get('gated')}; license {(info.get('cardData') or {}).get('license')}")
        except Exception as e:
            print(f"  download size: not available ({str(e)[:80]})")
    print(f"  => {changed} field(s) changed in the mapped groups; read QWEN4_READY.md section 2 (port map) for the files each group touches")


if __name__ == "__main__":
    main()
