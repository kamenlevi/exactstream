"""gdn_bytes.py (Fable B, mission B1): the GDN layer's bytes, FLOPs and floors from the model's shapes, plus a lossless-pack
ceiling read off the REAL 4-bit weights (order-0 entropy of the codes and of the bf16 scales / biases). CPU only, numpy,
no mlx. Reads slices of ~/BigMoE/models/bare/model-00016-of-00017.safetensors by memmap (layers 1 / 16 / 32 / 46).
Run: nice -n 15 ~/BigMoE/.venv/bin/python gdn_bytes.py > gdn_bytes.out"""
import json, struct, sys, time
import numpy as np

ROOT = "/Users/kamen/BigMoE/models/bare"
H, HK, HV, DK, DV, KC, NL = 2560, 16, 48, 128, 128, 4, 36
KEY, VAL = HK * DK, HV * DV                       # 2048, 6144
CONV = 2 * KEY + VAL                               # 10240
S = 16                                             # rows per check
BW = [258e9, 280e9, 294e9]                         # bw_probe: 258-287 GB/s medians, best 294; kern23 used 280


def q4_bytes(n, k, g=64):
    codes = n * k // 2
    scales = n * (k // g) * 2
    biases = n * (k // g) * 2
    return codes, scales, biases


def ms(b, bw):
    return 1e3 * b / bw


# ---- 1. per-layer bytes / FLOPs at 16 rows -------------------------------------------------------------------------
print("== 1. one GDN layer at %d rows: weights, activations, FLOPs [P from the config's shapes] ==" % S)
W = {}
W["in_proj_qkv"] = q4_bytes(CONV, H)
W["in_proj_z"] = q4_bytes(VAL, H)
W["in_proj_b"] = q4_bytes(HV, H)
W["in_proj_a"] = q4_bytes(HV, H)
W["out_proj"] = q4_bytes(H, VAL)
tot_w = 0
for k, (c, s, b) in W.items():
    tot_w += c + s + b
    print("  %-12s codes %6.2f MB  scales %5.2f MB  biases %5.2f MB  = %6.2f MB" % (k, c / 1e6, s / 1e6, b / 1e6, (c + s + b) / 1e6))
small = 2 * KC * CONV + 2 * DV + 2 * 2 * HV   # conv1d weight bf16 [10240, 4], norm weight [128], A_log + dt_bias [48] x 2
print("  conv1d + norm + A_log / dt_bias (bf16): %.3f MB" % (small / 1e6))
state = HV * DV * DK * 4                           # fp32 recurrent state
conv_state = (KC - 1) * CONV * 2                   # bf16 [3, 10240]
print("  recurrent state fp32 %.2f MB (read + written = %.2f MB); conv state bf16 %.3f MB (read + written)" % (state / 1e6, 2 * state / 1e6, conv_state / 1e6))
# activations at 16 rows (bf16): x in, qkv, z, b, a, q, k, v, y (gated+normed), out
act = {"x (HC read out)": S * H, "qkv": S * CONV, "z": S * VAL, "b, a": 2 * S * HV, "q, k (normed)": 2 * S * KEY, "v": S * VAL,
       "y (pre out_proj)": S * VAL, "out": S * H}
tot_a = 0
for k, n in act.items():
    tot_a += 2 * n
print("  activations bf16 (each written once, read once or twice): " + ", ".join("%s %.0f KB" % (k, 2 * n / 1e3) for k, n in act.items()))
print("  activation bytes per layer ~%.2f MB written; ~x2-3 at DRAM level incl. reads" % (tot_a / 1e6))
flop_in = 2 * S * H * (CONV + VAL + 2 * HV)
flop_out = 2 * S * VAL * H
flop_rec = S * HV * (DV * DK * 6)                  # per row per head: state *= g, kv += s*k, s += k*delta, out += s*q (~6 flop per element)
flop_conv = S * CONV * KC * 2
flop_norm = S * (2 * KEY + VAL) * 4
print("  FLOPs: in_proj %.3f G, out_proj %.3f G, recurrence %.3f G, conv %.4f G, norms %.4f G = %.3f GFLOP per layer, %.1f per check"
      % (flop_in / 1e9, flop_out / 1e9, flop_rec / 1e9, flop_conv / 1e9, flop_norm / 1e9,
         (flop_in + flop_out + flop_rec + flop_conv + flop_norm) / 1e9, NL * (flop_in + flop_out + flop_rec + flop_conv + flop_norm) / 1e9))
per_layer_bytes = tot_w + small + 2 * state + 2 * conv_state + 2.5 * tot_a
print("  bytes per layer: weights %.2f MB + state %.2f + conv state %.2f + activations ~%.2f = %.2f MB; x %d layers = %.3f GB per check"
      % (tot_w / 1e6, 2 * state / 1e6, 2 * conv_state / 1e6, 2.5 * tot_a / 1e6, per_layer_bytes / 1e6, NL, NL * per_layer_bytes / 1e9))
print("  arithmetic intensity: %.0f FLOP per byte (the M5 Pro's ~%d-%d GB/s against tens of TFLOPS on the tensor units: byte-bound at 16 rows)"
      % (NL * (flop_in + flop_out + flop_rec + flop_conv + flop_norm) / (NL * per_layer_bytes), 258, 294))

print("\n== 2. floors per check (36 layers) at 258 / 280 / 294 GB/s [P on K bw_probe] ==")
for name, b in (("weights (codes + scales + biases, all five projections)", NL * tot_w),
                ("  of which in_proj_qkv", NL * sum(W["in_proj_qkv"])), ("  of which in_proj_z", NL * sum(W["in_proj_z"])),
                ("  of which out_proj", NL * sum(W["out_proj"])), ("  of which in_proj_a + b", NL * (sum(W["in_proj_a"]) + sum(W["in_proj_b"]))),
                ("recurrent state read + write", NL * 2 * state), ("conv state + small weights", NL * (2 * conv_state + small)),
                ("activations (~2.5 touches of each)", NL * 2.5 * tot_a), ("ALL", NL * per_layer_bytes)):
    print("  %-56s %7.1f MB  %5.2f / %5.2f / %5.2f ms" % (name, b / 1e6, ms(b, BW[0]), ms(b, BW[1]), ms(b, BW[2])))

# ---- 3. the real weights: order-0 entropy of the 4-bit codes and of the scales / biases ---------------------------
print("\n== 3. lossless-pack ceiling from the REAL weights (order-0 entropy) [K on the model files] ==")
idx = json.load(open(ROOT + "/model.safetensors.index.json"))["weight_map"]


def open_st(fn):
    with open(fn, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        hdr = json.loads(fh.read(n))
    return hdr, 8 + n


_files = {}


def tensor(name):
    fn = ROOT + "/" + idx[name]
    if fn not in _files:
        hdr, base = open_st(fn)
        _files[fn] = (hdr, base, np.memmap(fn, dtype=np.uint8, mode="r"))
    hdr, base, mm = _files[fn]
    h = hdr[name]
    a, b = h["data_offsets"]
    raw = mm[base + a: base + b]
    dt = {"U32": np.uint32, "BF16": np.uint16}[h["dtype"]]
    return np.frombuffer(raw, dtype=dt).reshape(h["shape"])


def H0(counts):
    p = counts[counts > 0] / counts.sum()
    return float(-(p * np.log2(p)).sum())


def code_entropy(w):
    # MLX 4-bit pack: 8 codes per uint32, code j = (word >> 4 j) & 15
    cnt = np.zeros(16, np.int64)
    for j in range(8):
        cnt += np.bincount(((w >> (4 * j)) & 15).ravel(), minlength=16)
    return H0(cnt), cnt


def bf16_entropy(s):
    cnt = np.bincount(s.ravel().astype(np.int64), minlength=65536)
    h_raw = H0(cnt)
    # within-row delta coding of the bf16 bit pattern (as int16), order-0 entropy of the deltas
    d = np.diff(s.astype(np.int32), axis=1).ravel()
    cnt_d = np.bincount(d - d.min(), minlength=1)
    return h_raw, H0(cnt_d)


t0 = time.time()
summary = []
for L in (1, 16, 32, 46):
    for part in ("in_proj_qkv", "in_proj_z", "out_proj"):
        base = "language_model.model.layers.%d.linear_attn.%s" % (L, part)
        w = tensor(base + ".weight"); sc = tensor(base + ".scales"); bi = tensor(base + ".biases")
        hc, cnt = code_entropy(w)
        hs, hsd = bf16_entropy(sc); hb, hbd = bf16_entropy(bi)
        n_codes = w.size * 8
        bytes_now = w.size * 4 + sc.size * 2 + bi.size * 2
        bytes_pack = n_codes * hc / 8 + sc.size * min(hs, hsd) / 8 + bi.size * min(hb, hbd) / 8
        summary.append((L, part, hc, hs, hsd, hb, hbd, bytes_now, bytes_pack))
        top = np.argsort(cnt)[::-1][:4]
        print("  L%2d %-12s codes H0 %.3f bits (top codes %s: %s)  scales H0 %.2f / delta %.2f bits  biases H0 %.2f / delta %.2f bits  -> %.1f%% of today's bytes"
              % (L, part, hc, list(top), ["%.1f%%" % (100 * cnt[t] / cnt.sum()) for t in top], hs, hsd, hb, hbd, 100 * bytes_pack / bytes_now))
hc_m = np.mean([s[2] for s in summary]); r = sum(s[8] for s in summary) / sum(s[7] for s in summary)
print("  mean code entropy %.3f bits of 4 (%.1f%% of the code bytes); whole-pack ceiling (codes + scales + biases, order-0) %.1f%% of today's bytes"
      % (hc_m, 100 * hc_m / 4, 100 * r))
wb = NL * tot_w
print("  => the dense GDN weights 1.174 GB per check could shrink to ~%.3f GB: the weight floor %.2f -> %.2f ms at 280 GB/s (-%.2f ms upper bound, if the decode is free)"
      % (wb * r / 1e9, ms(wb, 280e9), ms(wb * r, 280e9), ms(wb, 280e9) - ms(wb * r, 280e9)))
print("  (%.1f s)" % (time.time() - t0))
