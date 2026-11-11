"""head_bound.py (Fable B, mission B3, cloud idea #3 "seeded exact argmax"): what a Cauchy-Schwarz (or a code-box) bound
can prune of the 125B's 248,320 x 2,560 4-bit LM head. CPU only, numpy, no mlx.
Real parts [K]: the head's dequantized rows (codes + scales + biases from model-00016), their global and per-chunk norms,
the head's energy spectrum (W^T W), and the target's top-32 log-probs on the 24 HumanEval windows (Leap's l118_tgt
captures: the logit gaps between the top-1 and the runners-up). No capture of the trunk's final hidden state exists on
disk (l118_tgt's h is the drafter's context input), so the hidden state is synthetic [P/G]: flat (rms 1 per element,
L1/L2 ~40 as every normed state measured here) with a planted cosine c to the candidate row; c is swept and the value
that reproduces the measured top-1 - top-2 gap is read off. Pruning fractions are then computed on the REAL head rows.
Run: nice -n 15 ~/BigMoE/.venv/bin/python head_bound.py > head_bound.out  (~2 min, ~1.5 GB)"""
import glob, json, struct, time
import numpy as np

ROOT = "/Users/kamen/BigMoE/models/bare/"
idx = json.load(open(ROOT + "model.safetensors.index.json"))["weight_map"]
rng = np.random.default_rng(0)


def st(name):
    fn = ROOT + idx[name]
    with open(fn, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]; hdr = json.loads(fh.read(n))
    h = hdr[name]; a, b = h["data_offsets"]
    mm = np.memmap(fn, dtype=np.uint8, mode="r")
    dt = {"U32": np.uint32, "BF16": np.uint16}[h["dtype"]]
    return np.frombuffer(mm[8 + n + a: 8 + n + b], dtype=dt).reshape(h["shape"])


def bf16(u16):
    return (u16.astype(np.uint32) << 16).view(np.float32)


t0 = time.time()
W = st("language_model.lm_head.weight"); S = st("language_model.lm_head.scales"); B = st("language_model.lm_head.biases")
V, K, G, NCH = W.shape[0], 2560, 64, 40
Sf = bf16(S); Bf = bf16(B)
print("== the real head: %d rows x %d, 4-bit g64 affine: codes %.1f MB + scales %.1f MB + biases %.1f MB = %.1f MB [K] =="
      % (V, K, W.nbytes / 1e6, S.nbytes / 1e6, B.nbytes / 1e6, (W.nbytes + S.nbytes + B.nbytes) / 1e6))
CH = 16384
Wd = np.empty((V, K), np.float16)          # the dequantized head, once (1.27 GB)
cn = np.empty((V, NCH), np.float32); cn256 = np.empty((V, 10), np.float32)
gram = np.zeros((K, K), np.float64)
for r0 in range(0, V, CH):
    r1 = min(V, r0 + CH); w = W[r0:r1]
    codes = np.empty((r1 - r0, K), np.float32)
    for j in range(8):
        codes[:, j::8] = ((w >> (4 * j)) & 15).astype(np.float32)
    d = (codes.reshape(-1, NCH, G) * Sf[r0:r1, :, None] + Bf[r0:r1, :, None]).reshape(-1, K)
    Wd[r0:r1] = d.astype(np.float16)
    cn[r0:r1] = np.sqrt((d.reshape(-1, NCH, G) ** 2).sum(2)); cn256[r0:r1] = np.sqrt((d.reshape(-1, 10, 256) ** 2).sum(2))
    gram += d.T.astype(np.float64) @ d.astype(np.float64)
gn = np.sqrt((cn ** 2).sum(1))
print("row norms ||w_v||: median %.3f, p1 %.3f, p10 %.3f, p90 %.3f, p99 %.3f, min %.4f, max %.3f; CV %.2f  (%.0f s)"
      % (np.median(gn), *np.percentile(gn, [1, 10, 90, 99]), gn.min(), gn.max(), gn.std() / gn.mean(), time.time() - t0))
print("rows with norm < 0.25 / 0.5 / 0.75 x the median: %.2f %% / %.2f %% / %.2f %%" % tuple(100 * (gn < f * np.median(gn)).mean() for f in (0.25, 0.5, 0.75)))
print("chunk-norm flatness of the rows: median (max chunk norm / mean chunk norm) %.2f; for a flat h the 40-chunk bound / the global bound = %.3f (median over rows)"
      % (np.median(cn.max(1) / cn.mean(1)), np.median(cn.sum(1) / (gn * np.sqrt(NCH)))))
ev = np.linalg.eigvalsh(gram)[::-1]; cum = np.cumsum(ev) / ev.sum()
print("head energy spectrum (eigenvalues of W^T W): top 1 / 10 / 100 / 256 / 640 / 1280 of 2560 dims hold %.1f / %.1f / %.1f / %.1f / %.1f / %.1f %% of the energy [K]"
      % tuple(100 * cum[[0, 9, 99, 255, 639, 1279]]))


def head_matmul(H, k0=0, k1=K):
    """logits [V, n] = Wd[:, k0:k1] @ H[k0:k1, :], chunked in rows (fp32 accumulate)"""
    out = np.empty((V, H.shape[1]), np.float32)
    for r0 in range(0, V, CH):
        out[r0:r0 + CH] = Wd[r0:r0 + CH, k0:k1].astype(np.float32) @ H[k0:k1]
    return out


# ---- the measured logit gaps of the target (top-32 log-probs on 24 HumanEval windows) [K] ----
gaps1, gaps2, p1 = [], [], []
for fn in sorted(glob.glob("/Users/kamen/Leap/data/l118_tgt/tf_code_*.npz")):
    z = np.load(fn); lp = z["lp"].astype(np.float32); m = z["asst"][1:].astype(bool)
    lp = lp[m] if m.shape[0] == lp.shape[0] else lp
    gaps1.append(lp[:, 0] - lp[:, 1]); gaps2.append(lp[:, 0] - lp[:, 31]); p1.append(np.exp(lp[:, 0]))
gaps1 = np.concatenate(gaps1); gaps2 = np.concatenate(gaps2); p1 = np.concatenate(p1)
print("\n== the target's logit gaps on code (assistant positions of 24 HumanEval windows, n = %d) [K l118_tgt] ==" % gaps1.size)
print("top-1 - top-2 (logits): median %.2f, p10 %.2f, p25 %.2f, p75 %.2f, p90 %.2f; top-1 - top-32: median %.2f, p10 %.2f; p(top-1): median %.3f, p10 %.3f, p25 %.3f"
      % (np.median(gaps1), *np.percentile(gaps1, [10, 25, 75, 90]), np.median(gaps2), np.percentile(gaps2, 10), np.median(p1), np.percentile(p1, 10), np.percentile(p1, 25)))

# ---- synthetic flat hidden states with a planted cosine to a real candidate row ----
NS = 64
rows = rng.choice(V, NS, replace=False)
hn = np.sqrt(K)
print("\n== pruning on the real head for a flat h (rms 1, ||h|| = %.1f, L1/L2 ~%.0f) with a planted cosine c to the candidate row [P on K] ==" % (hn, np.sqrt(2 / np.pi) * np.sqrt(K)))
print("candidate = the planted row (the drafted token is the argmax: the bound's best case); margin = 2 bf16 ulps of the candidate logit; medians over %d rows" % NS)
print("%5s | %7s %7s %7s | %7s | %7s %7s %7s %7s | %7s %7s %7s" % ("c", "gap1", "gap32", "cand", "cosmax", "global", "chnk64", "chk256", "box", "prg25%", "prg50%", "prg75%"))
for c in (0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.70, 0.90):
    Hs = np.empty((K, NS), np.float32)
    for i, r in enumerate(rows):
        u = Wd[r].astype(np.float32); u /= np.linalg.norm(u)
        g = rng.standard_normal(K).astype(np.float32); g -= (g @ u) * u; g /= np.linalg.norm(g)
        Hs[:, i] = hn * (c * u + np.sqrt(1 - c * c) * g)
    lg = head_matmul(Hs)                                            # [V, NS]
    srt = np.sort(lg, axis=0)
    s_c = lg[rows, np.arange(NS)]
    top2 = np.where(srt[-1] == s_c, srt[-2], srt[-1])
    margin = 2 * 2.0 ** -8 * np.abs(s_c); thr = s_c - margin        # [NS]
    hnorm = np.linalg.norm(Hs, axis=0)
    hc = np.sqrt((Hs.reshape(NCH, G, NS) ** 2).sum(1)); hc256 = np.sqrt((Hs.reshape(10, 256, NS) ** 2).sum(1))
    pr_g = (gn[:, None] * hnorm[None, :] < thr[None, :]).mean(0)
    pr_c64 = (cn @ hc < thr[None, :]).mean(0)
    pr_c256 = (cn256 @ hc256 < thr[None, :]).mean(0)
    # the code-box bound from the quantization structure (scales / biases only): per group g, max over codes in [0, 15]
    Hg = Hs.reshape(NCH, G, NS)
    Hp = np.maximum(Hg, 0).sum(1); Hm = np.minimum(Hg, 0).sum(1); Hsum = Hg.sum(1)   # [NCH, NS]
    box = np.where(Sf[:, :, None] >= 0, Sf[:, :, None] * 15 * Hp[None], Sf[:, :, None] * 15 * Hm[None]).sum(1) + Bf @ Hsum
    pr_box = (box < thr[None, :]).mean(0)
    cosmax = (lg / (gn[:, None] * hnorm[None, :])).max(0)
    prg = []
    for f in (0.25, 0.5, 0.75):
        kk = int(NCH * f) * G
        part = head_matmul(Hs, 0, kk)
        rem = cn[:, int(NCH * f):] @ hc[int(NCH * f):]
        prg.append(((part + rem) < thr[None, :]).mean(0))
    print("%5.2f | %7.2f %7.2f %7.2f | %7.3f | %7.4f %7.4f %7.4f %7.4f | %7.4f %7.4f %7.4f"
          % (c, np.median(s_c - top2), np.median(s_c - srt[-32]), np.median(s_c), np.median(cosmax), np.median(pr_g), np.median(pr_c64), np.median(pr_c256), np.median(pr_box),
             np.median(prg[0]), np.median(prg[1]), np.median(prg[2])))
print("columns: gap1 / gap32 = the synthetic top-1 - top-2 / top-32 gaps (compare with the measured medians %.2f / %.2f); cand = the candidate's logit; cosmax = the largest cosine any row reaches;" % (np.median(gaps1), np.median(gaps2)))
print("global / chnk64 / chk256 = fraction of the vocabulary pruned by ||h|| ||w_v||, by sum over 40 or 10 chunks of ||h_c|| ||w_{v,c}||; box = the code-box bound from scales / biases alone;")
print("prg f = exact partial sum over the first f of the dims + Cauchy-Schwarz on the rest. A pruned row's codes (1,280 B) and scales / biases (160 B) are not read; the 40-chunk norm table costs 80 B per row (%.0f MB), the 10-chunk one 20 B (%.0f MB), the box bound reads the 40 MB of scales / biases."
      % (V * 80 / 1e6, V * 20 / 1e6))
print("(%.0f s)" % (time.time() - t0))
