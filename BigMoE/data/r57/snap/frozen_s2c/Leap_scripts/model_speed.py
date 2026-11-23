"""Rough decode tok/s on Kamen's M5 Pro 64 GB for any MoE / dense model at 4-bit (Leap, 2026-10-23).
Calibrated on OUR measurements:
  RAM: plain decode = bytes_per_token / (eta x 280 GB/s), eta 0.55 (35B-A3B 4-bit: 2.00 GB/tok, 75-83 tok/s measured
       vs 140 roofline; 125B in RAM ~34 plain / ~65 with MTP).
  SSD: misses_per_token x expert_MB / 13 GB/s, exposed (24 GB 125B is SSD-wait bound).
  Hit rate vs cache fraction c (LFU-ish, 125B traces): h(0.5) 0.956 (21.2 misses/tok of 480), h(0.25) ~0.80
       (trace replay 222-347 misses / 4-row pass), interpolated in log c; assumed to transfer to other MoEs.
  Speculation: x1.9 in RAM with a native MTP head (125B 65 / 34), x1.55 streamed (125B 41.6 / 26.8), x1.3 with
       lookup only (35B-A3B 103 / 75 overall).
bytes/param at 4-bit g64 incl. scales: 0.5625; dense (non-routed) weights always resident; KV/state ~0.3 GB/token read
at ~10K context for hybrid models, more for full-attention ones (set kv_gb)."""
import math

BW, ETA, SSD = 280.0, 0.55, 13.0
BPP = 0.5625

def hit(c):
    if c >= 1.0: return 1.0
    if c <= 0.0: return 0.0
    pts = [(0.03, 0.30), (0.10, 0.55), (0.25, 0.80), (0.50, 0.956), (0.75, 0.99), (1.0, 1.0)]  # 0.03/0.10 extrapolated
    for (c0, h0), (c1, h1) in zip(pts, pts[1:]):
        if c0 <= c <= c1:
            t = (math.log(c) - math.log(c0)) / (math.log(c1) - math.log(c0))
            return h0 + t * (h1 - h0)
    return pts[0][1] * c / pts[0][0]

def speed(m, budget_gb, overhead_gb=1.5):
    """m: total_b, active_b, routed_total_b, routed_active_b, expert_mb (4-bit), kv_gb, spec ('mtp'|'lookup'|None)"""
    total_gb = m["total_b"] * BPP
    dense_gb = (m["total_b"] - m["routed_total_b"]) * BPP
    ram_tok = m["active_b"] * BPP + m.get("kv_gb", 0.3)
    t_ram = ram_tok / (ETA * BW)
    if budget_gb is None or total_gb + overhead_gb <= budget_gb:     # fully resident
        plain, mode = 1.0 / t_ram, "RAM"
        mult = {"mtp": 1.9, "lookup": 1.3, None: 1.0}[m.get("spec")]
    else:
        cache_gb = budget_gb - dense_gb - overhead_gb
        if cache_gb <= 0 or m["routed_total_b"] == 0:
            return None, None, "no fit"
        c = cache_gb / (m["routed_total_b"] * BPP)
        n_active = m["routed_active_b"] * 1e3 / (m["expert_mb"] / BPP)   # active experts per token
        misses = n_active * (1 - hit(c))
        t_ssd = misses * m["expert_mb"] / 1e3 / SSD
        plain, mode = 1.0 / (t_ram + t_ssd), f"c={c:.2f} miss/tok {misses:.0f}"
        mult = {"mtp": 1.55, "lookup": 1.2, None: 1.0}[m.get("spec")]
    return plain, plain * mult, mode

if __name__ == "__main__":
    # self-check against our measured 125B points (4-bit, exact): expect ~33 / ~63 RAM-equivalent, ~27 / ~42 at 34 GB, ~19 / ~30 at 24 GB
    q125 = dict(total_b=125, active_b=6.0, routed_total_b=120.8, routed_active_b=2.36, expert_mb=2.77, kv_gb=0.3, spec="mtp")
    q125["active_b"] = 4.7 / BPP - 0.3 / BPP     # measured dense ~3.4 GB/pass incl. bf16 parts -> 4.7 GB/token total
    for b in (None, 34, 24, 12):
        print(b, speed(q125, b))
