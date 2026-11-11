"""Decode tok/s cost model for Qwen3.8-Flash-Next at 4-bit on this Mac (M5 Pro, 64 GB),
built on: parameter counts from the checkpoint headers (cfg/shapes.json), measured SSD and
GPU numbers (data/ssd.txt, MoEPrefetch, SpecDecode), and the lossless cache simulation on
proxy routing traces (data/sim_*.json). Writes data/costmodel.txt."""
import json, os, sys, math
H = os.path.expanduser("~/BigMoE")
# ---- sizes (MLX affine 4-bit, group 64 => 4.5 bits/weight; router 8-bit) -------------
BPW4, BPW8 = 4.5 / 8, 8.5 / 8
P = dict(linear_attn=2.087, full_attn=0.617, shared=0.236, hyper_conn=0.641, ple=0.033,
         lm_head=0.636, embed=0.636, router=0.063, routed=120.796, mtp=2.607, ngram=51.2)  # B params
NL, E, K = 48, 512, 10
EXP_MB = 3 * 2560 * 640 * BPW4 / 1e6                         # 2.765 MB per expert
dense_read = (P['linear_attn'] + P['full_attn'] + P['shared'] + P['hyper_conn'] + P['ple'] + P['lm_head']) * BPW4 + P['router'] * BPW8
routed_read = NL * K * EXP_MB / 1e3
experts_total = P['routed'] * BPW4
resident_fixed = dense_read + P['embed'] * BPW4 + P['mtp'] * BPW4 + 0.8 + 0.12 + 1.0   # +KV@32k, GDN state, scratch
# ---- hardware (measured) ----------------------------------------------------------------
BW = 280 * 0.92          # GB/s effective streaming read (35B-A3B 4-bit runs at ~0.92 of 280 after launch overhead)
T_FIXED = 2.4e-3 * (NL / 40) * 1.4   # launch/dispatch: 2.4 ms on 35B-A3B (40 layers), x1.4 for hyper-connection ops
SSD = 13.4               # GB/s, 2.77 MB random reads at QD>=4 (data/ssd.txt)
LAT1 = 0.45e-3           # s, first 2.77 MB read at QD1 (p50)
WAKE = 30e-6            # HYPOTHETICAL in-kernel GPU wait (not achievable on Metal: data/coherence*.txt)
HS = float(os.environ.get("HS_US", 24)) * 1e-6   # per-layer exact gate: 24 us measured with bigmoe_ext (data/test_gate.txt); 155 us with MTLSharedEvent GPU signal
LAT1 = 0.37e-3          # measured extra per stalled layer incl. one cold 2.77 MB pread (data/eventwait.txt)
SYNC = 0.23e-3           # s, GPU idle per per-layer CPU sync (MoEPrefetch sync_cost)
TAX = 0.0076             # decode slowdown per GB/s of SSD traffic (MoEPrefetch contention)
MTP_DRAFT = 0.6e-3       # s per drafted token (one extra layer + its 10 experts)
def t_compute(S, union):  # one forward pass over S tokens, all weights resident
    return T_FIXED + (dense_read + NL * union * EXP_MB / 1e3) / BW
def load(f):
    return [r for r in json.load(open(f))]
sims = load(f"{H}/data/sim_f60.json")
if os.path.exists(f"{H}/data/sim_f50_f70.json"): sims += load(f"{H}/data/sim_f50_f70.json")
def scale(r):
    """Proxy (40 layers, top-8 of 256) -> target (48 layers, top-10 of 512) at the same resident fraction."""
    uses = 8 * 40 * r['S']; q = r['stall'] / uses                   # miss rate per expert use
    stall = q * K * NL * r['S'] * (r['union'] / (8 * r['S']))         # misses scale with distinct experts
    lay_frac = r['stall_layers'] / 40
    q_l = 1 - (1 - lay_frac) ** (1 / max(r['union'], 1))              # per-distinct-expert layer-miss prob
    sl = NL * (1 - (1 - q_l) ** (r['union'] * K / 8))
    pre = r['prefetch'] / 40 * NL * K / 8
    return stall, sl, pre, r['union'] * K / 8
def step_time(r, mode):
    stall, sl, pre, uni = scale(r)
    tc = t_compute(r['S'], uni)
    if mode == "hs":     # exact: every layer handshakes (router out -> CPU checks/loads -> GPU continues)
        ts = NL * HS + sl * LAT1 + max(0, stall - sl) * EXP_MB / 1e3 / SSD
    else:                # GPU-side wait: only stalled layers pay
        ts = sl * (WAKE + LAT1) + max(0, stall - sl) * EXP_MB / 1e3 / SSD
    t_pre = pre * EXP_MB / 1e3 / SSD
    t = tc + ts + max(0, t_pre - tc)
    io = (stall + pre) * EXP_MB / 1e3 / t
    return t * (1 + TAX * io), io, stall, sl, pre
out = []
out.append(f"Target: Qwen3.8-Flash-Next, 4-bit (4.5 bpw). Expert {EXP_MB:.3f} MB; experts total {experts_total:.1f} GB; "
           f"per-token read: dense {dense_read:.2f} GB + routed {routed_read:.2f} GB = {dense_read+routed_read:.2f} GB")
out.append(f"Resident non-expert (dense+embed+MTP+KV32k+state+scratch): {resident_fixed:.1f} GB; n-gram table ({P['ngram']}B) on SSD")
for budget in (44, 46, 48, 52):
    out.append(f"  GPU budget {budget} GB -> experts resident {budget-resident_fixed:.1f} GB = {(budget-resident_fixed)/experts_total:.0%} of experts")
t1 = t_compute(1, K)
out.append(f"Fully resident (needs ~{experts_total+resident_fixed:.0f} GB): {t1*1e3:.1f} ms/token = {1/t1:.0f} tok/s (model; launch {T_FIXED*1e3:.1f} ms)")
out.append("")
out.append("Lossless streaming, per decode step (S = tokens verified per pass; tok/s assumes MTP acceptance a per draft position)")
out.append(f"{'res':>4} {'policy':28s} {'S':>2} {'miss/step':>9} {'stall lay':>9} {'pref/step':>9} {'SSD GB/s':>8} {'ms/step':>7} | tok/s: S=1 or a=.6/.75/.9")
for r in sorted(sims, key=lambda r: (r['frac'], r['B'], r['S'])):
    modes = ["hs", "wait"] if r['B'] in (0, 12) else ["hs"]
    for mode in modes:
        t, io, stall, sl, pre = step_time(r, mode)
        pol = f"{'handshake' if mode=='hs' else '(ideal wait)'} LRU" + (f"+lookahead{r['B']}" if r['B'] else "")
        if r['S'] == 1:
            tk = f"{1/t:5.1f}"
        else:
            D = r['S'] - 1; t_step = t + D * MTP_DRAFT
            tk = " / ".join(f"{((1-a**(D+1))/(1-a))/t_step:5.1f}" for a in (0.6, 0.75, 0.9))
        out.append(f"{r['frac']:4.0%} {pol:28s} {r['S']:2d} {stall:9.1f} {sl:9.1f} {pre:9.1f} {io:8.2f} {t*1e3:7.1f} | {tk}")
# fully-resident MTP reference
out.append("")
for D in (1, 2, 3):
    uni = {1: 12.88, 2: 17.03, 3: 20.76}[D] * K / 8
    t = t_compute(D + 1, uni) + D * MTP_DRAFT
    out.append(f"fully resident + MTP depth {D}: " + " / ".join(f"{((1-a**(D+1))/(1-a))/t:5.1f}" for a in (0.6, 0.75, 0.9)) + " tok/s (a=.6/.75/.9)")
open(f"{H}/data/costmodel.txt", "w").write("\n".join(out) + "\n"); print("\n".join(out))
