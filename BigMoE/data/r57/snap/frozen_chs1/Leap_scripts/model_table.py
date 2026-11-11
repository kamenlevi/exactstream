> (a quote from the project owner, kept private)
import sys, os
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_chs1/Leap_scripts"))
import model_speed as ms

def mb(dim, ffn): return 3 * dim * ffn / 1e6 * ms.BPP        # one routed expert at 4-bit, MB

M = {  # name: (AA index, spec dict); routed_* in B params; dense models: routed_total_b = 0
 "Qwen3.8-Flash-Next 125B-A6B": (39.8, dict(total_b=125, active_b=4.7 / ms.BPP - 0.3 / ms.BPP, routed_total_b=120.8, routed_active_b=2.36, expert_mb=2.77, spec="mtp")),
 "Qwen3.8-27B (dense)": (33.7, dict(total_b=28.6, active_b=28.6, routed_total_b=0, routed_active_b=0, expert_mb=1, spec="mtp_dense", dense=True)),
 "Qwen3.6-35B-A3B": (18.2, dict(total_b=35, active_b=2.97, routed_total_b=32.2, routed_active_b=1.0, expert_mb=mb(2048, 512), spec="mtp")),
 "Nemotron 3.5 Bolt 74B-A9B": (21.4, dict(total_b=74, active_b=8.7, routed_total_b=67, routed_active_b=1.74, expert_mb=2.72, spec="mtp")),
 "GLM-5.3-Flash 320B-A18B": (42, dict(total_b=320, active_b=18, routed_total_b=305, routed_active_b=8.47, expert_mb=mb(4096, 2048), spec="mtp")),
 "Ling-3.1-flash 560B-A25B": (41, dict(total_b=560, active_b=25, routed_total_b=547, routed_active_b=8.6, expert_mb=mb(4096, 1280), spec="mtp")),
 "DeepSeek-V4.1-Flash-1017 552B-A16B": (45, dict(total_b=552, active_b=16, routed_total_b=544, routed_active_b=7.5, expert_mb=mb(5120, 2304), spec="mtp")),
 "GLM-5.3 744B-A40B": (45, dict(total_b=744, active_b=40, routed_total_b=720, routed_active_b=20.1, expert_mb=mb(6144, 2048), spec="mtp")),
 "Beam 1.0 501B-A23B": (30, dict(total_b=501, active_b=23.4, routed_total_b=493, routed_active_b=15.4, expert_mb=mb(5120, 2560), spec="mtp")),
 "Muse Spark Open 816B-A42B": (39, dict(total_b=816, active_b=42, routed_total_b=799, routed_active_b=25, expert_mb=mb(8192, 4096), spec="mtp")),
 "MiMo-V2.6-Pro 1.02T-A42B": (46, dict(total_b=1020, active_b=42, routed_total_b=999, routed_active_b=20.8, expert_mb=mb(6144, 2048), spec="mtp")),
 "Kimi K3 2.8T-A104B": (44, dict(total_b=2780, active_b=104, routed_total_b=2720, routed_active_b=43.2, expert_mb=mb(7168, 3072), spec="mtp")),
}

def est(m, budget):
    if m.get("dense"):
        gb = m["total_b"] * ms.BPP
        if budget is not None and gb + 1.5 > budget: return None
        plain = 0.75 * ms.BW / (gb + 0.2)                 # big dense GEMVs run closer to the bandwidth than hybrid MoE
        return plain, plain * 2.5                          # MTP / DFlash2 on a dense target (verify ~ one read)
    fits = m["total_b"] * ms.BPP + 1.5 <= (budget if budget is not None else 50)
    if budget is None and not fits: return None
    p, s, mode = ms.speed(m, budget if budget is not None else 999)
    return None if p is None else (p, s)

print(f"{'model':38s} {'AA':>5s} {'4-bit GB':>8s} | {'RAM <=50 GB':>13s} | {'24 GB':>13s} | {'12 GB':>13s}")
for name, (aa, m) in M.items():
    row = []
    for b in (None, 24, 12):
        r = est(m, b)
        row.append("   -   " if r is None else f"{r[0]:5.1f} / {r[1]:5.1f}")
    print(f"{name:38s} {aa:5.1f} {m['total_b'] * ms.BPP:8.0f} | {row[0]:>13s} | {row[1]:>13s} | {row[2]:>13s}")
