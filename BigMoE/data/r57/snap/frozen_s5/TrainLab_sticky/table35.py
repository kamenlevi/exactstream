"""sticky35's results (data/sticky35_results.json) as one table: per policy U(16), U(4), switch rate, KLD (prose / gsm /
code) and the chat segment, per-layer MoE output error, and the KLD of the zero-shot caps at equal union (log-
interpolated between caps 1-6, anchored at natural routing with the control's floor KLD). CPU. -> stdout"""
import json, os, re
import numpy as np

r = json.load(open(os.path.expanduser("~/TrainLab/data/sticky35_results.json")))


def sw(x):
    m = re.search(r"switch ([0-9.]+)", x.get("cache") or "")
    return float(m.group(1)) if m else float("nan")


def at(u, Z):
    pts = sorted(Z)
    for (u0, k0), (u1, k1) in zip(pts, pts[1:]):
        if u0 <= u <= u1:
            f = (u - u0) / (u1 - u0) if u1 > u0 else 0.0
            return float(np.exp(np.log(k0) + f * (np.log(k1) - np.log(k0))))
    return None


floor = max(r["control (fp32 W0) exact"]["kld"], 1e-4) if "control (fp32 W0) exact" in r else 1e-4
caps = [f"zero-shot cap{m}" for m in range(1, 7) if f"zero-shot cap{m}" in r]
Z16 = [(r[c]["U16"], r[c]["kld"]) for c in caps] + [(r["zero-shot exact"]["U16"], floor)]
Z4 = [(r[c]["U4"], r[c]["kld"]) for c in caps] + [(r["zero-shot exact"]["U4"], floor)]
print(f"| policy | U(16) | U(4) | switch | KLD prose/gsm/code | chat | per-layer err | caps' KLD at equal union -> cut |")
print("|---|---|---|---|---|---|---|---|")
for tag, x in r.items():
    if "band" in tag:
        continue
    four = any(s in tag for s in ("budget4", "oea4"))
    z = at(x["U4"] if four else x["U16"], Z4 if four else Z16)
    own = tag.startswith("zero-shot cap") or tag == "zero-shot exact"
    cut = "" if own or not z or x["kld"] <= 0 else f"{z:.4f} at equal U({4 if four else 16}) -> {z / x['kld']:.2f}x"
    print(f"| {tag} | {x['U16']:.1f} | {x['U4']:.1f} | {sw(x):.3f} | {x['kld']:.4f} | {x['per'].get('chat', float('nan')):.4f} | "
          f"{x['err']:.5f} | {cut} |")
bands = [t for t in r if "band" in t]
if bands:
    print("\nbands (KLD with the change in one band of 10 layers only):")
    for t in bands:
        print(f"  {t}: {r[t]['kld']:.4f}")
