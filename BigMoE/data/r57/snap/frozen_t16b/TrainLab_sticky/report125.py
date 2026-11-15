"""budget125_tf's results (data/budget125_tf_R4.json) per arm and Eval-suite part, against the per-part target (the
accepted build ar4's teacher-forced KLD x 1.10, margins.json) and against frugal-10 %'s step KLD (w2f_r10's report),
with the exact 4-row floor's KLD next to them and the distinct experts per pass vs natural. CPU. -> stdout
usage: report125.py [R] [variant]   (variant as budget125_tf's TL_VARIANT, default ref)"""
import json, os, sys

R = int(sys.argv[1]) if len(sys.argv) > 1 else 4
VT = sys.argv[2] if len(sys.argv) > 2 else "ref"                  # e.g. plan4433333333 (turn15's combined screen)
r = json.load(open(os.path.expanduser(f"~/TrainLab/data/budget125_tf_R{R}{'' if VT == 'ref' else '_' + VT}.json")))
TARGET = {"tf_code": 0.0023 * 1.10, "tf_gsm": 0.00408 * 1.10, "tf_ifeval": 0.01856 * 1.10, "tf_multi": 0.01868 * 1.10}
FRUGAL10 = {"tf_code": 0.01469, "tf_gsm": 0.03068, "tf_ifeval": 0.11532, "tf_multi": 0.09244}
parts = ["tf_code", "tf_gsm", "tf_ifeval", "tf_multi"]
print(f"| arm | distinct experts / pass / layer (natural) | " + " | ".join(
    f"{p[3:]} KLD (target {TARGET[p]:.4f}, frugal-10 {FRUGAL10[p]:.4f})" for p in parts) + " |")
print("|---|---|" + "---|" * len(parts))
for arm, x in r.items():
    u = x.get("_union")
    cells = []
    for p in parts:
        if p not in x:
            cells.append("-")
            continue
        k = x[p]["kld"]
        flag = "<= target" if k <= TARGET[p] else ("<= frugal-10" if k <= FRUGAL10[p] else "> frugal-10")
        cells.append(f"{k:.4f} top-1 {x[p]['agree']:.3f} n {x[p]['n']} ({flag})")
    uni = f"{u['used']:.1f} ({u['natural']:.1f}, {u['used'] / u['natural']:.2f}x)" if u else "natural"
    print(f"| {arm} | {uni} | " + " | ".join(cells) + " |")
