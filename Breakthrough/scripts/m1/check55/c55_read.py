"""c55_read.py (Breakthrough, CHECK55 step 0): the attribution table from a c55_profile.py run, against the pre-registration
(docs/CHECK55_PROFILE.md). Usage: [venv] python -I -B c55_read.py <tag>  -> data/m1/check55/<tag>_read.txt"""
import json
import os
import sys

import numpy as np

TAG = sys.argv[1]
D = os.path.expanduser("~/Breakthrough/data/m1/check55")
r = json.load(open(os.path.join(D, TAG + ".json")))
L = []


def say(s=""):
    print(s, flush=True)
    L.append(s)


T, K, KL, FL = r["T"], r["K"], r["K_layers"], r["floors"]
arms = [a for a in ("A16", "C32", "A16k", "C32k") if a in T]
ms = lambda v: 1e3 * v
# Leap's groups (the orchestrator's list) from the pass-K classes
GROUPS = [("GDN core", ["gdn_core"]), ("attention core", ["att_core"]), ("dense projections (GDN + attention)", ["gdn_proj", "att_proj"]),
          ("MoE experts", ["moe_experts"]), ("MoE dense side (router + shared + tail)", ["moe_router", "moe_shared", "moe_tail"]),
          ("HC (read + write)", ["hc_read", "hc_write"]), ("head (mixer + lm_head)", ["head_mixer", "head_lm"]), ("PLE", ["ple"]),
          ("glue (embed, tile, between classes)", ["other"]), ("read + commit", ["read", "commit"])]
CI = ["disp", "gpu", "gpu_net", "build", "enc", "flushes", "cbs"]


def cls(a, c, f):
    return K[a][c][CI.index(f)]


say(f"CHECK55 step 0 read-out of {TAG} ({r['when']}; {r['kinds']} slice, {r['rounds']} timed rounds, {r['krounds']} attribution "
    f"rounds; SB {r['sbn']}; gdn_body {r['gdn_body']})")
say("")
say("## 1. Time budget per check (pass T, medians; ms)")
say("| arm | wall | CPU build | CPU encode (SB) | CPU read phase | CPU commit phase | CPU total | GPU busy | GPU idle | GPU first start | "
    "floor | GPU busy / floor | CPU / wall | dispatches | CPU us per dispatch |")
say("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
CPU = {}
for a in arms:
    t = T[a]
    cpu = t["cpu_build"] + t["cpu_enc_sb"] + t["cpu_read"] + t["cpu_commit"]
    CPU[a] = cpu
    say(f"| {a} | {ms(t['wall']):.2f} | {ms(t['cpu_build']):.2f} | {ms(t['cpu_enc_sb']):.2f} | {ms(t['cpu_read']):.2f} | {ms(t['cpu_commit']):.2f} | "
        f"{ms(cpu):.2f} | {ms(t['gpu']):.2f} | {ms(t['idle']):.2f} ({t['idle'] / t['wall']:.0%}) | {ms(t['gpu_first']):.2f} | "
        f"{FL[a]['floor_ms']:.2f} | {ms(t['gpu']) / FL[a]['floor_ms']:.2f} | {cpu / t['wall']:.2f} | {t['disp']:.0f} | {1e6 * cpu / t['disp']:.1f} |")
say("(CPU = the Python thread's CPU time: graph build, MLX encode, the read and the commit. Waiting on the GPU costs none.)")
say("")
say("## 2. By class (pass K: each class flushed into its own command buffers; per check)")
for a in arms:
    gk = sum(cls(a, c, "gpu") for c in K[a])
    sc = ms(T[a]["gpu"]) / gk if gk else 1.0
    ek = sum(cls(a, c, "enc") for c in K[a])
    enc_t = ms(T[a]["cpu_enc_sb"] + T[a]["cpu_read"])        # production encode ~ SB submits + the read phase's tail encode
    se = enc_t / ek if ek else 1.0
    say(f"### {a}: wall {ms(T[a]['wall']):.2f} ms; GPU scale pass T / pass K {sc:.2f}; encode scale {se:.2f}")
    say("| group | dispatches | GPU ms (scaled) | CPU build ms | CPU encode ms (scaled) | CPU ms | share of CPU | share of GPU |")
    say("|---|---|---|---|---|---|---|---|")
    tot_c = sum(cls(a, c, "build") + se * cls(a, c, "enc") for c in K[a])
    tot_g = sum(sc * cls(a, c, "gpu") for c in K[a])
    for g, cs in GROUPS:
        d = sum(cls(a, c, "disp") for c in cs)
        gg = sum(sc * cls(a, c, "gpu") for c in cs)
        b = sum(cls(a, c, "build") for c in cs)
        e = sum(se * cls(a, c, "enc") for c in cs)
        say(f"| {g} | {d:.0f} | {gg:.2f} | {b:.2f} | {e:.2f} | {b + e:.2f} | {(b + e) / tot_c:.0%} | {gg / tot_g:.0%} |")
    say(f"| total | {sum(cls(a, c, 'disp') for c in K[a]):.0f} | {tot_g:.2f} | {sum(cls(a, c, 'build') for c in K[a]):.2f} | "
        f"{sum(se * cls(a, c, 'enc') for c in K[a]):.2f} | {tot_c:.2f} | | |")
    say("")
say("## 3. The 17-row step (C32 - A16, and C32k - A16k): where the extra time goes")
for a16, c32 in (("A16", "C32"), ("A16k", "C32k")):
    if a16 not in T or c32 not in T:
        continue
    dW = ms(T[c32]["wall"] - T[a16]["wall"])
    dC = ms(CPU[c32] - CPU[a16])
    dG = ms(T[c32]["gpu"] - T[a16]["gpu"])
    dI = ms(T[c32]["idle"] - T[a16]["idle"])
    say(f"### {c32} - {a16}: wall +{dW:.2f} ms (x{T[c32]['wall'] / T[a16]['wall']:.3f}); CPU +{dC:.2f}; GPU busy +{dG:.2f}; GPU idle "
        f"{dI:+.2f}; dispatches +{T[c32]['disp'] - T[a16]['disp']:.0f}")
    sc32 = ms(T[c32]["gpu"]) / sum(cls(c32, c, "gpu") for c in K[c32])
    sc16 = ms(T[a16]["gpu"]) / sum(cls(a16, c, "gpu") for c in K[a16])
    say("| group | +dispatches | +GPU ms (scaled) | +CPU build ms | +CPU encode ms (raw) |")
    say("|---|---|---|---|---|")
    for g, cs in GROUPS:
        dd = sum(cls(c32, c, "disp") - cls(a16, c, "disp") for c in cs)
        dg = sum(sc32 * cls(c32, c, "gpu") - sc16 * cls(a16, c, "gpu") for c in cs)
        db = sum(cls(c32, c, "build") - cls(a16, c, "build") for c in cs)
        de = sum(cls(c32, c, "enc") - cls(a16, c, "enc") for c in cs)
        say(f"| {g} | {dd:+.0f} | {dg:+.2f} | {db:+.2f} | {de:+.2f} |")
    say("")
say("## 4. Dispatches per layer (pass K), by layer kind")
for a in arms:
    say(f"- {a}: " + ", ".join(f"{k} {v:.1f}" for k, v in sorted(KL[a].items(), key=lambda z: (z[0][0], -z[1])) if v >= 0.5))
say("")
say("## 5. Kernels per class (pass K, dispatches per check, top names)")
for a in arms:
    say(f"- {a}:")
    for c, lst in r["K_kernels"][a].items():
        if lst:
            say(f"  - {c}: " + "; ".join(f"{nm[:50]} {v:.1f}" for nm, v in lst))
say("")
say(f"calibration: {r['cal']}; bits K-73r vs stock (info): {r['bits']}; floors: " + "; ".join(
    f"{a} {FL[a]['dense_gb']:.2f} + {FL[a]['experts_gb']:.2f} GB (union {FL[a]['union_mean']:.1f}) = {FL[a]['floor_ms']:.2f} ms" for a in arms))
open(os.path.join(D, TAG + "_read.txt"), "w").write("\n".join(L) + "\n")
