"""COLD-CUT's byte claim on the map's own blocks (Breakthrough, Tue 2026-11-03 19:4x; Fable C's UNION_STACK.md: HYB-COLD
4/0.05 + t3 reads x0.35 of EXACT's SSD bytes at 45 GB vs u8 + t3's x0.46, because cold experts are the ones that miss).
dm_cold_masks.py found the opposite without a cache on code (union x0.471 vs x0.440: COLD keeps 9.3 of an unsure row's 10
experts, u8 8.5). This replays the cache: per layer a plain LRU base pool of ~/MoonShots/data/laysz45_256.json's C_m (the
45 GB default, mean 295 of 512; NewIdeas' qom replay), carried across a domain's windows in walk order, the prompt rows'
top-10 warm it; each live check loads its union (the arm's base sets, the rows after the first miss by the stand-in p + 97)
and pays the MSB-3 misses x 2,162,688 B; the LSB-1 sidecar (ranks 0-1 of the 4-bit rows) has its own LRU of the same size
and pays 770,048 B per miss. Ratios vs EXACT under the same model. CPU, 1 core, < 1 min. No popularity prior in the LRU
(Fable C used LRU + 30 x popularity), so the cold experts' miss rate here is if anything higher, which favours COLD.
Usage (BigMoE venv): python dm_cold_lru.py -> out/COLD_lru.txt"""
import json, os
from collections import OrderedDict
import numpy as np
from dm_q1u_masks import OUT, base
from dm_cold_masks import MSB, LSB, RULES, windows, segments, prior
RULES = dict(RULES, oea_k8=(None, 8, None, False, None),  # chat context: OEA k8 at the same live checks (no live map run of it)
             u3=(0.9, 10, None, True, None))                # U3 = Q1-UNION 0.9 + t3 (unsure rows exact): the engine's FAST mode
CL = json.load(open(os.path.expanduser("~/MoonShots/data/laysz45_256.json")))["C_m"]
SFX = ""
SKIP = set()
if os.environ.get("COLD_HELDOUT"):                        # FLOOR's 78 minus the map's 36 (12b3d63)
    SKIP = set(json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "out", "baseline.json")))); SFX = "_heldout"
if os.environ.get("COLD_LRU_C"):                          # e.g. 51: the 12 GB engine's slots per layer (Tue 22:3x)
    CL = [int(os.environ["COLD_LRU_C"])] * 48; SFX = "_c" + os.environ["COLD_LRU_C"]

def miss(c, es, C):
    m = 0
    for e in es:
        if e in c: c.move_to_end(e)
        else:
            m += 1; c[e] = 1
            if len(c) > C: c.popitem(last=False)
    return m

lines = []
def say(s): print(s, flush=True); lines.append(s)
res = {}
for dom in ("code", "ifeval", "multi"):
    for arm, (thr, r, pth, t3, pk) in RULES.items():
        cm = [OrderedDict() for _ in range(48)]; cl = [OrderedDict() for _ in range(48)]; b_ = 0
        for d, i, nm, Z, ids, g in windows():
            if d != dom or nm in SKIP: continue
            P = prior(pk, d, nm, ids) if pk else None
            for p in range(base[nm]["a"]):                    # prompt rows (prefill, all ten at 4 bits) warm both pools
                for l in range(48): miss(cm[l], ids[p, l].tolist(), CL[l]); miss(cl[l], ids[p, l, :2].tolist(), CL[l])
            for rows, cf, nr in segments(Z, i, len(ids), g):
                sure = ~(cf < thr) if thr is not None else np.zeros(len(rows), bool)
                for l in range(48):
                    E = ids[rows, l]
                    b = np.zeros(E.shape, bool); b[sure, :2] = True; b[~sure, :r] = True
                    if pth is not None: b[~sure] |= P[l][E[~sure]] >= pth
                    b_ += miss(cm[l], np.unique(E[b]).tolist(), CL[l]) * MSB
                    b_ += miss(cl[l], np.unique(E[~sure if t3 else slice(None), :2]).tolist(), CL[l]) * LSB
        res[(dom, arm)] = b_
    ex = res[(dom, "exact")]
    say(f"{dom:6s} cached bytes vs EXACT ({'45 GB pools' if not SFX else 'C ' + SFX[2:] + ' per layer'}, plain LRU): " + "; ".join(f"{a} x{res[(dom, a)] / ex:.3f}" for a in RULES if a != "exact")
        + f"  (EXACT {ex / 1e9:.1f} GB over the domain's checks)")
say("Fable C's claim on code: HYB-COLD 4/0.05 + t3 x0.35 vs u8 + t3 x0.46 at 45 GB (cap45 replay, LRU + 30 x popularity)")
open(os.path.join(OUT, f"COLD_lru{SFX}.txt"), "w").write("\n".join(lines) + "\n")
json.dump({a: {d: res[(d, a)] / res[(d, "exact")] for d in ("code", "ifeval", "multi")} for a in RULES},
          open(os.path.join(OUT, f"COLD_lru{SFX}.json"), "w"), indent=1)
