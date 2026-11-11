"""VICTIM-TIER replay (NewIdeas, Tue 3 Nov 2026; CPU only): does BigMoE's planned volatile tier (PURGE-1: records evicted from
C 256's resident pools go to a per-layer LRU victim tier; a resident miss that hits the tier costs no SSD read and moves the
record back) reach the per-layer-fitted bigger pools' savings? Same machinery and inputs as band45_oea.py (cap45's live
checks, D-23 cf transplants, laysz45 replay rules: LRU + 30 x popularity, strict, res0 forced). Arms per rule, all six
prompts, SSD MB per kept token: C 256 alone; fitted C 300 / C 320 pools (laysz45_300 / _320); C 256 + a victim tier with
the same extra records per layer (C300 - C256, C320 - C256 for the base and sidecar pools). Writes nothing.
Usage (BigMoE venv): BAND45_SEEDS=2 python victim_tier.py"""
import os, sys, json, heapq, time
from collections import OrderedDict
import multiprocessing as mp
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import band45_oea as B  # noqa: E402  (module import only defines rules and helpers)
LZ = B.LZ
S256 = json.load(open(os.path.join(LZ.OUT, "laysz45_256.json")))
S300 = json.load(open(os.path.join(LZ.OUT, "laysz45_300.json")))
S320 = json.load(open(os.path.join(LZ.OUT, "laysz45_320.json")))


def replay2(task):
    """B.replay's rules for one layer's resident pool, plus an LRU victim tier of V records"""
    key, K, V, calls, pops = task
    miss = np.zeros(len(calls)); st, i = {}, 0; tier = OrderedDict()
    for p, cl in enumerate(calls):
        pop = pops[p]
        for t, need in cl:
            i += 1
            if t >= 0:
                for e in need:
                    if e not in st:
                        if e in tier: del tier[e]
                        else: miss[p] += 1
            else:
                for e in need: tier.pop(e, None)
            for e in need:
                st[e] = float(i)
            if len(st) > K:
                kf = lambda e: st[e] + 30.0 * pop[e]
                vic = heapq.nsmallest(len(st) - K, (e for e in st if e not in need), key=kf)
                if len(st) - len(vic) > K:
                    vic += heapq.nsmallest(len(st) - len(vic) - K, [e for e in st if e not in vic], key=kf)
                for e in vic:
                    del st[e]
                    if V > 0:
                        tier[e] = 1; tier.move_to_end(e)
                        if len(tier) > V: tier.popitem(last=False)
    return miss


def run(rules, seeds):
    # B.CACHE is laysz45.load()'s own output, written locally by MoonShots' / band45's pipeline (trusted, same as band45_oea)
    if os.path.exists(B.CACHE): import pickle; P, agr = pickle.load(open(B.CACHE, "rb"))
    else: P, agr = LZ.load()
    D = B.d23(); KEPT = np.array([sum(c["kept"] for c in cks) for _, cks, _ in P], float)
    arms = {"C 256": (S256, None), "C 300 fitted": (S300, None), "C 320 fitted": (S320, None),
            "C 256 + tier(300)": (S256, S300), "C 256 + tier(320)": (S256, S320)}
    if os.environ.get("VT_BASE"):                          # any base sizes file (e.g. uniform C 160) + VT_PLUS "64/32" slots
        SB = json.load(open(os.environ["VT_BASE"])); pm, pl = (int(x) for x in os.environ.get("VT_PLUS", "64/32").split("/"))
        U = {"C_m": [k + pm for k in SB["C_m"]], "C_l": [k + pl for k in SB["C_l"]]}
        arms = {"base": (SB, None), f"base + tier {pm}/{pl}": (SB, U)}
    if os.environ.get("VT_ARMS") == "224":                # BigMoE's tier smoke vt1_224: C 224 + 64 base / 32 sidecar slots
        S224 = json.load(open(os.path.join(LZ.OUT, "laysz45_224.json")))
        U = {"C_m": [k + 64 for k in S224["C_m"]], "C_l": [k + 32 for k in S224["C_l"]]}
        arms = {"C 224": (S224, None), "C 224 + tier 64/32": (S224, U), "C 256": (S256, None)}
    out = {}
    with mp.get_context("fork").Pool(2) as pool:
        for s in range(seeds):
            T = B.transplant(P, D, s)
            Qs = {r: B.needs(P, T, f) for r, f in rules.items()}
            POP = {key: LZ.popularity(Qs["EXACT"], key) for key in ("m", "l")}
            for r, Q in Qs.items():
                for an, (SZ, SX) in arms.items():
                    jobs = []
                    for key in ("m", "l"):
                        ck = "C_m" if key == "m" else "C_l"
                        for l in range(B.NL):
                            calls = [([(-1, res0[l])] if key == "m" else []) + [(t, c[key][l]) for t, c in enumerate(cks)]
                                     for _, cks, res0 in Q]
                            K = SZ[ck][l]; V = (SX[ck][l] - K) if SX else 0
                            jobs.append((key, K, max(V, 0), calls, [POP[key][p][l] for p in range(len(Q))]))
                    res = pool.map(replay2, jobs, chunksize=8)
                    mm = sum(res[: B.NL]); ll = sum(res[B.NL:])
                    mb = (mm * B.MR + ll * B.LR) / 1e6
                    out.setdefault((r, an), []).append(mb.sum() / KEPT.sum())
            print(f"  seed {s} done", flush=True)
    return out


if __name__ == "__main__":
    t0 = time.time(); seeds = int(os.environ.get("BAND45_SEEDS", "2"))
    rules = {r: B.RULES[r] for r in ("EXACT", "Q1 0.9") if r in B.RULES}
    out = run(rules, seeds)
    bk = "base" if os.environ.get("VT_BASE") else ("C 224" if os.environ.get("VT_ARMS") == "224" else "C 256")
    base = np.mean(out[("EXACT", bk)])
    print(f"| rule | arm | SSD MB per kept token (all six prompts) | vs {bk} EXACT |")
    print("|---|---|---|---|")
    for (r, an), v in out.items():
        print(f"| {r} | {an} | {np.mean(v):.1f} | {100 * (np.mean(v) / base - 1):+.1f} % |")
    print(f"({seeds} seeds, {time.time() - t0:.0f} s)")
