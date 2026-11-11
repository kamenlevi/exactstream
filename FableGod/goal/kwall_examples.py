"""kwall_examples.py: the code stops of CD16 (convention B) on MBPP, to read. python -I."""
import importlib.util, os, sys, math, random
HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("kwr", f"{HERE}/kwall_replay.py"); K = importlib.util.module_from_spec(spec); spec.loader.exec_module(K)
sets = K.load_sets()
random.seed(1)
for label, S in sets[:2]:
    cen = []
    K.run(S, K.CD16, "B", census=cen)
    cm = [c for c in cen if c["stop"][0].startswith("code")]
    print(f"\n=== {label}: {len(cm)} code stops of {len(cen)} checks")
    pairs = {}
    for c in cm:
        x, q = c["x"], c["q"]
        i = x["idx"][c["a"]]
        d = int(x["dt"][i][c["k"]])
        key = (K.dec(x["ids"][q]), K.dec(d))
        pairs[key] = pairs.get(key, 0) + 1
    top = sorted(pairs.items(), key=lambda kv: -kv[1])[:40]
    print("  most frequent (125B token -> draft token): " + "; ".join(f"{a!r}->{b!r} x{n}" for (a, b), n in top))
    random.shuffle(cm)
    for c in cm[:70 if label.startswith("MBPP") else 30]:
        x, q = c["x"], c["q"]
        i = x["idx"][c["a"]]
        d = int(x["dt"][i][c["k"]])
        ctx = "".join(K.dec(t) for t in x["ids"][max(x["gen"], q - 14):q])
        nxt = "".join(K.dec(t) for t in x["ids"][q + 1:q + 6])
        dn = "".join(K.dec(int(t)) for t in x["dt"][i][c["k"] + 1:c["k"] + 5])
        p1 = K.p1_of(x, q); dl, rk = K.PI.top8(x, q, d)
        print(f"  [{c['stop'][0][6:]:10s} k={c['k']:2d} p1={'%.2f' % p1 if p1 is not None else ' n/a'} rank={rk if rk < 99 else '>8'}] "
              f"{ctx[-60:]!r} | 125B {K.dec(x['ids'][q])!r}+{nxt!r} | draft {K.dec(d)!r}+{dn!r}")
