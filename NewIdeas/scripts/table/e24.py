import numpy as np, glob, math, json, os
geo = lambda xs: math.exp(sum(math.log(x) for x in xs) / len(xs))
d0 = "/Users/kamen/BigMoE/data/e24/e24a"
f = sorted(glob.glob(d0 + "/p*.npz"))
z = np.load(f[0])
print("keys:", list(z.keys())[:20])
res = {}
for run in ("e24a", "e24b"):
    for p in sorted(glob.glob(f"/Users/kamen/BigMoE/data/e24/{run}/p*.npz")):
        z = np.load(p)
        name = os.path.basename(p)
        t = None
        for k in ("dec_s", "decode_s", "t_dec", "dec", "wall"):
            if k in z:
                t = float(z[k]); break
        n = None
        for k in ("n", "tokens", "out"):
            if k in z:
                v = z[k]; n = int(v) if v.ndim == 0 else int(v.shape[-1]); break
        res.setdefault(run, {})[name] = (n, t)
for run, d in res.items():
    vals = [(nm, n, t) for nm, (n, t) in d.items()]
    print(run, len(vals), vals[:2])
