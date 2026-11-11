import numpy as np, sys
for p in sys.argv[1:]:
    print("==", p)
    d = np.load(p, allow_pickle=False)
    for k in d.files:
        a = d[k]
        extra = ""
        if a.size and a.dtype.kind in "iuf":
            extra = f"min={a.min()} max={a.max()}"
        print(f"  {k}: shape={a.shape} dtype={a.dtype} {extra}")
