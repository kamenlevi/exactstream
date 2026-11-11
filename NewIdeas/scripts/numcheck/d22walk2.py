import numpy as np, glob, os, json, collections
D = "/Users/kamen/TrainLab/data"
tj = json.load(open("/Users/kamen/BigMoE/models/bare/tokenizer.json"))
END = {t["id"] for t in tj.get("added_tokens", []) if t["content"] in ("<|im_end|>", "<|endoftext|>")}
def tpp(out, mtp, K):
    L = len(out)
    for i, t in enumerate(out):
        if t in END:
            L = i + 1; break
    else:
        L = len(out)
    p, tok, n = 1, 0, 0
    while p < L:
        k = 0
        while k < K and p + k < L and mtp[p, k] >= 0 and mtp[p, k] == out[p + k]:
            k += 1
        tok += k + 1; n += 1; p += k + 1
    return tok, n
for run, base in (("d22r_b3", "d17"), ("d22r_b4", "d17")):
    acc = collections.defaultdict(lambda: [0, 0, 0, 0, 0])
    for f in sorted(glob.glob(f"{D}/{run}/*.npz")):
        b = f"{D}/{base}/{os.path.basename(f)}"
        if not os.path.exists(b):
            continue
        z, s = np.load(f), np.load(b)
        assert list(z["out"]) == list(s["out"])
        dom = str(z["domain"])
        t1, n1 = tpp(list(z["out"]), z["mtp"], 15); t0, n0 = tpp(list(s["out"]), s["mtp"], 15)
        a = acc[dom]; a[0] += t1; a[1] += n1; a[2] += t0; a[3] += n0; a[4] += 1
    print(run, "vs", base, "K15:", " | ".join(f"{d} ({a[4]} answers) {a[2] / a[3]:.2f} -> {a[0] / a[1]:.2f} = x{(a[0] / a[1]) / (a[2] / a[3]):.3f}" for d, a in sorted(acc.items())))
print("END ids", sorted(END))
