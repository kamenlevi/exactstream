"""REF2b merge check (Leap, Fri 2026-10-30; runner 42): bitwise, ref2.py's merged key-bucket _verify_sdpa == ref2_kb's wrapper (STEP 4c) with buckets on, and == ref2's
128-key split alone (old behaviour, via ref2_kb with KB off) with key_buckets off; synthetic 16-row calls"""
import os, sys
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_p4s/Leap_scripts"))
import mlx.core as mx
import ref2, ref2_kb

ref2._ORIG["vsdpa"] = ref2.Q._verify_sdpa                 # MTPLX's own (ref2 not applied: no other patches)
ref2._ON[0] = True
merged = ref2._verify_sdpa
ref2_kb.install()                                         # ref2._verify_sdpa -> ref2_kb's; _R2["fn"] = merged
eq = lambda a, b: bool(mx.all(mx.view(a.reshape(-1), mx.uint16) == mx.view(b.reshape(-1), mx.uint16)).item())
mx.random.seed(1)
res = []
for N in (100, 120, 127, 300, 500, 505, 512, 600, 1015, 1020, 1100):
    q = (mx.random.normal((1, 24, 16, 256)) * 0.8).astype(mx.bfloat16)
    k = (mx.random.normal((1, 2, N, 256)) * 0.8).astype(mx.bfloat16)
    v = mx.random.normal((1, 2, N, 256)).astype(mx.bfloat16)
    qpos = mx.arange(N - 16, N); tpos = mx.arange(N)
    mask = (tpos[None, :] <= qpos[:, None])[None, None]
    ref2.CFG["key_buckets"] = True; ref2_kb.KB["on"] = True
    a = merged(q, k, v, scale=0.0625, mask=mask)
    b = ref2_kb._verify_sdpa(q, k, v, scale=0.0625, mask=mask)
    ref2.CFG["key_buckets"] = False
    c = merged(q, k, v, scale=0.0625, mask=mask)
    ref2_kb.KB["on"] = False                              # -> _R2["fn"] = merged with key_buckets off: same function
    d = ref2.Q._verify_sdpa(q, k, v, scale=0.0625, mask=mask)   # MTPLX's plain call
    mx.eval(a, b, c, d)
    res.append((N, eq(a, b), eq(c, d) if N < 112 or N >= 128 else "split"))
print("N, merged == ref2_kb (buckets on), merged w/o buckets == MTPLX plain (no class crossing):", res)
print("kb_pad", ref2.COUNTS["kb_pad"], "sdpa_split", ref2.COUNTS["sdpa_split"])
