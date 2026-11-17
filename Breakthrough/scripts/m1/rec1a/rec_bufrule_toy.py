"""rec_bufrule_toy.py (Breakthrough, Tue 17 Nov 2026 01:3x; MODULE 1a milestone 2 groundwork, docs/MODULE1A.md):
per-call input arrays found and patched THROUGH THE RECORD, pre-registered in docs/MODULE1A.md (Tue 01:3x).
  y = x * offs[0] + offs[2],  offs = mx.array([st(s), 7, s]) built inside the call,  st(s) = ((s * 48 + 7) mod 128) * 17
(1) a contents diff of the small bound buffers (<= 4 KB) between records at s = 5, 6 finds both seq words of offs;
(2) with buffer-word rules (and x written through its recorded binding) the mode-2 and ICB replays are bitwise == MLX
on 16 new s; (3) the stale control misses 16/16. Tiny memory."""
import os
import resource
import struct
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mtlrec as R
R.load()
import mlx.core as mx
import numpy as np

mx.set_cache_limit(0)
N, NL, LL, SMALL = 4096, 48, 7, 4096
SEQS = list(range(7, 23))
rng = np.random.default_rng(20261117)
OPC = {v: k for k, v in R.OPN.items()}


def st(s):
    return ((s * NL + LL) % 128) * 17


def f(s, x):
    offs = mx.array([st(s), 7, s], dtype=mx.int32)       # built per call, like the engine's offs
    o = offs.astype(mx.float32)
    return x * o[0] + o[2]


def u8(a):
    return np.frombuffer(memoryview(a).cast("B"), dtype=np.uint8)


def newx():
    x = mx.array(rng.standard_normal(N).astype(np.float32))
    mx.eval(x)
    return x


def record(s):
    x = newx()
    mx.synchronize()
    R.start()
    y = f(s, x)
    mx.eval(y)
    mx.synchronize()
    R.stop()
    ops = R.ops_dump()
    first = {}
    for i in np.where(ops[:, 0] == OPC["BUF"])[0]:
        first.setdefault(int(ops[i, 10]), int(i))           # canonical buffers in order of first binding
    bufs = []
    for c, (ptr, i) in enumerate(first.items()):
        L = R.buf_len(i)
        bufs.append(dict(c=c, op=i, len=L, data=R.buf_read(i) if 0 <= L <= SMALL else None))
    return dict(s=s, x=x, y=y, ops=ops, bufs=bufs, names=R.disp_names())


A = record(5)
B = record(6)
print(f"[bufr] kernels: {B['names']}")
print(f"[bufr] canonical buffers: {len(A['bufs'])} / {len(B['bufs'])}; lengths "
      f"{[b['len'] for b in B['bufs']]}")
same = (len(A["bufs"]) == len(B["bufs"]) and all(a["len"] == b["len"] for a, b in zip(A["bufs"], B["bufs"]))
        and np.array_equal(A["ops"][:, 0], B["ops"][:, 0]))
changed = []
for a, b in zip(A["bufs"], B["bufs"]):
    if a["data"] is None or b["data"] is None:
        continue
    for off in range(0, len(a["data"]) - len(a["data"]) % 4, 4):
        if a["data"][off:off + 4] != b["data"][off:off + 4]:
            changed.append(dict(c=a["c"], op=b["op"], off=off, i5=struct.unpack("<i", a["data"][off:off + 4])[0],
                                i6=struct.unpack("<i", b["data"][off:off + 4])[0],
                                f5=struct.unpack("<f", a["data"][off:off + 4])[0],
                                f6=struct.unpack("<f", b["data"][off:off + 4])[0]))
rules, other = [], []
for c in changed:
    if c["i5"] == st(5) and c["i6"] == st(6):
        rules.append((c["op"], c["off"], dict(a=NL, b=LL, m=128, s=17)))
    elif c["i5"] == 5 and c["i6"] == 6:
        rules.append((c["op"], c["off"], dict(a=1, b=0)))
    else:
        other.append(c)
for c in changed:
    print(f"[bufr] changed word: buffer {c['c']} (BUF op {c['op']}) off {c['off']}: int32 {c['i5']} -> {c['i6']}, "
          f"float32 {c['f5']:.6g} -> {c['f6']:.6g}")
line1 = same and len(rules) == 2 and len({r[0] for r in rules}) == 1
print(f"[bufr] (1) same structure {same}; {len(rules)} seq words matched to formulas (one buffer: "
      f"{len({r[0] for r in rules}) == 1}); {len(other)} other changed words (info) -> {'PASS' if line1 else 'FAIL'}")

# the base record = B (s = 6): find x's binding by its contents, build the ICB, register the buffer-word rules
xb = bytes(u8(B["x"]))
xops = [b["op"] for b in B["bufs"] if b["len"] >= len(xb) and R.buf_read(b["op"], 0, len(xb)) == xb]
print(f"[bufr] x's binding found by contents: BUF ops {xops}")
n_icb, err = R.build_icb()
print(f"[bufr] ICB: {n_icb} commands {('(error: ' + err + ')') if n_icb < 0 else ''}")
for op, off, kw in rules:
    print(f"[bufr] rule bufword32 op {op} off {off} {kw} -> {R.rule_add2('bufword32', op, off=off, **kw)}")
ok2 = oki = miss = 0
for s in SEQS:
    xn = newx()
    ref = f(s, xn)
    mx.eval(ref)
    rb = u8(ref).copy()
    R.buf_write(xops[0], bytes(u8(xn)))
    R.replay(2)                                            # control: the previous check's words
    miss += not np.array_equal(u8(B["y"]), rb)
    R.rules_apply(s)
    rc2, _ = R.replay(2)
    m2 = rc2 == 0 and np.array_equal(u8(B["y"]), rb)
    u8(B["y"])[:] = 0
    rci, _ = R.replay_icb()
    mi = rci == 0 and np.array_equal(u8(B["y"]), rb)
    ok2 += m2
    oki += mi
    print(f"[bufr] s {s} (st {st(s)}): mode 2 {m2}, ICB {mi}")
line2 = len(xops) == 1 and ok2 == len(SEQS) and oki == len(SEQS)
line3 = miss == len(SEQS)
print(f"[bufr] (2) mode 2 {ok2}/{len(SEQS)}, ICB {oki}/{len(SEQS)} -> {'PASS' if line2 else 'FAIL'}; "
      f"(3) control misses {miss}/{len(SEQS)} -> {'PASS' if line3 else 'FAIL'}")
verdict = "PASS" if (line1 and line2 and line3) else ("KILL" if not line2 else "FAIL")
print(f"[bufr] RESULT: {verdict}: (1) {line1} (2) {line2} (3) {line3}; lifetime peak "
      f"{resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30:.2f} GB")
