"""rec_patch_toy.py (Breakthrough, Tue 17 Nov 2026 01:0x; MODULE 1a milestone 2 groundwork, docs/MODULE1A.md): per-check
patching of seq-derived setBytes words on a toy MLX graph, pre-registered in docs/MODULE1A.md (Tue 01:0x).
  y = x * arange(st(s), st(s) + N) + arange(s, s + N) + sum(offs),  st(s) = ((s * 48 + 7) mod 128) * 17
(1) the diff of two records (s = 5, 6) finds exactly the two arange start words; (2) with rules applied per check, the
mode-2 replay and the ICB replay are bitwise == MLX on 16 new seqs with new x and offs; (3) control: the previous check's
values do not match; (4) rec_rules_apply < 2 us per call from Python. Tiny memory."""
import os
import resource
import struct
import sys
import time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mtlrec as R
R.load()
import mlx.core as mx
import numpy as np

mx.set_cache_limit(0)
NL, LL, N = 48, 7, 4096
SEQS = list(range(7, 23))
rng = np.random.default_rng(20261117)


def st(s):
    return ((s * NL + LL) % 128) * 17


def f(s, x, offs):
    a = mx.arange(st(s), st(s) + N, dtype=mx.int32)
    b = mx.arange(s, s + N, dtype=mx.float32)
    return x * a.astype(mx.float32) + b + offs.astype(mx.float32).sum()


def inputs():
    x = mx.array(rng.standard_normal((N,)).astype(np.float32))
    offs = mx.array(rng.integers(0, 1000, 3).astype(np.int32))
    mx.eval(x, offs)
    return x, offs


def record(s):
    x, offs = inputs()
    mx.synchronize()
    R.start()
    y = f(s, x, offs)
    mx.eval(y)
    mx.synchronize()
    R.stop()
    return x, offs, y


def u8(a):
    return np.frombuffer(memoryview(a).cast("B"), dtype=np.uint8)


def peak_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30


# ---- (1) two records, diff the setBytes payloads word by word
_x5, _o5, _y5 = record(5)
d5, ops5, names5 = R.bytes_dump(), R.opcounts(), R.disp_names()
x6, o6, y6 = record(6)
d6, ops6, names6 = R.bytes_dump(), R.opcounts(), R.disp_names()
print(f"[ptoy] record s=5: {ops5}; s=6: {ops6}")
print(f"[ptoy] dispatch kernels (s=6): {names6}")
same_shape = (ops5 == ops6 and names5 == names6 and len(d5) == len(d6)
              and all(a["length"] == b["length"] and a["bind"] == b["bind"] and a["disp"] == b["disp"]
                      for a, b in zip(d5, d6)))
print(f"[ptoy] same op sequence, kernels and payload layout in both records: {same_shape}")
changed = []
for a, b in zip(d5, d6):
    for off in range(0, a["length"] - a["length"] % 4, 4):
        wa, wb = a["data"][off:off + 4], b["data"][off:off + 4]
        if wa != wb:
            changed.append(dict(k=a["k"], off=off, disp=a["disp"], kernel=names6[a["disp"]] if a["disp"] >= 0 else "?",
                                i5=struct.unpack("<i", wa)[0], i6=struct.unpack("<i", wb)[0],
                                f5=struct.unpack("<f", wa)[0], f6=struct.unpack("<f", wb)[0]))
for c in changed:
    print(f"[ptoy] changed word: setBytes k={c['k']} off={c['off']} (dispatch {c['disp']}: {c['kernel']}): "
          f"int32 {c['i5']} -> {c['i6']}, float32 {c['f5']:.6g} -> {c['f6']:.6g}")
rules, unexplained = [], []
for c in changed:
    if c["i5"] == st(5) and c["i6"] == st(6):
        rules.append((c["k"], c["off"], "int32", NL, LL, 128, 17, 0))
    elif c["f5"] == 5.0 and c["f6"] == 6.0:
        rules.append((c["k"], c["off"], "float32", 1, 0, 0, 1, 0))
    else:
        unexplained.append(c)
line1 = same_shape and len(changed) == 2 and len(rules) == 2 and not unexplained
print(f"[ptoy] (1) diff: {len(changed)} changed words, {len(rules)} matched to a seq rule, {len(unexplained)} "
      f"unexplained -> {'PASS' if line1 else 'FAIL'}")
for r in rules:
    n = R.rule_add(*r)
    print(f"[ptoy] rule: k={r[0]} off={r[1]} {r[2]} = ((seq*{r[3]} + {r[4]}) mod {r[5]}) * {r[6]} + {r[7]} -> {n} rules")

# ---- (2) + (3): new seqs, new inputs in place; control (previous values) then patched, mode 2 and ICB
n_icb, err = R.build_icb()
print(f"[ptoy] ICB build: {n_icb} commands {('(error: ' + err + ')') if n_icb < 0 else ''}")
ok2 = ok_icb = ctl_miss = 0
for s in SEQS:
    xn, on = inputs()
    ref = f(s, xn, on)
    mx.eval(ref)
    ref_np = np.array(ref).copy()
    u8(x6)[:] = u8(xn)
    u8(o6)[:] = u8(on)
    u8(y6)[:] = 0
    rc0, _ = R.replay(2)                         # control: still the previous check's seq words
    ctl = np.array_equal(np.array(y6), ref_np)
    ctl_miss += (not ctl)
    R.rules_apply(s)
    u8(y6)[:] = 0
    rc2, st2 = R.replay(2)
    m2 = rc2 == 0 and np.array_equal(np.array(y6), ref_np)
    mi = False
    if n_icb > 0:
        u8(y6)[:] = 0
        rci, sti = R.replay_icb()
        mi = rci == 0 and np.array_equal(np.array(y6), ref_np)
    ok2 += m2
    ok_icb += mi
    print(f"[ptoy] s={s} st={st(s)}: control (stale words) bitwise {ctl}; patched mode 2 {m2} (rc {rc2}); "
          f"patched ICB {mi}")
line2 = ok2 == len(SEQS) and ok_icb == len(SEQS)
line3 = ctl_miss == len(SEQS)
print(f"[ptoy] (2) patched replays bitwise == MLX: mode 2 {ok2}/{len(SEQS)}, ICB {ok_icb}/{len(SEQS)} -> "
      f"{'PASS' if line2 else 'FAIL'}")
print(f"[ptoy] (3) control misses: {ctl_miss}/{len(SEQS)} -> {'PASS' if line3 else 'FAIL'}")

# ---- (4) cost of one rules_apply call from Python
ts = []
for i in range(2000):
    t0 = time.perf_counter_ns()
    R.rules_apply(1000 + i)
    ts.append(time.perf_counter_ns() - t0)
ts = np.array(ts[200:]) / 1e3
line4 = float(np.median(ts)) < 2.0
print(f"[ptoy] (4) rules_apply per call ({len(rules)} rules, ctypes included): median {np.median(ts):.2f} us, "
      f"min {ts.min():.2f}, p99 {np.percentile(ts, 99):.2f} -> {'PASS' if line4 else 'FAIL'}")
R.rules_apply(SEQS[-1])
verdict = "PASS" if (line1 and line2 and line3 and line4) else ("KILL" if not line2 else "FAIL")
print(f"[ptoy] RESULT: {verdict}: (1) {line1} (2) {line2} (3) {line3} (4) {line4}; lifetime peak {peak_gb():.2f} GB")
