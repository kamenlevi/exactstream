"""CPU test of sticky/b10_extrap.py on synthetic per-position files with known answers (TL_DATA = a temp dir).
Isolated design: F = 0.3 V; layer 2 r = (2 - 0.01) / (1 - 0.01) with a dense floor x, layer 24 r = 1.5 ->
mean r 1.755, all layers x(0.3 + 1.755 x 0.7) = x1.529. vs v3: excesses +0.01 and +0.02 -> x(1 + 48 x 0.015) = x1.72.
usage: python tests/test_b10_extrap.py"""
import os, subprocess, sys, tempfile
import numpy as np

rng = np.random.default_rng(0)
parts = np.array(["tf_gsm", "tf_code"])
seq = np.repeat(np.arange(16), 20).astype(np.int32)
pos = np.tile(np.arange(100, 120), 16).astype(np.int32)
part = np.where(seq < 10, 0, 1).astype(np.int8)
kv = (0.002 * rng.gamma(2.0, 1.0, seq.size)).astype(np.float32)
V = {pi: kv[part == pi].mean() for pi in (0, 1)}
Q48 = np.array([0.7 * V[pi] / 48 for pi in part], np.float32)
w = rng.gamma(4.0, 0.25, seq.size).astype(np.float32)          # per-position damage spread, mean ~1
arm = {"p4444444444": dict(kld=0.3 * np.array([V[pi] for pi in part], np.float32), lkld=np.zeros(seq.size, np.float32)),
       "d2p4444444444": dict(lkld=Q48 * w), "t2p4444444444": dict(lkld=2 * Q48 * w), "x2p4444444444": dict(lkld=0.01 * Q48 * w),
       "d24p4444444444": dict(lkld=1.2 * Q48 * w), "t24p4444444444": dict(lkld=1.8 * Q48 * w),
       "t2": dict(kld=1.01 * kv), "t24": dict(kld=1.02 * kv)}
out = {"parts": parts}
for a, f in arm.items():
    f.setdefault("kld", kv)
    for k, x in dict(f, seq=seq, pos=pos, part=part).items():
        out[f"{a}__{k}"] = x
with tempfile.TemporaryDirectory() as td:
    np.savez(f"{td}/budget125_tf_R4_plan4433333333_mtp_b2_pos.npz", parts=parts, p4433333333__seq=seq, p4433333333__pos=pos,
             p4433333333__part=part, p4433333333__kld=kv)
    np.savez(f"{td}/budget125_tf_R4_test_pos.npz", **out)
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "sticky", "b10_extrap.py")
    r = subprocess.run([sys.executable, script, "test"], env=dict(os.environ, TL_DATA=td), capture_output=True, text=True)
    first = [k for k in out if k.startswith("p4444444444__")] + ["parts"]
    for L, extra in (("2", ("t2", "t24")), ("24", ())):            # the same arms split into one run per layer
        keep = [k for k in out if k.split("__")[0] in {f"d{L}p4444444444", f"t{L}p4444444444", f"x{L}p4444444444", *extra}]
        np.savez(f"{td}/budget125_tf_R4_testL{L}_pos.npz", **{k: out[k] for k in first + keep})
    r2 = subprocess.run([sys.executable, script, "testL2,testL24"], env=dict(os.environ, TL_DATA=td), capture_output=True,
                        text=True)
    merged_file = os.path.exists(f"{td}/b10_extrap_testLall.txt")
    bad = dict(out); bad["p4444444444__kld"] = out["p4444444444__kld"] * 1.001
    np.savez(f"{td}/budget125_tf_R4_testBad_pos.npz", **{k: bad[k] for k in first})
    r3 = subprocess.run([sys.executable, script, "testL2,testBad"], env=dict(os.environ, TL_DATA=td), capture_output=True,
                        text=True)
print(r.stdout, r.stderr)
ok = r.returncode == 0
same = r2.returncode == 0 and r2.stdout.split("\n")[1:] == r.stdout.split("\n")[1:]
print(f"{'ok  ' if same else 'FAIL'} one run per layer (merged) == one run; written as b10_extrap_testLall.txt: {merged_file}")
print(f"{'ok  ' if r3.returncode != 0 and 'differs between the runs' in r3.stderr else 'FAIL'} a reference arm that differs between runs stops the read-out")
ok &= same and merged_file and r3.returncode != 0
for want in ("x1.72 of v3", "excess +0.0100", "excess +0.0200", "floor F", "= 30% of v3", "r 2.01", "r 1.50",
             "mean r 1.76", "x1.53 of v3"):
    hit = r.stdout.count(want) >= 2                                  # both parts
    ok &= hit
    print(f"{'ok  ' if hit else 'FAIL'} {want!r} in both parts")
print("ALL OK" if ok else "SOME FAILED")
