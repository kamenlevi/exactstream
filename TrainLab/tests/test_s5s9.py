"""CPU test for sticky/s5s9.py on synthetic per-position data (4 parts x 8 sequences x 50 positions; ar4 = v3 / 0.55, so v3
sits at x0.55 of ar4 as in turn18):
A. HC8 = v3 x 1.02, top-8 = v3 x 1.4 (x0.77 of ar4), v3 from 'turn18' -> S5 FREE, S9 PASS, x of v3 printed right.
B. HC8 = v3 x 1.08, top-8 = v3 x 2.5 (x1.38 of ar4), v3 in the run itself -> S5 NOT FREE, S9 FAIL.
C. R-26: a DYNK arm q060 run on 3 parts only (ifeval / code / multi, its own part order), v3 from a 4-part 'turn18':
   v3's part indices are mapped by name (gsm left out), q060 = v3 x 1.3 -> PASS with x of v3 1.30; x 2.5 -> FAIL.
usage: [BigMoE venv] python tests/test_s5s9.py"""
import os, re, subprocess, sys, tempfile
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
PARTS = np.array(["tf_gsm", "tf_ifeval", "tf_code", "tf_multi"])
ok_all = True


def check(name, cond):
    global ok_all
    ok_all &= bool(cond)
    print(f"{'ok  ' if cond else 'FAIL'} {name}")


def num(pat, o, i=0):
    f = re.findall(pat, o)
    return float(f[i]) if len(f) > i else float("nan")


def arm(name, kld, agree, seq, pos, part):
    return {f"{name}__kld": kld.astype(np.float32), f"{name}__agree": agree.astype(np.uint8), f"{name}__seq": seq,
            f"{name}__pos": pos, f"{name}__part": part}


def case(hc, tk, v3_in_run):
    g = np.random.default_rng(3)
    seq = np.repeat(np.arange(32, dtype=np.int32), 50)
    pos = np.tile(np.arange(100, 150, dtype=np.int32), 32)
    part = (seq // 8).astype(np.int8)
    v3 = g.lognormal(-6, 1.0, seq.size)
    agree = (g.random(seq.size) > 0.03).astype(np.uint8)
    noise = lambda: 1 + 0.01 * g.standard_normal(seq.size)
    d = tempfile.mkdtemp(dir=os.environ.get("TMPDIR"))
    run = {"parts": PARTS, **arm("p4433333333h", v3 * hc * noise(), agree, seq, pos, part),
           **arm("p44333333", v3 * tk * noise(), agree, seq, pos, part),
           **arm("p44333333h", v3 * tk * hc * noise(), agree, seq, pos, part)}
    v3a = arm("p4433333333", v3, agree, seq, pos, part)
    if v3_in_run:
        run.update(v3a)
    else:
        np.savez(f"{d}/budget125_tf_R4_plan4433333333_mtp_b2_pos.npz", parts=PARTS, **v3a)
    np.savez(f"{d}/budget125_tf_R4_test_pos.npz", **run)
    np.savez(f"{d}/ar4.npz", seq=seq, pos=pos, kld=(v3 / 0.55).astype(np.float32), agree=agree)
    r = subprocess.run([PY, os.path.join(HERE, "..", "sticky", "s5s9.py"), "test"], capture_output=True, text=True,
                       env=dict(os.environ, TL_DATA=d, TL_AR4=f"{d}/ar4.npz", PYTHONDONTWRITEBYTECODE="1"))
    print(r.stdout + r.stderr)
    return r.stdout, open(f"{d}/s5s9_test.txt").read() if r.returncode == 0 else ""


o, f = case(1.02, 1.4, False)
check("A: v3 taken from turn18", "v3 from turn18 (b2)" in o)
check("A: HC8 FREE at x1.02", abs(num(r"S5 HC8: v3 \+ HC8 = x([\d.]+) of v3", o) - 1.02) < 0.005
      and "FREE (<= 5 %)" in o and "NOT FREE" not in o)
check("A: top-8 passes the strict rules at x0.77 of ar4", "S9 top-8: the strict rules PASS" in o and "v3 + top-8        PASS" in o)
mp = r"mean of the parts x([\d.]+)"
check("A: x of v3 printed (HC8 x1.02, top-8 x1.40, both x1.43, HC8 on top-8 x1.02)",
      max(abs(num(mp, o, i) - w) for i, w in enumerate((1.02, 1.4, 1.428))) < 0.005
      and abs(num(r"\| mean x([\d.]+)", o) - 1.02) < 0.005)
check("A: the output file = stdout", f.strip() == o.strip())
o, f = case(1.08, 2.5, True)
check("B: v3 taken from the run", "v3 from this run" in o and "v3                PASS" in o)
check("B: HC8 NOT FREE at x1.08", abs(num(r"S5 HC8: v3 \+ HC8 = x([\d.]+) of v3", o) - 1.08) < 0.005
      and "NOT FREE (> 5 %)" in o)
check("B: top-8 fails the strict rules at x1.38 of ar4", "S9 top-8: the strict rules FAIL" in o)


def case_q(qmul):
    g = np.random.default_rng(4)
    seq = np.repeat(np.arange(32, dtype=np.int32), 50)
    pos = np.tile(np.arange(100, 150, dtype=np.int32), 32)
    part = (seq // 8).astype(np.int8)                                  # turn18's order: gsm, ifeval, code, multi
    v3 = g.lognormal(-6, 1.0, seq.size)
    agree = (g.random(seq.size) > 0.03).astype(np.uint8)
    d = tempfile.mkdtemp(dir=os.environ.get("TMPDIR"))
    np.savez(f"{d}/budget125_tf_R4_plan4433333333_mtp_b2_pos.npz", parts=PARTS, **arm("p4433333333", v3, agree, seq, pos, part))
    m = part >= 1                                                      # R-26: ifeval / code / multi, in its own order
    np.savez(f"{d}/budget125_tf_R4_test_pos.npz", parts=PARTS[1:], **arm(
        "q060", v3[m] * qmul * (1 + 0.01 * g.standard_normal(m.sum())), agree[m], seq[m], pos[m], (part[m] - 1).astype(np.int8)))
    np.savez(f"{d}/ar4.npz", seq=seq, pos=pos, kld=(v3 / 0.55).astype(np.float32), agree=agree)
    r = subprocess.run([PY, os.path.join(HERE, "..", "sticky", "s5s9.py"), "test"], capture_output=True, text=True,
                       env=dict(os.environ, TL_DATA=d, TL_AR4=f"{d}/ar4.npz", PYTHONDONTWRITEBYTECODE="1"))
    print(r.stdout + r.stderr)
    return r.stdout


o = case_q(1.3)
check("C: q060 on 3 parts, v3 from a 4-part turn18: both lines hold exactly ifeval / code / multi (400 positions, "
      "8 sequences each), q060 PASS, x of v3 1.30, the R-26 verdict line",
      re.search(r"v3 \+ DYNK 0\.060 +PASS", o) and len(re.findall(r"\(n 400, 8 seqs\)", o)) == 6 and "gsm: KLD" not in o
      and abs(num(mp, o) - 1.3) < 0.01 and "R-26 v3 + DYNK 0.060: the strict rules PASS in the parts run (ifeval, code, multi" in o)
o = case_q(2.5)
check("C: q060 at x2.5 of v3 (x1.38 of ar4) fails", "R-26 v3 + DYNK 0.060: the strict rules FAIL" in o)
print("ALL OK" if ok_all else "SOME FAILED")
