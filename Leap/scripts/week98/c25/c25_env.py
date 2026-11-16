"""c25_env.py (Leap, Mon 16 Nov 2026 12:5x; CHECK25): prints the c25 loop's environment as shell export lines.
= BigMoE's 1c-tiers set t1cx45t1c1 (data/stack/t1cx45t1c1_env.txt: lossy-45 + R57_TIERS=1c at lambda 1.6e-2; parsed as
EvalSuite's runner does, t1c_keytest.env_pairs) with run_env()'s extras (R57_PRESET 45, R57_REPS 1, R57_WARM 1, a tag, a
key), MINUS REF2_GDN_LAZY (v32 refuses it; it changes no bits outside GDNB), PLUS the G32T test mode as E14 ran it
(BIGMOE_MAXK 320, ext build_g32: the QOM-skip and the G32 overflow guard, BIGMOE_OVERFLOW_FLAG 1, V32_MOE1 1, the windows,
the rounds) and C25_ARMS. Windows default to E14's three G32T windows (dd__mc_27921, dd__mc_2753, dd__mc_28056).
usage: eval "$(python3 c25_env.py)"   (env: C25_ROUNDS, C25_WARM, C25_KEYS, C25_ARMS, C25_TAG override the defaults)"""
import os
import shlex

BM = os.path.expanduser("~/BigMoE")
ENVF = os.path.join(BM, "data/stack/t1cx45t1c1_env.txt")


def env_pairs(path):
    line = [l for l in open(path).read().splitlines() if l and not l.startswith("#")][-1]
    out = []
    for t in line.split(","):
        if "=" in t or not out:
            out.append(t)
        else:
            out[-1] += "," + t
    assert all("=" in t for t in out), path
    return [t.split("=", 1) for t in out]


env = dict(env_pairs(ENVF))
assert env.get("R57_TIERS") == "1c" and env.get("R57_FAST") == "QOM", env
env.pop("REF2_GDN_LAZY", None)
env.update(R57_PRESET="45", R57_REPS="1", R57_WARM="1", R57_TAG=os.environ.get("C25_TAG", "c25"),
           R57_KEYS="humaneval/HumanEval/1",
           BIGMOE_MAXK="320", BIGMOE_EXT_BUILD="build_g32", BIGMOE_OVERFLOW_FLAG="1", V32_MOE1="1",
           R57_G32TEST=os.environ.get("C25_KEYS", "dd__mc_27921,dd__mc_2753,dd__mc_28056"),
           R57_G32TIME=os.environ.get("C25_ROUNDS", "12"), R57_G32TIME_WARM=os.environ.get("C25_WARM", "2"),
           C25_ARMS=os.environ.get("C25_ARMS", "F16,F32s,F32o,F32d,F32df,F32if,F32if6,F32c5,F32c6i"))
for k in sorted(env):
    print(f"export {k}={shlex.quote(env[k])}")
print("unset REF2_GDN_LAZY REF2_SEAM_HI")
