"""bench.py (Breakthrough, Tue 17 Nov 2026 11:3x; MOE BLOCK proof): the pre-registered block bench (docs/MOEBLOCK.md,
f6f806e). For each layer (default 20 and 44) and row count (32, 16): NB real blocks (the decision map's code windows:
the layer's MoE input x, fp16 -> bf16). Each block's routing comes from the layer's own router, and its floor is
block_bytes / 280 GB/s. Arms run interleaved with the order rotated per round. Per block and arm, REPS back-to-back calls
run in one graph and the time is wall / REPS (exp32's sustained way, per block); per block the median over ROUNDS.
Efficiency = floor / time per block; the verdict is the median over blocks (PASS >= 0.85, BETWEEN 0.75-0.85, else FAIL),
required at both row counts and on both layers.
Arms: today (chain_today, knob off), knob (chain_today with xknob MODE tgl16eu, frozen knob_v2), plus any arm a
candidate module registers in ARMS.
Usage: ~/BigMoE/.venv/bin/python -I -B scripts/moeblk/bench.py [--layers 20,44] [--nb 12] [--reps 6] [--rounds 5]
       [--arms today,knob] [--tag T]   -> data/moeblk/bench[_T].txt / .json   (MEM ~5 GB; quiet for the verdict)"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.expanduser("~/Leap/third_party/mtlprof"))
import mtlprof as MP                                        # before mlx

MP.load(2)
import numpy as np                                          # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import moeblk as MB                                         # noqa: E402

mx = MB.mx
KNOB = os.path.expanduser(os.environ.get("MOEBLK_KNOB", "~/Breakthrough/data/xrw/knob_v2"))
OUTD = os.environ.get("MOEBLK_OUTD") or os.path.join(HERE, "..", "..", "data", "moeblk")
MAP = os.path.expanduser("~/FableGod/decision_map/out")
FLOOR_BPS = 280e9


def blocks(L, nb, rows):
    basej = json.load(open(os.path.join(MAP, "baseline.json")))
    code = [f for f in sorted(os.listdir(os.path.join(MAP, "xcap"))) if basej[f]["dom"] == "code"]
    out = []
    for f in code:
        z = np.load(os.path.join(MAP, "xcap", f))
        n = z["ids"].shape[0]
        for b0 in range(0, n - 31, 32):
            out.append((f, b0, z["x"][b0:b0 + rows, L].astype(np.float32), z["ids"][b0:b0 + rows, L].astype(np.int64)))
            if len(out) >= nb:
                return out
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="20,44")
    ap.add_argument("--nb", type=int, default=12)
    ap.add_argument("--reps", type=int, default=6)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--arms", default="today,knob")
    ap.add_argument("--rows", default="32,16")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    cfg = MB.load_cfg()
    sys.path.insert(0, KNOB)
    import xknob
    assert os.path.dirname(os.path.abspath(xknob.__file__)) == os.path.abspath(KNOB), xknob.__file__
    xknob.install()
    arms = a.arms.split(",")
    ARMS = {"today": lambda lay, x: (xknob.ON.__setitem__(0, False), MB.chain_today(lay, x))[1],
            "knob": lambda lay, x: (xknob.ON.__setitem__(0, True), xknob.MODE.__setitem__(0, "tgl16eu"),
                                    MB.chain_today(lay, x))[2]}
    for arm in arms:
        if arm not in ARMS:                                 # a candidate module cand_<arm>.py (block), or component arms
            if arm.startswith("c_"):
                import cand_comp
                f_ = getattr(cand_comp, arm)
                ARMS[arm] = (lambda f__: (lambda lay, x: (xknob.ON.__setitem__(0, True), xknob.MODE.__setitem__(0, "tgl16eu"),
                                                          f__(lay, x))[2]))(f_)
            else:
                mod = __import__(f"cand_{arm}")
                ARMS[arm] = mod.block
    L_ = []

    def say(s=""):
        print(s, flush=True)
        L_.append(s)
    say(f"\nMOEBLK BENCH {time.strftime('%a %d %b %H:%M:%S')}: arms {arms}; knob {KNOB}; layers {a.layers}; rows {a.rows}; "
        f"{a.nb} blocks x {a.reps} back-to-back calls x {a.rounds} rounds (rotated); floor = block bytes / 280 GB/s "
        f"(pre-registered f6f806e)")
    res = {}
    slot0 = 0
    for L in [int(v) for v in a.layers.split(",")]:
        lay = MB.Layer(cfg, L, slot0)
        slot0 = lay.next_slot
        for rows in [int(v) for v in a.rows.split(",")]:
            if "cand_comp" in sys.modules:
                sys.modules["cand_comp"].reset()
            BL = blocks(L, a.nb, rows)
            xs, floors, agree = [], [], []
            for (f, b0, xf, ids_cap) in BL:
                x = mx.array(xf).astype(mx.bfloat16)
                mx.eval(x)
                ids, sc0, sc, idn, nl = MB.routing_np(lay, x)
                agree.append(float(np.mean([set(ids[m].tolist()) == set(ids_cap[m].tolist()) for m in range(rows)])))
                tot, routed, other = MB.block_bytes(lay, idn, nl)
                floors.append(1e6 * tot / FLOOR_BPS)
                xs.append(x)
            for arm in list(arms):                          # build + warm (a failing arm is dropped, not fatal)
                try:
                    mx.eval([ARMS[arm](lay, x) for x in xs[:2]])
                except Exception as e:                      # noqa: BLE001
                    say(f"  L{L} {rows} rows: arm {arm} FAILED to build ({type(e).__name__}: {str(e)[:160]}); dropped")
                    arms = [v for v in arms if v != arm]
            mx.synchronize()
            for arm in [v for v in arms if v.endswith("c") and v[:-1] in arms]:   # a compiled arm must equal its source
                same = all(bool(mx.all(ARMS[arm](lay, x).view(mx.uint16) == ARMS[arm[:-1]](lay, x).view(mx.uint16)).item())
                           for x in xs)
                say(f"  L{L} {rows} rows: {arm} == {arm[:-1]} bitwise on {len(xs)} blocks: {same}")
                if not same:
                    arms = [v for v in arms if v != arm]
            T = {arm: [[] for _ in xs] for arm in arms}
            GB = {arm: [[] for _ in xs] for arm in arms}          # GPU busy per call (libmtlprof: command buffers' union)
            CP = {arm: [[] for _ in xs] for arm in arms}          # CPU per call: (graph build, encode + commit), syncs
            for rd in range(a.rounds):
                order = arms[rd % len(arms):] + arms[:rd % len(arms)]
                for bi, x in enumerate(xs):
                    for arm in order:
                        mx.synchronize()
                        c0 = MP.counts()["ncb"]
                        m0 = MP.now()
                        t0 = time.perf_counter()
                        outs = [ARMS[arm](lay, x) for _ in range(a.reps)]
                        t1 = time.perf_counter()
                        cb1 = MP.counts()["ncb"]                     # > c0: the arm synced while building
                        mx.async_eval(outs)
                        t2 = time.perf_counter()
                        mx.synchronize()
                        t3 = time.perf_counter()
                        T[arm][bi].append(1e6 * (t3 - t0) / a.reps)
                        m1 = MP.now()
                        GB[arm][bi].append(1e6 * MP.busy(MP.cbs(c0, MP.counts()["ncb"]), m0, m1) / a.reps)
                        CP[arm][bi].append((1e6 * (t1 - t0) / a.reps, 1e6 * (t2 - t1) / a.reps, cb1 - c0))
            xknob.ON[0] = False
            fl = np.array(floors)
            say(f"  L{L} {rows} rows: {len(xs)} blocks; router == the capture's routing on {100 * np.mean(agree):.1f} % of rows; "
                f"floor mean {fl.mean():.1f} us (bytes {fl.mean() * 1e-6 * FLOOR_BPS / 1e9:.4f} GB)")
            for arm in arms:
                tb = np.array([np.median(v) for v in T[arm]])
                gb_ = np.array([np.median(v) for v in GB[arm]])
                bd = np.array([np.median([c[0] for c in v]) for v in CP[arm]])
                en = np.array([np.median([c[1] for c in v]) for v in CP[arm]])
                ns_ = int(max(max(c[2] for c in v) for v in CP[arm]))
                say(f"    {arm:8s}: CPU build {np.median(bd):6.1f} + encode {np.median(en):6.1f} = {np.median(bd + en):6.1f} us / call; "
                    f"GPU busy {np.median(gb_):6.1f}; wall {np.median(tb):6.1f}; GPU idle (wall - busy) {np.median(tb - gb_):6.1f}; "
                    f"floor / busy {np.median(fl / gb_):.3f}; command buffers during the build {ns_}")
                res[f"L{L}_{rows}_{arm}_split"] = dict(build=bd.tolist(), encode=en.tolist(), busy=gb_.tolist(), wall=tb.tolist())
                eff = fl / tb
                v = "PASS" if np.median(eff) >= 0.85 else ("BETWEEN" if np.median(eff) >= 0.75 else "FAIL")
                say(f"    {arm:8s}: {np.median(tb):7.1f} us / call (median over blocks); efficiency floor / time median "
                    f"{np.median(eff):.3f} [p10 {np.percentile(eff, 10):.3f}, p90 {np.percentile(eff, 90):.3f}] -> {v}; "
                    f"x{np.median(tb / (fl * 280 / 305)):.2f} the 305 GB/s ceiling")
                res[f"L{L}_{rows}_{arm}"] = dict(t=tb.tolist(), floor=fl.tolist(), eff_med=float(np.median(eff)), verdict=v)
            if "today" in arms:
                t0_ = np.array([np.median(v) for v in T["today"]])
                for arm in arms:
                    if arm != "today":
                        r_ = np.array([np.median(v) for v in T[arm]]) / t0_
                        say(f"    {arm} / today per block: median {np.median(r_):.3f} [p10 {np.percentile(r_, 10):.3f}, "
                            f"p90 {np.percentile(r_, 90):.3f}]")
    sfx = f"_{a.tag}" if a.tag else ""
    os.makedirs(OUTD, exist_ok=True)
    open(os.path.join(OUTD, f"bench{sfx}.txt"), "a").write("\n".join(L_) + "\n")
    json.dump(res, open(os.path.join(OUTD, f"bench{sfx}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
