"""WHOLE-CHIP screen (Leap, Tue 2026-11-03 17:1x; pre-registered in ~/Leap/TABLE_350.md 'WHOLE-CHIP'): the CPU's
SME / NEON and the Neural Engine beside the GPU's expert kernel.
- G: NAX6-N on data/native_sample layer 20, S 16, real routing (8 windows: code + chat, as data/wc/routing_L20.bin),
  dependent chains of 16 calls (+ combine), cycling the windows.
- Units (separate processes, started and stopped per phase): native/wc_cpu (bw 1 / 2 / 4 threads, q4 4 threads,
  sgemm alone only) and scripts/wc_ane.py (a8 / a4 Neural Engine models).
- Each unit alone first; then phases alternating "GPU alone" and "GPU + unit". s = the GPU's time per call in the phase
  over the mean of the two neighbouring alone phases, per window, median. The unit's rate comes from its own log
  during the phase. gain = R_u' / R_g + 1 / (1 + s) - 1 (R in experts per ms; C-BW in GB/s against the GPU's GB/s).
Usage: [BigMoE venv] wc_screen.py -> ~/Leap/data/wc_screen.txt (GPU ~2.5 GB + one unit <= 1.5 GB; share lock)
WHOLE-CHIP-2 S (the sustained re-run): WC_PH=15 WC_WARM=60 WC_ALONE=0 WC_UNITS=a4x4,q4,a4x4+q4,a4x4,q4,a4x4+q4,a4x4,q4,a4x4+q4
  (a4x4 cycles four 32-expert models with different weights; "+" runs units together, their experts/ms summed)"""
import json, os, signal, subprocess, sys, threading, time
import numpy as np
import mlx.core as mx
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_p4s/Leap_third_party/ms_gdn"))
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_p4s/Leap_scripts"))
from tierglu import PROJ
import tierglu as TG
import tierglu_nax6 as N6
import tierglu_nax6n as NN

LEAP = os.path.expanduser("~/Leap")
OUT = f"{LEAP}/data/wc_screen.txt"
PY = os.path.expanduser("~/BigMoE/.venv/bin/python")
CPU = f"{LEAP}/native/wc_cpu"
ROUT = f"{LEAP}/data/wc/routing_L20.bin"
PH = float(os.environ.get("WC_PH", "4"))
WARM = float(os.environ.get("WC_WARM", "0"))
ALONE = os.environ.get("WC_ALONE", "1") != "0"
UNITS = os.environ.get("WC_UNITS", "bw4,q4,a4,a8,bw2,q4,a4,bw1").split(",")


def say(s):
    print(s, flush=True)
    open(OUT, "a").write(s + "\n")


def swap_gb():
    out = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout
    return float(out.split("used = ")[1].split("M")[0]) / 1024


CMD = {"bw1": [CPU, "bw", "1", "60"], "bw2": [CPU, "bw", "2", "60"], "bw4": [CPU, "bw", "4", "60"],
       "q4": [CPU, "q4", "4", "60", ROUT], "q4x2": [CPU, "q4", "2", "60", ROUT],
       "sgemm1": [CPU, "sgemm", "1", "60"], "sgemm4": [CPU, "sgemm", "4", "60"],
       "a8": [PY, f"{LEAP}/scripts/wc_ane.py", "run", f"{LEAP}/data/wc/ane_e32_a8.mlpackage", "60"],
       "a4": [PY, f"{LEAP}/scripts/wc_ane.py", "run", f"{LEAP}/data/wc/ane_e32_a4.mlpackage", "60"],
       "a4x4": [PY, f"{LEAP}/scripts/wc_ane.py", "run",
                ",".join(f"{LEAP}/data/wc/ane_e32_a4_s{i}.mlpackage" for i in (1, 2, 3, 4)), "60"]}
KEY = {"bw": "gbps", "q4": "exp_per_ms", "sg": "exp_per_ms", "a8": "exp_per_ms", "a4": "exp_per_ms"}


class Unit:
    def __init__(self, name):
        self.name, self.lines = name, []
        self.p = subprocess.Popen(CMD[name], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, cwd=LEAP)
        self.ready = threading.Event()
        threading.Thread(target=self._read, daemon=True).start()
        if not self.ready.wait(120):
            self.stop()
            raise RuntimeError(f"{name} never became ready")

    def _read(self):
        for ln in self.p.stdout:
            if ln.startswith("ready"):
                self.ready.set()
                continue
            f = ln.split()
            try:
                self.lines.append((float(f[0]), {f[i]: float(f[i + 1]) for i in range(1, len(f) - 1, 2)}))
            except (ValueError, IndexError):
                pass

    def rate(self, t0, t1, key=None):
        key = key or KEY[self.name[:2]]
        v = [d[key] for t, d in self.lines if t0 + 0.99 <= t <= t1 + 0.05 and key in d]
        return float(np.mean(v)) if v else float("nan"), len(v)

    def stop(self):
        if self.p.poll() is None:
            self.p.send_signal(signal.SIGTERM)
            try:
                self.p.wait(5)
            except subprocess.TimeoutExpired:
                self.p.kill()
                self.p.wait()


say(f"\nWHOLE-CHIP screen {time.strftime('%a %H:%M')} ({os.environ.get('WC_COND', 'share lock')}); phases {PH:.0f} s; units {UNITS}")
swap0 = swap_gb()

# ------------------------------------------------------------------ units alone
alone = {}
for name in (("bw1", "bw2", "bw4", "q4", "sgemm1", "sgemm4", "a8", "a4") if ALONE else ()):
    u = Unit(name)
    t0 = time.time()
    while time.time() < t0 + 5.2:
        time.sleep(0.1)
    alone[name] = u.rate(t0, time.time())
    extra = ""
    if name.startswith("sgemm"):
        extra = f", {u.rate(t0, time.time(), 'gflops')[0]:.0f} GFLOP/s, fp32 weights {u.rate(t0, time.time(), 'gbps_fp32')[0]:.1f} GB/s"
    if name == "q4":
        extra = f", {u.rate(t0, time.time(), 'gbps')[0]:.1f} GB/s of 4-bit expert bytes"
    u.stop()
    say(f"  alone {name}: {KEY[name[:2]]} {alone[name][0]:.3f} ({alone[name][1]} s){extra}")

# ------------------------------------------------------------------ GPU arm
RD = os.path.expanduser("~/BigMoE/models/bare/records3")
assert all(v.get("layout", "row-major") == "row-major" for v in json.load(open(f"{RD}/index.json"))["layers"].values()), \
    "records3 is converted (native+p11+pad, Thu 11-05): this script reads row-major bytes only"   # Leap 18:5x guard
ND = f"{LEAP}/data/native_sample"
SH = {"gate_proj": (640, 2560), "up_proj": (640, 2560), "down_proj": (2560, 640)}
PLAN = "4433333333"
R = len(PLAN)
r = json.load(open(f"{RD}/index.json"))["layers"]["20"]
raw = np.memmap(f"{ND}/{r['file']}", dtype=np.uint8, mode="r", shape=(r["E"], r["recsz"]))
msb, lsb = {}, {}
E = r["E"]
for p, pr in enumerate(PROJ):
    o, i = SH[pr]
    ow, os_, ob = r["msb_piece_off"][3 * p:3 * p + 3]
    lw, lb = r["lsb_piece_off"][2 * p:2 * p + 2]
    msb[pr] = {"w3": mx.array(np.ascontiguousarray(raw[:, ow:ow + o * i * 3 // 8]).view(np.uint32).reshape(E, o, i * 3 // 32)),
               "s3": mx.array(np.ascontiguousarray(raw[:, os_:os_ + o * i // 32]).view(np.uint16).reshape(E, o, i // 64)).view(mx.bfloat16),
               "b3": mx.array(np.ascontiguousarray(raw[:, ob:ob + o * i // 32]).view(np.uint16).reshape(E, o, i // 64)).view(mx.bfloat16)}
    lsb[pr] = {"l1": mx.array(np.ascontiguousarray(raw[:, lw:lw + o * i // 8]).view(np.uint32).reshape(E, o, i // 32)),
               "b": mx.array(np.ascontiguousarray(raw[:, lb:lb + o * i // 32]).view(np.uint16).reshape(E, o, i // 64)).view(mx.bfloat16)}
mx.eval([v for d in (msb, lsb) for pr in d for v in d[pr].values()])
del raw
MSB_B, LSB_B = 3 * (614400 + 2 * 51200), 3 * (204800 + 51200)
mx.random.seed(0)
rng = np.random.default_rng(3)
wins = []
for t in ("code", "chat"):
    tr = np.load(os.path.expanduser(f"~/Breakthrough/data/expA/traces125/{t}_800_g.npz"))["L20"].astype(np.int32)
    for t0 in np.linspace(0, tr.shape[0] - 16, 4).astype(int):
        ids = tr[t0:t0 + 16, :10].copy()
        ms_, ls_ = mx.array(ids), mx.array(ids[:, :2].copy())
        x0 = (mx.random.normal((16, 2560)) * 0.5).astype(mx.bfloat16)
        sc = mx.softmax(mx.random.normal((16, R)), axis=-1).astype(mx.bfloat16)
        one = mx.ones((16, R), mx.int32)
        mx.eval(x0, sc, one)
        nu, nl = len(np.unique(ids)), len(np.unique(ids[:, :2]))
        def g(x, ms_=ms_, ls_=ls_, sc=sc, one=one):
            y = NN.tier_glu_nax6n(x, msb, lsb, ms_, ls_, plan=PLAN, grp=N6.gpu_grp(ms_, ls_, PLAN)).reshape(-1, 2560)
            return TG.combine_phase(y, y, one, sc)
        wins.append(dict(name=f"{t}@{t0}", f=g, x0=x0, nu=nu, bytes=nu * MSB_B + nl * LSB_B))
NCH = 16
def chain(w):
    out = None
    for _ in range(NCH):
        out = w["f"](w["x0"] if out is None else mx.depends(w["x0"], out))
    return out
for w in wins:
    for _ in range(3):
        mx.eval(chain(w))


def gpu_phase(secs):
    ts = {i: [] for i in range(len(wins))}
    t_end = time.time() + secs
    i = 0
    while time.time() < t_end:
        w = wins[i]
        mx.synchronize()
        t1 = time.perf_counter()
        mx.eval(chain(w))
        ts[i].append((time.perf_counter() - t1) / NCH)
        i = (i + 1) % len(wins)
    return {k: float(np.median(v)) for k, v in ts.items()}


if WARM > 0:                                                      # to steady temperature first
    tw0 = time.time()
    warm = gpu_phase(WARM)
    say(f"  warm-up {time.time() - tw0:.0f} s: GPU {1e3 * np.mean(list(warm.values())):.3f} ms per call at its end")

# ------------------------------------------------------------------ phases
sched = ["none"]
for u in UNITS:
    sched += [u, "none"]
res = []
for ph in sched:
    if swap_gb() - swap0 > 2.0:
        say(f"  STOP: swap grew {swap_gb() - swap0:.2f} GB")
        break
    us = [Unit(n) for n in ph.split("+")] if ph != "none" else []
    if us:
        time.sleep(1.0)
    t0 = time.time()
    tw = gpu_phase(PH)
    t1 = time.time()
    rs = [u.rate(t0, t1) for u in us]
    rate = (float(sum(r[0] for r in rs)), min(r[1] for r in rs)) if us else (float("nan"), 0)
    for u in us:
        u.stop()
    res.append((ph, tw, rate))
    say(f"  phase {ph:8s}: GPU {1e3 * np.mean(list(tw.values())):.3f} ms per call (mean over windows)" +
        (f"; {ph} {KEY[ph[:2]]} {rate[0]:.3f} ({rate[1]} s" + (", " + " + ".join(f"{r[0]:.2f}" for r in rs) if len(rs) > 1 else "") + ")"
         if us else ""))

# ------------------------------------------------------------------ verdict
nw = len(wins)
gpu_alone = {k: float(np.mean([res[j][1][k] for j in range(len(res)) if res[j][0] == "none"])) for k in range(nw)}
Rg = float(np.mean([wins[k]["nu"] / (1e3 * gpu_alone[k]) for k in range(nw)]))       # experts per ms
Bg = float(np.mean([wins[k]["bytes"] / gpu_alone[k] / 1e9 for k in range(nw)]))      # GB/s
say(f"  GPU alone: {1e3 * np.mean(list(gpu_alone.values())):.3f} ms per call, {Rg:.1f} experts/ms, {Bg:.0f} GB/s of tier bytes "
    f"(unique experts per window {np.mean([w['nu'] for w in wins]):.1f})")
summary = {}
for j, (ph, tw, rate) in enumerate(res):
    if ph == "none":
        continue
    nb = [res[k][1] for k in (j - 1, j + 1) if 0 <= k < len(res) and res[k][0] == "none"]
    s = float(np.median([tw[k] / np.mean([b[k] for b in nb]) - 1 for k in range(nw)]))
    if ph.startswith("bw"):
        gain = rate[0] / Bg + 1 / (1 + s) - 1
        what = f"{rate[0]:.1f} GB/s beside (alone {alone.get(ph, (float('nan'), 0))[0]:.1f}); bound gain {100 * gain:+.1f} %"
    else:
        gain = rate[0] / Rg + 1 / (1 + s) - 1
        what = f"{rate[0]:.2f} experts/ms beside (alone {alone.get(ph, (float('nan'), 0))[0]:.2f}); gain {100 * gain:+.1f} %"
    summary.setdefault(ph, []).append((s, gain))
    say(f"  {ph:5s}: GPU slowed {100 * s:+.1f} %; {what}")
say("  per unit (mean of its phases): " + "; ".join(f"{k} s {100 * np.mean([a for a, _ in v]):+.1f} % gain {100 * np.mean([b for _, b in v]):+.1f} %"
                                               for k, v in summary.items()))
best = {k: (float(np.mean([a for a, _ in v])), float(np.mean([b for _, b in v]))) for k, v in summary.items() if not k.startswith("bw")}
passed = [k for k, (s, g) in best.items() if g >= 0.20 and s < 0.05]
between = [k for k, (s, g) in best.items() if g >= 0.10 and s < 0.05]
say(f"  WHOLE-CHIP verdict: {'PASS ' + str(passed) if passed else ('BETWEEN ' + str(between) if between else 'KILL')} "
    f"(swap {swap_gb() - swap0:+.2f} GB)")
