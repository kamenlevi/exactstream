"""Breakthrough R-19 W24-12 (NewIdeas Tue 10-27 19:5x, QUEUE 'RAPID SCREENS'; written before any run): price MoonShots'
exact packs (W24-sp 11-bit scale / bias + W24-ec rANS MSB-3 codes: an MSB piece 2,211,840 -> ~1,900,544 B, -14.07 %; the
LSB piece 720,896 B unchanged) on bt_b12's AR decode (b12_generate's path, shipped env) at 2.6K and 8.5K tokens.
Predicted gain = SSD share x byte cut. SSD share = (T_cold - T_hot) / T_cold per decode token (steps 2..DEC):
  cold (R19_ARM=cold): the shipped decode, C_m 51 and PFR on, every decode starting from empty pools;
  hot (R19_ARM=hot, run with B12_CM=160 B12_PFR=0): each prompt decoded twice, the second time with its experts already
  resident (bytes read ~0), so its step time is the decode without the SSD.
Byte cut = 14.07 % x the MSB share of the cold run's disk bytes (MSB / LSB reads solved from its bytes and read count).
Line (QUEUE R-19): PASS >= +8 % decode at 8.5K; KILL < +3 %. One RESULT line per (prompt, run).
Usage: [BigMoE venv] R19_ARM=cold|hot [B12_CM=160 B12_PFR=0] r19_price.py TAG"""
import ctypes, hashlib, json, os, sys, time

ENV = {"B12_CACHE_MB": "64", "BIGMOE_TIER_PF_GROUP": "64", "BIGMOE_PF_GROUP": "64", "BIGMOE_MTP_C": "32",
       "BIGMOE_ASYNC_EVERY": "1", "B12_LATE3": "2,10", "B12_NORM": "mass"}          # pk64.py's shipped env
for k, v in ENV.items():
    os.environ.setdefault(k, v)
TAG = sys.argv[1] if len(sys.argv) > 1 else "r19"
ARM = os.environ.get("R19_ARM", "cold")
DEC = int(os.environ.get("R19_DEC", "64"))
LENS = [int(v) for v in os.environ.get("R19_LENS", "2600,8474").split(",")]


class _RU(ctypes.Structure):
    _fields_ = [("uuid", ctypes.c_uint8 * 16)] + [(f"u{i}", ctypes.c_uint64) for i in range(40)]


_LIBC = ctypes.CDLL(None)


def ru():                                            # (footprint now GB, lifetime peak GB, disk bytes read)
    r = _RU()
    if _LIBC.proc_pid_rusage(os.getpid(), 4, ctypes.byref(r)) != 0:
        return float("nan"), float("nan"), 0
    return r.u7 / 1e9, r.u28 / 1e9, r.u16


import mlx.core as mx
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bt_b12 as B
rt, st = B.load_b12(0.10, mtp=False)
from bigmoe_v3f.fuse import reset_generation_state
from bigmoe_v3f.gated import ext
model, tok = rt.model, rt.tokenizer
ALL = tok.encode(open(os.path.expanduser("~/Breakthrough/data/prompts/ctx64k.txt")).read())
print(f"[{TAG}] arm {ARM}: engine {B.ENGINE}, AR path, C_m {B.CFG['cm']} C_l {B.CFG['cl']}, B12_PFR "
      f"{os.environ.get('B12_PFR', '1')}, pools {st.resident_gb():.2f} GB, decode {DEC} tokens per run | footprint "
      f"{ru()[0]:.2f} GB", flush=True)


def one(p, run):
    reset_generation_state(model); B.STASH.clear(); B.PENDH.clear()
    cache = model.make_cache()
    t0 = time.perf_counter()
    B.b12_prefill(model, mx.array([p[:-1]]), cache, 512, 4096)
    mx.eval([a for c in cache for a in (c.state if hasattr(c, "state") else []) if isinstance(a, mx.array)])
    tp = time.perf_counter() - t0
    ext.reset_stats(); ext.defer_reset()
    m0, dk0 = ext.stats()["misses"], ru()[2]
    y, out, TT = p[-1], [], []
    for i in range(DEC):
        lg = model(mx.array([[y]]), cache=cache)
        y = int(mx.argmax((lg[0] if isinstance(lg, tuple) else lg)[0, -1]).item())
        out.append(y)
        TT.append(time.perf_counter())
        if i == 0:                                   # steps 2..DEC: their reads and bytes only
            m1, dk1 = ext.stats()["misses"], ru()[2]
    st.gate.check()
    rows, waited, late, over = ext.defer_stats()
    n = DEC - 1
    dt = (TT[-1] - TT[0]) / n
    reads, disk = ext.stats()["misses"] - m1, ru()[2] - dk1
    sha = hashlib.sha1(json.dumps(out).encode()).hexdigest()[:10]
    print(f"[{TAG}] RESULT arm {ARM} n {len(p) - 1} run {run} | prefill {tp:.1f} s | steps 2-{DEC}: {dt * 1e3:.2f} ms/token "
          f"({1 / dt:.2f} tok/s), {disk / n / 1e6:.1f} MB/token, reads/token {reads / n:.1f} | whole decode: waited "
          f"{waited / DEC:.1f}, late {late / DEC:.1f}, overflow {over / DEC:.2f} per token, first-step reads {m1 - m0}, "
          f"{(dk1 - dk0) / 1e6:.0f} MB | footprint {ru()[0]:.2f} GB, lifetime peak {ru()[1]:.2f} GB | sha {sha}", flush=True)
    del cache


for L in LENS:
    p = ALL[:L + 1]
    assert len(p) == L + 1, len(p)
    for run in ((1, 2) if ARM == "hot" else (1,)):
        one(p, run)
print(f"[{TAG}] done", flush=True)
