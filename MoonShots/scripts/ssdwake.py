> (a quote from the project owner, kept private)
in ~/NewIdeas/QUEUE.md M-20 before any timing): does the internal SSD charge a wake-up (power-state) penalty on the first
reads after the short idle gaps the 45 GB route leaves between layers? Each layer's demand misses go out as one burst
after its router (~3.6 misses per layer per check at C 256, 9 pieces per miss), and the SSD idles while the GPU computes
(~1 ms per layer; ~10-15 ms around draft + commit). If it does, can a trickle of tiny keepalive reads remove it?
Prior-art check (22:0x): no MoE / LLM-offload work on SSD idle states; no published Apple Silicon wake-latency numbers. PC
NVMe drives pay 2-60 ms after deep idle (APST / ASPM, usually entered after 10s-100s of ms), so the prior is low for ~1 ms.
Method: F_NOCACHE preads into page-aligned buffers from BigMoE's records3 store (read-only), each read at a fresh random
expert's MSB part (2,162,688 B at e x 2,932,736) in a random layer file, never reused. Burst = 4 experts x 9 parallel pieces
on 36 threads; single = one expert as one read. Gaps between bursts (busy-waited): 0, 0.25, 0.5, 1, 2, 5, 10, 20 ms.
Arms: plain; keepalive (a thread reading 4 KB F_NOCACHE at a random offset every ~200 us). Gap order shuffled per round.
Default QoS on purpose (taskpolicy -b throttles disk I/O).
Line: PASS if the burst after a 1 ms gap is >= 15 % and >= 50 us slower than back to back AND the keepalive removes >= 70 %
of that penalty; KILL if the 1 ms-gap penalty is < 5 % or < 20 us for both bursts and single reads; BETWEEN otherwise.
usage: [BigMoE venv] ssdwake.py -> data/ssdwake.txt"""
import fcntl, mmap, os, random, sys, threading, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
sys.setswitchinterval(5e-5)                                      # the keepalive thread must run during busy-waited gaps

STORE = os.path.expanduser("~/BigMoE/models/bare/records3")
EXP, MSB, NPIECE, NEXP = 2_932_736, 2_162_688, 9, 4
GAPS = (0.0, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0)
ROUNDS = 40
F_NOCACHE = 48
OUT = os.path.expanduser("~/MoonShots/data/ssdwake.txt")
lines = []


def say(s):
    print(s, flush=True); lines.append(s)


files = [os.path.join(STORE, f"L{l:02d}.bin") for l in range(48)]
fds = []
for p in files:
    fd = os.open(p, os.O_RDONLY)
    fcntl.fcntl(fd, F_NOCACHE, 1)
    fds.append(fd)
NE = os.path.getsize(files[0]) // EXP
used = set()
rng = random.Random(20261102)


def fresh():
    while True:
        k = (rng.randrange(48), rng.randrange(NE))
        if k not in used:
            used.add(k)
            return k


PIECE = -(-MSB // NPIECE // 16384) * 16384                       # 16 KB-aligned pieces (last one shorter)
bufs = [mmap.mmap(-1, PIECE) for _ in range(NEXP * NPIECE)]
one = mmap.mmap(-1, MSB)
pool = ThreadPoolExecutor(NEXP * NPIECE)


def read_piece(args):
    fd, off, n, b = args
    return os.preadv(fd, [memoryview(b)[:n]], off)


def burst():
    jobs = []
    for i in range(NEXP):
        l, e = fresh()
        for j in range(NPIECE):
            o = j * PIECE
            jobs.append((fds[l], e * EXP + o, min(PIECE, MSB - o), bufs[i * NPIECE + j]))
    t0 = time.perf_counter()
    list(pool.map(read_piece, jobs))
    return time.perf_counter() - t0


def single():
    l, e = fresh()
    t0 = time.perf_counter()
    os.preadv(fds[l], [memoryview(one)], e * EXP)
    return time.perf_counter() - t0


def spin(ms):
    t = time.perf_counter() + ms / 1e3
    while time.perf_counter() < t:
        time.sleep(0)


stop, alive = threading.Event(), threading.Event()
KA = [4096, 0.0002]                                              # keepalive read size, interval (follow-up arms change it)


def keepalive():
    b = mmap.mmap(-1, 65536)
    r = random.Random(7)
    while not stop.is_set():
        if alive.is_set():
            os.preadv(fds[r.randrange(48)], [memoryview(b)[:KA[0]]], r.randrange(NE) * EXP + 16384 * r.randrange(100))
        time.sleep(KA[1])


def follow():
    """exploratory (not pre-registered), right after the screen: a CPU-only spinner (no I/O) separates host wake-ups from
    the SSD path; faster / larger keepalives show how much more is recoverable. Blocks of ROUNDS rounds, plain between."""
    import subprocess
    rng.seed(7777)                                               # other experts than the screen read
    th = threading.Thread(target=keepalive, daemon=True)
    th.start()
    for _ in range(3):
        burst(); single()
    say(f"== M-20 follow-up ({time.strftime('%a %H:%M')}, exploratory): blocks of {ROUNDS} rounds; 1 ms and 10 ms gaps vs back to back")
    for arm in ("plain", "cpuspin", "plain", "ka4k/50us", "plain", "ka64k/200us", "plain", "ka4k/200us"):
        sp = subprocess.Popen([sys.executable, "-c", "while True: pass"]) if arm == "cpuspin" else None
        if arm.startswith("ka"):
            size, iv = arm[2:].split("/")
            KA[0], KA[1] = int(size[:-1]) * 1024, int(iv[:-2]) * 1e-6
            alive.set()
        else:
            alive.clear()
        spin(20)
        R = {g: ([], []) for g in (0.0, 1.0, 10.0)}
        for r in range(ROUNDS):
            for g in random.Random(r).sample(list(R), len(R)):
                spin(g); R[g][0].append(burst())
                spin(g); R[g][1].append(single())
        if sp:
            sp.kill()
        b0, s0 = np.median(R[0.0][0]), np.median(R[0.0][1])
        say(f"  {arm:12s}: burst {1e6 * b0:5.0f} us, +{1e6 * (np.median(R[1.0][0]) - b0):4.0f} us at 1 ms, +{1e6 * (np.median(R[10.0][0]) - b0):4.0f} at 10 ms | "
            f"single {1e6 * s0:4.0f} us, +{1e6 * (np.median(R[1.0][1]) - s0):4.0f} at 1 ms, +{1e6 * (np.median(R[10.0][1]) - s0):4.0f} at 10 ms")
    stop.set(); th.join()
    open(OUT, "a").write("\n".join(lines) + "\n")


def main():
    t0 = time.time()
    say(f"== M-20 SSDWAKE ({time.strftime('%a %H:%M')}): records3, {NEXP} x {NPIECE} pieces of {PIECE} B per burst, {ROUNDS} rounds "
        f"per gap and arm (arms alternate within each round); us median (p90)")
    th = threading.Thread(target=keepalive, daemon=True)
    th.start()
    for _ in range(3):                                           # warm-up: threads, page tables
        burst(); single()
    res = {a: {g: ([], []) for g in GAPS} for a in ("plain", "keepalive")}
    for r in range(ROUNDS):
        arms = ["plain", "keepalive"]
        random.Random(1000 + r).shuffle(arms)
        for arm in arms:
            alive.set() if arm == "keepalive" else alive.clear()
            spin(5)
            order = list(GAPS)
            random.Random(r).shuffle(order)
            for g in order:
                spin(g); res[arm][g][0].append(burst())
                spin(g); res[arm][g][1].append(single())
    stop.set(); th.join()
    for arm, R in res.items():
        b0, s0 = np.median(R[0.0][0]), np.median(R[0.0][1])
        for g in GAPS:
            b, s = np.array(R[g][0]), np.array(R[g][1])
            say(f"  {arm:9s} gap {g:5.2f} ms: burst {1e6 * np.median(b):7.0f} ({1e6 * np.percentile(b, 90):7.0f}) {100 * (np.median(b) / b0 - 1):+6.1f} % | "
                f"single {1e6 * np.median(s):6.0f} ({1e6 * np.percentile(s, 90):6.0f}) {100 * (np.median(s) / s0 - 1):+6.1f} %")
    pb = {a: np.median(res[a][1.0][0]) - np.median(res[a][0.0][0]) for a in res}
    ps = {a: np.median(res[a][1.0][1]) - np.median(res[a][0.0][1]) for a in res}
    rel = pb["plain"] / np.median(res["plain"][0.0][0])
    removed = 1 - pb["keepalive"] / pb["plain"] if pb["plain"] > 0 else float("nan")
    verdict = ("PASS" if rel >= 0.15 and pb["plain"] >= 50e-6 and removed >= 0.70 else
               "KILL" if (rel < 0.05 or pb["plain"] < 20e-6) and (ps["plain"] / np.median(res["plain"][0.0][1]) < 0.05 or ps["plain"] < 20e-6)
               else "BETWEEN")
    say(f"  1 ms-gap penalty: burst {1e6 * pb['plain']:+.0f} us ({100 * rel:+.1f} %), single {1e6 * ps['plain']:+.0f} us; "
        f"keepalive removes {100 * removed:.0f} % of the burst penalty -> {verdict}  ({time.time() - t0:.0f} s, {len(used)} reads "
        f"of fresh experts)")
    open(OUT, "a").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    follow() if sys.argv[1:] == ["follow"] else main()
