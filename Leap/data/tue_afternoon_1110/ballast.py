"""36 GB Mac simulation (orchestrator Tue 13:3x): hold <GB> of anonymous memory locked in RAM (mlock: wired, never paged
out or compressed) until SIGTERM / SIGINT, so that 64 - <GB> GB is left for macOS and the engine. No sysctl, no sudo:
mlock is within the per-user wire limit (vm.user_wire_limit 56.3 GB; ulimit -l unlimited). Exiting releases it."""
import ctypes, mmap, signal, sys, time

n = int(float(sys.argv[1]) * 1e9)
n -= n % 16384
m = mmap.mmap(-1, n, flags=mmap.MAP_PRIVATE | mmap.MAP_ANON)
buf = ctypes.c_char.from_buffer(m)
libc = ctypes.CDLL(None, use_errno=True)
t0 = time.time()
if libc.mlock(ctypes.c_void_p(ctypes.addressof(buf)), ctypes.c_size_t(n)) != 0:
    print(f"ballast: mlock failed, errno {ctypes.get_errno()}", flush=True)
    sys.exit(2)
for off in range(0, n, 16384):                               # dirty every page too (nothing left to a lazy zero page)
    m[off] = 1
print(f"ballast: {n / 1e9:.2f} GB locked in {time.time() - t0:.1f} s", flush=True)
signal.signal(signal.SIGTERM, lambda *a: sys.exit(0))
signal.signal(signal.SIGINT, lambda *a: sys.exit(0))
while True:
    time.sleep(5)
