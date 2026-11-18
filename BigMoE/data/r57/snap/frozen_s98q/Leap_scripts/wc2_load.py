"""Background-load sampler beside WHOLE-CHIP-2 when it runs with Kamen on the Mac (Leap, Tue 2026-11-03 19:0x; the
orchestrator 18:5x: "record the background load beside H and S"). Read-only: ps, ioreg, sysctl, vm_stat; no sudo.
  sample ROOT OUT [INTERVAL]  every INTERVAL s until ROOT exits, one JSON line: CPU per app (deltas of ps cumulative CPU
                              time, % of one core) and GPU per app (deltas of ioreg AGXDeviceUserClient
                              accumulatedGPUTime, ns -> % of wall time), split into ROOT's process tree ("mine") and the
                              rest ("bg"; other Claude sessions' tools are "claude sessions"); the GPU's device and
                              renderer utilization; swap used; available memory (free + inactive + speculative +
                              purgeable).
  summary OUT MARKS           per window of MARKS (lines "label epoch": base0/1, H0/1, S0/1): mean / max of the background
                              CPU (cores) and GPU (%), the device utilization, and the top background apps."""
import json, os, plistlib, subprocess, sys, time


def secs(tm):
    d, _, hms = tm.rpartition("-")
    s = 0.0
    for p in hms.split(":"):
        s = s * 60 + float(p)
    return s + (int(d) * 86400 if d else 0)


def app_name(comm):
    i = comm.find(".app/")
    return os.path.basename(comm[:i] if i >= 0 else comm) or comm


def procs():
    """pid -> [ppid, cpu seconds, name]."""
    out = subprocess.run(["ps", "-A", "-o", "pid=,ppid=,time=,comm="], capture_output=True, text=True).stdout
    d = {}
    for ln in out.splitlines():
        p = ln.split(None, 3)
        if len(p) == 4:
            try:
                d[int(p[0])] = [int(p[1]), secs(p[2]), app_name(p[3])]
            except ValueError:
                pass
    return d


def labels(d, root):
    """pid -> 'mine' (ROOT's tree), 'claude sessions' (under another claude process) or the app's name."""
    out = {}
    for pid in d:
        lab, p, n = d[pid][2], pid, 0
        while p in d and p > 1 and n < 64:
            if p == root:
                lab = "mine"
                break
            if p != pid and d[p][2] == "claude":
                lab = "claude sessions"
                break
            p, n = d[p][0], n + 1
        out[pid] = lab
    return out


def gpu_ns():
    """pid -> accumulated GPU time (ns) over its AGX user clients."""
    raw = subprocess.run(["ioreg", "-r", "-c", "AGXDeviceUserClient", "-a"], capture_output=True).stdout
    acc = {}

    def walk(n):
        if isinstance(n, dict):
            cr = n.get("IOUserClientCreator", "")
            if cr.startswith("pid "):
                pid = int(cr[4:].split(",")[0])
                acc[pid] = acc.get(pid, 0) + sum(u.get("accumulatedGPUTime", 0) for u in n.get("AppUsage") or [])
            for c in n.get("IORegistryEntryChildren") or []:
                walk(c)
        elif isinstance(n, list):
            for c in n:
                walk(c)
    try:
        walk(plistlib.loads(raw))
    except Exception:
        pass
    return acc


def util():
    raw = subprocess.run(["ioreg", "-r", "-d", "1", "-c", "IOAccelerator", "-a"], capture_output=True).stdout
    try:
        for n in plistlib.loads(raw):
            ps = n.get("PerformanceStatistics") or {}
            if "Device Utilization %" in ps:
                return ps.get("Device Utilization %"), ps.get("Renderer Utilization %")
    except Exception:
        pass
    return None, None


def mem():
    sw = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout
    try:
        swap = float(sw.split("used = ")[1].split("M")[0])
    except (IndexError, ValueError):
        swap = None
    vm = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout.splitlines()
    page = int(vm[0].split("page size of ")[1].split()[0])
    n = sum(int(l.split(":")[1].strip().rstrip(".")) for l in vm[1:]
            if l.split(":")[0] in ("Pages free", "Pages inactive", "Pages speculative", "Pages purgeable"))
    return swap, round(n * page / 1e9, 1)


def sample(root, out, interval):
    prev_t, prev_c, prev_g = time.time(), procs(), gpu_ns()
    with open(out, "a") as f:
        while True:
            time.sleep(interval)
            try:
                os.kill(root, 0)
            except OSError:
                break
            t, c, g = time.time(), procs(), gpu_ns()
            dt = t - prev_t
            lab = labels(c, root)
            cpu, gpu = {}, {}
            for pid, (_, s, _) in c.items():
                d = s - prev_c[pid][1] if pid in prev_c else s
                if d > 0:
                    cpu[lab[pid]] = cpu.get(lab[pid], 0) + 100 * d / dt
            for pid, ns in g.items():
                d = ns - prev_g.get(pid, ns)
                if d > 0:
                    k = lab.get(pid, "exited")
                    gpu[k] = gpu.get(k, 0) + 100 * d / 1e9 / dt
            dev, ren = util()
            swap, avail = mem()
            top = lambda m: dict(sorted(((k, round(v, 1)) for k, v in m.items() if k != "mine"), key=lambda kv: -kv[1])[:10])
            f.write(json.dumps({"t": round(t, 2), "dt": round(dt, 2),
                                "cpu_mine": round(cpu.get("mine", 0), 1), "cpu_bg": round(sum(v for k, v in cpu.items() if k != "mine"), 1),
                                "gpu_mine": round(gpu.get("mine", 0), 1), "gpu_bg": round(sum(v for k, v in gpu.items() if k != "mine"), 1),
                                "dev": dev, "ren": ren, "swap_mb": swap, "avail_gb": avail,
                                "cpu": top(cpu), "gpu": top(gpu)}) + "\n")
            f.flush()
            prev_t, prev_c, prev_g = t, c, g


def summary(out, marks):
    rows = [json.loads(l) for l in open(out) if l.strip()]
    mk = {}
    for l in open(marks):
        p = l.split()
        if len(p) == 2:
            mk[p[0]] = float(p[1])
    print(f"background load beside WHOLE-CHIP-2 ({len(rows)} samples; CPU in cores, GPU in % of wall time; "
          f"'mine' = the runner's process tree)")
    for w in ("base", "H", "S"):
        a, b = mk.get(w + "0"), mk.get(w + "1")
        if a is None or b is None:
            continue
        r = [x for x in rows if a < x["t"] <= b + 1]
        if not r:
            print(f"{w}: no samples")
            continue
        mean = lambda k: sum(x[k] or 0 for x in r) / len(r)
        mx = lambda k: max(x[k] or 0 for x in r)
        apps = lambda key: sorted(((k, sum(x[key].get(k, 0) for x in r) / len(r)) for k in {k for x in r for k in x[key]}),
                                  key=lambda kv: -kv[1])[:6]
        print(f"{w} ({b - a:.0f} s, {len(r)} samples): bg CPU {mean('cpu_bg') / 100:.2f} cores (max {mx('cpu_bg') / 100:.2f}), "
              f"bg GPU {mean('gpu_bg'):.1f} % (max {mx('gpu_bg'):.1f}); mine CPU {mean('cpu_mine') / 100:.2f} cores, "
              f"GPU {mean('gpu_mine'):.0f} %; device util {mean('dev'):.0f} %; swap {r[0]['swap_mb']:.0f} -> "
              f"{r[-1]['swap_mb']:.0f} MB; available min {min(x['avail_gb'] for x in r):.1f} GB")
        print("   bg CPU: " + ", ".join(f"{k} {v / 100:.2f}" for k, v in apps("cpu")))
        print("   bg GPU: " + ", ".join(f"{k} {v:.1f} %" for k, v in apps("gpu")))
        frz = [(x["t"] - x["dt"] - a, x["dt"]) for x in r if x["dt"] > 3 * min(y["dt"] for y in rows)]
        if frz:   # the sampler is in the runner's tree: a long interval = the tree was frozen (heat watcher SIGSTOP)
            print(f"   FROZEN {sum(d for _, d in frz):.0f} s: " + ", ".join(f"from +{s:.0f} s for {d:.0f} s" for s, d in frz)
                  + f" ({w} is not a clean measurement)")


if __name__ == "__main__":
    if sys.argv[1] == "sample":
        sample(int(sys.argv[2]), sys.argv[3], float(sys.argv[4]) if len(sys.argv) > 4 else 2.0)
    else:
        summary(sys.argv[2], sys.argv[3])
