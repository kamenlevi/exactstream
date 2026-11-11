"""Facts for the 3-hourly laptop + research health review (standing duty taken over from the orchestrator, 2026-10-25).
Read-only: prints what the review judges; the reviewer decides, fixes what is clearly inside RULES.md, and appends one
line to ~/.local/share/quiet/health_reviews.log. Checklist: ~/BigMoE/ops/HEALTH_REVIEW.md.
usage: taskpolicy -b python3 ~/BigMoE/ops/health_review.py"""
import os, re, subprocess, time

Q = os.path.expanduser("~/.local/share/quiet")
LOGS = os.path.expanduser("~/Library/Logs")
REVIEWS = os.path.join(Q, "health_reviews.log")


def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60).stdout
    except Exception as e:
        return f"<{e}>"


def tail(path, n):
    try:
        with open(path, errors="replace") as f:
            return f.readlines()[-n:]
    except OSError:
        return []


print("== now", time.strftime("%Y-%m-%d %H:%M %Z"))
last = tail(REVIEWS, 1)
last_ts = last[0][:16] if last else "1970-01-01 00:00"
m = re.search(r"swap ([0-9.]+) GB", last[0]) if last else None
last_swap = float(m.group(1)) if m else None
print(f"last review {last_ts}" + (f" (swap then {last_swap} GB)" if last_swap is not None else ""))

print("\n== HEALTH.md")
for l in tail(os.path.join(Q, "HEALTH.md"), 40):
    if l.startswith(("# ", "**Verdict", "**Now", "**Last", "**Protective", "**SSD")):
        print(l.rstrip())

print("\n== health_alerts.log since the last review")
al = [l.rstrip() for l in tail(os.path.join(LOGS, "health_alerts.log"), 400) if l[:16] >= last_ts]
print("\n".join(al[-15:]) or "(none)")

print("\n== fanwatch.log (last ~3.3 h): events + fan swings")
fw = tail(os.path.join(LOGS, "fanwatch.log"), 600)
ev = [l.rstrip() for l in fw if re.search(r"HOT|WARM|SAFETY|PAUSE|STOP|resume|cap|COOL", l)]
print("\n".join(ev[-12:]) or "(no events)")
fans = []
for l in fw:
    mm = re.match(r"(\d\d):(\d\d):(\d\d) .* fans (\d+) (\d+)", l)
    if mm:
        fans.append((int(mm.group(1)) * 3600 + int(mm.group(2)) * 60 + int(mm.group(3)), max(int(mm.group(4)), int(mm.group(5))), l[:8]))
# fanwatch lines carry no date: anchor the last sample at the log's mtime, walk back, count from the last review on
# (a stopped controller, e.g. in a quiet window, leaves no sample after the review: then nothing is new)
ab, start = [], len(fans)
if fans:
    mt = os.path.getmtime(os.path.join(LOGS, "fanwatch.log"))
    d0 = time.mktime(time.localtime(mt)[:3] + (0, 0, 0, 0, 0, -1))
    a = d0 + fans[-1][0] if d0 + fans[-1][0] <= mt + 5 else d0 - 86400 + fans[-1][0]
    for i in range(len(fans) - 1, -1, -1):
        ab.append(a)
        if i:
            a -= (fans[i][0] - fans[i - 1][0]) % 86400
    ab.reverse()
    try:
        lt = time.mktime(time.strptime(last_ts, "%Y-%m-%d %H:%M"))
    except ValueError:
        lt = 0
    start = next((i for i, a in enumerate(ab) if a >= lt), len(fans))
big, worst, swings = 0, (0, ""), []
for i, (t, f, lab) in enumerate(fans):
    if i < start:
        continue
    for j in range(i - 1, -1, -1):
        dt = (t - fans[j][0]) % 86400
        if dt > 60:
            break
        if fans[j][1] == 0 or f == 0:           # a controller restart logs 0 rpm: not a swing
            continue
        d = abs(f - fans[j][1])
        if d > worst[0]:
            worst = (d, lab)
        if d > 500:
            big += 1
            swings.append(f"{lab}{'+' if f > fans[j][1] else '-'}{d}")
            break
if fans:
    age = time.time() - ab[-1]
    now = "now" if age < 120 else f"last sample {fans[-1][2]} ({age / 3600:.1f} h ago):"
    print(f"fan samples {len(fans)}: {now} {fans[-1][1]} rpm, range {min(x[1] for x in fans)}-{max(x[1] for x in fans)}; "
          f"swings > 500 rpm within 60 s since the last review: {big} (largest {worst[0]} rpm at {worst[1]})")
if swings:                                     # + = up, - = down; alternating signs = oscillation (what Kamen hates)
    ups, downs = sum(x[8] == "+" for x in swings), sum(x[8] == "-" for x in swings)
    print(f"  swings: {ups} up / {downs} down: " + " ".join(swings[:14]) + (" ..." if len(swings) > 14 else ""))
print("PAUSE file:", "PRESENT" if os.path.exists(os.path.join(Q, "PAUSE")) else "absent",
      "| FAN_CAP:", (open(os.path.join(Q, "FAN_CAP")).read().strip() if os.path.exists(os.path.join(Q, "FAN_CAP")) else "absent"))
# controller liveness (10-25 19:27-22:13 it was dead and the fans sat at its last fixed RPM under full GPU load)
ctl = sh("pgrep -f fanwatch_ctl.py").split()
wrap = sh("ps -axo pid=,command= | grep '[f]anwatch.sh'").strip().splitlines()
fwp = os.path.join(LOGS, "fanwatch.log")
age = time.time() - os.path.getmtime(fwp) if os.path.exists(fwp) else 1e9
st = [i for i, l in enumerate(fw) if re.search(r"fanwatch v\d+ start", l)]
fb = [l for l in fw[st[-1]:] if "did not follow fixed mode" in l] if st else []
try:
    import json
    mode = ",".join(sorted({str(x.get("mode")) for x in json.load(open("/Library/Application Support/Baro/helper-status.json")).get("fans", [])}))
except Exception:
    mode = "?"
quiet_win = sh("ps -axo command= | grep -E '[q]uiet_window.sh|[q]uiet_auto.sh|[n]ight_silent.sh'").strip()
bad = not ctl or age > 120 or fb
print(f"fan controller: {'running pid ' + ' '.join(ctl) if ctl else 'NOT RUNNING'}, log age {age:.0f} s, Baro mode {mode}, "
      + (f"FALLBACK {fb[-1][:19]}" if fb else "no fixed-mode fallback since its start")
      + (" | quiet window running" if quiet_win else "")
      + (("  (expected: the window stops it and restarts it at its end)" if quiet_win else
          "  -> DEAD/STALE: HEALTH_REVIEW.md step 1 (restart rule)") if bad else ""))
for w in wrap:
    if "quiet/fanwatch.sh" not in w:              # quiet windows stop it with pkill -f quiet/fanwatch.sh: this one escapes
        print(f"  wrapper started without its full path ({w.strip()}): quiet windows' pkill won't stop it")

print("\n== swap / disk / memory")
sw = sh("sysctl vm.swapusage").strip()
ms = re.search(r"used = ([0-9.]+)M", sw)
swap_gb = float(ms.group(1)) / 1024 if ms else None
print(sw, f"-> {swap_gb:.1f} GB" if swap_gb is not None else "",
      f"(change since last review {swap_gb - last_swap:+.1f} GB)" if swap_gb is not None and last_swap is not None else "")
print(sh("df -g / | tail -1").strip(), "  (GB; floor 100, stop big writes at 110)")
print(sh("memory_pressure | tail -1").strip())

print("\n== processes: >= 50 % CPU, >= 4 GB RSS, or stopped (T)")
skip = re.compile(r"WindowServer|kernel_task|Google Chrome|/Claude.app/|Spotify|mds_stores|backupd|softwareupdate")
rows = []
for l in sh("ps -axo pid=,ppid=,%cpu=,rss=,etime=,stat=,command=").splitlines():
    p = l.split(None, 6)
    if len(p) < 7:
        continue
    pid, ppid, cpu, rss, et, stat, cmd = p
    cpu, rss = float(cpu), int(rss) / 1048576
    if skip.search(cmd):
        continue
    if cpu >= 50 or rss >= 4 or "T" in stat:
        cwd = sh(f"lsof -a -p {pid} -d cwd -Fn 2>/dev/null | grep '^n' | head -1").strip()[1:]
        rows.append((cpu, f"pid {pid} ppid {ppid} {cpu:5.0f}% {rss:5.1f} GB {stat:4s} {et:>11s} cwd={cwd.replace(os.path.expanduser('~'), '~')}  {cmd[:110]}"))
for _, r in sorted(rows, reverse=True)[:15]:
    print(r)
over4 = [r for c, r in rows if c > 400]
print(f"CPU jobs above 4 cores: {len(over4)}")

print("\n== peer: GPU queue / locks / CPU users")
for l in sh("peer ls 2>&1").splitlines():
    if re.search(r"waiting for a quiet|benchmarking|lock|Machine now|Biggest users", l):
        print(l[:300])
