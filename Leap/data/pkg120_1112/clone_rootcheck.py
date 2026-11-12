"""clone_rootcheck.py <clone>: every tracked file that differs from HEAD must differ only by install.sh's @ROOT@ rewrite
(the clone's absolute path where the package says @ROOT@). Prints '<n> changed, <m> by the @ROOT@ rewrite only' and exits 1
if any other change is found (and names it)."""
import hashlib, os, subprocess, sys
cl = os.path.realpath(sys.argv[1])
ch = [l[3:] for l in subprocess.run(["git", "-C", cl, "status", "--porcelain", "--untracked-files=no"], capture_output=True,
                                     text=True, check=True).stdout.splitlines() if l.strip()]
bad = []
for f in ch:
    want = subprocess.run(["git", "-C", cl, "rev-parse", f"HEAD:{f}"], capture_output=True, text=True).stdout.strip()
    try:
        b = open(os.path.join(cl, f), "rb").read().replace(cl.encode(), b"@ROOT@")
    except FileNotFoundError:
        bad.append(f + " (deleted)"); continue
    if hashlib.sha1(b"blob %d\0" % len(b) + b).hexdigest() != want:
        bad.append(f)
print(f"{len(ch)} tracked files changed, {len(ch) - len(bad)} by the @ROOT@ rewrite only" + (f"; OTHER CHANGES: {bad[:10]}" if bad else ""))
sys.exit(1 if bad else 0)
