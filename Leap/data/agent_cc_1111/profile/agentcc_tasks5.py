"""EvalSuite's Claude Code tasks (PREREG_agentcc_2026-11-11) against a running serve.py. Per task: the template copied to a
fresh folder with git init + one commit; a fresh copy of the base Claude Code config folder; claude -p (the launcher,
sandboxed, network only to localhost, private home folders unreadable) with stream-json output, 25 turns at most, under
the manifest's timeout for this setting (the whole process group is killed at the timeout, and after a normal end too);
then the task's check, copied in after claude ends and run in the task folder, sandboxed.
Writes <run>/<id>.jsonl, <id>.stderr, <id>.check.txt, <run>/sandbox_<id>.sb and one line per task to <run>/results.jsonl.
v2: one profile per task = the run's sandbox.sb + read denies (file contents and listings; metadata stays readable for
path lookups) of EvalSuite's agentcc folder and of the run folder outside the task's own folders; zsh's TMPPREFIX in
claude's environment (a .zshrc export does not reach Claude Code's Bash tool). v3: CLAUDE_BIN from the environment.
v4 (a security review of Leap 68ec944): the agent writes only in its own task folder, its own config folder and its own
temp folder <run>/cctmp/<id> (CLAUDE_CODE_TMPDIR and TMPPREFIX point there; HOME is read-only); the run's base profile
must not allow writes to <run> (checked); every file this unsandboxed runner writes in <run> is opened with O_NOFOLLOW
(O_EXCL when new), and the check is copied into the task folder only after any entry at that name is removed, with
O_CREAT|O_EXCL|O_NOFOLLOW, so a link the agent planted is never followed outside the sandbox.
v5 (a second review): after claude ends, every sandboxed process with its cwd or an open file in the task's folders is
killed (sandbox_check(pid) == 1 + lsof; rounds until none; logged to <run>/lingering.jsonl), and the check no longer runs
in the agent's folder: the runner copies the task folder to <run>/checks/<id> (regular files and folders; links copied
as links, never followed; other file types skipped and logged), places the check there through a directory fd, and runs
it sandboxed with <run>/sandbox_check_<id>.sb, whose only writable folder is that copy (+ /private/var/folders, /dev).
A process that escaped the sweep can still write in the agent's own folders, but not in the copy the check grades.
Usage: agentcc_tasks5.py <run> <GB> <manifest> <port> <sandbox.sb> <base config dir> <launcher> [first task id]"""
import ctypes, json, os, shutil, signal, stat, subprocess, sys, time

RUN, GB, MAN, PORT, SB, CCB, LAUNCH = sys.argv[1:8]
RUN = os.path.realpath(RUN)
START = sys.argv[8] if len(sys.argv) > 8 else None
tasks = json.load(open(MAN))["tasks"]
if START:
    tasks = tasks[[t["id"] for t in tasks].index(START):]
BASE = open(SB).read()
if f'(subpath "{RUN}")' in BASE:
    sys.exit(f"agentcc_tasks4.py: the base profile {SB} names the run folder; the per-task rules add the task's own folders")
for d in ("tasks", "cc_home", "cctmp", "checks"):
    os.makedirs(os.path.join(RUN, d), exist_ok=True)
DENY = sorted({os.path.realpath(os.path.dirname(MAN)), "/Users/kamen/EvalSuite/agentcc"})   # EvalSuite's agentcc + the manifest's folder
NEW = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW


def wopen(path, flags=NEW, mode="w"):
    """a runner output, opened without following a link at its name (O_EXCL: it must not exist yet)"""
    return os.fdopen(os.open(path, flags, 0o644), mode)


def killgroup(p):
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(p.pid, sig)
        except ProcessLookupError:
            return
        time.sleep(5)


LIBSYS = ctypes.CDLL("/usr/lib/libSystem.B.dylib")


def sandboxed(pid):
    try:
        return LIBSYS.sandbox_check(ctypes.c_int(pid), None, ctypes.c_int(0)) == 1
    except Exception:
        return False


def lingering(dirs):
    """this user's sandboxed processes whose cwd or an open file is in one of dirs (the unsandboxed server is not one)"""
    out = subprocess.run(["lsof", "-n", "-P", "-w", "-u", str(os.getuid()), "-F", "pn"], capture_output=True, text=True,
                         timeout=120).stdout
    pids, pid = set(), None
    for line in out.splitlines():
        if line.startswith("p"):
            pid = int(line[1:])
        elif line.startswith("n") and pid is not None and any(line[1:] == d or line[1:].startswith(d + "/") for d in dirs):
            pids.add(pid)
    return sorted(q for q in pids if q != os.getpid() and sandboxed(q))


def sweep(tid, dirs):
    """kill what the task left running; returns the pids still there after 5 rounds"""
    killed, left = [], []
    for _ in range(5):
        left = lingering(dirs)
        if not left:
            break
        for q in left:
            try:
                os.kill(q, signal.SIGKILL)
            except ProcessLookupError:
                pass
        killed += left
        time.sleep(1)
    else:
        left = lingering(dirs)
    if killed or left:
        with wopen(os.path.join(RUN, "lingering.jsonl"), os.O_WRONLY | os.O_CREAT | os.O_APPEND | os.O_NOFOLLOW) as lf:
            lf.write(json.dumps({"id": tid, "killed": sorted(set(killed)), "left": left}) + "\n")
    return left


def snapshot(src, dst):
    """copy the agent's folder for the check, every step through directory fds opened with O_NOFOLLOW (a link swapped in
    mid-copy is never followed): folders, regular files (mode bits & 0o777 and times kept), links as links; other file
    types skipped"""
    skipped = []

    def copy_dir(sfd, dpath, rel):
        for e in os.scandir(sfd):
            name, r, d = e.name, os.path.join(rel, e.name), os.path.join(dpath, e.name)
            try:
                st = os.stat(name, dir_fd=sfd, follow_symlinks=False)
            except FileNotFoundError:
                continue
            if stat.S_ISLNK(st.st_mode):
                os.symlink(os.readlink(name, dir_fd=sfd), d)
            elif stat.S_ISDIR(st.st_mode):
                try:
                    cfd = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=sfd)
                except OSError:
                    skipped.append(r)
                    continue
                try:
                    os.mkdir(d)
                    copy_dir(cfd, d, r)
                finally:
                    os.close(cfd)
            elif stat.S_ISREG(st.st_mode):
                try:
                    fi = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=sfd)
                except OSError:
                    skipped.append(r)
                    continue
                try:
                    fst = os.fstat(fi)
                    if not stat.S_ISREG(fst.st_mode):
                        skipped.append(r)
                        continue
                    with wopen(d, mode="wb") as fo, os.fdopen(os.dup(fi), "rb") as fr:
                        shutil.copyfileobj(fr, fo)
                        os.fchmod(fo.fileno(), fst.st_mode & 0o777)
                    os.utime(d, ns=(fst.st_atime_ns, fst.st_mtime_ns), follow_symlinks=False)
                finally:
                    os.close(fi)
            else:
                skipped.append(r)

    os.mkdir(dst)
    rfd = os.open(src, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        copy_dir(rfd, dst, ".")
    finally:
        os.close(rfd)
    return skipped


def place_check(src, folder, name):
    """put the check in the runner's copy through a directory fd: any entry at its name is removed, never followed"""
    if os.sep in name or name in ("", ".", ".."):
        raise ValueError(f"check_copy_as {name!r} is not a plain file name")
    dfd = os.open(folder, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        if not stat.S_ISDIR(os.fstat(dfd).st_mode):
            raise ValueError(f"{folder} is not a folder")
        try:
            if stat.S_ISDIR(os.stat(name, dir_fd=dfd, follow_symlinks=False).st_mode):
                raise ValueError(f"{name} is a folder")
            os.unlink(name, dir_fd=dfd)
        except FileNotFoundError:
            pass
        data = open(src, "rb").read()
        with os.fdopen(os.open(name, NEW, 0o644, dir_fd=dfd), "wb") as f:
            f.write(data)
    finally:
        os.close(dfd)


for t in tasks:
    T = os.path.join(RUN, "tasks", t["id"])
    shutil.copytree(t["template"], T)
    for c in (["git", "init", "-q"], ["git", "add", "-A"],
              ["git", "-c", "user.email=agentcc@localhost", "-c", "user.name=agentcc", "commit", "-q", "-m", "template"]):
        subprocess.run(c, cwd=T, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    cch = os.path.join(RUN, "cc_home", t["id"])
    shutil.copytree(CCB, cch, ignore=shutil.ignore_patterns("projects", "sessions", "backups", "shell-snapshots", "todos"))
    CT = os.path.join(RUN, "cctmp", t["id"])
    os.makedirs(CT)
    tsb = os.path.join(RUN, "sandbox_" + t["id"] + ".sb")
    with wopen(tsb) as f:
        f.write(BASE.rstrip() + "\n"
                f'(allow file-write* (subpath "{T}") (subpath "{cch}") (subpath "{CT}"))\n'
                '(deny file-read-data ' + ' '.join(f'(subpath "{d}")' for d in DENY) + f' (subpath "{RUN}"))\n'
                f'(allow file-read-data (subpath "{T}") (subpath "{cch}") (subpath "{CT}")'
                f' (subpath "{os.path.join(RUN, "home")}"))\n')
    env = dict(os.environ, CLAUDE_CONFIG_DIR=cch, EXACTSTREAM_PORT=PORT, CLAUDE_BIN=os.environ.get("CLAUDE_BIN", "/Users/kamen/.local/bin/claude"),
               CLAUDE_CODE_TMPDIR=CT, TMPPREFIX=os.path.join(CT, "zsh"))
    cmd = ["sandbox-exec", "-f", tsb, LAUNCH, "-p", t["prompt"], "--output-format", "stream-json", "--verbose",
           "--max-turns", "25", "--allowedTools", "Bash", "Edit", "Glob", "Grep", "Read", "Write"]
    timeout = int(t["timeout_s"][GB])
    t0, timed_out = time.time(), False
    print(f"{time.strftime('%H:%M:%S')} {t['id']}: start (timeout {timeout} s)", flush=True)
    with wopen(os.path.join(RUN, t["id"] + ".jsonl")) as out, wopen(os.path.join(RUN, t["id"] + ".stderr")) as err:
        p = subprocess.Popen(cmd, cwd=T, env=env, stdin=subprocess.DEVNULL, stdout=out, stderr=err, start_new_session=True)
        try:
            rc = p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            killgroup(p)
            rc = p.wait()
        wall = time.time() - t0
        killgroup(p)                                      # anything the task left running in its group stops here
    left = sweep(t["id"], [T, cch, CT])                   # and any sandboxed process still in its folders
    K = os.path.join(RUN, "checks", t["id"])
    note = None
    if left:
        note = f"check not run: processes still in the task's folders after 5 rounds: {left}"
    else:
        try:
            skipped = snapshot(T, K)
            if skipped:
                print(f"{t['id']}: snapshot skipped {len(skipped)} non-regular entries: {skipped[:5]}", flush=True)
            place_check(t["check_file"], K, t["check_copy_as"])
        except Exception as e:                            # OSError, ValueError, RecursionError on a very deep tree
            note = f"check not placed: {type(e).__name__}: {e}"
    ksb = os.path.join(RUN, "sandbox_check_" + t["id"] + ".sb")
    with wopen(ksb) as f:
        f.write(BASE.rstrip() + "\n"
                f'(allow file-write* (subpath "{K}"))\n'
                '(deny file-read-data ' + ' '.join(f'(subpath "{d}")' for d in DENY) + f' (subpath "{RUN}"))\n'
                f'(allow file-read-data (subpath "{K}") (subpath "{os.path.join(RUN, "home")}"))\n')
    with wopen(os.path.join(RUN, t["id"] + ".check.txt")) as co:
        if note is None:
            c = subprocess.Popen(["sandbox-exec", "-f", ksb, "/bin/sh", "-c", t["check"]], cwd=K, stdout=co, stderr=subprocess.STDOUT,
                                 stdin=subprocess.DEVNULL, start_new_session=True)
            try:
                crc = c.wait(timeout=300)
            except subprocess.TimeoutExpired:
                killgroup(c)
                c.wait()
                crc = 124
        else:
            co.write(note + "\n")
            crc = 125
    sweep(t["id"] + ":check", [K])
    rec = {"id": t["id"], "claude_rc": rc, "timed_out": timed_out, "wall_s": round(wall, 1), "check_rc": crc}
    with wopen(os.path.join(RUN, "results.jsonl"), os.O_WRONLY | os.O_CREAT | os.O_APPEND | os.O_NOFOLLOW) as rf:
        rf.write(json.dumps(rec) + "\n")
    print(f"{time.strftime('%H:%M:%S')} {t['id']}: claude rc {rc}{' TIMED OUT' if timed_out else ''}, {wall:.0f} s, check rc {crc}", flush=True)
print("TASKS DONE", flush=True)
