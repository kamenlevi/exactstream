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
Usage: agentcc_tasks4.py <run> <GB> <manifest> <port> <sandbox.sb> <base config dir> <launcher> [first task id]"""
import json, os, shutil, signal, subprocess, sys, time

RUN, GB, MAN, PORT, SB, CCB, LAUNCH = sys.argv[1:8]
RUN = os.path.realpath(RUN)
START = sys.argv[8] if len(sys.argv) > 8 else None
tasks = json.load(open(MAN))["tasks"]
if START:
    tasks = tasks[[t["id"] for t in tasks].index(START):]
BASE = open(SB).read()
if f'(subpath "{RUN}")' in BASE:
    sys.exit(f"agentcc_tasks4.py: the base profile {SB} names the run folder; the per-task rules add the task's own folders")
for d in ("tasks", "cc_home", "cctmp"):
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


def place_check(src, T, name):
    """copy the check into the task folder, which the agent could write: never through an entry the agent left there"""
    if os.sep in name or name in ("", ".", ".."):
        raise ValueError(f"check_copy_as {name!r} is not a plain file name")
    if os.path.islink(T) or os.path.realpath(T) != T:
        raise ValueError(f"task folder {T} is not a plain folder")
    dst = os.path.join(T, name)
    if os.path.lexists(dst):
        if os.path.isdir(dst) and not os.path.islink(dst):
            raise ValueError(f"{dst} is a folder")
        os.unlink(dst)
    data = open(src, "rb").read()
    with wopen(dst, mode="wb") as f:
        f.write(data)


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
    try:
        place_check(t["check_file"], T, t["check_copy_as"])
        note = None
    except (OSError, ValueError) as e:
        note = f"check not placed: {e}"
    with wopen(os.path.join(RUN, t["id"] + ".check.txt")) as co:
        if note is None:
            c = subprocess.Popen(["sandbox-exec", "-f", tsb, "/bin/sh", "-c", t["check"]], cwd=T, stdout=co, stderr=subprocess.STDOUT,
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
    rec = {"id": t["id"], "claude_rc": rc, "timed_out": timed_out, "wall_s": round(wall, 1), "check_rc": crc}
    with wopen(os.path.join(RUN, "results.jsonl"), os.O_WRONLY | os.O_CREAT | os.O_APPEND | os.O_NOFOLLOW) as rf:
        rf.write(json.dumps(rec) + "\n")
    print(f"{time.strftime('%H:%M:%S')} {t['id']}: claude rc {rc}{' TIMED OUT' if timed_out else ''}, {wall:.0f} s, check rc {crc}", flush=True)
print("TASKS DONE", flush=True)
