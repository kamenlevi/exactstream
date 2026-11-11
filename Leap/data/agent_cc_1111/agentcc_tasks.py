"""EvalSuite's Claude Code tasks (PREREG_agentcc_2026-11-11) against a running serve.py. Per task: the template copied to a
fresh folder with git init + one commit; a fresh copy of the base Claude Code config folder; claude -p (the launcher,
sandboxed: writes only in the run folder and temp, network only to localhost, private home folders unreadable) with
stream-json output, 25 turns at most, under the manifest's timeout for this setting (the whole process group is killed at
the timeout); then the task's check, copied in after claude ends and run in the task folder, sandboxed.
Writes <run>/<id>.jsonl, <id>.stderr, <id>.check.txt and one line per task to <run>/results.jsonl.
Usage: agentcc_tasks.py <run> <GB> <manifest> <port> <sandbox.sb> <base config dir> <launcher> [first task id]"""
import json, os, shutil, signal, subprocess, sys, time

RUN, GB, MAN, PORT, SB, CCB, LAUNCH = sys.argv[1:8]
START = sys.argv[8] if len(sys.argv) > 8 else None
tasks = json.load(open(MAN))["tasks"]
if START:
    tasks = tasks[[t["id"] for t in tasks].index(START):]
os.makedirs(os.path.join(RUN, "tasks"), exist_ok=True)
os.makedirs(os.path.join(RUN, "cc_home"), exist_ok=True)


def killgroup(p):
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(p.pid, sig)
        except ProcessLookupError:
            return
        time.sleep(5)


for t in tasks:
    T = os.path.join(RUN, "tasks", t["id"])
    shutil.copytree(t["template"], T)
    for c in (["git", "init", "-q"], ["git", "add", "-A"],
              ["git", "-c", "user.email=agentcc@localhost", "-c", "user.name=agentcc", "commit", "-q", "-m", "template"]):
        subprocess.run(c, cwd=T, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    cch = os.path.join(RUN, "cc_home", t["id"])
    shutil.copytree(CCB, cch, ignore=shutil.ignore_patterns("projects", "sessions", "backups", "shell-snapshots", "todos"))
    env = dict(os.environ, CLAUDE_CONFIG_DIR=cch, EXACTSTREAM_PORT=PORT, CLAUDE_BIN="/Users/kamen/.local/bin/claude")
    cmd = ["sandbox-exec", "-f", SB, LAUNCH, "-p", t["prompt"], "--output-format", "stream-json", "--verbose",
           "--max-turns", "25", "--allowedTools", "Bash", "Edit", "Glob", "Grep", "Read", "Write"]
    timeout = int(t["timeout_s"][GB])
    t0, timed_out = time.time(), False
    print(f"{time.strftime('%H:%M:%S')} {t['id']}: start (timeout {timeout} s)", flush=True)
    with open(os.path.join(RUN, t["id"] + ".jsonl"), "w") as out, open(os.path.join(RUN, t["id"] + ".stderr"), "w") as err:
        p = subprocess.Popen(cmd, cwd=T, env=env, stdin=subprocess.DEVNULL, stdout=out, stderr=err, start_new_session=True)
        try:
            rc = p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            killgroup(p)
            rc = p.wait()
    wall = time.time() - t0
    shutil.copy(t["check_file"], os.path.join(T, t["check_copy_as"]))
    with open(os.path.join(RUN, t["id"] + ".check.txt"), "w") as co:
        c = subprocess.Popen(["sandbox-exec", "-f", SB, "/bin/sh", "-c", t["check"]], cwd=T, stdout=co, stderr=subprocess.STDOUT,
                             stdin=subprocess.DEVNULL, start_new_session=True)
        try:
            crc = c.wait(timeout=300)
        except subprocess.TimeoutExpired:
            killgroup(c)
            c.wait()
            crc = 124
    rec = {"id": t["id"], "claude_rc": rc, "timed_out": timed_out, "wall_s": round(wall, 1), "check_rc": crc}
    with open(os.path.join(RUN, "results.jsonl"), "a") as rf:
        rf.write(json.dumps(rec) + "\n")
    print(f"{time.strftime('%H:%M:%S')} {t['id']}: claude rc {rc}{' TIMED OUT' if timed_out else ''}, {wall:.0f} s, check rc {crc}", flush=True)
print("TASKS DONE", flush=True)
