"""Per-call summary of an agent run's Claude Code transcripts, for the record (the 04:0x rule: raw stream-json transcripts
and request logs stay out of git; commit this summary instead). One line per task: id, num_turns, the result subtype and
its calls in order, each with the tool name, error flag, an error class (a fixed label, never the text),
dangerouslyDisableSandbox, and outside_task: whether the call names a path outside the task's own folder: anything in
the run folder outside tasks/<id>/ (other tasks, transcripts, logs, checks), ~ or $HOME, a ../ step, or an absolute
path other than system prefixes and scratch temp (/tmp, /private/var/folders); glob patterns (find -path '*/.git/*')
are not paths. No inputs or outputs are kept.
Usage: summarize_calls.py <run folder> <out.jsonl>  (reads <run>/t??_*.jsonl; the task folders are <run>/tasks/<id>/)"""
import glob, json, os, re, sys

RUN, OUT = sys.argv[1].rstrip("/"), sys.argv[2]
SYS = ("/usr/", "/bin/", "/sbin/", "/opt/homebrew/", "/System/", "/Library/Developer/", "/dev/", "/private/var/folders/",
       "/var/folders/", "/etc/")
PATH_RE = re.compile(r"(?<![\w.])(/[^\s'\"`;|&<>(){}$,]+)")


def err_class(text):
    for pat, lab in (("EPERM: operation not permitted, mkdir", "EPERM mkdir"),
                     ("can't create temp file for here document", "heredoc temp file denied"),
                     ("has not been read yet", "file not read first"), ("command not found", "command not found"),
                     ("illegal option", "illegal option"), ("No such file or directory", "no such file"),
                     ("Operation not permitted", "operation not permitted"), ("Permission denied", "permission denied"),
                     ("timed out", "timeout")):
        if pat in text:
            return lab
    m = re.search(r"Exit code (\d+)", text)
    return f"exit {m.group(1)}" if m else "other"


def outside(tool, inp, task_dir):
    vals = []
    for k in ("file_path", "path", "notebook_path"):
        if isinstance(inp.get(k), str):
            vals.append(inp[k])
    cmd = inp.get("command") if tool == "Bash" else None
    if isinstance(cmd, str):
        if re.search(r"(^|[\s='\"])~(/|\s|$)|\$HOME|\$\{HOME\}|(^|[\s/'\"])\.\./", cmd):
            return True
        vals += PATH_RE.findall(cmd)
    run = os.path.dirname(os.path.dirname(task_dir))
    for p in vals:
        if not p.startswith("/") or "*" in p or "?" in p:
            continue
        if p == "/tmp" or p.startswith("/tmp/"):
            p = "/private" + p
        if p == task_dir or p.startswith(task_dir + "/"):
            continue
        if p == run or p.startswith(run + "/"):
            return True
        if p.startswith(SYS) or p == "/private/tmp" or p.startswith("/private/tmp/"):
            continue
        return True
    return False


rows = []
for f in sorted(glob.glob(f"{RUN}/t[0-9][0-9]_*.jsonl")):
    tid = os.path.basename(f)[:-6]
    task_dir = f"{RUN}/tasks/{tid}"
    calls, idx, turns, sub = [], {}, None, None
    for line in open(f):
        try:
            d = json.loads(line)
        except Exception:
            continue
        if d.get("type") == "assistant":
            for c in d.get("message", {}).get("content", []) or []:
                if c.get("type") == "tool_use":
                    inp = c.get("input") or {}
                    idx[c.get("id")] = len(calls)
                    calls.append(dict(tool=c.get("name"), error=False, error_class=None,
                                      disable_sandbox=bool(inp.get("dangerouslyDisableSandbox", False)),
                                      outside_task=outside(c.get("name"), inp, task_dir)))
        elif d.get("type") == "user" and isinstance(d.get("message", {}).get("content"), list):
            for c in d["message"]["content"]:
                if c.get("type") == "tool_result" and c.get("tool_use_id") in idx and c.get("is_error"):
                    t = c.get("content")
                    t = t if isinstance(t, str) else json.dumps(t)
                    k = idx[c["tool_use_id"]]
                    calls[k]["error"] = True
                    calls[k]["error_class"] = err_class(t)
        elif d.get("type") == "result":
            turns, sub = d.get("num_turns"), d.get("subtype")
    rows.append(dict(id=tid, num_turns=turns, subtype=sub, calls=calls))
with open(OUT, "w") as o:
    for r in rows:
        o.write(json.dumps(r) + "\n")
n = sum(len(r["calls"]) for r in rows)
print(f"{OUT}: {len(rows)} tasks, {n} calls, errors {sum(c['error'] for r in rows for c in r['calls'])}, "
      f"outside_task {sum(c['outside_task'] for r in rows for c in r['calls'])} "
      f"({', '.join(sorted({r['id'] for r in rows for c in r['calls'] if c['outside_task']}))})")
