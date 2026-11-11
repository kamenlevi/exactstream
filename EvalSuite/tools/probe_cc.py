#!/usr/bin/env python3
"""Claude Code tool-call probe (docs/PREREG_agentcc_2026-11-11.md): 30 calls on serve.py's Anthropic endpoint with Claude Code's
real tools array (data/agentcc/cc_tools_2.1.321.json: 14 tools, taken verbatim from a Claude Code 2.1.321 session by Leap).
Each case wants exactly one of Read / Bash / Grep / Glob / Write / Edit (5 each; the Edit cases carry a prior Read turn, as
Claude Code requires one). A call PASSES when: a tool_use came back; every tool name used is in the tools array (no invented
tool); the first tool_use is the expected tool; its input fits that tool's input_schema (required keys, types, enums); the
case's own checks hold; file_path is absolute for Read / Write / Edit. Usage: probe_cc.py <port> <out.jsonl>"""
import hashlib, json, os, re, sys, time, urllib.request

PORT, OUT = int(sys.argv[1]), sys.argv[2]
TF = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "agentcc", "cc_tools_2.1.321.json")
TOOLS = json.load(open(TF))
NAMES = {t["name"] for t in TOOLS}
SCHEMA = {t["name"]: t["input_schema"] for t in TOOLS}
W = "/work/demo"
SYSTEM = (f"You are Claude Code, a command-line coding assistant. Use the tools to act on the user's request. The working "
          f"directory is {W}, a git repository containing src/app.py, src/utils.py, tests/test_app.py, README.md and "
          f"config.json. Always use absolute paths with file tools.")
end = lambda suf: (lambda a: str(a.get("file_path", "")).endswith(suf))
has = lambda k, *subs: (lambda a: all(s.lower() in str(a.get(k, "")).lower() for s in subs))
both = lambda *fs: (lambda a: all(f(a) for f in fs))
# amendment 1 (Wed 11-11, after the 32 GB run): any ls flag group with a / A, or --all ("ls -la" was rejected)
lsall = lambda a: bool(re.search(r"\bls\b[^|;&]*\s(-[A-Za-z]*[aA][A-Za-z]*|--all)\b", str(a.get("command", ""))))
CASES = [  # (prompt, expected tool, check on its input, prior Read (path, content) for Edit cases)
    ("Show me src/app.py.", "Read", end("src/app.py"), None),
    ("Open the README.", "Read", end("README.md"), None),
    ("What's in config.json?", "Read", end("config.json"), None),
    ("Read the first 20 lines of src/utils.py.", "Read", both(end("src/utils.py"), lambda a: a.get("limit") in (None, 20)), None),
    ("Look at tests/test_app.py.", "Read", end("tests/test_app.py"), None),
    ("Run the tests with pytest.", "Bash", has("command", "pytest"), None),
    ("Show me the git status.", "Bash", has("command", "git status"), None),
    ("Count the lines in src/app.py with wc.", "Bash", has("command", "wc", "app.py"), None),
    ("Install the requests package with pip.", "Bash", has("command", "pip", "install", "requests"), None),
    ("List the files in this directory, including hidden ones.", "Bash", lsall, None),
    ("Find every TODO comment in the project.", "Grep", has("pattern", "TODO"), None),
    ("Which files mention database_url?", "Grep", has("pattern", "database_url"), None),
    ("Search the src folder for functions whose names start with parse_.", "Grep", has("pattern", "parse_"), None),
    ("Find every call to print( in the Python files.", "Grep", has("pattern", "print"), None),
    ("Find lines containing FIXME, ignoring case.", "Grep", has("pattern", "fixme"), None),
    ("List all Python files in the project.", "Glob", has("pattern", ".py"), None),
    ("Find all JSON files.", "Glob", has("pattern", ".json"), None),
    ("Find every test file in the tests folder.", "Glob", lambda a: "test" in (str(a.get("pattern", "")) + str(a.get("path", ""))), None),
    ("Find all Markdown files.", "Glob", has("pattern", ".md"), None),
    ("Find all YAML files, both .yml and .yaml.", "Glob", lambda a: "y" in str(a.get("pattern", "")).lower() and "ml" in str(a.get("pattern", "")).lower(), None),
    ("Create a file notes.txt containing exactly: hello", "Write", both(end("notes.txt"), lambda a: str(a.get("content", "")).strip() == "hello"), None),
    ("Create a .gitignore with the single line __pycache__/", "Write", both(end(".gitignore"), lambda a: str(a.get("content", "")).strip() == "__pycache__/"), None),
    ("Create src/version.py that sets VERSION = '1.0.0'.", "Write", both(end("src/version.py"), has("content", "VERSION", "1.0.0")), None),
    ("Create todo.md whose first line is the heading # Todo", "Write", both(end("todo.md"), lambda a: str(a.get("content", "")).lstrip().startswith("# Todo")), None),
    ("Create data/sample.csv with the header line name,age", "Write", both(end("data/sample.csv"), lambda a: str(a.get("content", "")).strip().startswith("name,age")), None),
    ("In src/utils.py, change the default retries from 3 to 5.", "Edit",
     both(end("src/utils.py"), has("old_string", "retries=3"), has("new_string", "retries=5")),
     ("src/utils.py", "def connect(host, retries=3):\n    return host, retries\n")),
    ("In src/app.py, rename the variable cnt to count.", "Edit",
     both(end("src/app.py"), has("old_string", "cnt"), has("new_string", "count")),
     ("src/app.py", "def tally(xs):\n    cnt = len(xs)\n    return cnt\n")),
    ("In config.json, set debug to false.", "Edit",
     both(end("config.json"), has("old_string", "true"), has("new_string", "false")),
     ("config.json", '{\n  "name": "demo",\n  "debug": true\n}\n')),
    ("In README.md, change the title to # Demo App", "Edit",
     both(end("README.md"), has("new_string", "# Demo App")),
     ("README.md", "# demo\n\nA demo project.\n")),
    ("In tests/test_app.py, fix the expected value: tally([1, 2]) is 2, not 3.", "Edit",
     both(end("tests/test_app.py"), has("old_string", "3"), has("new_string", "2")),
     ("tests/test_app.py", "from src.app import tally\n\n\ndef test_tally():\n    assert tally([1, 2]) == 3\n")),
]
TYPES = {"string": str, "number": (int, float), "integer": int, "boolean": bool, "array": list, "object": dict}


def fits(name, a):
    s = SCHEMA[name]
    if not isinstance(a, dict):
        return "input is not an object"
    for k in s.get("required", []):
        if k not in a:
            return f"missing {k}"
    for k, v in a.items():
        p = s["properties"].get(k)
        if p is None:
            if s.get("additionalProperties") is False:
                return f"unknown key {k}"
            continue
        t = TYPES.get(p.get("type"))
        if t and (not isinstance(v, t) or (p.get("type") in ("number", "integer") and isinstance(v, bool))):
            return f"{k} is not {p.get('type')}"
        if "enum" in p and v not in p["enum"]:
            return f"{k} not in enum"
    return ""


def post(body):
    r = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/messages", data=json.dumps(body).encode(),
                               headers={"content-type": "application/json", "anthropic-version": "2023-06-01", "x-api-key": "local"})
    with urllib.request.urlopen(r, timeout=900) as f:
        return json.loads(f.read())


def main():
    sha = hashlib.sha256(open(TF, "rb").read()).hexdigest()[:12]
    n_pass = n_inv = 0
    with open(OUT, "w") as out:
        for i, (q, want, ok, prior) in enumerate(CASES, 1):
            msgs = [{"role": "user", "content": q}]
            if prior:
                tid = f"toolu_probe{i:02d}"
                msgs += [{"role": "assistant", "content": [{"type": "tool_use", "id": tid, "name": "Read",
                                                             "input": {"file_path": f"{W}/{prior[0]}"}}]},
                         {"role": "user", "content": [{"type": "tool_result", "tool_use_id": tid,
                                                        "content": "\n".join(f"{n:6d}\t{l}" for n, l in enumerate(prior[1].splitlines(), 1))}]}]
            t0 = time.time()
            try:
                r = post({"model": "exactstream", "max_tokens": 1024, "system": SYSTEM, "tools": TOOLS, "messages": msgs})
                err = ""
            except Exception as e:  # a server error is a failed call
                r, err = {"content": []}, f"{type(e).__name__}: {e}"
            uses = [b for b in r.get("content", []) if b.get("type") == "tool_use"]
            inv = [u["name"] for u in uses if u.get("name") not in NAMES]
            first = uses[0] if uses else None
            why = err or ("no tool_use" if not first else f"invented {inv}" if inv else
                          f"tool {first['name']} != {want}" if first["name"] != want else fits(want, first.get("input")))
            if not why and want in ("Read", "Write", "Edit") and not str(first["input"].get("file_path", "")).startswith("/"):
                why = "file_path not absolute"
            if not why and not ok(first["input"]):
                why = "case check failed"
            n_pass += not why
            n_inv += bool(inv)
            out.write(json.dumps({"case": i, "prompt": q, "want": want, "pass": not why, "why": why, "invented": inv,
                                  "uses": [{"name": u.get("name"), "input": u.get("input")} for u in uses],
                                  "text": " ".join(b.get("text", "") for b in r.get("content", []) if b.get("type") == "text")[:400],
                                  "stop_reason": r.get("stop_reason"), "s": round(time.time() - t0, 1)}) + "\n")
            out.flush()
            print(f"  {i:2d} {want:5s} {'PASS' if not why else 'FAIL: ' + why} ({time.time() - t0:.0f} s)", flush=True)
    print(f"PROBE_CC: {n_pass} / {len(CASES)} pass; calls with an invented tool {n_inv}; tools file {sha}")


if __name__ == "__main__":
    main()
