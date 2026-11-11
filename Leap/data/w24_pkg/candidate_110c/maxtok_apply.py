"""Raise chat.py's and serve.py's --max-tokens default 16384 -> 65536 in one installed clone (exact old lines only; each file
written to a temp file and renamed over the old one, so a chat.py starting at that moment reads the old or the new file).
Usage: maxtok_apply.py <clone dir>"""
import os, sys
D = sys.argv[1]
SUBS = [
  ("  .venv/bin/python chat.py                            # the 32 GB chat config, up to 16,384 tokens per answer",
   "  .venv/bin/python chat.py                            # the 32 GB chat config, up to 65,536 tokens per answer"),
  ('    ap.add_argument("--max-tokens", type=int, default=16384, help="the most one answer may generate (default 16384)")',
   '    ap.add_argument("--max-tokens", type=int, default=65536,\n                    help="the most one answer may generate (default 65536; a long answer uses memory as a long prompt does)")'),
  ('    ap.add_argument("--max-tokens", type=int, default=16384, help="the most any request may generate")',
   '    ap.add_argument("--max-tokens", type=int, default=65536,\n                    help="the most any request may generate (default 65536; a request\'s own max_tokens can ask for less)")'),
]
for f, want in (("chat.py", 2), ("serve.py", 1)):
    p = os.path.join(D, f)
    t = open(p).read()
    n = 0
    for old, new in SUBS:
        if old in t:
            t, n = t.replace(old, new), n + 1
    if n == 0 and "default=65536" in t:
        print(f, "already 65536"); continue
    assert n == want, (f, n)
    compile(t, p, "exec")
    tmp = p + ".tmp_maxtok"
    open(tmp, "w").write(t)
    os.chmod(tmp, os.stat(p).st_mode)
    os.replace(tmp, p)
    print(f, f"{n} line(s) changed")
