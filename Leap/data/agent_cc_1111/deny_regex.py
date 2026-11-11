"""Rewrite a sandbox deny rule so it names no account folder: the subpaths ~/.claude, ~/.claude-<a>, ~/.claude-<a>2
become one regex rule that covers every ~/.claude* entry (orchestrator 04:0x: no literal account names in tracked files).
Usage: deny_regex.py FILE [--record]   (--record: also a comment that the as-run profile listed them one by one)"""
import re, sys
p = sys.argv[1]; s = open(p).read()
pat = re.compile(r'\(subpath "(\$H|/Users/[a-z]+)/\.claude"\) \(subpath "(?:\$H|/Users/[a-z]+)/\.claude-[a-z]+"\)\n  '
                 r'\(subpath "(?:\$H|/Users/[a-z]+)/\.claude-[a-z]+2"\) ')
n = len(pat.findall(s))
assert n == 1, (p, n)
s = pat.sub(lambda m: '(regex #"^/Users/[^/]+/\\.claude[^/]*(/|$)")\n  ', s)
if "--record" in sys.argv:
    s = ("; Leap 04:0x record copy: as run, the read-deny rule listed ~/.claude and the two account config folders one by "
         "one;\n; here they are written as the one regex that covers them (no account names in tracked files).\n") + s
open(p, "w").write(s)
print("rewrote", p)
