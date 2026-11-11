import json, os, re, subprocess, sys
sys.path.insert(0, os.getcwd())
def run(*a):
    return subprocess.run(list(a), capture_output=True, text=True, timeout=60)
def unit():
    r = run('/usr/bin/python3', '-B', '-m', 'unittest', 'discover', '-q', '-s', 'tests', '-t', '.')
    assert r.returncode == 0, 'visible tests fail: ' + r.stderr[-400:]
unit()
left = [os.path.join(d, f) for d, _, fs in os.walk('.') if '.git' not in d for f in fs
        if f.endswith('.py') and f != '.agentcc_check.py' and 'get_user' in open(os.path.join(d, f)).read()]
assert not left, left
from app.users import fetch_user
from app.api import handle
assert handle(2) == {'user': 'bo'}
r = run('/usr/bin/python3', '-c', 'from app.cli import main; main(1)')
assert r.stdout.strip() == 'ann', r.stderr[-300:]
