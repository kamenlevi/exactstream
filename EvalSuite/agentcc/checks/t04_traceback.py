import json, os, re, subprocess, sys
sys.path.insert(0, os.getcwd())
def run(*a):
    return subprocess.run(list(a), capture_output=True, text=True, timeout=60)
def unit():
    r = run('/usr/bin/python3', '-B', '-m', 'unittest', 'discover', '-q', '-s', 'tests', '-t', '.')
    assert r.returncode == 0, 'visible tests fail: ' + r.stderr[-400:]
from inventory import total_value
assert total_value([{'price': 2.0}]) == 2.0
assert total_value([{'price': 2.0, 'qty': 3}]) == 6.0
r = run('/usr/bin/python3', 'report.py')
assert r.returncode == 0 and r.stdout.strip() == '23.0', (r.returncode, r.stdout, r.stderr[-300:])
