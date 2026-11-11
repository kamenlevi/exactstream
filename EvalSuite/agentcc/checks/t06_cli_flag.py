import json, os, re, subprocess, sys
sys.path.insert(0, os.getcwd())
def run(*a):
    return subprocess.run(list(a), capture_output=True, text=True, timeout=60)
def unit():
    r = run('/usr/bin/python3', '-B', '-m', 'unittest', 'discover', '-q', '-s', 'tests', '-t', '.')
    assert r.returncode == 0, 'visible tests fail: ' + r.stderr[-400:]
r = run('/usr/bin/python3', 'greet.py', 'Ann')
assert r.stdout == 'Hello, Ann\n', repr(r.stdout)
r = run('/usr/bin/python3', 'greet.py', 'Ann', '--shout')
assert r.stdout == 'HELLO, ANN\n', repr(r.stdout + r.stderr[-200:])
