import json, os, re, subprocess, sys
sys.path.insert(0, os.getcwd())
def run(*a):
    return subprocess.run(list(a), capture_output=True, text=True, timeout=60)
def unit():
    r = run('/usr/bin/python3', '-B', '-m', 'unittest', 'discover', '-q', '-s', 'tests', '-t', '.')
    assert r.returncode == 0, 'visible tests fail: ' + r.stderr[-400:]
unit()
from textutil import slugify
assert slugify('Hello, World!') == 'hello-world'
assert slugify('  a--b  ') == 'a-b' and slugify('Already-slug') == 'already-slug' and slugify('X 2 y') == 'x-2-y'
assert 'slugify' in open('tests/test_textutil.py').read(), 'no test added'
