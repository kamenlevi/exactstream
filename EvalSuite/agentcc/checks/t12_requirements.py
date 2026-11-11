import json, os, re, subprocess, sys
sys.path.insert(0, os.getcwd())
def run(*a):
    return subprocess.run(list(a), capture_output=True, text=True, timeout=60)
def unit():
    r = run('/usr/bin/python3', '-B', '-m', 'unittest', 'discover', '-q', '-s', 'tests', '-t', '.')
    assert r.returncode == 0, 'visible tests fail: ' + r.stderr[-400:]
ls = [l.strip() for l in open('requirements.txt') if l.strip()]
assert ls == ['attrs==23.2.0', 'click==8.1.7', 'Jinja2==3.1.4', 'numpy==1.26.4', 'requests==2.32.3', 'urllib3==2.2.2'], ls
