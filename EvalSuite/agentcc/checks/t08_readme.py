import json, os, re, subprocess, sys
sys.path.insert(0, os.getcwd())
def run(*a):
    return subprocess.run(list(a), capture_output=True, text=True, timeout=60)
def unit():
    r = run('/usr/bin/python3', '-B', '-m', 'unittest', 'discover', '-q', '-s', 'tests', '-t', '.')
    assert r.returncode == 0, 'visible tests fail: ' + r.stderr[-400:]
s = open('README.md').read()
assert s.startswith('# csvtool') and 'A small tool that summarises CSV files.' in s
assert re.search(r'(?m)^#+\s*Usage\b', s), 'no Usage heading'
blocks = re.findall(r'```[^\n]*\n(.*?)```', s, re.S)
assert any('python3 tool.py --input data.csv' in b for b in blocks), 'no fenced command'
