import json, os, re, subprocess, sys
sys.path.insert(0, os.getcwd())
def run(*a):
    return subprocess.run(list(a), capture_output=True, text=True, timeout=60)
def unit():
    r = run('/usr/bin/python3', '-B', '-m', 'unittest', 'discover', '-q', '-s', 'tests', '-t', '.')
    assert r.returncode == 0, 'visible tests fail: ' + r.stderr[-400:]
import ast, math
t = ast.parse(open('geometry.py').read())
fs = [n for n in t.body if isinstance(n, ast.FunctionDef)]
assert sorted(f.name for f in fs) == ['circle_area', 'hypot', 'rect_area'], [f.name for f in fs]
for f in fs:
    assert f.returns is not None and all(a.annotation is not None for a in f.args.args), f.name
import geometry as G
assert abs(G.circle_area(2) - 4 * math.pi) < 1e-9 and G.rect_area(2, 3) == 6 and G.hypot(3, 4) == 5
