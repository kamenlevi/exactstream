import json, os, re, subprocess, sys
sys.path.insert(0, os.getcwd())
def run(*a):
    return subprocess.run(list(a), capture_output=True, text=True, timeout=60)
def unit():
    r = run('/usr/bin/python3', '-B', '-m', 'unittest', 'discover', '-q', '-s', 'tests', '-t', '.')
    assert r.returncode == 0, 'visible tests fail: ' + r.stderr[-400:]
assert not os.path.exists('shop/pricing.py'), 'a file was moved or added'
assert os.path.exists('shop/core/pricing.py')
from shop.cart import cart_total
assert cart_total(['apple', 'pear', 'apple']) == 1.75
