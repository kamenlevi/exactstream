import json, os, re, subprocess, sys
sys.path.insert(0, os.getcwd())
def run(*a):
    return subprocess.run(list(a), capture_output=True, text=True, timeout=60)
def unit():
    r = run('/usr/bin/python3', '-B', '-m', 'unittest', 'discover', '-q', '-s', 'tests', '-t', '.')
    assert r.returncode == 0, 'visible tests fail: ' + r.stderr[-400:]
import report as R
assert R.format_money(1234.5) == '$1,234.50'
assert R.line_item('Pen', 1234.5) == 'Pen: $1,234.50' and R.total_line(99) == 'Total: $99.00'
assert R.tax_line(200, 0.2) == 'Tax: $40.00'
assert open('report.py').read().count(':,.2f') == 1, 'still duplicated'
