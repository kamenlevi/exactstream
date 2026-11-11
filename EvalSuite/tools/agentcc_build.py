#!/usr/bin/env python3
"""The Claude Code agent test's task suite (docs/PREREG_agentcc_2026-11-11.md): 12 small coding tasks in throwaway repos.
Writes agentcc/tasks/<id>/template (the repo the agent gets), agentcc/tasks/<id>/solution (a reference fix, overlaid on a
copy of the template only by --validate), agentcc/checks/<id>.py (the hidden check: Leap's runner copies it into the task
folder as .agentcc_check.py AFTER claude ends; exit 0 = pass) and agentcc/manifest.json for Leap's runner.
  python3 tools/agentcc_build.py [--validate]   (--validate: each check must FAIL on the template and PASS on template+solution)
Checks run with /usr/bin/python3 (3.9, standard library only: unittest, json, ast, subprocess)."""
import json, os, shutil, subprocess, sys, tempfile

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "agentcc")
ROOT = os.path.abspath(ROOT)
PY = "/usr/bin/python3"
CHECK = f"{PY} -I -B .agentcc_check.py"
TIMEOUT = {"24": 1200, "32": 1200, "45": 3000}       # s per task; 45 GB re-prefills ~95 s per turn (Leap 23:1x)
HEAD = ("import json, os, re, subprocess, sys\nsys.path.insert(0, os.getcwd())\n"
        "def run(*a):\n    return subprocess.run(list(a), capture_output=True, text=True, timeout=60)\n"
        "def unit():\n    r = run('" + PY + "', '-B', '-m', 'unittest', 'discover', '-q', '-s', 'tests', '-t', '.')\n"
        "    assert r.returncode == 0, 'visible tests fail: ' + r.stderr[-400:]\n")
T = []   # (id, prompt, template files, solution files, check body)

T.append(("t01_fix_test", "The test suite fails. Find the bug in the code and fix it so the tests pass. Do not change the tests. "
          "Run the tests to confirm.",
          {"calc.py": "def mean(xs):\n    return sum(xs) / (len(xs) - 1)\n",
           "tests/__init__.py": "", "tests/test_calc.py": "import unittest\nfrom calc import mean\n\n\nclass T(unittest.TestCase):\n"
           "    def test_mean(self):\n        self.assertEqual(mean([1, 2, 3]), 2)\n\n\nif __name__ == '__main__':\n    unittest.main()\n"},
          {"calc.py": "def mean(xs):\n    return sum(xs) / len(xs)\n"},
          "unit()\nfrom calc import mean\nassert mean([2, 4]) == 3 and mean([5]) == 5\n"
          "assert 'assertEqual(mean([1, 2, 3]), 2)' in open('tests/test_calc.py').read(), 'test changed'\n"))

T.append(("t02_add_function", "Add a function slugify(text) to textutil.py: lowercase the text, replace every run of characters that "
          "are not letters or digits with a single hyphen, and strip hyphens from both ends. Add a unit test for it to "
          "tests/test_textutil.py, and make sure all tests pass.",
          {"textutil.py": "def title_case(s):\n    return ' '.join(w.capitalize() for w in s.split())\n",
           "tests/__init__.py": "", "tests/test_textutil.py": "import unittest\nfrom textutil import title_case\n\n\nclass T(unittest.TestCase):\n"
           "    def test_title(self):\n        self.assertEqual(title_case('a b'), 'A B')\n"},
          {"textutil.py": "import re\n\n\ndef title_case(s):\n    return ' '.join(w.capitalize() for w in s.split())\n\n\n"
           "def slugify(text):\n    return re.sub(r'[^a-z0-9]+', '-', text.lower()).strip('-')\n",
           "tests/test_textutil.py": "import unittest\nfrom textutil import title_case, slugify\n\n\nclass T(unittest.TestCase):\n"
           "    def test_title(self):\n        self.assertEqual(title_case('a b'), 'A B')\n\n"
           "    def test_slug(self):\n        self.assertEqual(slugify('Hello, World!'), 'hello-world')\n"},
          "unit()\nfrom textutil import slugify\nassert slugify('Hello, World!') == 'hello-world'\n"
          "assert slugify('  a--b  ') == 'a-b' and slugify('Already-slug') == 'already-slug' and slugify('X 2 y') == 'x-2-y'\n"
          "assert 'slugify' in open('tests/test_textutil.py').read(), 'no test added'\n"))

T.append(("t03_config", "In config.json, raise the timeout to 30 and turn the cache feature on. Leave every other setting unchanged.",
          {"config.json": json.dumps({"name": "svc", "timeout": 10, "retries": 3, "features": {"cache": False, "metrics": True}}, indent=2) + "\n"},
          {"config.json": json.dumps({"name": "svc", "timeout": 30, "retries": 3, "features": {"cache": True, "metrics": True}}, indent=2) + "\n"},
          "c = json.load(open('config.json'))\n"
          "assert c == {'name': 'svc', 'timeout': 30, 'retries': 3, 'features': {'cache': True, 'metrics': True}}, c\n"))

T.append(("t04_traceback", "Running `python3 report.py` fails with this error:\n\nTraceback (most recent call last):\n"
          "  File \"report.py\", line 4, in <module>\n    print(total_value(load_items()))\n"
          "  File \"inventory.py\", line 2, in total_value\n    return sum(i[\"price\"] * i[\"qty\"] for i in items)\n"
          "KeyError: 'qty'\n\nItems without a qty should count as quantity 1. Fix it.",
          {"inventory.py": "def total_value(items):\n    return sum(i[\"price\"] * i[\"qty\"] for i in items)\n\n\n"
           "def load_items():\n    return [{\"price\": 2.5, \"qty\": 4}, {\"price\": 10.0}, {\"price\": 1.0, \"qty\": 3}]\n",
           "report.py": "from inventory import total_value, load_items\n\nif __name__ == '__main__':\n    print(total_value(load_items()))\n"},
          {"inventory.py": "def total_value(items):\n    return sum(i[\"price\"] * i.get(\"qty\", 1) for i in items)\n\n\n"
           "def load_items():\n    return [{\"price\": 2.5, \"qty\": 4}, {\"price\": 10.0}, {\"price\": 1.0, \"qty\": 3}]\n"},
          "from inventory import total_value\nassert total_value([{'price': 2.0}]) == 2.0\n"
          "assert total_value([{'price': 2.0, 'qty': 3}]) == 6.0\nr = run('" + PY + "', 'report.py')\n"
          "assert r.returncode == 0 and r.stdout.strip() == '23.0', (r.returncode, r.stdout, r.stderr[-300:])\n"))

T.append(("t05_rename", "Rename the function get_user to fetch_user everywhere in this project: the definition and every call site, "
          "including the tests. The tests must still pass.",
          {"app/__init__.py": "", "app/users.py": "USERS = {1: 'ann', 2: 'bo'}\n\n\ndef get_user(uid):\n    return USERS.get(uid)\n",
           "app/api.py": "from app.users import get_user\n\n\ndef handle(uid):\n    return {'user': get_user(uid)}\n",
           "app/cli.py": "from app import users\n\n\ndef main(uid):\n    print(users.get_user(uid))\n",
           "tests/__init__.py": "", "tests/test_users.py": "import unittest\nfrom app.users import get_user\n\n\nclass T(unittest.TestCase):\n"
           "    def test_get(self):\n        self.assertEqual(get_user(1), 'ann')\n"},
          {"app/users.py": "USERS = {1: 'ann', 2: 'bo'}\n\n\ndef fetch_user(uid):\n    return USERS.get(uid)\n",
           "app/api.py": "from app.users import fetch_user\n\n\ndef handle(uid):\n    return {'user': fetch_user(uid)}\n",
           "app/cli.py": "from app import users\n\n\ndef main(uid):\n    print(users.fetch_user(uid))\n",
           "tests/test_users.py": "import unittest\nfrom app.users import fetch_user\n\n\nclass T(unittest.TestCase):\n"
           "    def test_get(self):\n        self.assertEqual(fetch_user(1), 'ann')\n"},
          "unit()\nleft = [os.path.join(d, f) for d, _, fs in os.walk('.') if '.git' not in d for f in fs\n"
          "        if f.endswith('.py') and f != '.agentcc_check.py' and 'get_user' in open(os.path.join(d, f)).read()]\n"
          "assert not left, left\nfrom app.users import fetch_user\nfrom app.api import handle\nassert handle(2) == {'user': 'bo'}\n"
          "r = run('" + PY + "', '-c', 'from app.cli import main; main(1)')\nassert r.stdout.strip() == 'ann', r.stderr[-300:]\n"))

T.append(("t06_cli_flag", "Add a --shout flag to greet.py that prints the greeting in upper case. Without the flag the output must stay "
          "exactly the same.",
          {"greet.py": "import argparse\n\np = argparse.ArgumentParser()\np.add_argument('name')\na = p.parse_args()\nprint(f'Hello, {a.name}')\n"},
          {"greet.py": "import argparse\n\np = argparse.ArgumentParser()\np.add_argument('name')\np.add_argument('--shout', action='store_true')\n"
           "a = p.parse_args()\nmsg = f'Hello, {a.name}'\nprint(msg.upper() if a.shout else msg)\n"},
          "r = run('" + PY + "', 'greet.py', 'Ann')\nassert r.stdout == 'Hello, Ann\\n', repr(r.stdout)\n"
          "r = run('" + PY + "', 'greet.py', 'Ann', '--shout')\nassert r.stdout == 'HELLO, ANN\\n', repr(r.stdout + r.stderr[-200:])\n"))

T.append(("t07_import", "Importing shop.cart fails with: ModuleNotFoundError: No module named 'shop.pricing'. The pricing module now "
          "lives in shop/core/. Fix the import without moving any file.",
          {"shop/__init__.py": "", "shop/core/__init__.py": "",
           "shop/core/pricing.py": "PRICES = {'apple': 0.5, 'pear': 0.75}\n\n\ndef price_of(item):\n    return PRICES[item]\n",
           "shop/cart.py": "from shop.pricing import price_of\n\n\ndef cart_total(items):\n    return sum(price_of(i) for i in items)\n"},
          {"shop/cart.py": "from shop.core.pricing import price_of\n\n\ndef cart_total(items):\n    return sum(price_of(i) for i in items)\n"},
          "assert not os.path.exists('shop/pricing.py'), 'a file was moved or added'\nassert os.path.exists('shop/core/pricing.py')\n"
          "from shop.cart import cart_total\nassert cart_total(['apple', 'pear', 'apple']) == 1.75\n"))

T.append(("t08_readme", "Add a 'Usage' section to README.md that shows how to run the tool with `python3 tool.py --input data.csv`, "
          "in a fenced code block. Keep the existing text.",
          {"README.md": "# csvtool\n\nA small tool that summarises CSV files.\n", "tool.py": "print('ok')\n"},
          {"README.md": "# csvtool\n\nA small tool that summarises CSV files.\n\n## Usage\n\n```sh\npython3 tool.py --input data.csv\n```\n"},
          "s = open('README.md').read()\nassert s.startswith('# csvtool') and 'A small tool that summarises CSV files.' in s\n"
          "assert re.search(r'(?m)^#+\\s*Usage\\b', s), 'no Usage heading'\n"
          "blocks = re.findall(r'```[^\\n]*\\n(.*?)```', s, re.S)\n"
          "assert any('python3 tool.py --input data.csv' in b for b in blocks), 'no fenced command'\n"))

T.append(("t09_refactor", "report.py repeats the same money-formatting code in three places. Extract it into one helper function named "
          "format_money(amount) and use it in all three places. The output must not change.",
          {"report.py": "def line_item(name, amount):\n    return f'{name}: ${amount:,.2f}'\n\n\n"
           "def total_line(amount):\n    return f'Total: ${amount:,.2f}'\n\n\n"
           "def tax_line(amount, rate):\n    return f'Tax: ${amount * rate:,.2f}'\n"},
          {"report.py": "def format_money(amount):\n    return f'${amount:,.2f}'\n\n\n"
           "def line_item(name, amount):\n    return f'{name}: {format_money(amount)}'\n\n\n"
           "def total_line(amount):\n    return f'Total: {format_money(amount)}'\n\n\n"
           "def tax_line(amount, rate):\n    return f'Tax: {format_money(amount * rate)}'\n"},
          "import report as R\nassert R.format_money(1234.5) == '$1,234.50'\n"
          "assert R.line_item('Pen', 1234.5) == 'Pen: $1,234.50' and R.total_line(99) == 'Total: $99.00'\n"
          "assert R.tax_line(200, 0.2) == 'Tax: $40.00'\nassert open('report.py').read().count(':,.2f') == 1, 'still duplicated'\n"))

T.append(("t10_type_hints", "Add type hints to every function in geometry.py, for all parameters and return values. Use float for "
          "numbers. Do not change what the functions do.",
          {"geometry.py": "import math\n\n\ndef circle_area(r):\n    return math.pi * r * r\n\n\n"
           "def rect_area(w, h):\n    return w * h\n\n\ndef hypot(a, b):\n    return math.sqrt(a * a + b * b)\n"},
          {"geometry.py": "import math\n\n\ndef circle_area(r: float) -> float:\n    return math.pi * r * r\n\n\n"
           "def rect_area(w: float, h: float) -> float:\n    return w * h\n\n\ndef hypot(a: float, b: float) -> float:\n    return math.sqrt(a * a + b * b)\n"},
          "import ast, math\nt = ast.parse(open('geometry.py').read())\nfs = [n for n in t.body if isinstance(n, ast.FunctionDef)]\n"
          "assert sorted(f.name for f in fs) == ['circle_area', 'hypot', 'rect_area'], [f.name for f in fs]\n"
          "for f in fs:\n    assert f.returns is not None and all(a.annotation is not None for a in f.args.args), f.name\n"
          "import geometry as G\nassert abs(G.circle_area(2) - 4 * math.pi) < 1e-9 and G.rect_area(2, 3) == 6 and G.hypot(3, 4) == 5\n"))

T.append(("t11_shell", "count.sh should print the total number of lines in all .txt files under data/, including subfolders, but it "
          "prints the wrong number. Fix count.sh.",
          {"count.sh": "#!/bin/sh\ncat data/*.txt | wc -l | tr -d ' '\n", "data/a.txt": "1\n2\n3\n", "data/b.txt": "x\n",
           "data/sub/c.txt": "p\nq\n", "data/sub/deeper/d.txt": "z\n", "data/notes.md": "not\ncounted\n"},
          {"count.sh": "#!/bin/sh\nfind data -name '*.txt' -type f -exec cat {} + | wc -l | tr -d ' '\n"},
          "r = run('/bin/sh', 'count.sh')\nassert r.stdout.strip() == '7', repr(r.stdout + r.stderr[-200:])\n"))

T.append(("t12_requirements", "Add requests==2.32.3 to requirements.txt. Keep the file sorted alphabetically (ignoring case) and do not "
          "change the other lines.",
          {"requirements.txt": "attrs==23.2.0\nclick==8.1.7\nJinja2==3.1.4\nnumpy==1.26.4\nurllib3==2.2.2\n"},
          {"requirements.txt": "attrs==23.2.0\nclick==8.1.7\nJinja2==3.1.4\nnumpy==1.26.4\nrequests==2.32.3\nurllib3==2.2.2\n"},
          "ls = [l.strip() for l in open('requirements.txt') if l.strip()]\n"
          "assert ls == ['attrs==23.2.0', 'click==8.1.7', 'Jinja2==3.1.4', 'numpy==1.26.4', 'requests==2.32.3', 'urllib3==2.2.2'], ls\n"))


def write(base, files):
    for p, s in files.items():
        f = os.path.join(base, p)
        os.makedirs(os.path.dirname(f), exist_ok=True)
        open(f, "w").write(s)


def check(folder, tid):
    shutil.copy(os.path.join(ROOT, "checks", f"{tid}.py"), os.path.join(folder, ".agentcc_check.py"))
    r = subprocess.run(CHECK.split(), cwd=folder, capture_output=True, text=True, timeout=120)
    os.remove(os.path.join(folder, ".agentcc_check.py"))
    return r.returncode


def main():
    if os.path.exists(ROOT):
        shutil.rmtree(ROOT)
    man = []
    for tid, prompt, tpl, sol, body in T:
        write(os.path.join(ROOT, "tasks", tid, "template"), tpl)
        write(os.path.join(ROOT, "tasks", tid, "solution"), sol)
        os.makedirs(os.path.join(ROOT, "checks"), exist_ok=True)
        open(os.path.join(ROOT, "checks", f"{tid}.py"), "w").write(HEAD + body)
        man.append({"id": tid, "prompt": prompt, "template": os.path.join(ROOT, "tasks", tid, "template"),
                    "check_file": os.path.join(ROOT, "checks", f"{tid}.py"), "check_copy_as": ".agentcc_check.py",
                    "check": CHECK, "timeout_s": TIMEOUT})
    json.dump({"suite": "agentcc", "n": len(man), "tasks": man}, open(os.path.join(ROOT, "manifest.json"), "w"), indent=1)
    print(f"{len(man)} tasks -> {ROOT}/manifest.json")
    if "--validate" in sys.argv:
        bad = 0
        for tid, *_ in T:
            with tempfile.TemporaryDirectory() as d:
                w = os.path.join(d, "w")
                shutil.copytree(os.path.join(ROOT, "tasks", tid, "template"), w)
                pre = check(w, tid)
                shutil.copytree(os.path.join(ROOT, "tasks", tid, "solution"), w, dirs_exist_ok=True)
                post = check(w, tid)
            ok = pre != 0 and post == 0
            bad += not ok
            print(f"  {tid}: template check rc {pre} (must fail), template + solution rc {post} (must pass): {'OK' if ok else 'BAD'}")
        print("VALIDATE:", "all OK" if not bad else f"{bad} BAD")
        sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
