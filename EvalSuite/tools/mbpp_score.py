"""MBPP pass@1 (the orchestrator Fri 09:5x; docs/PREREG_mbpp_*): the answer's longest ```python block, then the task's
test_setup_code and its 3 asserts, in an isolated child process (python -I in a temp dir, evalplus's reliability_guard,
a 10 s timeout); pass = exit 0. Rows' text comes from their completion_ids (BigMoE's tokenizer) when they carry none.
  .venv/bin/python tools/mbpp_score.py selftest                 # the dataset's own solutions must pass
  .venv/bin/python tools/mbpp_score.py score <run>              # pass@1 of one run
  .venv/bin/python tools/mbpp_score.py d4 <arm run> <reference run>   # decision 4 on MBPP (tools/d4x.py's rule)
Last line of d4: `MBPP: PASS | FAIL | DIFFERS (HIGHER)`; exit 0 PASS / DIFFERS, 1 FAIL, 2 rows missing."""
import json, os, re, subprocess, sys, tempfile
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
from evalsuite import data as D

PY = os.path.join(ROOT, ".venv", "bin", "python")
CHILD = ('import sys\nfrom evalplus.eval.utils import reliability_guard\nsrc = open(sys.argv[1]).read()\n'
         'reliability_guard(maximum_memory_bytes=None)\nexec(compile(src, "answer", "exec"), {"__name__": "__main__"})\n')
_TOK = {}


def text_of(r):
    if r.get("text"):
        return r["text"]
    if "t" not in _TOK:
        from tokenizers import Tokenizer
        _TOK["t"] = Tokenizer.from_file(os.path.expanduser("~/BigMoE/models/bare/tokenizer.json"))
    return _TOK["t"].decode([int(i) for i in r["completion_ids"]], skip_special_tokens=True)


def code_of(text):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    blocks = re.findall(r"```(?:python)?\n(.*?)```", text, re.S)
    return max(blocks, key=len) if blocks else text


def run_tests(code, it, timeout=10):
    prog = (it.get("test_setup_code") or "") + "\n" + code + "\n\n" + "\n".join(it["test_list"]) + "\n"
    with tempfile.TemporaryDirectory(prefix="mbpp_") as d:
        f = os.path.join(d, "answer.py")
        open(f, "w").write(prog)
        try:
            r = subprocess.run([PY, "-I", "-c", CHILD, f], cwd=d, capture_output=True, timeout=timeout,
                               env={"PATH": "/usr/bin:/bin", "HOME": d})
            return r.returncode == 0
        except subprocess.TimeoutExpired:
            return False


def items():
    return {it["key"]: it for it in D.load("dec_mbpp")}


def score(rows, its=None):
    its = its or items()
    return {k: (ok, float(ok)) for k, r in rows.items() if k in its for ok in [run_tests(code_of(text_of(r)), its[k])]}


def load(run):
    p = os.path.join(ROOT, "results", run, "decode.jsonl")
    return {r["key"]: r for r in map(json.loads, open(p)) if r["key"].startswith("mbpp/")}


def main():
    cmd = sys.argv[1]
    its = items()
    if cmd == "selftest":
        bad = [k for k, it in its.items() if not run_tests(it["ref_code"], it)]
        print(f"MBPP selftest: the dataset's own solutions pass {len(its) - len(bad)}/{len(its)}" + (f"; FAIL {bad}" if bad else ""))
        sys.exit(1 if bad else 0)
    if cmd == "score":
        s = score(load(sys.argv[2]), its)
        print(f"{sys.argv[2]}: MBPP pass@1 {100 * sum(v[1] for v in s.values()) / max(len(s), 1):.1f} % ({len(s)} rows)")
        return
    if cmd == "d4":
        import d4x
        a, r = load(sys.argv[2]), load(sys.argv[3])
        keys = sorted(set(a) & set(r) & set(its))
        if len(keys) < len(its):
            print(f"mbpp: {len(keys)} of {len(its)} rows in both {sys.argv[2]} and {sys.argv[3]}")
            sys.exit(2)
        v = d4x.line("mbpp", score({k: a[k] for k in keys}, its), score({k: r[k] for k in keys}, its), keys)
        print(f"MBPP: {v}")
        sys.exit(1 if v == "FAIL" else 0)


if __name__ == "__main__":
    main()
