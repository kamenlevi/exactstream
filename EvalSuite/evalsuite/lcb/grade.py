"""Grade LiveCodeBench rows with the official run_test (testing_util.py, vendored verbatim), CPU, one row at a time.

Per row: LCB's extract_code for chat models (the lines between the last two ``` lines; fewer than two: no code, a fail),
then compute_code_generation_metrics.check_correctness's semantics: one process runs every public + private test (6 s
each, the official default), joined with the global timeout (6 + 1) x tests + 5 s (a timeout: every test -1); pass =
every result > 0 (pass_k_utils). EvalSuite's additions, outside the official path: the process starts in an empty
temporary directory with its output to /dev/null, a watchdog kills it above RSS_GB of resident memory (macOS ignores
RLIMIT_AS; also every test -1), and verdicts are cached by (question, sha256 of the code) in results/lcb6_grades.jsonl,
so identical code always gets the same verdict (an arm's row that equals the reference's cannot flip on a timeout).
Private tests in test6.jsonl are base64 + zlib + a pickled JSON string, unpickled by a loader that admits no globals."""
import base64, hashlib, io, json, os, pickle, shutil, tempfile, time, zlib

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW = os.path.join(ROOT, "data", "raw", "lcb6", "test6.jsonl")
CACHE = os.environ.get("EVALSUITE_LCB_CACHE") or os.path.join(ROOT, "results", "lcb6_grades.jsonl")   # env: a dry run's own
TIMEOUT = 6
RSS_GB = float(os.environ.get("EVALSUITE_LCB_RSS_GB", "3"))
VERSION = 1


class _NoGlobals(pickle.Unpickler):
    def find_class(self, module, name):
        raise pickle.UnpicklingError(f"a test blob asks for {module}.{name}")


def _tests(s):
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        blob = _NoGlobals(io.BytesIO(zlib.decompress(base64.b64decode(s.encode("utf-8"))))).load()
        assert isinstance(blob, str), type(blob)
        return json.loads(blob)


def offsets():
    out = {}
    with open(RAW, "rb") as f:
        while True:
            off, line = f.tell(), f.readline()
            if not line:
                return out
            out[json.loads(line)["question_id"]] = off


def sample(qid, off):
    """the official evaluation sample (get_evaluation_sample): public + private tests, fn_name from the metadata"""
    with open(RAW, "rb") as f:
        f.seek(off)
        r = json.loads(f.readline())
    assert r["question_id"] == qid, (qid, r["question_id"])
    ts = json.loads(r["public_test_cases"]) + _tests(r["private_test_cases"])
    return {"input_output": json.dumps({"inputs": [t["input"] for t in ts], "outputs": [t["output"] for t in ts],
                                        "fn_name": json.loads(r["metadata"]).get("func_name")})}


def extract_code(text):
    """lcb_runner/utils/extraction_utils.extract_code for chat models"""
    lines = text.split("\n")
    idx = [i for i, line in enumerate(lines) if "```" in line]
    return "\n".join(lines[idx[-2] + 1:idx[-1]]) if len(idx) >= 2 else ""


CHILD = r"""
import json, os, sys
sys.path.insert(0, sys.argv[1])
job = json.load(sys.stdin)
res_fd = os.dup(1)
nul = os.open(os.devnull, os.O_WRONLY)
os.dup2(nul, 1)
os.dup2(nul, 2)
import numpy as np
from evalsuite.lcb.testing_util import run_test
try:
    res, meta = run_test(job["sample"], test=job["code"], debug=False, timeout=job["timeout"])
    out = []
    for e in res:
        if isinstance(e, np.ndarray):
            e = e.item(0)
        out.append(bool(e) if isinstance(e, (bool, np.bool_)) else int(e))
    msg = [out, {k: str(v)[:300] for k, v in (meta or {}).items()}]
except BaseException as e:      # the official driver: an exception in the run counts as failed tests
    msg = [[-5], {"error": repr(e)[:300]}]
os.write(res_fd, (json.dumps(msg) + "\n").encode())
"""


def check(smp, code):
    """-> (results, metadata, seconds), check_correctness's form; the run is `python -I` in its own process group"""
    import psutil, signal, subprocess, sys
    n = len(json.loads(smp["input_output"])["inputs"])
    d = tempfile.mkdtemp(prefix="lcb6_")
    p = subprocess.Popen([sys.executable, "-I", "-c", CHILD, ROOT], cwd=d, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=subprocess.DEVNULL, start_new_session=True)
    t0, why = time.time(), None
    try:
        try:
            p.stdin.write(json.dumps({"sample": smp, "code": code, "timeout": TIMEOUT}).encode())
            p.stdin.close()
        except BrokenPipeError:
            pass
        pp = psutil.Process(p.pid)
        while p.poll() is None:
            if time.time() - t0 > (TIMEOUT + 1) * n + 5:
                why = "global timeout"
                break
            try:
                rss = pp.memory_info().rss + sum(c.memory_info().rss for c in pp.children(recursive=True))
            except psutil.Error:
                rss = 0
            if rss > RSS_GB * 2 ** 30:
                why = f"resident memory above {RSS_GB} GB"
                break
            time.sleep(0.05)
    finally:
        if p.poll() is None:
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            p.wait(10)
        out = p.stdout.read() if why is None else b""
        p.stdout.close()
        shutil.rmtree(d, ignore_errors=True)
    try:
        res, meta = json.loads(out.decode().strip().splitlines()[-1])
    except (ValueError, IndexError):
        res, meta = [-1] * n, {"error": why or f"no result (exit {p.returncode})"}
    return res, meta, round(time.time() - t0, 2)


def score(texts, its, log=None):
    """texts: key -> response text (lcb6 rows). -> key -> (pass, float(pass)), via the cache"""
    cache = {}
    if os.path.exists(CACHE):
        for r in map(json.loads, open(CACHE)):
            if r.get("v") == VERSION:
                cache[(r["qid"], r["sha"])] = r["pass"]
    todo, out, offs = {}, {}, None
    for k, t in texts.items():
        qid = its[k]["question_id"]
        code = extract_code(t.split("</think>")[-1] if "</think>" in t else t)
        sha = hashlib.sha256(code.encode()).hexdigest()[:16]
        if (qid, sha) in cache:
            out[k] = cache[(qid, sha)]
        else:
            todo.setdefault((qid, sha), (code, []))[1].append(k)
    if todo:
        offs = offsets()
        os.makedirs(os.path.dirname(CACHE), exist_ok=True)
        with open(CACHE, "a") as f:
            for (qid, sha), (code, ks) in sorted(todo.items()):
                if not code:
                    res, meta, sec = [-2], {"error": "no fenced code"}, 0.0
                else:
                    res, meta, sec = check(sample(qid, offs[qid]), code)
                ok = bool(res) and all(x > 0 for x in res)
                f.write(json.dumps(dict(v=VERSION, qid=qid, sha=sha, **{"pass": ok}, n=len(res),
                                        failed=[i for i, x in enumerate(res) if not x > 0][:3], codes=sorted({str(x) for x in res}),
                                        meta=meta, seconds=sec)) + "\n")
                f.flush()
                for k in ks:
                    out[k] = ok
                if log:
                    log(f"lcb6 {qid} {sha}: {'pass' if ok else 'fail'} ({len(res)} results, {sec} s)")
    return {k: (ok, float(ok)) for k, ok in out.items()}
