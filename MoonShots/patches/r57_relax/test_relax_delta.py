"""Unit tests for relax_delta.py (CPU, no model): python3 test_relax_delta.py. relax_core is BigMoE's vendored copy (read-only)."""
import importlib.util, json, math, os, random, sys, tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import relax_delta as RLD

_sp = importlib.util.spec_from_file_location("relax_core", os.path.expanduser("~/BigMoE/scripts/relax_core.py"))
RXC = importlib.util.module_from_spec(_sp); _sp.loader.exec_module(RXC)


def exact_k(dr, am):
    k = 0
    while k < len(dr) and dr[k] == am[k]:
        k += 1
    return k


def test_exact_when_delta_tiny():
    rng = random.Random(1)
    for _ in range(2000):
        n = rng.randint(0, 15)
        am = [rng.randint(0, 9) for _ in range(n)]
        dr = [a if rng.random() < 0.8 else (a + 1) % 10 for a in am]
        dlm = [0.0 if d == a else -rng.uniform(0.01, 5) for d, a in zip(dr, am)]
        k, rel = RLD.relaxed_k(dr, am, dlm, 1e-9)
        assert k == exact_k(dr, am) and rel == [], (dr, am, dlm, k)


def test_r1_window():
    am = [5, 5, 5, 5, 5]
    dr = [5, 6, 5, 7, 5]
    dlm = [0.0, -0.4, 0.0, -0.6, 0.0]
    assert RLD.relaxed_k(dr, am, dlm, 0.5) == (3, [1])           # row 3 is 0.6 nat below: stop there
    assert RLD.relaxed_k(dr, am, dlm, 0.6) == (5, [1, 3])
    assert RLD.relaxed_k(dr, am, dlm, 0.3) == (1, [])


def test_gate_and_fast():
    am, dr, dlm = [1, 1, 1], [2, 2, 1], [-0.1, -0.1, 0.0]
    assert RLD.relaxed_k(dr, am, dlm, 0.5, may=[False, True, True]) == (0, [])
    assert RLD.relaxed_k(dr, am, dlm, 0.5, may=[True, False, True]) == (1, [0])
    assert RLD.relaxed_k(dr, am, dlm, 0.5, may=[True, False, True], fast_ok=lambda j: j == 1) == (3, [0])
    assert RLD.relaxed_k(dr, am, [-math.inf] * 3, 9.9) == (0, [])  # a draft outside the stored top-k never relaxes


def test_parse_gate():
    assert RLD.parse_gate("") == (None, False)
    assert RLD.parse_gate("cf:0.9") == (0.9, False)
    assert RLD.parse_gate("copy") == (None, True)
    assert RLD.parse_gate("cf:0.5+copy") == (0.5, True)
    try:
        RLD.parse_gate("cf0.5")
        raise AssertionError("bad spec accepted")
    except ValueError:
        pass


def test_gate_flags():
    assert RLD.gate_flags(RXC, None, False, [0.5] * 15, [0] * 8, list(range(15)), None) is None
    cf = [0.95 if j % 2 == 0 else 0.5 for j in range(15)]
    g = RLD.gate_flags(RXC, 0.9, False, cf, [0] * 8, list(range(100, 115)), None)
    assert g == [j % 2 == 0 for j in range(15)], g
    tail, drafts = [1, 2, 3, 4, 5, 6, 7, 8], list(range(100, 115))
    grams = RXC.prompt_grams([9, 9] + tail + [100, 101])         # the window ending at draft 0's row occurs in the prompt
    g = RLD.gate_flags(RXC, None, True, [0.1] * 15, tail, drafts, grams)
    assert g[0] and not any(g[3:]), g
    g = RLD.gate_flags(RXC, 0.9, False, [0.95] * 31, [0] * 8, list(range(31)), None)   # a 32-row block
    assert len(g) == 31 and all(g[:15]) and not any(g[15:]), g


def test_log():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "rlx.jsonl")
        lg = RLD.Log(p)
        lg.write(prompt="x", check=3, row=1, draft=7, argmax=9, dlm=-0.2, rank=1, cf=0.97, gate="cf")
        rows = [json.loads(l) for l in open(p)]
        assert rows == [dict(prompt="x", check=3, row=1, draft=7, argmax=9, dlm=-0.2, rank=1, cf=0.97, gate="cf")]
        RLD.Log("").write(a=1)                                     # no path: a no-op


if __name__ == "__main__":
    n = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn(); n += 1
    print(f"relax_delta: {n} tests OK")
