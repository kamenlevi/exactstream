"""Desk test of the screens' _gate_block (Mon 16 Nov; EvalSuite tier 0a's crash, 05:0x): R57_FAST's row gate on a drafted
block of any length. No model, no GPU. The function is taken from each screen's source (ast), not re-typed here.

  .venv/bin/python scripts/test_gate_block.py [screen.py ...]   (default: the working copy + snapshot rpl2)

Checks:
  T0  the bug: relax_core.gate_rows on a short block raises (IndexError in U3's cf gate; AssertionError with copy grams);
  T1  a full window (cf >= 15, drafts 15 or longer for U3's cf-only call) -> exactly gate_rows' rows (random cases);
  T2  a short block (0..14 drafts) -> 16 rows, no error; rows from the block's end on are 0 (exact); rows inside the block
      equal gate_rows' rows on any full block that starts with it (they read only cf[j] and the row's own inputs);
  T3  the helper's source is identical in every screen given, and no screen calls RXC.gate_rows outside it.
"""
import ast
import importlib.util
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("relax_core", os.path.join(HERE, "relax_core.py"))
RXC = importlib.util.module_from_spec(spec); spec.loader.exec_module(RXC)
SCREENS = sys.argv[1:] or [os.path.join(HERE, "r57_screen_fast.py"),
                           os.path.join(HERE, "..", "data/r57/snap/rpl2/r57_screen_rpl2.py")]


def helper_of(path):
    src = open(path).read()
    tree = ast.parse(src)
    fns = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_gate_block"]
    assert len(fns) == 1, (path, len(fns))
    seg = ast.get_source_segment(src, fns[0])
    ns = {"RXC": RXC}
    exec(compile(ast.Module(body=[fns[0]], type_ignores=[]), path, "exec"), ns)
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
             and n.func.attr == "gate_rows" and isinstance(n.func.value, ast.Name) and n.func.value.id == "RXC"]
    inside = [n for n in ast.walk(fns[0]) if n in calls]
    return ns["_gate_block"], seg, len(calls) - len(inside)


def case(rng, n, thr, with_grams):
    tail = [rng.randrange(50) for _ in range(8)]
    dr = [rng.randrange(50) for _ in range(n)]
    cf = [rng.choice([0.0, 0.1, 0.3, 0.5, 0.9, 0.99, 1.0, rng.random()]) for _ in range(n)]
    grams = None
    if with_grams:                                       # a prompt holding some of this block's own 8-token windows
        w = tail + dr
        prompt = [rng.randrange(50) for _ in range(40)]
        for _ in range(rng.randrange(4)):
            j = rng.randrange(max(1, len(w) - 7))
            prompt += w[j:j + 8] + [rng.randrange(50)]
        grams = RXC.prompt_grams(prompt)
    return cf, tail, dr, thr, grams


def main():
    out = dict(screens=[os.path.relpath(p, os.path.join(HERE, "..")) for p in SCREENS], checks={})
    rng = random.Random(1116)
    THRS = (None, 0.0, 0.3, 0.9, 1.0)
    # T0: the bug as EvalSuite saw it
    cf, tail, dr, _, grams = case(rng, 9, 0.0, True)
    t0 = {}
    for name, g in (("u3", None), ("copy", grams)):
        try:
            RXC.gate_rows(cf, tail, dr, 0.0, g)
            t0[name] = "no error"
        except (IndexError, AssertionError) as e:
            t0[name] = type(e).__name__
    assert t0 == {"u3": "IndexError", "copy": "IndexError"} or t0 == {"u3": "IndexError", "copy": "AssertionError"}, t0
    out["checks"]["T0_gate_rows_short_block"] = t0
    srcs = []
    for path in SCREENS:
        gb, seg, outside = helper_of(path)
        srcs.append(seg)
        assert outside == 0, (path, "RXC.gate_rows called outside _gate_block", outside)
        n1 = n2 = n2r = 0
        for _ in range(4000):                            # T1: full windows
            thr = rng.choice(THRS); wg = rng.random() < 0.5
            n = 15 if wg else rng.choice([15, 15, 16, 31])  # U3's cf-only call also sees longer cf (it reads cf[0..14])
            cf, tail, dr, thr, grams = case(rng, n, thr, wg)
            if thr is None and grams is None:
                continue
            assert gb(cf, tail, dr, thr, grams) == RXC.gate_rows(cf, tail, dr, thr, grams)
            n1 += 1
        for _ in range(4000):                            # T2: short blocks
            thr = rng.choice(THRS); wg = rng.random() < 0.5; n = rng.randrange(15)
            cf, tail, dr, thr, grams = case(rng, n, thr, wg)
            if thr is None and grams is None:
                continue
            g = gb(cf, tail, dr, thr, grams)
            assert len(g) == RXC.R and not any(g[n:]), (n, g)
            for _ in range(3):                           # any full block starting with this one agrees inside it
                ext_cf = cf + [rng.choice([0.0, 0.5, 1.0]) for _ in range(15 - n)]
                ext_dr = dr + [rng.randrange(50) for _ in range(15 - n)]
                assert g[:n] == RXC.gate_rows(ext_cf, tail, ext_dr, thr, grams)[:n], (n, thr)
                n2r += 1
            n2 += 1
        out["checks"][os.path.basename(path)] = dict(T1_full_equal=n1, T2_short_ok=n2, T2_rows_vs_full_blocks=n2r,
                                                      gate_rows_calls_outside_helper=outside)
    assert len(set(srcs)) == 1, "the helper differs between the screens"
    out["checks"]["T3_helper_identical"] = len(SCREENS)
    out["verdict"] = "PASS"
    print(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    main()
