"""WEEK98 J(A), as pre-registered in LEVERS_A_MOONSHOTS.md (MoonShots, Sun 15 Nov 2026): the BRANCH test on Qwen3.5-2B.
At up to 3 in-span rows per answer (first 50 of Sandbox98's MBPP answers, the 2B's own greedy), take the 2B's rank-2 token
there if it passes the start-AND-end guard, decode the 2B greedily for 64 tokens from prefix + that token, and compare the
code that follows the comment / docstring with the original's: the first two code lines after the span ends.
LINE: >= 90 % unchanged -> H's approximation holds for code rows; < 70 % -> H's lossy numbers aren't trustworthy.
Read-only: Sandbox98's data/mbpp and sb/core.py (model loader). Writes ~/MoonShots/week98/.
usage: ~/BigMoE/.venv/bin/python -I branch_2b.py"""
import glob, json, os, sys
import numpy as np

H = os.path.expanduser
sys.path.insert(0, H("~/Sandbox98")); sys.path.insert(0, H("~/MoonShots/week98"))
from span_tracker import SpanTracker                      # noqa: E402

SLOTS = SpanTracker.__slots__[1:]


def code_after_span(tr, toks, table, n_lines=2):
    """step tr character by character over the decoded toks; at the character that ends the span (a comment's newline,
    a docstring's third closing quote), take the rest of the text; return its first n_lines COMPLETE non-blank lines
    (a line counts only if a newline follows it). None if the span never ends."""
    text = "".join(table[t] for t in toks)
    out_ = tr.mode not in ("K", "T")                      # the first token may already have ended the span
    code = []
    for ch in text:
        was_code = out_ and tr.region == "C" and tr.mode in ("N", "S") and tr.qrun == 0
        tr.step(ch)
        if not out_:
            out_ = tr.mode not in ("K", "T")              # the comment's newline / the docstring's third closing quote
            continue
        if was_code and tr.mode in ("N", "S"):            # code characters only: later comments / docstrings are skipped
            code.append(ch)
        elif ch == "\n":
            code.append("\n")
    if not out_:
        return None
    segs = "".join(code).split("\n")[:-1]                 # the last segment may be cut off by the window: drop it
    return [s.rstrip() for s in segs if s.strip()][:n_lines]


def main():
    from sb import core as C
    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache
    C.limit_memory(6)
    model, tok = C.load_target()

    class Lazy(dict):
        def __missing__(self, k):
            v = tok.decode([int(k)]); self[k] = v; return v
    table = Lazy()
    files = sorted(glob.glob(H("~/Sandbox98/data/mbpp/mbpp_*.npz")))
    files = [f for f in files if not f.endswith(".taps.npz")][:50]
    events = []
    for f in files:
        z = np.load(f); ids = [int(x) for x in z["ids"]]; gen = int(z["gen"]); tk = np.asarray(z["tk"])
        tr = SpanTracker(table); cands = []
        for t in range(gen, len(ids)):
            snap = tuple(getattr(tr, s) for s in SLOTS)
            if tr.in_span() and t - 1 >= 0:
                d = int(tk[t - 1, 1]); g = int(tk[t - 1, 0])
                if g == ids[t] and d != g:
                    probe = SpanTracker(table)
                    for s, v in zip(SLOTS, snap):
                        setattr(probe, s, v)
                    probe.update(d)
                    if probe.in_span():
                        cands.append((t, d, snap))
            tr.update(ids[t])
        if cands:
            pick = sorted({0, len(cands) // 2, len(cands) - 1})
            events += [(f, ids, gen) + cands[i] for i in pick]
    print(f"{len(events)} events from {len(files)} answers", flush=True)
    out, same, diff, undet = [], 0, 0, 0
    for (f, ids, gen, t, d, snap) in events:
        prefix = ids[:t] + [d]
        cache = make_prompt_cache(model)
        lg = model(mx.array(prefix)[None], cache=cache)
        nxt = int(mx.argmax(lg[0, -1]).item()); cont = [nxt]
        for _ in range(63):
            lg = model(mx.array([[nxt]]), cache=cache); nxt = int(mx.argmax(lg[0, -1]).item()); cont.append(nxt)
        def tr_from(snap_, first):
            p = SpanTracker(table)
            for s, v in zip(SLOTS, snap_):
                setattr(p, s, v)
            p.update(first)
            return p
        cd = code_after_span(tr_from(snap, d), cont, table)
        cg = code_after_span(tr_from(snap, ids[t]), ids[t + 1:], table)      # the original: its whole remaining answer
        k = min(len(cd), len(cg)) if (cd is not None and cg is not None) else 0
        if k == 0:
            undet += 1; verdict = "undetermined"
        elif cd[:k] == cg[:k]:
            same += 1; verdict = "same"
        else:
            diff += 1; verdict = "changed"
        out.append(dict(file=os.path.basename(f), t=t, relaxed=table[d], greedy=table[ids[t]], verdict=verdict,
                        code_relaxed=cd, code_greedy=cg))
        mx.clear_cache()
    det = same + diff
    res = dict(events=len(events), determined=det, same=same, changed=diff, undetermined=undet,
               unchanged_pct=round(100 * same / max(det, 1), 1),
               line=("HOLDS (>= 90 %)" if det and same / det >= 0.9 else "NOT TRUSTWORTHY (< 70 %)" if det and same / det < 0.7 else "BETWEEN: the gate decides"),
               examples_changed=[e for e in out if e["verdict"] == "changed"][:5])
    print(json.dumps({k: v for k, v in res.items() if k != "examples_changed"}, indent=1))
    for e in res["examples_changed"][:3]:
        print("--", e["file"], repr(e["greedy"]), "->", repr(e["relaxed"]), "\n   greedy code:", repr(e["code_greedy"]), "\n   relaxed code:", repr(e["code_relaxed"]))
    json.dump(dict(res, all=out), open(H("~/MoonShots/week98/branch_2b.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
