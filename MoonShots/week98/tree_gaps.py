"""WEEK98 TREE-E2E (MoonShots, Sun 15 Nov 2026; LEVERS_A 'TREE-E2E', reported item (2)): for every key where two arms'
answers differ, the 2B's top-2 logit gap at the FIRST differing answer position. One full-sequence forward of the
prompt + the common prefix (sb.core.target_states, read-only), so the gap is read under a third numerics (a [1, S]
forward), which tells near-ties from real disagreements: a numerics flip sits at a small gap with the two arms' tokens
as the 2B's top 2.
usage: ~/BigMoE/.venv/bin/python -I -B tree_gaps.py A.jsonl B.jsonl OUT.json"""
import json, os, sys
import numpy as np

H = os.path.expanduser
sys.path.insert(0, H("~/Sandbox98")); sys.path.insert(0, H("~/MoonShots/week98"))
import jcc_decode as JD                                   # noqa: E402  (panel_items)


def main():
    pa, pb, out = sys.argv[1:4]
    from sb import core as C
    from sb.gen import chat_ids
    import mlx.core as mx
    C.limit_memory(3.5)
    _, tok = C.load_target()
    t = C.tm()
    its = {it["key"]: it for it in JD.panel_items()}
    A = {r["key"]: r for r in map(json.loads, open(pa))}
    B = {r["key"]: r for r in map(json.loads, open(pb))}
    res = []
    for k in sorted(set(A) & set(B)):
        a, b = A[k]["ids"], B[k]["ids"]
        if a == b:
            continue
        fd = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
        if fd >= min(len(a), len(b)):
            res.append(dict(key=k, first=fd, kind="length", n=(len(a), len(b))))
            continue
        ids = chat_ids(tok, its[k]["user"]) + a[:fd]
        _, post, _, _ = C.target_states(ids, taps=(), logits=False)
        lg = t.embed_tokens.as_linear(post[-1:])[0].astype(mx.float32)
        i2 = mx.argpartition(-lg, 1)[:2]
        mx.eval(lg, i2)
        i2 = sorted(np.array(i2).tolist(), key=lambda i: -float(lg[i].item()))
        l1, l2 = float(lg[i2[0]].item()), float(lg[i2[1]].item())
        res.append(dict(key=k, first=fd, kind="token", a_tok=a[fd], b_tok=b[fd], top2=i2, gap=round(l1 - l2, 4),
                        a_dl=round(float(lg[a[fd]].item()) - l1, 4), b_dl=round(float(lg[b[fd]].item()) - l1, 4),
                        pair_is_top2=sorted([a[fd], b[fd]]) == sorted(i2)))
        mx.clear_cache()
    tk = [r for r in res if r["kind"] == "token"]
    g = np.array([r["gap"] for r in tk]) if tk else np.zeros(0)
    summ = dict(differ=len(res), token=len(tk), length=len(res) - len(tk),
                pair_is_top2=sum(r["pair_is_top2"] for r in tk),
                gap_median=round(float(np.median(g)), 4) if len(g) else None,
                gap_lt_0_1=int((g < 0.1).sum()), gap_lt_0_5=int((g < 0.5).sum()), gap_max=round(float(g.max()), 4) if len(g) else None)
    json.dump(dict(a=pa, b=pb, summary=summ, rows=res), open(out, "w"), indent=1)
    print(json.dumps(summ), flush=True)


if __name__ == "__main__":
    main()
