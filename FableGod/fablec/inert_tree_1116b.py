"""THINK_1116b audit (Fable C, Mon 16 Nov 2026, ~18:1x): what #13 (INERT comment / docstring acceptance) and #8 (span-
conditional close-call) add ON TOP of today's lossy base (tree-32 + close-call d1 = 16.91 kept on HE+), replayed on the
same real t30b tree dumps and cells that produced 16.91 (BigMoE ceiling dumps; NewIdeas tree_cc32.py's accounting, the
chain-style arm walk: after a lossy acceptance the walk goes on along the true text's cells, the same approximation
as the booked close-call numbers). Row classes from NewIdeas slot_prior.py's lexer (fresh at the answer start), with
triple-quoted strings split out as docstrings. CPU only, read-only, one process, seconds.
usage: ~/BigMoE/.venv/bin/python -I -B inert_tree_1116b.py
"""
import os, sys, json, glob, keyword
import numpy as np
from tokenizers import Tokenizer
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.expanduser("~/EvalSuite/tools"))
import certify98 as C98

H = os.path.expanduser("~")
CE = f"{H}/BigMoE/data/deepdraft/ceiling"
BANK = f"{H}/datasets/deepdraft/bank"
TOKJ = sorted(glob.glob(f"{H}/.cache/huggingface/hub/models--mlx-community--Qwen3.5-2B-MLX-4bit/snapshots/*/tokenizer.json"))[0]
OUT = f"{H}/FableGod/fablec/inert_tree_1116b.out"
SETS = {"he": ("heplus24", 13.77), "mbpp": ("mbppbank", 6.92)}
DEP, NODES = 31, 31
KW = set(keyword.kwlist) | {"True", "False", "None", "self"}
PUN = set("()[]{}:,.;=+-*/<>!&|^%~@")
lines = []
def say(s=""):
    print(s, flush=True); lines.append(s)


def lex_classes(pieces):
    """slot_prior.py's lexer, + 'in docstring' for triple-quoted strings; class = state AFTER the token"""
    out = []; state = None; quote = None; fence = False; has_fence = "```" in "".join(pieces)
    for pc in pieces:
        i = 0; s = pc
        while i < len(s):
            ch = s[i]
            if has_fence and s.startswith("```", i) and state is None:
                fence = not fence; i += 3; continue
            if state == "comment":
                if ch == "\n": state = None
                i += 1; continue
            if state == "string":
                if ch == "\\": i += 2; continue
                if s.startswith(quote, i): state = None; i += len(quote); continue
                i += 1; continue
            if has_fence and not fence: i += 1; continue
            if ch == "#": state = "comment"; i += 1; continue
            if s.startswith('"""', i) or s.startswith("'''", i): state = "string"; quote = s[i:i + 3]; i += 3; continue
            if ch in "\"'": state = "string"; quote = ch; i += 1; continue
            i += 1
        t = pc.strip()
        if state == "comment": c = "in comment"
        elif state == "string": c = "in docstring" if len(quote) == 3 else "in string"
        elif has_fence and not fence and not pc.startswith("```"): c = "prose"
        elif "\n" in pc or t == "": c = "newline/indent"
        elif all(x in PUN for x in t): c = "punctuation"
        elif t.replace("_", "").isdigit() or t.replace(".", "", 1).isdigit(): c = "number"
        elif t in KW: c = "keyword"
        elif t.replace("_", "a").isalnum(): c = "identifier"
        else: c = "other"
        out.append(c)
    return out


def load(tag, published, tok):
    cz, tz = np.load(f"{CE}/cells_t30b_{tag}.npz"), np.load(f"{CE}/topk_t30b_{tag}.npz")
    for k in ("depth", "fi", "row", "right"):
        assert (cz[k] == tz[k]).all()
    dep, right, yr = tz["depth"].astype(int), tz["right"].astype(int), tz["y_rank"].astype(int)
    yr = np.where((yr == 0) & (right == 0), 1, yr)
    row, fi, yy = tz["row"].astype(np.int64), tz["fi"].astype(np.int64), tz["y"].astype(np.int64)
    mm, dl = cz["m"].astype(np.float32), cz["dlp"].astype(np.float32)
    files = [str(f) for f in cz["files"]]
    st = np.flatnonzero((dep == 0) & np.append(True, dep[:-1] != 0)); en = np.append(st[1:], len(dep))
    ref = C98.windows(f"{CE}/cells_t30b_{tag}.npz")
    W, k32 = [], []
    off_hits = {0: 0, 1: 0, -1: 0}
    for wi, (s, t) in enumerate(zip(st, en)):
        f_ = int(fi[s]); b = np.load(os.path.join(BANK, files[f_]))  # bank npz written by our own BigMoE session; plain arrays, no pickle needed
        ids = [int(x) for x in b["ids"]]; gen = int(b["gen"])
        pieces = [tok.decode([i]) for i in ids]
        st0 = max(gen - 1, 0)
        cls = [None] * st0 + lex_classes(pieces[st0:])
        dw = tz["depth"][s:t]; n0 = int((dw == 0).sum())
        rt = np.full((n0 + 1, DEP), -1, np.int8); R = np.full((n0 + 1, DEP), -1, np.int16)
        M = np.full((n0 + 1, DEP), np.nan, np.float32); DL = np.full((n0 + 1, DEP), np.nan, np.float32)
        ROW = np.full((n0 + 1, DEP), -1, np.int64)
        for j in range(DEP):
            m = dw == j; c = int(m.sum())
            rt[:c, j] = right[s:t][m]; R[:c, j] = yr[s:t][m]; M[:c, j] = mm[s:t][m]; DL[:c, j] = dl[s:t][m]
            ROW[:c, j] = row[s:t][m]
            if j == 0:  # settle the token offset: is the cell's true token ids[row] or ids[row + 1]?
                for o in (0, 1, -1):
                    rr = row[s:t][m] + o
                    ok = [(0 <= r < len(ids)) and ids[r] == y for r, y in zip(rr, yy[s:t][m])]
                    off_hits[o] += int(sum(ok))
        assert (rt == ref[wi][0]).all()
        R[rt < 0] = -1
        k32 += C98.kept_walk(rt, C98.R0, 31)
        an = row[s:t][dw == 0] - 1
        W.append({"rt": rt, "R": R, "M": M, "DL": DL, "ROW": ROW, "fi": f_, "an": an, "n": n0 + 1, "cls": cls, "ids": ids})
    say(f"SELF-CHECK {tag}: chain-32 {np.mean(k32):.2f} vs {published} -> {'PASS' if abs(np.mean(k32) - published) < 0.01 else 'FAIL'}")
    say(f"  token offset hits (ids[row + o] == y at depth 0): {off_hits}")
    return W


def treepass(W, tag):
    tz = np.load(f"{CE}/tree_t30b_{tag}.npz")
    nfi, nan_, nnode, ndep, ontrue = (tz[k] for k in ("fi", "an", "node", "depth", "on_true"))
    rowof = {X["fi"]: {int(a): i for i, a in enumerate(X["an"])} for X in W}
    TP = {X["fi"]: np.zeros((X["n"], DEP), bool) for X in W}
    sel = ontrue.astype(bool) & (nnode < NODES)
    for f_, a_, d_ in zip(nfi[sel], nan_[sel], ndep[sel]):
        r = rowof[int(f_)].get(int(a_))
        if r is not None and d_ < DEP: TP[int(f_)][r, int(d_)] = True
    return TP


def arm_acc(X, TP, use_tree, cc_code, cc_span, inert, off, spanset):
    n = X["n"]; out = np.zeros(n, np.int64); R, DL, ROW, cls = X["R"], X["DL"], X["ROW"], X["cls"]
    for p in range(n):
        a = 0; on_path = True
        for d in range(min(DEP, NODES)):
            if R[p, d] < 0: break
            tr = int(ROW[p, d]) + off
            cb = cls[tr - 1] if 0 <= tr - 1 < len(cls) else None
            ca = cls[tr] if 0 <= tr < len(cls) else None
            in_span = (cb is not None) and (cb == ca) and (ca in spanset)   # wholly inside: the span's ends stay exact
            ok = R[p, d] == 0
            if not ok and use_tree and on_path and TP[p, d]:
                ok = True
            elif not ok:
                delta = cc_span if in_span else cc_code
                if delta > 0 and not np.isnan(DL[p, d]) and DL[p, d] >= -delta:
                    ok = True; on_path = on_path and bool(use_tree and TP[p, d])
                elif inert and in_span:
                    ok = True; on_path = on_path and bool(use_tree and TP[p, d])
            if not ok: break
            a = d + 1
        out[p] = a
    return out


def walk(acc, n):
    p = C98.R0; ks = nc = 0; starts = []
    while p < n:
        a = int(acc[p]); starts.append(p); ks += a + 1; nc += 1; p += a + 1
    return ks, nc, starts


def main():
    tok = Tokenizer.from_file(TOKJ)
    RES = {}
    for s, (tag, pub) in SETS.items():
        if not os.path.exists(f"{CE}/tree_t30b_{tag}.npz"):
            say(f"== {s}: no tree dump, skipped"); continue
        W = load(tag, pub, tok)
        TP = treepass(W, tag)
        # offset: the cell's token is ids[row + off]; choose from the depth-0 hits (printed above)
        off = 1   # settled by the depth-0 hits: the cell token is ids[row + 1]
        # row-class shares on the answer rows and at the tree-32 walk's first misses
        CD = {"in comment", "in docstring"}; CDS = CD | {"in string"}
        tot = {}; fm = {}
        for X in W:
            cls = X["cls"]
            for c in cls:
                if c is not None: tot[c] = tot.get(c, 0) + 1
        say(f"== {s} ({tag}): answer rows by class: " + ", ".join(f"{c} {100 * v / sum(tot.values()):.1f} %" for c, v in sorted(tot.items(), key=lambda kv: -kv[1])))
        acc_t = {X["fi"]: arm_acc(X, TP[X["fi"]], True, 0, 0, False, off, CD) for X in W}
        for X in W:
            _, _, st = walk(acc_t[X["fi"]], X["n"])
            for p in st:
                a = int(acc_t[X["fi"]][p])
                if a >= NODES or X["R"][p, a] < 0: continue
                tr = int(X["ROW"][p, a]) + off; cb = X["cls"][tr - 1]; ca = X["cls"][tr]
                key = ca if (cb == ca and ca in CDS) else ("span edge" if (ca in CDS or cb in CDS) else "code")
                fm[key] = fm.get(key, 0) + 1
        say(f"   tree-32 walk's first misses by class: " + ", ".join(f"{k} {100 * v / sum(fm.values()):.1f} %" for k, v in sorted(fm.items(), key=lambda kv: -kv[1])) + f" (n {sum(fm.values())})")
        ARMS = [
            ("chain-32", False, 0, 0, False, CD),
            ("chain-32 + CC d1", False, 1, 1, False, CD),
            ("chain-32 + INERT c+d", False, 0, 0, True, CD),
            ("chain-32 + CC d1 + INERT c+d", False, 1, 1, True, CD),
            ("tree-32", True, 0, 0, False, CD),
            ("tree-32 + CC d1", True, 1, 1, False, CD),
            ("tree-32 + #8 (d1 code, d3 c+d)", True, 1, 3, False, CD),
            ("tree-32 + INERT c+d", True, 0, 0, True, CD),
            ("tree-32 + CC d1 + INERT c+d", True, 1, 1, True, CD),
            ("tree-32 + CC d1 + INERT c+d+strings", True, 1, 1, True, CDS),
            ("tree-32 + CC d2 + INERT c+d", True, 2, 2, True, CD),
            ("tree-32 + INERT comments only", True, 0, 0, True, {"in comment"}),
            ("tree-32 + INERT docstrings only", True, 0, 0, True, {"in docstring"}),
        ]
        base = None; r = {}
        for name, ut, ccc, ccs, ine, sp in ARMS:
            ks = nc = 0
            for X in W:
                a_ = arm_acc(X, TP[X["fi"]], ut, ccc, ccs, ine, off, sp)
                k_, n_, _ = walk(a_, X["n"]); ks += k_; nc += n_
            kp = ks / nc; r[name] = kp
            if base is None: base = kp
            capn = capc = 0
            for X in W:
                a_ = arm_acc(X, TP[X["fi"]], ut, ccc, ccs, ine, off, sp); _, _, stt = walk(a_, X["n"])
                for p_ in stt:
                    capc += 1; capn += int(a_[p_] >= NODES or X["R"][p_, min(int(a_[p_]), DEP - 1)] < 0)
            say(f"   {name:38s} kept/32-row check {kp:7.3f}  x{kp / base:.3f} vs chain-32   (checks ending at the 32-row cap or the window end: {100 * capn / max(capc, 1):.0f} %)")
        t, tc = r["tree-32"], r["tree-32 + CC d1"]
        for nm in ("tree-32 + #8 (d1 code, d3 c+d)", "tree-32 + INERT c+d", "tree-32 + CC d1 + INERT c+d", "tree-32 + CC d1 + INERT c+d+strings", "tree-32 + CC d2 + INERT c+d"):
            say(f"   {nm:38s} x{r[nm] / tc:.3f} vs tree-32 + CC d1,  x{r[nm] / t:.3f} vs tree-32")
        gi = r["chain-32 + INERT c+d"] - r["chain-32"]; gt = r["tree-32 + CC d1"] - r["chain-32"]; gb = r["tree-32 + CC d1 + INERT c+d"] - r["chain-32"]
        say(f"   additivity: gain(tree + CC + INERT) {gb:.3f} vs gain(tree + CC) + gain(INERT on the chain) {gt + gi:.3f} -> {gb / max(gt + gi, 1e-9):.2f}")
        RES[s] = {"kept": r, "first_miss_classes": fm, "row_classes": tot}
    open(OUT, "w").write("\n".join(lines) + "\n")
    json.dump(RES, open(OUT.replace(".out", ".json"), "w"), indent=1)

if __name__ == "__main__":
    main()
