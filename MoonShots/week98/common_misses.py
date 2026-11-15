"""WEEK98 (MoonShots, Sun 15 Nov 2026, orchestrator 01:3x): the COMMON MISSES at depth 23-30, rows where t30b's head AND the
2B (teacher-forced) are both wrong. They are the gap between the oracle chooser (HE+ 96.29, MBPP 89.96) and 98 %.
Per set, compared with all 23-30 cells and with the fixable ones (head wrong, 2B right):
- the 125B's margin bin (m < 0.25 nat: near-ties, which no exact method can fix),
- the true token's class (newline / indent, other whitespace, punctuation / operator, number, identifier or keyword,
  inside a string, inside a comment),
- its place in the line (starts the line, first word, mid-line, ends the line),
- whether the true token appeared earlier: in the last 64 / 256 tokens / anywhere before, and whether a bigram lookup
  (the last two tokens seen earlier, followed by this token) predicts it, which is what a copy lever can reach.
Inputs read-only: the cells (BigMoE's MBPP, MTP Head's HE+), the bank files (ids), rows2b_*.npz, the 2B tokenizer.
usage: ~/BigMoE/.venv/bin/python -I common_misses.py"""
import json, os, sys
import numpy as np

H = os.path.expanduser
sys.path.insert(0, H("~/MoonShots/week98"))
import chooser as CH                                      # noqa: E402  (the cell loaders and the 2B join)

BANK = H("~/datasets/deepdraft/bank")
CLS = ["newline/indent", "space", "punct/op", "number", "ident/keyword", "string text", "comment text", "other"]
POS = ["starts line", "first word", "mid-line", "ends line"]


def annotate(ids, gen, tok):
    """per token (index t >= gen): class and place in the line. A character scan of the decoded answer marks code,
    string (', ", and triple-quoted) and comment characters (Python-like; enough for HumanEval+ / MBPP answers)."""
    pieces = [tok.decode([int(i)]) for i in ids[gen:]]
    text = "".join(pieces); n = len(text)
    state = np.zeros(n + 1, np.int8)                      # 0 code, 1 string, 2 comment
    i, mode, q = 0, 0, ""
    while i < n:
        ch = text[i]; state[i] = mode
        if mode == 0:
            if ch == "#":
                mode = 2; state[i] = 2
            elif text.startswith('"""', i) or text.startswith("'''", i):
                q = text[i:i + 3]; state[i:i + 3] = 1; i += 3; mode = 1; continue
            elif ch in "'\"":
                q = ch; mode = 1; state[i] = 1
        elif mode == 1:
            if ch == "\\":
                state[i:i + 2] = 1; i += 2; continue
            if len(q) == 3 and text.startswith(q, i):
                state[i:i + 3] = 1; i += 3; mode = 0; continue
            if len(q) == 1 and ch in (q, "\n"):
                mode = 0
        elif ch == "\n":
            mode = 0
        i += 1
    cls = np.full(len(ids), 7, np.int64); pos = np.full(len(ids), 2, np.int64)
    off = 0
    for k, s in enumerate(pieces):
        t = gen + k
        lead = len(s) - len(s.lstrip())                   # the token's first non-space character decides string / comment
        st_ = int(state[min(off + lead, n)]) if s.strip() else 0
        line_start = text.rfind("\n", 0, off) + 1
        before = text[line_start:off]
        stp = s.strip()
        if "\n" in s or (stp == "" and before.strip() == ""):
            c = 0
        elif st_ == 2:
            c = 6
        elif st_ == 1:
            c = 5
        elif stp == "":
            c = 1
        elif stp.isdigit():
            c = 3
        elif any(ch.isalpha() or ch == "_" for ch in stp):
            c = 4
        else:
            c = 2
        cls[t] = c
        pos[t] = 3 if "\n" in s else (0 if off == line_start else (1 if before.strip() == "" else 2))
        off += len(s)
    return cls, pos


def main():
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(CH.TOKJ)
    out = {}
    for name, cells, r2b_file in (("mbpp", CH.mbpp_cells(), "rows2b_mbpp.npz"), ("heplus", CH.heplus_cells(), "rows2b_heplus.npz")):
        c = CH.attach_2b(cells, H(f"~/MoonShots/week98/{r2b_file}"), lambda t: 5)
        if name == "mbpp":
            Z = np.load(H("~/BigMoE/data/deepdraft/ceiling/cells_t30b_mbpp100.npz")); m125 = Z["m"]
        else:
            Z = np.load(H("~/BigMoE/data/deepdraft/ceiling/cells_t30b_heplus.npz"), allow_pickle=True); m125 = Z["m"]
        deep = (c["depth"] >= 23) & (c["depth"] <= 30)
        feats = {k: np.zeros(len(c["depth"]), np.int64) for k in ("cls", "pos", "seen64", "seen256", "seenany", "bigram")}
        for k, nm in enumerate(c["files"]):
            sel = np.flatnonzero((c["fi"] == k) & deep)
            if not len(sel):
                continue
            z = np.load(os.path.join(BANK, nm)); ids = np.asarray(z["ids"]); gen = int(z["gen"])
            cls, pos = annotate(ids, gen, tok)
            for i in sel:
                r = int(c["row"][i]); t = r + 1; tid = ids[t]
                feats["cls"][i], feats["pos"][i] = cls[t], pos[t]
                feats["seen64"][i] = int(tid in ids[max(0, t - 64):t]); feats["seen256"][i] = int(tid in ids[max(0, t - 256):t])
                feats["seenany"][i] = int(tid in ids[:t])
                a, b = ids[t - 2], ids[t - 1]               # bigram lookup: an earlier (a, b) followed by this token
                hits = np.flatnonzero((ids[:t - 2] == a) & (ids[1:t - 1] == b))
                feats["bigram"][i] = int(len(hits) > 0 and ids[hits[-1] + 2] == tid)
        right, r2b = c["right"], c["r2b"]
        groups = {"all 23-30": deep, "common miss (both wrong)": deep & ~right & ~r2b,
                  "fixable (head wrong, 2B right)": deep & ~right & r2b}
        res = {}
        mb = np.digitize(m125, [0.25, 1.0, 4.0])
        for g, s in groups.items():
            n = int(s.sum())
            res[g] = dict(
                cells=n, pt_of_deep=round(100 * n / max(int(deep.sum()), 1), 2),
                margin={lab: round(100 * float((mb[s] == j).mean()), 1) for j, lab in enumerate(["<0.25", "0.25-1", "1-4", ">=4"])},
                token_class={lab: round(100 * float((feats["cls"][s] == j).mean()), 1) for j, lab in enumerate(CLS)},
                line_place={lab: round(100 * float((feats["pos"][s] == j).mean()), 1) for j, lab in enumerate(POS)},
                seen={k: round(100 * float(feats[k][s].mean()), 1) for k in ("seen64", "seen256", "seenany", "bigram")})
        out[name] = res
        print(f"===== {name}")
        for g, r in res.items():
            print(f"-- {g}: {r['cells']} cells ({r['pt_of_deep']} pt of the 23-30 cells)")
            print("   125B margin %:", r["margin"]); print("   token class %:", r["token_class"])
            print("   place in line %:", r["line_place"]); print("   seen before %:", r["seen"])
    json.dump(out, open(H("~/MoonShots/week98/common_misses.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
