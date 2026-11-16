"""THINK_1115 #7 SLOT PRIOR (ZERO idea 3), NewIdeas, Sun 15 Nov 22:2x; v2 Mon 16 Nov 08:2x (v1 lexed from the prompt start and
called ~99 % of rows prose: VOID). CPU only, read-only.
P(the 125B's margin < 1 nat at the next token | the class of the prefix's last token / lexical state), per answer row of
the dev bank (HE+ 24 windows, MBPP 100). Margin = lp[row, 0] - lp[row, 1] (the bank's stored 125B top-2 log-probs; equal
to the cells' m). The 125B at row predicts ids[row + 1]; the class is that of ids[row] and the lexer state after it.
Classes: in comment, in string (incl. docstrings), prose (outside a ``` fence when the answer has fences),
newline / indent, punctuation, number, keyword, identifier, other.
LINE (ZERO's kill, written before the run): KILL unless some class has lift >= 2x AND covers >= 25 % of the ties (m < 1)
on HE+; MBPP reported.
usage: ~/BigMoE/.venv/bin/python -I -B slot_prior.py"""
import os, sys, glob, json, keyword
import numpy as np
from tokenizers import Tokenizer

H = os.path.expanduser("~")
BANK = f"{H}/datasets/deepdraft/bank"
SPLIT = f"{H}/BigMoE/data/deepdraft/bank/split.json"
TOKJ = sorted(glob.glob(f"{H}/.cache/huggingface/hub/models--mlx-community--Qwen3.5-2B-MLX-4bit/snapshots/*/tokenizer.json"))[0]
OUT = os.path.expanduser("~/NewIdeas/research/week98/slot_prior")
KW = set(keyword.kwlist) | {"True", "False", "None", "self"}
PUN = set("()[]{}:,.;=+-*/<>!&|^%~@")
CLASSES = ["in comment", "in string", "prose", "newline/indent", "punctuation", "number", "keyword", "identifier", "other"]


def lex_classes(pieces):
    """per token: the class of the token at that position, given the lexer state after it"""
    out = []; state = None; quote = None; fence = False; has_fence = "```" in "".join(pieces)
    text_tail = ""
    for pc in pieces:
        i = 0; s = pc
        while i < len(s):
            ch = s[i]
            if has_fence and s.startswith("```", i) and state is None:
                fence = not fence; i += 3; continue
            if state == "comment":
                if ch == "\n":
                    state = None
                i += 1; continue
            if state == "string":
                if ch == "\\":
                    i += 2; continue
                if s.startswith(quote, i):
                    state = None; i += len(quote); continue
                i += 1; continue
            if has_fence and not fence:
                i += 1; continue
            if ch == "#":
                state = "comment"; i += 1; continue
            if s.startswith('"""', i) or s.startswith("'''", i):
                state = "string"; quote = s[i:i + 3]; i += 3; continue
            if ch in "\"'":
                state = "string"; quote = ch; i += 1; continue
            i += 1
        t = pc.strip()
        if state == "comment":
            c = "in comment"
        elif state == "string":
            c = "in string"
        elif has_fence and not fence and not pc.startswith("```"):
            c = "prose"
        elif "\n" in pc or t == "":
            c = "newline/indent"
        elif all(x in PUN for x in t):
            c = "punctuation"
        elif t.replace("_", "").isdigit() or t.replace(".", "", 1).isdigit():
            c = "number"
        elif t in KW:
            c = "keyword"
        elif t.replace("_", "a").isalnum():
            c = "identifier"
        else:
            c = "other"
        out.append(c)
    return out


def main():
    tok = Tokenizer.from_file(TOKJ)
    sp = json.load(open(SPLIT))
    lines, res = [], {}
    def say(s=""):
        print(s, flush=True); lines.append(s)
    for name, files in (("he", sp["heplus"]), ("mbpp", sp["mbpp"])):
        cnt = {c: [0, 0] for c in CLASSES}                 # [rows, ties]
        for f in files:
            z = np.load(os.path.join(BANK, f), allow_pickle=True)
            ids = [int(x) for x in z["ids"]]; lp = np.asarray(z["lp"], np.float32); gen = int(z["gen"])
            pieces = [tok.decode([i]) for i in ids]
            st = max(gen - 1, 0)                              # fresh lexer at the answer: the prompt's odd fence count
            cls = [None] * st + lex_classes(pieces[st:])      # (3 HE+, 1 MBPP) and its strings must not leak in (v1 bug)
            for row in range(max(gen - 1, 0), min(len(lp), len(ids) - 1)):
                m = float(lp[row, 0] - lp[row, 1])
                c = cls[row]; cnt[c][0] += 1; cnt[c][1] += int(m < 1.0)
        rows = sum(v[0] for v in cnt.values()); ties = sum(v[1] for v in cnt.values()); base = ties / rows
        say(f"\n=== {name}: {rows} answer rows, ties (m < 1) {ties} = {100 * base:.2f} % ===")
        say(f"  {'class':16s} {'rows %':>7s} {'P(tie)':>8s} {'lift':>6s} {'ties covered':>13s}")
        r = {}
        for c in CLASSES:
            n, t = cnt[c]
            if n == 0:
                continue
            p = t / n; lift = p / base; cov = t / max(ties, 1)
            r[c] = {"rows_share": n / rows, "p_tie": p, "lift": lift, "tie_coverage": cov}
            say(f"  {c:16s} {100 * n / rows:6.1f}% {100 * p:7.2f}% {lift:6.2f} {100 * cov:12.1f}%")
        hit = [c for c, v in r.items() if v["lift"] >= 2 and v["tie_coverage"] >= 0.25]
        say(f"  classes with lift >= 2x AND >= 25 % of ties: {hit if hit else 'none'}")
        res[name] = {"base": base, "classes": r, "pass_classes": hit}
    v = "PASS" if res["he"]["pass_classes"] else "KILL"
    say(f"\nLINE (HE+: a class with lift >= 2x covering >= 25 % of ties): {v}")
    json.dump(res, open(OUT + ".json", "w"), indent=1)
    open(OUT + ".txt", "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
