"""nocomment_rest.py (Fable FULL, LOSSY350): what the first misses on CODE rows are (the misses a comment-free mode keeps).
Classes of the true token at the miss: inside a string / docstring; a NEW name at a binding site (assignment target at the
line start, for / def / class / as / lambda / with target, def parameter); another identifier start; a word continuation;
layout (newline, indent); number; punctuation. For new binding names: is the draft also an identifier start (a rename)?
The npz is the project's own recorded walk (allow_pickle only for its names array). CPU, seconds.
Run: nice -n 15 ~/BigMoE/.venv/bin/python -I nocomment_rest.py > nocomment_rest.out"""
import importlib.util, re, sys, keyword
import numpy as np
from tokenizers import Tokenizer
TOK = Tokenizer.from_file("/Users/kamen/BigMoE/models/bare/tokenizer.json")
spec = importlib.util.spec_from_file_location("relax_core", "/Users/kamen/EvalSuite/evalsuite/relax_core.py")
rc = importlib.util.module_from_spec(spec); spec.loader.exec_module(rc)
_d = {}
def dec(i):
    i = int(i)
    if i not in _d: _d[i] = TOK.decode([i])
    return _d[i]
WORD = re.compile(r"[A-Za-z_]\w*")
KW = re.compile(r"(^|[^\w.])(for|def|class|as|lambda|with|global|nonlocal|except\s+\w+\s+as)\s*$")
ASSIGN = re.compile(r"^\s*(=(?!=)|,|\+=|-=|\*=|/=|//=|%=|:\s*[\w\[\], .]+\s*=(?!=))")
def is_ident_start(s, prev):
    st = s.strip()
    if not st or not (st[0].isalpha() or st[0] == "_"): return False
    return s[0] in " \t\n" or prev is None or not (prev[-1:].isalnum() or prev[-1:] == "_")
for dom in sys.argv[1:] or ["code", "mbpp"]:
    W = np.load("/Users/kamen/TrainLab/data/d02b_walk_hpd0_%s.npz" % dom, allow_pickle=True)
    cnt = {}; total = 0; bind_swap = 0; bind_rows = 0; rows_code = 0; ex = []
    for wi in range(len(W["names"])):
        ids = W[f"{wi}_ids"].astype(int).tolist(); ck = W[f"{wi}_checks"]; dt = W[f"{wi}_dtok"]; tt = W[f"{wi}_ttok"]
        texts = [dec(i) for i in ids]
        sc = rc.CodeScan(); flag = []; instr = []
        for s in texts:
            before = bool(sc.code and sc.com); instr.append(bool(sc.code and (sc.str_ or sc.doc or sc.qn))); anyc = False
            for ch in s:
                sc._ch(ch); anyc = anyc or bool(sc.code and sc.com)
            flag.append(1 if anyc else (2 if before and s.startswith("\n") else (1 if before else 0)))
        cum = [0]
        for s in texts: cum.append(cum[-1] + len(s))
        full = "".join(texts)
        def klass(p, tok_text):
            """class of a token text placed at position p (true context before it)"""
            prev = texts[p - 1] if p > 0 else None
            if instr[p]: return "string / docstring content", None
            st = tok_text.strip()
            if "\n" in tok_text or st == "": return "layout (newline / indent)", None
            if st[0].isdigit(): return "number", None
            if st[0].isalpha() or st[0] == "_":
                if not is_ident_start(tok_text, prev): return "word continuation", None
                return "ident", None
            if st[0] in "\"'": return "string opener", None
            return "punctuation / operator", None
        for ci in range(ck.shape[0]):
            k, n, a = map(int, ck[ci]); d = dt[ci]; t = tt[ci]
            fm = next((j for j in range(n) if d[j] != t[j]), n)
            for j in range(min(fm + 1, n)):
                p = a + 1 + j
                if p < len(ids) and flag[p] == 0: rows_code += 1
            if fm >= n: continue
            p = a + 1 + fm
            if p >= len(ids) or flag[p] != 0: continue
            total += 1
            c, _ = klass(p, texts[p])
            if c == "ident":
                m = WORD.match(full, cum[p] + (len(texts[p]) - len(texts[p].lstrip())))
                word = m.group(0) if m else texts[p].strip()
                line_start = full.rfind("\n", 0, cum[p]) + 1
                prefix = full[line_start:cum[p]] + texts[p][:len(texts[p]) - len(texts[p].lstrip())]
                after = full[m.end():m.end() + 40] if m else ""
                seen = set(WORD.findall(full[:cum[p]]))
                new = word not in seen and not keyword.iskeyword(word)
                binding = bool(KW.search(prefix)) or (prefix.strip() == "" and bool(ASSIGN.match(after))) or bool(re.match(r"^\s*def \w+\([^)]*$", prefix)) \
                    or bool(re.search(r"\bfor\s+[\w, ]*,\s*$", prefix))
                if keyword.iskeyword(word): c = "keyword"
                elif new and binding:
                    c = "NEW name at a binding site"
                    dtxt = dec(d[fm]); dst = dtxt.strip()
                    swap = bool(dst) and (dst[0].isalpha() or dst[0] == "_") and is_ident_start(dtxt, texts[p - 1] if p else None) and not keyword.iskeyword(WORD.match(dst).group(0) if WORD.match(dst) else "")
                    bind_swap += int(swap)
                    if len(ex) < 8: ex.append((repr(prefix[-18:]), repr(texts[p]), repr(dtxt), swap))
                elif new: c = "new identifier, not a binding (call / attribute / use)"
                else: c = "identifier seen before in the window"
            cnt[c] = cnt.get(c, 0) + 1
    print(f"== {dom}: first misses on code rows {total} (tested code rows {rows_code}, hazard {total / max(1, rows_code):.4f})")
    for c, v in sorted(cnt.items(), key=lambda x: -x[1]):
        print(f"  {v:5d}  {100 * v / total:5.1f} %  {c}")
    nb = cnt.get("NEW name at a binding site", 0)
    print(f"  new binding names where the draft is also an identifier start (a rename, alpha-equivalent if fresh): {bind_swap} of {nb}")
    for x in ex: print("    line prefix %s | true %s | draft %s | rename %s" % x)
