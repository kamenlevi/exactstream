"""SHAPE NAMES (Lunatic, BREAK98c, Tue 17 Nov 11:4x). CPU only, read-only, one process, < 1 GB, ~1 min.
Decides the one family of decode shapes that the row law and the decision law (IDEAS_1117c) leave open in
assumption 5: misses whose 125B decision is SEMANTICS-FREE by construction, so a shape may accept or transfer it.
  FRESH-NAME: the 125B's token y starts an identifier that is NEW (not a whole word in the prompt + answer so far), and
    the head's top-1 d is also an identifier start that is new; neither is a keyword / builtin. A consistent rename of
    a fresh local cannot change what a program does (shadowing excluded by the builtin list).
  TEXT-PREFIX: the 125B's true text from the miss position starts with d's text (same characters, other tokenization).
Same cells, walk and lexer as lim800_slack.py (exec'd prefix). Check-ending misses on the chain-32 walk under the base
rule (exact, or CC d1 = the gated rung), then the walk with these misses counted as right (the same approximation as
the close-call walks after an accepted miss).
usage: ~/BigMoE/.venv/bin/python3 -I shape_names.py"""
import os, re, keyword, builtins
import numpy as np
SRC = open(os.path.expanduser("~/FableGod/lunatic/lim800_slack.py")).read().split("RES = {}")[0]
exec(SRC.replace('OUT = f"{H}/FableGod/lunatic/lim800_slack.out"', 'OUT = f"{H}/FableGod/lunatic/shape_names.out"'))
BANK = f"{H}/datasets/deepdraft/bank"
IDRE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$"); WRE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
RESERVED = set(keyword.kwlist) | set(dir(builtins)) | {"self", "cls"}

def prompt_text(tag):
    """per window: the prompt text before the answer (bank ids up to the first answer token), if the rows align"""
    tz = np.load(f"{CE}/topk_t30b_{tag}.npz"); dep = tz["depth"].astype(int)
    st = np.flatnonzero((dep == 0) & np.append(True, dep[:-1] != 0))
    files = [str(f) for f in tz["files"]]; out = []; ok = 0
    for s in st:
        z = np.load(f"{BANK}/{files[int(tz['fi'][s])]}"); r0 = int(tz["row"][s]); ids = z["ids"]
        good = int(ids[r0 + 1]) == int(tz["y"][s]) if r0 + 1 < len(ids) else False
        ok += good
        out.append(TOK.decode([int(t) for t in ids[:r0 + 1]], skip_special_tokens=False) if good else "")
    return out, ok, len(st)

def classes(W, PR):
    FN, TP = [], []
    for A, ptxt in zip(W, PR):
        y0 = A["Y"][:A["n0"], 0]
        pos_txt = np.cumsum([0] + [len(txt(t)) for t in y0])         # char offset of each answer token
        full = A["full"]; fn = np.zeros(A["rt"].shape, bool); tp = np.zeros(A["rt"].shape, bool)
        for i, j in zip(*np.nonzero(A["rt"] == 0)):
            p = i + j                                                   # answer position of the missed token
            if p >= len(y0) or A["S"][i, j] != 0: continue              # code cells only
            a, b = txt(A["DR"][i, j]), txt(A["Y"][i, j])
            rest = full[pos_txt[p]:pos_txt[p] + 64]
            tp[i, j] = a != b and len(a) > 0 and rest.startswith(a)
            sa, sb = a.strip(), b.strip()
            if not (IDRE.match(sa or "-") and IDRE.match(sb or "-")) or sa == sb: continue
            m = WRE.match(rest.lstrip()); yword = m.group(0) if m else sb
            prev = set(WRE.findall(ptxt + full[:pos_txt[p]]))
            if yword in RESERVED or sa in RESERVED or sb in RESERVED: continue
            if yword in prev: continue                                    # the 125B refers to an existing name
            if any(w.startswith(sa) for w in prev): continue              # the head's pick may refer to one
            fn[i, j] = True
        FN.append(fn); TP.append(tp)
    return FN, TP

for tag in ("heplus24", "mbppbank"):
    W = load(tag); PR, ok, n = prompt_text(tag)
    FN, TP = classes(W, PR)
    for A, f, t in zip(W, FN, TP): A["FN"], A["TP"] = f, t
    say(f"== {tag}: {len(W)} windows (prompt rows aligned on {ok}/{n})")
    cc1 = lambda A: np.nan_to_num(A["DL"], nan=-99) >= -1
    for base_name, base in (("exact", lambda A: np.zeros(A["rt"].shape, bool)), ("cc1", cc1)):
        rows = miss_table(W, base, base_name)
        # share of the check-ending misses that are FRESH-NAME / TEXT-PREFIX (recomputed on the same walk)
        nfn = ntp = ntot = 0
        for A in W:
            r = A["rt"].copy(); r[(r == 0) & base(A)] = 1; sub = r[C98.R0:]; p = 0
            while p < sub.shape[0]:
                k = 0
                while k < 31 and sub[p, k] == 1: k += 1
                if k < 31 and sub[p, k] == 0:
                    i = p + C98.R0; ntot += 1; nfn += bool(A["FN"][i, k]); ntp += bool(A["TP"][i, k])
                p += k + 1
        say(f"   {base_name}: check-ending misses {ntot}: FRESH-NAME {100*nfn/max(1,ntot):4.1f} %  TEXT-PREFIX {100*ntp/max(1,ntot):4.1f} %")
        k0 = walk(W, base)[0]
        for lab, fn in (("+fresh", lambda A: A["FN"]), ("+textprefix", lambda A: A["TP"]),
                        ("+both", lambda A: A["FN"] | A["TP"])):
            k1 = walk(W, lambda A, fn=fn: base(A) | fn(A))[0]
            say(f"   {base_name}{lab:12s} kept/32 {k1:6.2f}  x{k1/k0:5.3f} over {base_name} ({k0:.2f})")
    say()
open(OUT, "w").write("\n".join(lines) + "\n")
