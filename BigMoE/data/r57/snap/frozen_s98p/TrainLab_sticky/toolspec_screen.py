"""TOOLSPEC (a) (MTP Head, Tue 11-03 23:5x, pre-registered in REPORT 'TOOLSPEC'; MoonShots' WALL2_KEPT item 3, ToolSpec
arXiv 2604.13519; CPU, seconds): on the 12 held-out agentic turns (data/d02b_walk_d02a_agent.npz: DFlash2 L1-20's
official-loop walk on D-02a's agent captures, 160 answer tokens each, the 125B's XML tool-call format
<tool_call>\\n<function=N>\\n<parameter=P>\\nvalue\\n</parameter>\\n...</function>\\n</tool_call>), where do DFlash2's
first misses fall relative to the tool-call tags?
  tag  = the 125B's token at the first miss overlaps a tag (<tool_call>, <function=..>, <parameter=..>, </parameter>,
         </function>, </tool_call>, <|im_end|>, with the newlines that join them): the most a schema FSM could touch.
  fsm  = of those, the tokens a format FSM can fill without a model decision: NOT the token that ends a value (the start
         of '\\n</parameter>'), NOT the token after '</parameter>\\n' (next parameter or </function>: the model's
         choice; argument order here is the model's, not fixed by a schema), NOT the token that starts '<tool_call>' or
         '<|im_end|>' (call or stop), NOT a function / parameter NAME token (constrained by the schema, not determined).
Kept bound: each fsm-fillable first miss counted as one more kept token (no credit for the drafter's later depths).
LINE (MoonShots): PASS fsm >= 10 % of agent first misses AND kept >= x1.05; KILL tag < 4 % OR kept < x1.02; else BETWEEN.
Output: data/toolspec_screen.txt (appended)."""
import os, re, time
import numpy as np
from tokenizers import Tokenizer
tk = Tokenizer.from_file(os.path.expanduser("~/BigMoE/models/bare/tokenizer.json"))
z = np.load(os.path.expanduser("~/TrainLab/data/d02b_walk_d02a_agent.npz"))
names = [str(x) for x in z["names"]]
TAG = re.compile(r"\n?(<tool_call>|</tool_call>|<function=[^>\n]*>|</function>|<parameter=[^>\n]*>|</parameter>|<\|im_end\|>)\n?")
CTX = 48                                         # tokens decoded before the answer so a tag opened before gen is seen
lines = []
def say(s=""):
    print(s, flush=True); lines.append(s)


def forced_chars(text):
    """per char: None (not in a tag), False (in a tag, needs a model decision), True (format-forced)"""
    f = [None] * len(text)
    for m in TAG.finditer(text):
        s, e, g = m.start(), m.end(), m.group(1)
        gs = m.start(1)
        for c in range(s, e):
            f[c] = True
        if g == "</parameter>":                   # ending a value is the model's decision (leading newline + '<')
            for c in range(s, gs + 1):
                f[c] = False
        if g in ("<tool_call>", "<|im_end|>"):    # call or stop
            for c in range(s, gs + 1):
                f[c] = False
        if text[max(0, s - 13):s].endswith("</parameter>\n") and g in ("<parameter=", "</function>") or \
                (g.startswith("<parameter=") or g == "</function>") and text[max(0, s - 13):s].endswith("</parameter>\n"):
            for c in range(gs, min(gs + 2, e)):  # '<p' vs '</': next parameter or end of the function
                f[c] = False
        if g.startswith("<function=") or g.startswith("<parameter="):
            n0 = gs + g.index("=") + 1            # the NAME chars: constrained by the schema, not determined
            for c in range(n0, gs + len(g) - 1):
                f[c] = False
    return f


tot = dict(checks=0, fm=0, tag=0, fsm=0, toks=0, tagtoks=0, kept=0)
ex = []
for i, nm in enumerate(names):
    ids, off, gen, ck = z[f"{i}_ids"], int(z[f"{i}_off"]), int(z[f"{i}_gen"]), z[f"{i}_checks"]
    s0 = gen - off - CTX
    seq = ids[s0:].tolist()
    ends = [len(tk.decode(seq[:j + 1], skip_special_tokens=False)) for j in range(len(seq))]
    text = tk.decode(seq, skip_special_tokens=False)
    starts = [0] + ends[:-1]
    fc = forced_chars(text)

    def cls(j):                                   # 'none' / 'tag' (needs a decision) / 'fsm' (all its chars forced)
        cs = [fc[c] for c in range(starts[j], min(ends[j], len(fc)))]
        if not cs or all(x is None for x in cs):
            return "none"
        return "fsm" if all(x is True for x in cs if x is not None) and cs[0] is not None else "tag"

    for j in range(CTX, len(seq)):
        tot["toks"] += 1; tot["tagtoks"] += cls(j) != "none"
    for (k, m, a) in ck:
        tot["checks"] += 1; tot["kept"] += int(k) + 1
        if k >= m:
            continue                               # every checkable draft accepted: no first miss in this check
        p = int(a) + int(k) + 1 - off - s0         # seq index of the first-miss token (the 125B's token there)
        tot["fm"] += 1
        c_ = cls(p)
        if c_ == "none":
            continue
        tot["tag"] += 1; tot["fsm"] += c_ == "fsm"
        if len(ex) < 14:
            ex.append(f"{nm} check at {int(a) - gen:+d}: the 125B's {tk.decode([seq[p]], skip_special_tokens=False)!r} -> {c_}")
fm = max(tot["fm"], 1)
say(f"TOOLSPEC (a) {time.strftime('%m-%d %H:%M')}: 12 agentic turns, {tot['checks']} checks, kept per check "
    f"{tot['kept'] / tot['checks']:.2f}; tag tokens {tot['tagtoks']} of {tot['toks']} answer tokens ({tot['tagtoks'] / max(tot['toks'], 1):.1%})")
say(f"  first misses {tot['fm']}: on a tag token {tot['tag']} ({tot['tag'] / fm:.1%}, the upper bound); FSM-fillable "
    f"{tot['fsm']} ({tot['fsm'] / fm:.1%})")
kb = (tot["kept"] + tot["fsm"]) / tot["kept"]
kb_ub = (tot["kept"] + tot["tag"]) / tot["kept"]
say(f"  kept bound (+1 per filled first miss): FSM x{kb:.3f}; if every tag first miss were filled x{kb_ub:.3f}")
v = "PASS" if (tot["fsm"] / fm >= 0.10 and kb >= 1.05) else ("KILL" if (tot["tag"] / fm < 0.04 or kb < 1.02) else "BETWEEN")
say(f"  LINE (MoonShots: PASS FSM >= 10 % of first misses AND kept >= x1.05; KILL tag < 4 % OR kept < x1.02) -> {v}")
for e_ in ex:
    say("    e.g. " + e_)
open(os.path.expanduser("~/TrainLab/data/toolspec_screen.txt"), "a").write("\n".join(lines) + "\n\n")
