"""CPU unit test of bigmoe.repguard's R1 (LaTeX math spans masked) vs the shipped rule R0: a long answer whose formula
lines repeat must not arm R1 (R0 arms), and a plain repeated sentence must arm both.
Usage: ~/BigMoE/.venv/bin/python scripts/test_repguard.py   (CPU, ~10 s)"""
import os, sys
sys.path.insert(0, os.path.expanduser("~/BigMoE"))
from bigmoe import repguard

tok = repguard._tokenizer()
intro = "We work through each case carefully and check every step of the computation along the way. " + " ".join(f"Note {i} on the setup of this problem." for i in range(40)) + "\n\n"
words = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth"]
math = intro + "".join(f"Check {words[i % 10]} {i}:\n$$\\text{{Area}} = \\frac{{1}}{{2}} \\times 32 \\times (\\sqrt{{2}} \\times \\sqrt{{2}}) = 32$$\n"
                       for i in range(40))
loop = intro + "I love my sneakers because they are comfortable and they look great on me every single day. " * 40
ok = True
for name, text, want in (("math", math, {"R0": True, "R1": False}), ("loop", loop, {"R0": True, "R1": True})):
    ids = tok.encode(text, add_special_tokens=False)
    for mode in ("R0", "R1"):
        os.environ["BIGMOE_LOOP_GUARD_MATH"] = "1" if mode == "R1" else "0"
        g = repguard.make(tok)
        armed = False
        for c in range(1, len(ids) + 1):
            g.observe(ids[:c]); armed |= g.armed
        res = "armed" if armed else "never armed"
        good = armed == want[mode]; ok &= good
        print(f"{name:5s} {mode}: {res} ({len(ids)} tokens) {'ok' if good else 'UNEXPECTED'}")
print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
