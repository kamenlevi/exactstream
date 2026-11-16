"""fit12_c6c7_code_carry.py (Breakthrough, Mon 16 Nov 2026 12:0x; pre-registered in docs/LOSSY_CHECK_PRICE.md
"Carry check: Fit12's LOSSY32 C6 / C7 rows on code", b9f2a32). CPU, python -I -B.

Runs Fit12's ~/Fit12/scripts/lossy32_price.py (read-only; sha256 74f11370dd78) twice from its own source. Only two things
change: OUT (redirected below) and the two map 4-bit-eq inputs of C6 / C7 (its line 252).
  published: 33.6 / 54.8 (my pooled all-domain means, as Fit12 used them)
  code:      35.7 / 62.3 (code blocks: data/m1/lossy_check_domain_recount.out, 5fc4c34)
Self-check: the published variant's txt equals Fit12's data/lossy32/lossy32_price.txt byte for byte. Nothing in ~/Fit12
is written.
"""
import contextlib
import difflib
import hashlib
import io
import os

SRC = os.path.expanduser("~/Fit12/scripts/lossy32_price.py")
PUB = os.path.expanduser("~/Fit12/data/lossy32/lossy32_price.txt")
BASE = os.path.expanduser("~/Breakthrough/data/m1/fit12_c6c7_code")
OUTF = os.path.expanduser("~/Breakthrough/data/m1/fit12_c6c7_code_carry.out")
OUT_LINE = 'OUT = "/Users/kamen/Fit12/data/lossy32"'
LAM_LINE = ('("C6 C5 at lam 1.8e-2 (frontier)", 33.6, "1.8e-2"), '
            '("C7 1d pooled-safe (lam 8e-3) + V1 + V3", 54.8, "8e-3")')
VARIANTS = {"published": (33.6, 54.8), "code": (35.7, 62.3)}

raw = open(SRC, "rb").read()
sha = hashlib.sha256(raw).hexdigest()
assert sha.startswith("74f11370dd78"), sha
src = raw.decode("utf-8")
assert src.count(OUT_LINE) == 1 and src.count(LAM_LINE) == 1, "Fit12's script changed: STOP"

txt = {}
for name, (c6, c7) in VARIANTS.items():
    d = os.path.join(BASE, name)
    lam = LAM_LINE.replace("33.6", repr(c6)).replace("54.8", repr(c7))
    s = src.replace(OUT_LINE, f"OUT = {d!r}").replace(LAM_LINE, lam)
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(s, f"{SRC} [{name}]", "exec"), {"__name__": "__main__", "__file__": SRC})
    txt[name] = open(os.path.join(d, "lossy32_price.txt"), "rb").read()

L = []


def p(s=""):
    print(s)
    L.append(s)


pub = open(PUB, "rb").read()
same = txt["published"] == pub
p(f"Fit12 lossy32_price.py sha256 {sha[:12]}; published txt md5 {hashlib.md5(pub).hexdigest()}")
p(f"SELF-CHECK: published variant == Fit12's published txt byte for byte: {'PASS' if same else 'FAIL -> STOP, no claim'}")
if same:
    a, b = (txt[k].decode().splitlines() for k in ("published", "code"))
    p()
    p("Rows C5-C7, published (pooled 33.6 / 54.8) vs code (35.7 / 62.3):")
    for tag in ("| C5 ", "| C6 ", "| C7 "):
        for k, rows in (("published", a), ("code", b)):
            for r in rows:
                if r.startswith(tag):
                    p(f"  {k:9s} {r}")
    p()
    p("Every line that changes (unified diff, published -> code):")
    for r in difflib.unified_diff(a, b, "published", "code", n=0, lineterm=""):
        p(r)
open(OUTF, "w").write("\n".join(L) + "\n")
