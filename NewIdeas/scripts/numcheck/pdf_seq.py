"""NUMBER CHECK helper: ordered number-token diff between a .md and its built PDF / HTML (uses pdf_digits.py's readers).
Prints each span where the two sequences of number tokens differ, with the neighbouring numbers on each side.
usage: /usr/bin/python3 -I pdf_seq.py <paper.md> <paper.pdf> [--html] [--manifest <manifest.txt>]"""
import difflib, os, re, sys
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "pdf_digits.py")).read().split("\nmd, other = sys.argv[1]")[0]
exec(src)
a = nums(md_text(sys.argv[1])); b = nums(html_text(sys.argv[2]) if "--html" in sys.argv else pdf_text(sys.argv[2]))
sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
ops = [o for o in sm.get_opcodes() if o[0] != "equal"]
print(f"tokens md {len(a)}, build {len(b)}; matching {sum(i2 - i1 for t, i1, i2, j1, j2 in sm.get_opcodes() if t == 'equal')}; "
      f"{len(ops)} differing spans")
for t, i1, i2, j1, j2 in ops:
    print(f"  {t:7s} md[{i1}:{i2}] {a[i1:i2][:8]} -> build {b[j1:j2][:8]}   | before: {a[max(0, i1 - 4):i1]} after: {a[i2:i2 + 4]}")
