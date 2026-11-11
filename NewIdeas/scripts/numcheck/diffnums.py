"""NUMBER CHECK helper (NewIdeas, Mon 9 Nov): list the numeric lines that changed in each paper since a snapshot, so the
re-check after the 12:00 drafts only has to look at new or changed numbers.
usage: /usr/bin/python3 -I diffnums.py <snapshot dir> [<papers dir>]   (papers dir default ~/Papers)"""
import difflib, os, re, sys

snap = sys.argv[1]
papers = sys.argv[2] if len(sys.argv) > 2 else os.path.expanduser("~/Papers")
NUM = re.compile(r"\d")
for name in ("MAIN", "KERNELS", "DRAFTER", "EVALUATION", "SMALL_RAM", "RELEASE"):
    a_p, b_p = f"{snap}/{name}.md", f"{papers}/{name}.md"
    if not (os.path.exists(a_p) and os.path.exists(b_p)):
        print(f"== {name}: missing file"); continue
    a = open(a_p).read().splitlines(); b = open(b_p).read().splitlines()
    added = [(j + 1, l) for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes()
             if tag in ("replace", "insert") for j, l in zip(range(j1, j2), b[j1:j2]) if NUM.search(l)]
    print(f"== {name}: {len(a)} -> {len(b)} lines; {len(added)} new or changed lines with digits")
    for ln, l in added:
        print(f"  {name}:{ln}: {l[:200]}")
