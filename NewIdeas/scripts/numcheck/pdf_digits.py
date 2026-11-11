"""NUMBER CHECK (NewIdeas, Mon 9 Nov): are the digits in a built PDF the same as in its Markdown source?
Extracts every number token from the .md (markdown markup dropped) and from the PDF's text (pdftotext), normalises typography
(unicode minus / dashes, multiplication sign, thin and no-break spaces, ligatures, soft hyphens), and compares the two
multisets of numbers. Prints the counts and every number that appears a different number of times, with a little context.
usage: /usr/bin/python3 -I pdf_digits.py <paper.md> <paper.pdf> [--html]   (--html: the second file is an HTML page)"""
import collections, re, subprocess, sys, html

NUM = re.compile(r"(?<![\w.])[-+]?\d[\d,]*(?:\.\d+)?(?![\w])")


def norm(t):
    t = t.replace("−", "-").replace("–", "-").replace("—", "-").replace("×", "x")
    t = t.replace(" ", " ").replace(" ", " ").replace(" ", " ").replace("­", "")
    t = t.replace("ﬁ", "fi").replace("ﬂ", "fl").replace("ˆ", "^")
    t = re.sub(r"(?<![A-Za-z])x(?=\d)", "x ", t)            # ratios: "x1.08" -> "x 1.08"
    return t


def dropped_ranges():
    """--manifest <file>: the build's '[dropped: ...]' block, lines 'source lines A-B: ...' (1-based, inclusive)"""
    if "--manifest" not in sys.argv:
        return []
    out, on = [], False
    for l in open(sys.argv[sys.argv.index("--manifest") + 1], encoding="utf-8"):
        if l.startswith("[dropped"):
            on = True; continue
        if l.startswith("[") and on:
            break
        m = re.match(r"source lines (\d+)-(\d+):", l.strip())
        if on and m:
            out.append((int(m.group(1)), int(m.group(2))))
    return out


def md_text(p):
    lines = open(p, encoding="utf-8").read().split("\n")
    drop = dropped_ranges()
    t = "\n".join(l for i, l in enumerate(lines, 1) if not any(a <= i <= b for a, b in drop))
    t = re.sub(r"<!--.*?-->", " ", t, flags=re.S)          # comments
    t = re.sub(r"`([^`]*)`", r"\1", t)                      # inline code: keep its text
    t = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", t)          # links: keep the label, drop the URL
    t = re.sub(r"[*#>|]", " ", t)                           # emphasis, headings, quotes, table bars (underscores stay: keys)
    t = re.sub(r"(?<![\w])_([^_\n]+)_(?![\w])", r"\1", t)    # _emphasis_ only at word edges
    return norm(t)


def pdf_text(p):
    raw = subprocess.run(["pdftotext", "-enc", "UTF-8", p, "-"], capture_output=True, text=True, check=True).stdout
    pages = []
    for pno, pg in enumerate(raw.split("\f"), 1):             # drop the page's own number: a line equal to it near the end
        ls = pg.rstrip("\n").split("\n")
        idx = [i for i, l in enumerate(ls) if l.strip()]
        for i in reversed(idx[-4:]):
            if ls[i].strip() == str(pno):
                del ls[i]
                break
        pages.append("\n".join(ls))
    t = "\n".join(pages)
    t = re.sub(r"([/_=])[ \t]*\n[ \t]*", r"\1", t)            # the build breaks lines only after / _ = : re-join them
    return norm(t)


def html_text(p):
    t = open(p, encoding="utf-8").read()
    t = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", " ", t)
    t = re.sub(r"<[^>]+>", " ", t)
    return norm(html.unescape(t))


def nums(t):
    # join numbers a line break split inside a token (e.g. "1,2\n34" is rare; "12.\n5" never joined) - keep simple
    return [m.group(0).lstrip("+") for m in NUM.finditer(t)]


md, other = sys.argv[1], sys.argv[2]
a_t = md_text(md)
b_t = html_text(other) if "--html" in sys.argv else pdf_text(other)
A, B = collections.Counter(nums(a_t)), collections.Counter(nums(b_t))
diff = sorted(set(A) | set(B), key=lambda k: (-(abs(A[k] - B[k])), k))
bad = [(k, A[k], B[k]) for k in diff if A[k] != B[k]]
only_build = [b for b in bad if b[2] > b[1]]
bad.sort(key=lambda b: (b[2] <= b[1], -abs(b[1] - b[2])))   # build-only (changed digits?) first
print(f"{md} -> {other}: numbers in source {sum(A.values())} ({len(A)} distinct), in build {sum(B.values())} ({len(B)} distinct); "
      f"{len(bad)} distinct numbers differ in count; {len(only_build)} appear more often in the BUILD (changed digits?)")
def relaxed(t, k):
    """occurrences of the digits k with no digit (or '.digit') glued on either side; letters / underscores may touch it"""
    return len(re.findall(r"(?<![\d.])" + re.escape(k.lstrip("-")) + r"(?!\d|\.\d)", t))


real = [(k, x, y, relaxed(a_t, k), relaxed(b_t, k)) for k, x, y in bad]
real = [r for r in real if r[3] != r[4]]
print(f"  after a substring recount (layout: numbers glued to keys, table cells interleaved): {len(real)} numbers still differ")
for k, x, y, rx, ry in real:
    print(f"    STILL DIFFERENT {k!r}: source {rx}, build {ry}")
for k, x, y in bad[:60]:
    i = a_t.find(k) if x > y else b_t.find(k)
    src = a_t if x > y else b_t
    ctx = src[max(0, i - 50):i + 50].replace("\n", " ") if i >= 0 else ""
    print(f"  {k!r}: source {x}, build {y}   ... {ctx} ...")
