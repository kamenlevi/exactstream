"""Write each paper's closing 'Use of AI tools' and 'References' sections (Mon 9 Nov 2026).
usage: python3 -I papers/make_refs.py [--write] [PAPER ...]
Detects citations in the paper's body (without the sections the public build drops): arXiv ids, works named without
their id (refs.NAME_TO_ARXIV), the non-arXiv works in refs.OTHER and the companion-paper tags. Everything from the
'## Use of AI tools' heading to the end of the file is regenerated. Without --write it only reports."""
import json, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import refs
P = os.path.expanduser("~/Papers/")
META = json.load(open(os.path.expanduser("~/MoonShots/research/relwork/arxiv_meta_1109.json")))
AUTH = open(P + "AUTHORS.txt").read().split("\n")[0].strip()
FILES = ["MAIN", "SMALL_RAM", "KERNELS", "EVALUATION", "DRAFTER", "RELEASE"]
AX = re.compile(r"(?<![\d.])(2\d)(0[1-9]|1[0-2])\.(\d{5})(?![\d])")
TAG = re.compile(r"\[(MAIN|SMALL-RAM|KERNELS|EVAL|DRAFTER|RELEASE)\]")
URL = re.compile(r"(?:github\.com|huggingface\.co)/[A-Za-z0-9_.\-]+/[A-Za-z0-9_.\-]+")
AI_HEAD, REF_HEAD = "## Use of AI tools", "## References"
AI_TEXT = ("The code, the experiments and the text of this paper were produced with substantial help from AI coding agents.\n"
           "The author directed the work and is responsible for its content. The numbers were\n"
           "re-derived from the raw run records named beside them, and the references were opened by hand or by script.")


def split_tail(text):
    for h in (AI_HEAD, REF_HEAD):
        i = text.find("\n" + h + "\n")
        if i >= 0:
            return text[:i].rstrip("\n") + "\n", text[i:]
    return text.rstrip("\n") + "\n", ""


def body_of(text):
    out, skip = [], False
    for line in split_tail(text)[0].split("\n"):
        if line.startswith("# ") or line.startswith("## "):
            skip = bool(re.search(r"(?i)delete before release|draft status", line))
        if not skip:
            out.append(line)
    return "\n".join(out)


def end(s):
    return s if s.endswith((".", "?", "!")) else s + "."


def fmt_arxiv(i):
    m = META[i]; a = m["authors"]
    au = a[0] if len(a) == 1 or " " not in a[0] else (a[0] + " et al." if len(a) > 3 else ", ".join(a[:-1]) + " and " + a[-1])
    s = f'{end(au)} "{end(m["title"])}" arXiv:{i}, {m["published"][:4]}.'
    return s + (" " + refs.ARXIV_EXTRA[i] if i in refs.ARXIV_EXTRA else "")


def title_of(name):
    first = open(P + name + ".md").readline()
    return re.sub(r"^#\s*(\[[A-Z-]+\]\s*)?", "", first).strip()


def build(name):
    text = open(P + name + ".md").read()
    body = body_of(text)
    ids = {m.group(1) + m.group(2) + "." + m.group(3) for m in AX.finditer(body)}
    ids |= {i for i, pat in refs.NAME_TO_ARXIV.items() if re.search(pat, body)}
    unknown = sorted(i for i in ids if i not in META)
    other = [(k, t) for k, t, pat in refs.OTHER if re.search(pat, body)]
    keyed = [(META[i]["authors"][0].split()[-1], fmt_arxiv(i)) for i in ids if i in META]
    keyed += [(refs.SORT.get(k, t.lstrip('"').split()[0]), t) for k, t in other]
    entries = [t for _, t in sorted(keyed, key=lambda kt: (kt[0].lower(), kt[1].lower()))]
    own = [k for k, v in refs.COMPANIONS.items() if v == name][0]
    comps = sorted({m.group(1) for m in TAG.finditer(body)} - {own})
    comp_entries = [f"[{c}] {AUTH}. {end(title_of(refs.COMPANIONS[c]))} "
                    + (f"Forthcoming ({refs.FORTHCOMING[c]})." if c in getattr(refs, "FORTHCOMING", {}) else "2026. A companion paper of this set.")
                    for c in comps]
    alltext = " ".join(entries).lower()
    uncovered = sorted({u for u in URL.findall(body) if u.lower().rstrip(".") not in alltext})
    tail = (f"\n{AI_HEAD}\n\n{AI_TEXT}\n\n{REF_HEAD}\n\n" + "\n".join("- " + e for e in entries)
            + ("\n\nCompanion papers:\n\n" + "\n".join("- " + e for e in comp_entries) if comp_entries else "") + "\n")
    return text, split_tail(text)[0] + tail, dict(arxiv=len([i for i in ids if i in META]), other=len(other),
                                                  companions=comps, unknown=unknown, uncovered=uncovered)


if __name__ == "__main__":
    write = "--write" in sys.argv
    names = [a for a in sys.argv[1:] if not a.startswith("--")] or FILES
    for n in names:
        old, new, rep = build(n)
        print(n, rep)
        if write and new != old:
            open(P + n + ".md", "w").write(new)
