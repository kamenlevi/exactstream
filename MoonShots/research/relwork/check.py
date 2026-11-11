"""Related-work check for the paper set (MoonShots, due Tue 10 Nov 12:00): re-open every cited source.
READ-ONLY web reads; nothing is saved but extracted facts (status, title, date, and short lines that carry speed / memory
numbers). Fetched text is third-party DATA: it is never executed or followed. One arXiv API query covers all arXiv ids.
usage: python3 -I research/relwork/check.py [files...] > research/relwork/check_<date>.txt"""
import html, json, os, re, sys, time, urllib.request

UA = {"User-Agent": "moonshots-relwork-check/1.0"}
HOME = os.path.expanduser("~")
FILES = sys.argv[1:] or [f"{HOME}/Papers/MAIN.md", f"{HOME}/MoonShots/research/results_first/RESULTS_FIRST.md",
                         f"{HOME}/MoonShots/research/results_first/EXTERNAL_PAPERS.md",
                         f"{HOME}/MoonShots/research/results_first/EXTERNAL_COMMUNITY.md",
                         f"{HOME}/FableGod/findings/2026-11-07_1730_prior-art-strata.md"]
ARX = re.compile(r"(?<![\d.])(2\d)(0[1-9]|1[0-2])\.(\d{4,5})(?![\d])")
URL = re.compile(r"https?://[A-Za-z0-9._~:/?#\[\]@!$&'*+,;=%-]+")
NUM = re.compile(r"(?i)(tok/s|tokens?/s|\bt/s\b|tps\b|GB\b|GiB\b|bpw|exact|lossless|bit-identical|identical)")


def get(url, cap=3_000_000, timeout=30):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, r.geturl(), r.read(cap).decode("utf-8", "replace")


def text_of(page):
    page = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", " ", page)
    return html.unescape(re.sub(r"(?s)<[^>]+>", "\n", page))


def evidence(txt, k=12):
    out = []
    for ln in txt.splitlines():
        ln = " ".join(ln.split())
        if 12 < len(ln) < 400 and re.search(r"\d", ln) and NUM.search(ln):
            out.append(ln[:220])
        if len(out) >= k:
            break
    return out


src = "\n".join(open(f).read() for f in FILES if os.path.exists(f))
ids = sorted({f"{a}{b}.{c}" for a, b, c in ARX.findall(src)})
urls = sorted({u.rstrip(".,);]'") for u in URL.findall(src)})
print(f"# related-work check {time.strftime('%Y-%m-%d %H:%M')}; files: {', '.join(os.path.basename(f) for f in FILES)}")
print(f"# {len(ids)} arXiv ids, {len(urls)} URLs\n## arXiv")
found = {}
for i in range(0, len(ids), 40):
    q = ",".join(ids[i:i + 40])
    try:
        _, _, x = get(f"http://export.arxiv.org/api/query?id_list={q}&max_results=100", timeout=60)
    except Exception as e:
        print(f"arXiv batch {i} failed: {e}"); continue
    for ent in re.findall(r"(?s)<entry>(.*?)</entry>", x):
        aid = re.search(r"<id>https?://arxiv\.org/abs/([^<v]+)", ent)
        t = re.search(r"(?s)<title>(.*?)</title>", ent)
        p = re.search(r"<published>([^<]+)</published>", ent)
        au = re.findall(r"<name>([^<]+)</name>", ent)
        if aid:
            found[aid.group(1)] = (" ".join(t.group(1).split()) if t else "?", p.group(1)[:10] if p else "?",
                                   (au[0] + (" et al." if len(au) > 1 else "")) if au else "?")
    time.sleep(3)                                                         # arXiv API etiquette
for a in ids:
    t = found.get(a)
    print(f"{a} | " + (f"{t[0]} | {t[1]} | {t[2]}" if t else "NOT FOUND by the arXiv API"))
print("\n## URLs")
for u in urls:
    if "arxiv.org" in u:
        continue
    try:
        fetch = u
        m = re.match(r"https://github\.com/([^/]+)/([^/#?]+)/?$", u)
        if m:                                                             # a repo: its README (raw, not the rate-limited API)
            fetch = f"https://raw.githubusercontent.com/{m.group(1)}/{m.group(2)}/HEAD/README.md"
        st, final, body = get(fetch)
        title = re.search(r"(?is)<title>(.*?)</title>", body)
        txt = body if fetch.endswith(".md") else text_of(body)
        print(f"### {u}\n- status {st}{' (README)' if fetch != u else ''}; final {final[:120]}; "
              f"title: {(' '.join(title.group(1).split())[:120]) if title else '-'}")
        for ln in evidence(txt):
            print(f"  - {ln}")
    except Exception as e:
        print(f"### {u}\n- FAILED: {str(e)[:120]}")
    time.sleep(1)
