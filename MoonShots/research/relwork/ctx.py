"""Second pass of the related-work check: for each source, the lines that mention OUR model (Flash-Next / Qwen3.8 /
125B / 180B) with +-2 lines of context. Read-only; output is extracted facts only. usage: python3 -I ctx.py URL..."""
import html, re, sys, urllib.request
PAT = re.compile(r"(?i)flash[- ]?next|qwen3\.8|\b125b\b|\b180b\b")
for u in sys.argv[1:]:
    try:
        b = urllib.request.urlopen(urllib.request.Request(u, headers={"User-Agent": "moonshots-relwork-check/1.0"}),
                                   timeout=30).read(4_000_000).decode("utf-8", "replace")
    except Exception as e:
        print(f"### {u}\nFAILED {e}"); continue
    if not u.endswith(".md"):
        b = html.unescape(re.sub(r"(?s)<[^>]+>", "\n", re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", " ", b)))
    L = [" ".join(x.split()) for x in b.splitlines()]
    L = [x for x in L if x]
    hits = [i for i, x in enumerate(L) if PAT.search(x) and re.search(r"\d", x)]
    print(f"### {u}  ({len(hits)} lines on our model)")
    shown = set()
    for i in hits[:14]:
        for j in range(max(0, i - 1), min(len(L), i + 3)):
            if j not in shown:
                shown.add(j); print(f"  {j}: {L[j][:230]}")
