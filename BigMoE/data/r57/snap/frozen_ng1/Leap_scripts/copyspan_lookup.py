"""QUEUE L1-10 (Leap, Sat 2026-10-24, CPU): how much of the copy-span lever does prompt-lookup drafting already capture?
Sessions are rendered exactly as Chat1000's fshare.py (the 125B's output format, its tokenizer, the same session sample
and context rules); copy spans = verbatim spans >= 16 tokens found earlier in the context (fshare's Ctx.spans).
Lookup drafting is simulated as spec125's prompt lookup: the continuation after the most recent earlier occurrence of
the longest suffix (4, 3 or 2 tokens) of everything so far, up to K = 5 drafts per pass (its default), greedy
acceptance; a pass emits accepted + 1 tokens. Optionally the cross-check (lookup used only when the MTP head agrees on
its first token) is modelled as a random pass rate X.
Reported, over all visible output: the copy share; copy tokens lookup does NOT accept (QUEUE kill: < 5 % of output);
and passes spent inside copy spans vs one pass per span (the lever's pass saving).
Usage: [any python with tokenizers] copyspan_lookup.py [N_SESSIONS] [X]"""
import collections, glob, json, os, random, sys
from tokenizers import Tokenizer
TK = Tokenizer.from_file(os.path.expanduser("~/BigMoE/models/bare/tokenizer.json"))
enc = lambda s: TK.encode(s, add_special_tokens=False).ids if s else []
random.seed(3)                                       # fshare.py's sample
DIRS = ["~/.claude/projects"] + [d for d in os.environ.get("COPYSPAN_DIRS", "").split(":") if d]  # the recorded runs
# also read the research account's projects folder (its config dir + /projects), given here via COPYSPAN_DIRS
files = [f for d in DIRS
         for f in glob.glob(os.path.expanduser(d) + "/*/*.jsonl") if 300_000 < os.path.getsize(f) < 40_000_000]
random.shuffle(files); files = files[:int(sys.argv[1]) if len(sys.argv) > 1 else 80]
X = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
N, CAP, K, MINLEN = 8, 400_000, 5, 16
rng = random.Random(0)


class Ctx:                                           # fshare.py's context, plus a 2-4-gram index for lookup
    def __init__(self):
        self.t = []; self.idx = collections.defaultdict(list); self.ng = {}

    def add(self, ids):
        start = len(self.t); self.t += ids
        for i in range(max(0, start - N + 1), len(self.t) - N + 1):
            lst = self.idx[tuple(self.t[i:i + N])]; lst.append(i)
            if len(lst) > 16: del lst[0]
        for n in (2, 3, 4):
            for i in range(max(0, start - n + 1), len(self.t) - n + 1):
                self.ng[(n,) + tuple(self.t[i:i + n])] = i + n        # most recent occurrence: continuation start

    def spans(self, seg, minlen=MINLEN):
        out = []; i = 0
        while i <= len(seg) - minlen:
            cands = self.idx.get(tuple(seg[i:i + N])); best = 0
            for s in cands or []:
                L = N
                while i + L < len(seg) and s + L < len(self.t) and self.t[s + L] == seg[i + L]: L += 1
                best = max(best, L)
            if best >= minlen:
                out.append((i, best)); i += best
            else:
                i += 1
        return out


def strings_of_user(c):
    if isinstance(c, str): return [c]
    out = []
    for b in c or []:
        if not isinstance(b, dict): continue
        if b.get("type") == "text": out.append(b.get("text", ""))
        elif b.get("type") == "tool_result":
            cc = b.get("content")
            if isinstance(cc, str): out.append(cc)
            elif isinstance(cc, list): out += [x.get("text", "") for x in cc if isinstance(x, dict)]
    return out


tot = collections.Counter()


def decode(ctx, seg):
    """simulate lookup drafting over seg (its earlier context in ctx); count per-token outcomes inside / outside spans"""
    inspan = [False] * len(seg)
    sp = ctx.spans(seg)
    for i, L in sp:
        for j in range(i, i + L): inspan[j] = True
        tot["span_count"] += 1
    hist = ctx.t                                     # the context so far; seg tokens are appended as they are emitted
    base = len(hist)
    pos = 0
    while pos < len(seg):
        prop = []
        for n in (4, 3, 2):
            if base + pos >= n:
                suf = seg[pos - n:pos] if pos >= n else hist[base - (n - pos):] + seg[:pos]
                s = ctx.ng.get((n,) + tuple(suf))
                if s is not None and s < len(hist):
                    prop = hist[s:s + K]; break
        if prop and X < 1.0 and rng.random() > X:
            prop = []                                # cross-check rejected the lookup draft
        k = 0
        while k < len(prop) and pos + k < len(seg) and prop[k] == seg[pos + k]: k += 1
        n_emit = min(k + 1, len(seg) - pos)
        span_pass = any(inspan[pos:pos + n_emit])
        tot["passes"] += 1; tot["span_passes"] += span_pass
        for j in range(pos, pos + n_emit):
            lk = j < pos + k
            tot["tok"] += 1; tot["tok_lookup"] += lk
            if inspan[j]:
                tot["span_tok"] += 1; tot["span_tok_lookup"] += lk
        pos += n_emit


for f in files:
    ctx = Ctx(); events = []
    for line in open(f, errors="ignore"):
        try: d = json.loads(line)
        except Exception: continue
        if d.get("type") in ("user", "assistant"): events.append(d)
    i = 0
    while i < len(events):
        d = events[i]
        if d["type"] == "user":
            for s in strings_of_user(d.get("message", {}).get("content")):
                ctx.add(enc(s + "\n"))
            if len(ctx.t) > 2 * CAP:
                keep = ctx.t[-CAP:]; ctx = Ctx(); ctx.add(keep)
            i += 1; continue
        mid = d.get("message", {}).get("id"); blocks = []
        while i < len(events) and events[i]["type"] == "assistant" and events[i].get("message", {}).get("id") == mid:
            c = events[i].get("message", {}).get("content")
            if isinstance(c, list): blocks += [b for b in c if isinstance(b, dict)]
            i += 1
        for b in blocks:
            if b.get("type") == "text":
                seg = enc(b.get("text", "")); decode(ctx, seg); ctx.add(seg)
            elif b.get("type") == "tool_use":
                name = b.get("name", ""); inp = b.get("input", {}) or {}
                head = enc("<tool_call>\n<function=" + name + ">\n"); tot["forced"] += len(head); ctx.add(head)
                for k_, v in inp.items():
                    vs = v if isinstance(v, str) else json.dumps(v)
                    f1, f2, seg = enc("<parameter=" + k_ + ">\n"), enc("\n</parameter>\n"), enc(vs)
                    tot["forced"] += len(f1) + len(f2)
                    ctx.add(f1); decode(ctx, seg); ctx.add(seg); ctx.add(f2)
                tail = enc("</function>\n</tool_call>"); tot["forced"] += len(tail); ctx.add(tail)
T = tot["tok"] + tot["forced"]
unc = tot["span_tok"] - tot["span_tok_lookup"]
print(f"{len(files)} sessions, {T / 1e6:.2f} M output tokens (forced wrapper tokens {tot['forced'] / T:.1%}); lookup K={K}, "
      f"cross-check pass rate {X:.2f}")
print(f"copy spans >= {MINLEN}: {tot['span_count']} spans, {tot['span_tok'] / T:.1%} of output (fshare: 24.7 %)")
print(f"lookup accepts {tot['span_tok_lookup'] / max(1, tot['span_tok']):.1%} of copy-span tokens; NOT accepted "
      f"(drafted by nothing / emitted as a pass's own token): {unc / T:.1%} of output (QUEUE kill line: < 5 %)")
print(f"lookup over all value/text tokens: {tot['tok_lookup'] / max(1, tot['tok']):.1%} accepted from lookup drafts")
print(f"passes touching copy spans: {tot['span_passes']} of {tot['passes']} lookup-only passes ({tot['span_passes'] / tot['passes']:.1%}); "
      f"one pass per span instead: {tot['span_count']} -> saves {(tot['span_passes'] - tot['span_count']) / tot['passes']:.1%} "
      f"of these passes")
