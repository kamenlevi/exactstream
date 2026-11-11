"""QUEUE L1-10 stage 2 (Leap, Sat 2026-10-24, CPU): does WIDER prompt lookup capture the copy-span lever, and how much of it
fits today's engine (<= 9 rows per pass: the gated routing row carries <= 92 ids, top-10)?
Same sessions, rendering, copy-span finder and lookup rule as copyspan_lookup.py (spec125's prompt lookup: the
continuation after the most recent earlier occurrence of the longest 4/3/2-token suffix, greedy acceptance, a pass emits
accepted + 1). Width policies:
  fixed K: up to K drafts every pass (K = 5 is spec125's default);
  adaptive 5 -> W: 5 drafts, widened to W while the previous pass accepted every lookup draft it had (a copy run).
Pass cost model (SSD-streamed C192, Fit12 X1): verify time = a + b x unique experts per layer, with the union of r rows'
top-10 routes = 10 r^0.7185 (fit to Leap's agentic routes: 4 rows 28.2, 8 rows 46.2, 16 rows 73.3), and a, b from
Fit12's MTP 4-row (~94 ms) and DFlash2 8-row (~149 ms) verify passes. Reported per policy: tokens per pass, rows per
pass and modelled ms per token, inside copy spans and over all value / text tokens (lookup-only decoder: a pass without
an accepted lookup draft emits 1 token; in the real decoder MTP drafts there, which shrinks the all-token gains).
Usage: [any python with tokenizers] [COST=ram] copyspan_width.py [N_SESSIONS]   (COST=ram: in-RAM linear row cost)"""
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
N, CAP, MINLEN = 8, 400_000, 16
POL = [("fixed 5", 5, 5), ("fixed 8", 8, 8), ("adaptive 5->8", 5, 8), ("fixed 16", 16, 16), ("adaptive 5->16", 5, 16),
       ("fixed 32", 32, 32), ("adaptive 5->32", 5, 32)]
POL += [("L " + n, k0, kw) for n, k0, kw in POL]      # "L": longest-match lookup (SAM-decoding style): of the recent
# occurrences of the last 8 tokens, draft from the one whose match reaches furthest back (<= 64); else the 4/3/2 rule
UNION = lambda r: 10.0 * r ** 0.7185
B_MS = (149.0 - 94.0) / (UNION(8) - UNION(4)); A_MS = 94.0 - B_MS * UNION(4)
T_MS = lambda r: A_MS + B_MS * UNION(r)
if os.environ.get("COST") == "ram":                  # in RAM, today's exact 3c expert kernel: ~6 us per (row, rank) pair
    A_MS, B_MS = 15.0, 0.006 * 10 * 48             # whatever the union (Chat1000 REPORT 15:19), + ~15 ms dense / HC / head
    T_MS = lambda r: A_MS + B_MS * r


class Ctx:                                           # copyspan_lookup.py's context (fshare.py's, plus a 2-4-gram index)
    def __init__(self):
        self.t = []; self.idx = collections.defaultdict(list); self.ng = {}

    def add(self, ids):
        start = len(self.t); self.t += ids
        for i in range(max(0, start - N + 1), len(self.t) - N + 1):
            lst = self.idx[tuple(self.t[i:i + N])]; lst.append(i)
            if len(lst) > 16: del lst[0]
        for n in (2, 3, 4):
            for i in range(max(0, start - n + 1), len(self.t) - n + 1):
                self.ng[(n,) + tuple(self.t[i:i + n])] = i + n

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


tot = {p[0]: collections.Counter() for p in POL}
span_tok = [0]


def decode(ctx, seg):
    inspan = [False] * len(seg)
    for i, L in ctx.spans(seg):
        for j in range(i, i + L): inspan[j] = True
    span_tok[0] += sum(inspan)
    hist, base = ctx.t, len(ctx.t)
    for name, k0, kw in POL:
        c = tot[name]; pos, K = 0, k0
        while pos < len(seg):
            prop = []
            if name.startswith("L ") and base + pos >= N:
                suf = seg[pos - N:pos] if pos >= N else hist[base - (N - pos):] + seg[:pos]
                best, bs = -1, None
                for s0 in ctx.idx.get(tuple(suf)) or []:
                    L = N                         # extend the match backwards from the 8-gram
                    while L < 64 and s0 - (L - N) - 1 >= 0 and L + 1 <= base + pos:
                        a = s0 - (L - N) - 1; b = pos - L - 1
                        if (seg[b] if b >= 0 else hist[base + b]) != hist[a]: break
                        L += 1
                    if L >= best and s0 + N < len(hist): best, bs = L, s0
                if bs is not None:
                    prop = hist[bs + N:bs + N + K]
            for n in (4, 3, 2):
                if prop: break
                if base + pos >= n:
                    suf = seg[pos - n:pos] if pos >= n else hist[base - (n - pos):] + seg[:pos]
                    s = ctx.ng.get((n,) + tuple(suf))
                    if s is not None and s < len(hist):
                        prop = hist[s:s + K]; break
            k = 0
            while k < len(prop) and pos + k < len(seg) and prop[k] == seg[pos + k]: k += 1
            n_emit = min(k + 1, len(seg) - pos)
            rows = 1 + len(prop)
            sp = any(inspan[pos:pos + n_emit])
            c["passes"] += 1; c["rows"] += rows; c["ms"] += T_MS(rows); c["tok"] += n_emit
            if sp:
                ns = sum(inspan[pos:pos + n_emit])
                c["s_passes"] += 1; c["s_rows"] += rows; c["s_ms"] += T_MS(rows); c["s_tok"] += ns
            K = kw if (prop and k == len(prop)) else k0
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
                inp = b.get("input", {}) or {}
                ctx.add(enc("<tool_call>\n<function=" + b.get("name", "") + ">\n"))
                for k_, v in inp.items():
                    vs = v if isinstance(v, str) else json.dumps(v)
                    seg = enc(vs)
                    ctx.add(enc("<parameter=" + k_ + ">\n")); decode(ctx, seg); ctx.add(seg); ctx.add(enc("\n</parameter>\n"))
                ctx.add(enc("</function>\n</tool_call>"))
c5 = tot["fixed 5"]
print(f"{len(files)} sessions, {c5['tok'] / 1e6:.2f} M value/text tokens, copy spans >= {MINLEN}: {span_tok[0] / c5['tok']:.1%}; "
      f"cost model{' (in RAM, linear in rows)' if os.environ.get('COST') == 'ram' else ''}: {A_MS:.1f} + {B_MS:.2f} x {'rows' if os.environ.get('COST') == 'ram' else 'union(rows)'} ms (T1 {T_MS(1):.0f}, T6 {T_MS(6):.0f}, T9 {T_MS(9):.0f}, "
      f"T17 {T_MS(17):.0f}, T33 {T_MS(33):.0f} ms)")
print(f"{'policy':16s} | copy spans: tok/pass rows/pass ms/tok (vs fixed 5) | all tokens: tok/pass rows/pass ms/tok (vs fixed 5)")
for name, _, kw in POL:
    c = tot[name]
    st, at = c["s_ms"] / c["s_tok"], c["ms"] / c["tok"]
    print(f"{name:16s} | {c['s_tok'] / c['s_passes']:5.2f} {c['s_rows'] / c['s_passes']:5.1f} {st:6.1f} ({st / (c5['s_ms'] / c5['s_tok']):.3f}) "
          f"| {c['tok'] / c['passes']:5.2f} {c['rows'] / c['passes']:5.1f} {at:6.1f} ({at / (c5['ms'] / c5['tok']):.3f})"
          f"{'   <= 9 rows: runs on today' + chr(39) + 's engine' if kw <= 8 else ''}")
