"""The definitive 350 table (Leap, Sat 2026-10-24): code tok/s at the bytes floor of a verify pass, for the 125B and the
35B, route by route. tok/s = tokens per pass / (bytes per pass / bandwidth + drafting), one stream, F = 0 (code).
Inputs, all measured locally:
- Bytes from the weight files (safetensors headers):
  - 125B: routed experts 2.765 MB each at 4-bit (67.9 GB for 48 x 512); dense per pass 3.44 GB (hyper-connections 1.28,
    GDN 1.18, LM head 0.36, attention 0.36, shared expert 0.13, routers + PLE 0.13).
    - Dense diet 2.1 GB (Chat1000 REPORT).
    - v3 tiers: ranks 3-10 at 3 bits, ~2.15 MB per expert (NewIdeas); validated, not bit-exact.
  - 35B: 1.769 MB per expert (18.1 GB for 40 x 256); dense 1.10 GB.
- Expert union per pass:
  - 125B agentic routes 10 r^0.7185 per layer (8 rows 46.2, 16 rows 73.3; Leap L1-06);
  - 35B fresh-text routes 8 tokens 32.4, 16 tokens ~53 (SpecDecode 5.2, same exponent).
- Budget: union capped at 32 per layer (Chat1000's U16 = 32; non-exact).
- Tokens per pass on code (replay, fixed counting; the real loop gives 0.93-0.98x of it):
  - 125B DFlash2: block 8 6.87; block 16 fine-tuned 9.88; and block 16 fine-tuned with confidence truncation 9.38 at
    10.34 rows.
  - 35B z-lab DFlash, measured end to end: block 16 11.92, block 8 7.69.
- Bandwidth 280 GB/s (SpecDecode 5.1, measured read). Drafting 3.5 ms (125B) / 4 ms (35B) at the floor.
Usage: python3 table350.py  -> ~/Leap/data/table350.txt"""
import os

BW = 280e9
union125 = lambda r: 10.0 * r ** 0.7185
union35 = lambda r: 32.4 * (r / 8) ** 0.7185

def tps(tokens, rows, dense, mb, layers, union, draft_ms, cap=None):
    u = min(union(rows), cap) if cap else union(rows)
    gb = dense + u * layers * mb / 1e3
    ms = gb * 1e9 / BW * 1e3 + draft_ms
    return tokens / ms * 1e3, gb, ms

rows = []
def row(model, route, tokens, nrows, dense, mb, layers, union, draft, cap=None, resid="", note=""):
    t, gb, ms = tps(tokens, nrows, dense, mb, layers, union, draft, cap)
    rows.append((model, route, resid, f"{tokens:.2f} @ {nrows:g} rows", f"{gb:.1f}", f"{ms:.1f}", f"{t:.0f}", note))

E4, EV3, D, DD = 2.765, 2.15, 3.44, 2.1
for name, tok, nr in (("DFlash2 block 8", 6.87, 8), ("DFlash2 block 16 fine-tuned", 9.88, 16),
                      ("block 16 fine-tuned + confidence truncation", 9.38, 10.34)):
    row("125B", f"{name}, bit-exact 4-bit", tok, nr, D, E4, 48, union125, 3.5, resid="71.7 GB: cannot fit 64 GB")
    row("125B", f"{name}, bit-exact 4-bit + dense diet", tok, nr, DD, E4, 48, union125, 3.5, resid="71.7 GB: cannot fit")
    row("125B", f"{name}, v3 tiers + dense diet", tok, nr, DD, EV3, 48, union125, 3.5, resid="~56 GB: > 45 GB rule")
row("125B", "block 16 fine-tuned, v3 + diet + budget U=32", 9.88, 16, DD, EV3, 48, union125, 3.5, cap=32,
    resid="~56 GB: > 45 GB rule", note="non-exact twice (tiers, budget)")
row("35B", "z-lab DFlash block 8 (measured 207.6)", 7.69, 8, 1.10, 1.769, 40, union35, 4.0, resid="20.4 GB: fits")
row("35B", "z-lab DFlash block 16 (measured 218.9)", 11.92, 16, 1.10, 1.769, 40, union35, 4.0, resid="20.4 GB: fits")
head = ("model", "route", "residency", "tokens per pass", "GB/pass", "ms/pass", "tok/s at floor", "")
w = [max(len(str(r[i])) for r in rows + [head]) for i in range(len(head))]
out = [" | ".join(str(x).ljust(w[i]) for i, x in enumerate(r)) for r in [head] + rows]
for l in out:
    print(l)
open(os.path.expanduser("~/Leap/data/table350.txt"), "w").write("\n".join(out) + "\n")
