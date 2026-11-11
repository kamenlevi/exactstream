"""Main-track arithmetic: what does 80 tok/s exact at the 24 GB setting (C=128) need? Measured today (true reference,
24 GB mode): pass ~110 ms = loader stalls ~70 ms + compute/other ~40 ms; LRU misses ~261/pass (opt_bound125, 4 rows) x
2.93 MB (records3 record, exact mode) = 765 MB/pass -> the loader delivers ~11 GB/s effective. 3.1 tokens/pass.
Scenarios: SSD bytes per pass (relative), overlap (none = stalls add; full = max(SSD, compute)), compute ms.
usage: path80_125.py   (instant)"""
TOK, COMP, BW, MB0 = 3.1, 40.0, 11.0, 765.0
S = [("today (exact 4-bit records3, no overlap)", 1.00, False, COMP),
     ("3-bit tiers 4433333333 (bytes -22 %)", 0.78, False, COMP),
     ("+ out-of-order experts (-12.5 ms)", 0.78, False, COMP - 12.5),
     ("3-bit tiers + full loader/compute overlap", 0.78, True, COMP),
     ("+ caching at the online limit (-6 % misses)", 0.78 * 0.94, True, COMP),
     ("2-bit-heavy tiers 4222222222 on records4 (-45 %) + full overlap", 0.55, True, COMP),
     ("+ compute at the chain/byte floor (~29 ms)", 0.55, True, 29.0),
     ("Belady-class caching (x0.6) + 3-bit tiers + full overlap + 29 ms compute", 0.78 * 0.6, True, 29.0)]
for name, rel, ov, comp in S:
    ssd = MB0 * rel / BW
    T = max(ssd, comp) if ov else ssd * (70.0 / (MB0 / BW)) + comp      # no overlap: stalls scale with bytes from today's 70 ms
    print(f"{name:72s} SSD {ssd:5.1f} ms  T {T:6.1f} ms  {1000 * TOK / T:5.1f} tok/s")
