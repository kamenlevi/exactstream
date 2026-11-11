"""R-16 W24-ec bytes per token (CPU replay: wall24.py's engine-LFU model on the expA traces, 16-row checks accepting 10):
today's v3 split, W24-sp 11-bit (C_m 154 / C_l 49 in the same 24 GB), and W24-sp + W24-ec (same pools; an MSB miss reads
the rANS-coded codes, 1,843,200 B x b / 3 incl. states and offsets, + the 11-bit scale / bias blocks 211,248 B, rounded up
to 16 KB per expert; b = the screen's mean 2.7231 and its worst layer 2.7389). Then 12 GB for plan 4433333333 at
C_m 51 / C_l 20 (NewIdeas for Breakthrough's R-19): MSB miss 2,211,840 (bt_b12) or 2,162,688 (v3) -> W24-sp + W24-ec,
LSB miss 770,048 -> 720,896 (11-bit), same slots; side: the slots W24-sp's smaller MSB slot buys in the same RAM.
usage: python w24ec_replay.py -> stdout"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wall24 as W

blk = lambda n: -(-n // 16384) * 16384
base = W.bytes_per_token("lfu", Cm=147, Cl=46)[0]
W.MSB, W.LSB = 2064384, 720896
sp = W.bytes_per_token("lfu", Cm=154, Cl=49)[0]
print(f"today v3 (C_m 147 / C_l 46): {base / 1e6:.1f} MB / token")
print(f"W24-sp 11-bit (C_m 154 / C_l 49): {sp / 1e6:.1f} MB / token ({100 * (sp / base - 1):+.1f} %)")
for b in (2.7231, 2.7389):
    W.MSB = blk(round(1843200 * b / 3) + 211248)
    ec = W.bytes_per_token("lfu", Cm=154, Cl=49)[0]
    print(f"W24-sp + W24-ec at {b} b (MSB miss {W.MSB:,} B): {ec / 1e6:.1f} MB / token ({100 * (ec / sp - 1):+.1f} % vs "
          f"W24-sp, {100 * (ec / base - 1):+.1f} % vs today)")
ec_msb = blk(round(1843200 * 2.7231 / 3) + 211248)
print("12 GB, plan 4433333333, C_m 51 / C_l 20, 16-row checks accepting 10, LFU 0.99:")
r = {}
for nm, m, l in (("v3 records (MSB 2,162,688 / LSB 770,048)", 2162688, 770048),
                 ("bt_b12 MSB pieces (2,211,840 / LSB 770,048)", 2211840, 770048),
                 (f"W24-sp + W24-ec (MSB miss {ec_msb:,} / LSB 720,896)", ec_msb, 720896),
                 (f"W24-ec only (MSB miss {blk(round(1843200 * 2.7231 / 3) + 3 * 102400):,} / LSB 770,048)",
                  blk(round(1843200 * 2.7231 / 3) + 3 * 102400), 770048)):
    W.MSB, W.LSB = m, l
    r[nm] = W.bytes_per_token("lfu", Cm=51, Cl=20)[0]
    print(f"  {nm}: {r[nm] / 1e6:.1f} MB / token")
k = list(r)
print(f"  W24-sp + W24-ec vs v3 records {100 * (r[k[2]] / r[k[0]] - 1):+.1f} %, vs bt_b12 pieces {100 * (r[k[2]] / r[k[1]] - 1):+.1f} %; "
      f"W24-ec alone vs v3 {100 * (r[k[3]] / r[k[0]] - 1):+.1f} %")
cm, cl = (51 * 2162688 + 20 * 770048) // 2064384, 20
W.MSB, W.LSB = ec_msb, 720896
side = W.bytes_per_token("lfu", Cm=cm, Cl=cl)[0]
print(f"  side: W24-sp's smaller slots in the same RAM -> C_m {cm} / C_l {cl}: {side / 1e6:.1f} MB / token "
      f"({100 * (side / r[k[0]] - 1):+.1f} % vs v3 records)")
