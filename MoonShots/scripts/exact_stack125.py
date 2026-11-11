"""Projection: MoonShots' exact pieces stacked on the 24 GB setting (C=128, 17 GB of experts), per verify pass of 4 rows
and ~3.1 tokens. Base 131 ms/pass (pass model at 17 GB: compute 46 + between 17 + stalls ~68 at 0.30 ms x 226 misses;
measured 23.7 tok/s depth 2). Savings (ms/pass), measured where noted:
  3c+combine      routed GLU -25 %: 48 x (0.421 - 0.315) ms at M=4 (bench, measured)              -5.1
  3-bit tiers     SSD bytes per miss -22 % (MSB part + LSB for 2 of 10 ranks); stalls latency-bound, take -15 %  -10.2
  out-of-order    post-spin path at M=4: 0.314 -> 0.122 ms/layer at 30 % misses (measured); applies to layers that stall (~all at 17 GB)  -9.2
  exact GDN       prework measured 0.035 ms/layer x 36 + gated norm est. 0.012 x 36                    -1.7
The pieces touch different time (GLU compute, stall bytes, stall overlap, dense chain); OoO and 3c overlap a little
(OoO hides the 3c hit compute), so the stack is applied with 3c's saving counted on the miss pairs only under OoO.
usage: exact_stack125.py   (instant)"""
base, tok = 131.0, 3.1
s3c, st3, sooo, sgdn = 5.1, 10.2, 12.5, 1.7   # OoO at 24 GB: stalls ~1.5 ms/layer >> hit compute (round 32o)
rows = [("today (pass model, profile config)", 0.0), ("+ 3c+combine", s3c), ("+ 3-bit tiers (4433333333)", s3c + st3),
        ("+ out-of-order experts", s3c * 0.3 + st3 + sooo), ("+ exact GDN prework + gated norm", s3c * 0.3 + st3 + sooo + sgdn)]
for name, sav in rows:
    T = base - sav
    print(f"{name:40s} {T:6.1f} ms/pass  {1000 * tok / T:5.1f} tok/s  ({base / T - 1:+.0%})")
