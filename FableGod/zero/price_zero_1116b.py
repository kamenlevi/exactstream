# First-principles price of a kept token on the M5 Pro (real sizes, Leap 16 Nov): CPU desk calc
BW=280.0            # GB/s streaming
DENSE_GB={'4bit':2.13,'V4':1.80}; DENSE_MS={'bf16':8.53,'int8':4.93}   # compute at 32 rows (119 G MAC)
EXP_GB={'C5':5.05,'C6':4.60}; CORES=1.2
def bound(exp,dense,mma):
    d=max(DENSE_GB[dense]/BW*1000, DENSE_MS[mma]); e=EXP_GB[exp]/BW*1000
    return d+e+CORES, d, e
print("== bound per 32-row check at 100 % (ms) and per kept token at kept 16.91 ==")
for exp,dense,mma in [('C5','4bit','bf16'),('C6','4bit','int8'),('C6','V4','int8')]:
    b,d,e=bound(exp,dense,mma); k=16.91
    print(f"{exp}+{dense}+{mma}: bound {b:.1f} = dense {d:.1f} + experts {e:.1f} + cores {CORES}; per kept: {b/k:.2f} ms (dense {d/k:.2f}, experts {e/k:.2f}); zero-draft tok/s {1000*k/b:.0f}")
print("\n== kept needed for 775 / 800 (draft 3, overhead 1.5) by engine efficiency ==")
for exp,dense,mma in [('C6','4bit','int8'),('C6','V4','int8')]:
    b,_,_=bound(exp,dense,mma)
    for eff in (1.0,0.9,0.85,0.7,0.6):
        cyc=b/eff+3+1.5
        print(f"{exp}+{dense}+{mma} eff {eff:.2f}: check {b/eff:.1f} ms, cycle {cyc:.1f}: kept 775 -> {775*cyc/1000:.1f}, 800 -> {800*cyc/1000:.1f}")
print("\n== the measured and Friday checks: kept needed at 32 rows (bonus row counts; max 33) ==")
for chk in (56.4,42,36,30,29.6,25.2):
    cyc=chk+3+1.5; print(f"check {chk:.1f}: 775 needs kept {775*cyc/1000:.1f}; at kept 16.91 -> {1000*16.91/cyc:.0f} tok/s; at kept 25 -> {25000/cyc:.0f}")
print("\n== bytes per kept token (MB) today and the 775 line ==")
tot=DENSE_GB['4bit']+EXP_GB['C6']
print(f"C6+int8: {tot:.2f} GB per check; at kept 16.91 = {1000*tot/16.91:.0f} MB per kept; 775 at 85 % with 4.5 ms draft+overhead allows {(1/775*1000-4.5*16.91/16.91/16.91*0+0):.3f}")
for k in (16.91,20,23.15,25,26.6):
    ms_tok=1000/775; ms_check=ms_tok*k-4.5; allowed_GB=ms_check*0.85*BW/1000
    print(f"kept {k:5.2f}: 775 allows check {ms_check:.1f} ms = {allowed_GB:.2f} GB at 85 % (C6 needs {tot:.2f}, C6+V4 {DENSE_GB['V4']+EXP_GB['C6']:.2f})")
print("\n== where the expert bytes go (C5 per layer: 15.27 experts at 4-bit 2.784 MB + 30.73 at MSB-3 2.064 MB) ==")
b4=15.27*2.784; b3=30.73*2.064
print(f"per layer: 4-bit part {b4:.1f} MB ({100*b4/(b4+b3):.0f} %), MSB-3 part (sure rows' ranks 0-1) {b3:.1f} MB ({100*b3/(b4+b3):.0f} %); per check {48*(b4+b3)/1000:.2f} GB")
for cut,lab in ((0.5,'sure rows rank 0 only (half the MSB-3 distinct)'),(0.4,'rank 0 only, 60 % of MSB-3 distinct remain'),(0.33,'MSB-2 on sure rows (2.064->1.38 MB)')):
    sav=48*b3*cut/1000
    print(f"  {lab}: -{sav:.2f} GB per check = -{1000*sav/BW:.1f} ms at 100 %, -{1000*sav/BW/0.85:.1f} at 85 %")
print("\n== rows per check at today's fixed costs: 32 vs 64 (union x1.42 lossy [P], dense compute x2, GDN x2, launches x1) ==")
for fixed_excess in (0,10,20,30):
    c32=25.2+fixed_excess; c64=25.2+fixed_excess+ (4.60*0.42/BW*1000) + 4.93 + 3.5
    for k32,k64,lab in ((16.91,20.5,'tree-32 -> K64 [P, between 16.26 and 23.7]'),(16.91,23.7,'K128-like kept at 64 rows (early death)')):
        print(f"excess {fixed_excess:2d}: 32 rows {1000*k32/(c32+4.5):.0f} tok/s vs 64 rows {1000*k64/(c64+4.5):.0f} ({lab}; checks {c32:.1f} / {c64:.1f} ms)")
