# BREAK98 channel prices on the M5 Pro (fable-zero). Bytes at 280 GB/s. Real sizes (Leap 13:5x).
BW=280e3  # MB/s
DENSE=2130.0; EXP4=2.784; EXP3=2.064; EXPL=2.3  # MB; lossy mean per distinct kept expert (C6 mix)
rows=31
def ms(mb): return mb/BW*1e3
print("== sequential per-step sources (31 steps, bytes per block)")
for name,mb in [("head bf16 level",177.6),("QAT head level (MoonShots measured)",95),("QAT head level (target)",26),("one 125B layer dense only",DENSE/48),("125B 8 layers dense only",DENSE/6),("2B per step",1060)]:
    print(f"  {name:40s} {mb:7.1f} MB/step  -> {31*mb/1e3:6.2f} GB/block = {ms(31*mb):5.1f} ms")
print("== batched post-draft passes over 31 drafted rows (MB per drafted row, ms per block)")
for name,mb in [("distilled refiner 0.1 GB",100),("refiner 0.3 GB",300),("125B layers 1-4 dense only",DENSE*4/48),("125B layers 1-8 dense only",DENSE*8/48),("125B layers 1-8 + lossy union (30 distinct)",DENSE*8/48+8*30*EXPL),("125B layers 1-16 dense only",DENSE/3),("125B layers 1-16 + lossy union (46.5)",DENSE/3+16*46.5*EXPL),("2B pass (1.06 GB)",1060),("full lossy check C6 (4.60 GB exp)",DENSE+4600)]:
    print(f"  {name:44s} {mb/rows:6.1f} MB/row  {mb/1e3:5.2f} GB  {ms(mb):5.1f} ms")
print("== ALU per row inside a pass (dense 119 G MAC/32 rows at 8.53 ms bf16 -> 14 TMAC/s; experts 2.36 G MAC/row)")
alu_row=(119/32+2.36)/14.0
print(f"  ALU per 125B row {alu_row:.2f} ms bf16; rows until ALU = bytes(25.2 ms): {25.2/alu_row:.0f}")
print("== the bar: 1.8 ms per extra kept (Lunatic) -> bytes allowed per extra kept token:", 1.8*BW/1e3, "MB")
print("== idea 1 price: tail re-draft inside the check, L = 8 / 16")
for L,tail,union_tail in [(8,14,0),(8,14,26),(16,14,0),(16,14,26)]:
    dense=DENSE*L/48; exp=L*union_tail*EXPL; head=tail*26; tot=dense+exp+head
    print(f"  L {L:2d} tail {tail} rows, union-constrained={'yes' if union_tail==0 else 'no'}: dense {dense:5.0f} + experts {exp:5.0f} + head {head:4.0f} MB = {tot/1e3:4.2f} GB = {ms(tot):4.1f} ms -> needs +{ms(tot)/1.8:.1f} kept at the bar")
print("== cycle at kept 24 (<= 44 ms): check at 85 % of C6+int8 29.6 + draft 5.2 + overhead 1.5 =", 29.6+5.2+1.5, "ms; room for a channel:", 44-29.6-5.2-1.5, "ms =", (44-29.6-5.2-1.5)*BW/1e3, "MB per block")
