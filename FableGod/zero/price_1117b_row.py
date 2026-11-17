# BREAK98b: the marginal row on the M5 Pro, from measured parts (fable-zero). [K] = measured, else [P].
PEAK_BF16=14.0   # TMAC/s, measured matrix peak at the check's shapes (check25_bound)
INT8_RATE=1.64   # dn32q 1.55-1.73x bf16 [K]
BW=280.0         # GB/s
H=2560
gdn=36*(16384*H+6144*H); att=12*(12288*H+2*2*256*H+6144*H); shared=48*3*H*640; router=48*512*H; lm=248320*H
dense=gdn+att+shared+router+lm
print(f"dense MAC per row: GDN {gdn/1e9:.2f} G, attn {att/1e9:.2f}, shared {shared/1e9:.2f}, router {router/1e9:.3f}, lm_head {lm/1e9:.2f} -> {dense/1e9:.2f} G (config says 3.73 B incl. embeddings/norms)")
for name,mac in [("full row bf16",dense),("full row int8",dense),("row w/o lm_head (leaf) int8",dense-lm)]:
    rate=PEAK_BF16*(INT8_RATE if "int8" in name else 1)
    print(f"  {name:32s} {mac/1e9:5.2f} G MAC -> {mac/1e9/rate:.3f} ms at peak; 96 rows {96*mac/1e9/rate:5.1f} ms")
print("units: GPU matrix 14.0 bf16 / ~23 int8 TMAC/s [K]; ANE ~0.5 TMAC/s eff. [P from ANE-2b mlp 0.295 GB int8 x16 rows in 10 ms]; CPU ~0.1 [P from M-19 62 ms vs ANE 12.4]")
print(f"  125B dense row on ANE {dense/1e9/0.5:.1f} ms, on CPU {dense/1e9/0.1:.0f} ms")
print("experts per row: 10 x 3 x 2560 x 640 MAC x 48 =", 10*3*H*640*48/1e9, "G MAC; MMA tiles are 16 rows at ~5.2 pairs per segment [K]: ~67 % padding")
for K,union in [(32,109.7),(64,156.5),(96,187.4),(128,211.1)]:
    pairs=K*10/union
    print(f"  K {K:3d}: pairs per expert {pairs:4.1f} (<= 16: rides in the padded tile); union x{union/109.7:.2f} chain law, tree discount 15-25 % -> x{union/109.7*0.8:.2f}-{union/109.7*0.85:.2f}; expert ms at bound: {16.4*union/109.7*0.8:4.1f}-{16.4*union/109.7:4.1f}; union-constrained off-spine: 16.4")
print("cores: GDN state 226 MB read once per check = 0.81 ms [K]; sequential recurrence 4.3 ms at 32 rows [K] -> ~17 at 128 if per row (Leap: cores 35-46 ms at 128 rows [est]);")
intra=36*48*128*128*128  # tree-chunk term: 128x128 node pairs x dv 128 per head
print(f"  chunked tree form: intra-tree term {intra/1e9:.2f} G MAC = {intra/1e9/14:.2f} ms + state read 0.81 + one rank-128 update write 0.81 -> ~2 ms at any K <= 128")
print("== marginal row price (ms) and 96-row cost")
for name,ms in [("today (FAR: int8 dense 0.15 + union 0.08 + cores 0.04)",0.28),("int8 dense at peak + chain-law union + per-row cores",0.15+0.08+0.11),("int8 dense + union-constrained + chunked cores + no leaf lm_head",0.13),("brief's target",0.052)]:
    print(f"  {name:70s} {ms:.3f} -> 96 rows {96*ms:5.1f} ms")
print("== check at K rows at 100 % (C6+int8 25.2 at 32): dense 4.93*K/32 under bytes 7.6 until K ~49; above, compute")
import math
for K,kept in [(32,16.26),(48,18.86),(64,20.77),(96,22.5),(128,23.70)]:
    dense=max(7.6,4.93*K/32); dense_leaf=max(7.6,4.93*K/32*(1-0.4*0.17))
    exp_c=16.4; exp_free=16.4*min(1,0.0)+16.4  # union-constrained off-spine
    cores=2.0 if K>32 else 1.2
    chk=dense_leaf+exp_free+cores
    for draft,ov in [(5.2,1.5),(8.8,1.5)]:
        for eff in [1.0,0.85]:
            cyc=chk/eff+draft+ov
            print(f"  K {K:3d} kept {kept:5.2f}: check {chk:5.1f} (dense {dense_leaf:4.1f}) eff {eff:.2f} draft {draft}: cycle {cyc:5.1f} -> {1000*kept/cyc:4.0f} tok/s")
