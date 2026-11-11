"""b34 = S10c step 2 (NewIdeas Tue ~20:00, amended after R-22; rapid-screen size), scored from four runs' traces A1 B12 B16 A2.
A = the shipped 24 GB v3 decode; B12 / B16 = the same with TrainLab's frozen M1a lead-2 lists (top-12 / top-16 per row)
at the 8 traced pairs (L = 2, 8, .., 44 -> T = L + 2) replacing T's lead-1 router lists (bigmoe/m1a.py). All runs
BIGMOE_PRED_LINES=3 + PRED_CAP=61 + BIGMOE_TRACE=1. Prompts: prompt 4 of each domain (held out for M1a), 400 tokens.
PASS (each B arm vs A1 + A2 pooled): late + demand misses at the 8 target layers (MSB pool + LSB sidecar) per token
-15 % or more AND SSD bytes per token (rusage) up by no more than 2 %; OR SSD bytes per token at the 8 target layers
(reads in the ext trace x part size) -10 % or more at tok/s no worse than -1 %. KILL: misses < 5 % fewer, or rusage
bytes up > 5 %. Else BETWEEN. INVALID if any prompt's output sha differs between runs (the decode must stay exact).
usage: s10c_ab.py DIR  (DIR/A1, B12, B16, A2 each hold miss.bin + meta.json)"""
import json, os, sys
import numpy as np

D = sys.argv[1]
RUNS = ("A1", "B12", "B16", "A2")
T8 = (4, 10, 16, 22, 28, 34, 40, 46)
MSB, LSB = 2162688, 770048                                   # records3 parts (MoonShots' wall24.py)


def load(r):
    meta = json.load(open(os.path.join(D, r, "meta.json")))
    raw = np.fromfile(os.path.join(D, r, "miss.bin"), np.int32)
    k = np.zeros((256, 5), np.int64)       # per gate layer (trunk 0-47, MTP 48, LSB sidecars side0 + l): misses by kind
    i = 0                                  # 0 demand, 1 on time, 2 late, 3 queued; uses
    while i < len(raw):
        L, n = int(raw[i + 1]), int(raw[i + 2])
        v = raw[i + 3:i + 3 + n]
        m, kd = (v >> 16) & 1, (v >> 17) & 3
        k[L, :4] += np.bincount(kd[m == 1], minlength=4)[:4]
        k[L, 4] += n
        i += 3 + n
    P, s0 = meta["prompts"], meta["side0"]
    rd = np.zeros((256, 3), np.int64)                        # reads per gate layer by tag (demand, prefetch, joint)
    for p in P:
        for L, t, c in p["trace"].get("reads", []):
            rd[L, t] += c
    tgt = list(T8) + [s0 + T for T in T8]
    tok = sum(p["n_tok"] for p in P)
    ld = lambda a: a[..., 0] + a[..., 2] + a[..., 3]         # late + demand (queued counts as demand)
    return dict(tok=tok, sha={p["name"]: p["sha"] for p in P}, tps={p["name"]: p["tok_s"] for p in P},
                ld8=ld(k[tgt]).sum(), on8=k[tgt, 1].sum(), ld_rest=ld(k).sum() - ld(k[tgt]).sum(),
                disk=sum(p["disk_read"] for p in P), b8=rd[list(T8)].sum() * MSB + rd[[s0 + T for T in T8]].sum() * LSB,
                rd8=rd[tgt].sum(0), gw=sum(p["trace"].get("gate_wait_s", 0) for p in P),
                busy=sum(p["trace"].get("ssd_busy_s", 0) for p in P),
                iss=sum(p["ext"]["pf_issued"] for p in P), used=sum(p["ext"]["pf_used"] for p in P))


R = {r: load(r) for r in RUNS}
names = list(R["A1"]["sha"])
exact = all(R[r]["sha"] == R["A1"]["sha"] for r in RUNS)


def m(rs):
    """per-token metrics pooled over runs rs"""
    tok = sum(R[r]["tok"] for r in rs)
    s = lambda key: sum(R[r][key] for r in rs)
    return dict(ld8=s("ld8") / tok, on8=s("on8") / tok, ld_rest=s("ld_rest") / tok, mb=s("disk") / tok / 1e6,
                mb8=s("b8") / tok / 1e6, gw=1e3 * s("gw") / tok, busy=1e3 * s("busy") / tok, iss=s("iss") / tok,
                used=s("used") / tok, tps=float(np.mean([np.mean(list(R[r]["tps"].values())) for r in rs])),
                rd8=sum(R[r]["rd8"] for r in rs) / tok)


print("\n".join(__doc__.split("\n")[4:8]))
print(f"exact: {'yes, every prompt sha equal across the 4 runs' if exact else 'NO'}  ({', '.join(names)})")
print("run    late+dem@T8  on-time@T8  late+dem other  disk MB/tok  T8 MB/tok (dem/pf/joint reads)  gate wait ms/tok"
      "  SSD busy ms/tok  pf iss/used per tok  tok/s")
for r in RUNS + ("A",):
    q = m(("A1", "A2") if r == "A" else (r,))
    print(f"{r:5s} {q['ld8']:11.2f}  {q['on8']:10.2f}  {q['ld_rest']:14.2f}  {q['mb']:11.1f}  {q['mb8']:9.2f} "
          f"({q['rd8'][0]:.2f}/{q['rd8'][1]:.2f}/{q['rd8'][2]:.2f})  {q['gw']:16.2f}  {q['busy']:15.2f}  "
          f"{q['iss']:8.1f} / {q['used']:5.1f}  {q['tps']:6.2f}")
A, a1, a2 = m(("A1", "A2")), m(("A1",)), m(("A2",))
noise = {k: 100 * abs(a2[k] / a1[k] - 1) for k in ("ld8", "mb", "mb8", "gw", "tps")}
print("A1 vs A2 (run-to-run noise): " + ", ".join(f"{k} {v:.1f} %" for k, v in noise.items()))
out = {"exact": exact, "noise": noise, "A": A}
for b in ("B12", "B16"):
    q = m((b,))
    d = {k: 100 * (q[k] / A[k] - 1) for k in ("ld8", "ld_rest", "mb", "mb8", "gw", "tps")}
    v = ("INVALID (not exact)" if not exact else
         "PASS" if (d["ld8"] <= -15 and d["mb"] <= 2) or (d["mb8"] <= -10 and d["tps"] >= -1) else
         "KILL" if d["ld8"] > -5 or d["mb"] > 5 else "BETWEEN")
    print(f"{b} vs A: late + demand at T8 {d['ld8']:+.1f} %, other layers {d['ld_rest']:+.1f} %; disk bytes/tok "
          f"{d['mb']:+.1f} %; T8 bytes/tok {d['mb8']:+.1f} %; gate wait/tok {d['gw']:+.1f} %; tok/s {d['tps']:+.1f} %  "
          f"-> {v}")
    for n in names:
        a = np.mean([R[r]["tps"][n] for r in ("A1", "A2")])
        print(f"    {n}: tok/s A {a:.2f}  {b} {R[b]['tps'][n]:.2f}  ({100 * (R[b]['tps'][n] / a - 1):+.1f} %)")
    out[b] = dict(q=q, d=d, verdict=v)
json.dump(out, open(os.path.join(D, "ab.json"), "w"), indent=1, default=lambda o: o.tolist())
