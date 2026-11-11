"""b36 = R-27 LAYSZ config A/B (MoonShots' per-layer tier pools at the shipped pool RAM; NewIdeas' queue Tue ~20:45),
scored from four runs' traces A1 S1 S2 A2 (s10b_trace.py S10B_IDX=4: prompt 4 of each domain, 400 tokens, b33 settings).
A = the shipped uniform C_m 147 / C_l 46; S = BIGMOE_TIER_SIZES (MoonShots' per-layer C_m / C_l, same pool bytes).
PASS: whole-process SSD bytes per token (rusage) -5 % or more at tok/s no worse than -1 % (S1 + S2 vs A1 + A2).
KILL: bytes > -2 %. Else BETWEEN. INVALID if any prompt's output sha differs between runs, or the arms' pool GB differ by
more than 0.5 %. usage: r27_ab.py DIR"""
import json, os, re, sys
import numpy as np

D = sys.argv[1]
RUNS = ("A1", "S1", "S2", "A2")


def load(r):
    meta = json.load(open(os.path.join(D, r, "meta.json")))
    P = meta["prompts"]
    raw = np.fromfile(os.path.join(D, r, "miss.bin"), np.int32)
    k, i = np.zeros(4, np.int64), 0                          # misses by kind over all gate layers
    while i < len(raw):
        n = int(raw[i + 2]); v = raw[i + 3:i + 3 + n]
        k += np.bincount(((v >> 17) & 3)[((v >> 16) & 1) == 1], minlength=4)[:4]
        i += 3 + n
    txt = open(os.path.join(D, r + ".txt")).read()
    gb = re.findall(r"C_m=\S+ C_l=\S+ per layer, ([\d.]+) GB pools", txt)
    return dict(tok=sum(p["n_tok"] for p in P), sha={p["name"]: p["sha"] for p in P}, tps={p["name"]: p["tok_s"] for p in P},
                disk=sum(p["disk_read"] for p in P), gw=sum(p["trace"].get("gate_wait_s", 0) for p in P),
                busy=sum(p["trace"].get("ssd_busy_s", 0) for p in P), ld=k[0] + k[2] + k[3], on=k[1],
                pools=float(gb[-1]) if gb else float("nan"), peak=max((p.get("peak_gb", float("nan")) for p in P), default=float("nan")))


R = {r: load(r) for r in RUNS}
names = list(R["A1"]["sha"])
exact = all(R[r]["sha"] == R["A1"]["sha"] for r in RUNS)


def m(rs):
    tok = sum(R[r]["tok"] for r in rs)
    s = lambda key: sum(R[r][key] for r in rs)
    return dict(mb=s("disk") / tok / 1e6, gw=1e3 * s("gw") / tok, busy=1e3 * s("busy") / tok, ld=s("ld") / tok,
                on=s("on") / tok, tps=float(np.mean([np.mean(list(R[r]["tps"].values())) for r in rs])))


print("\n".join(__doc__.split("\n")[3:6]))
print(f"exact: {'yes, every prompt sha equal across the 4 runs' if exact else 'NO'}  ({', '.join(names)})")
print("run   disk MB/tok  gate wait ms/tok  SSD busy ms/tok  late+demand/tok  on-time/tok  tok/s   pools GB  MLX peak GB")
for r in RUNS:
    q = m((r,))
    print(f"{r:4s}  {q['mb']:11.1f}  {q['gw']:16.2f}  {q['busy']:15.2f}  {q['ld']:15.2f}  {q['on']:11.2f}  {q['tps']:6.2f}  "
          f"{R[r]['pools']:8.2f}  {R[r]['peak']:11.2f}")
A, S, a1, a2 = m(("A1", "A2")), m(("S1", "S2")), m(("A1",)), m(("A2",))
noise = {k: 100 * abs(a2[k] / a1[k] - 1) for k in ("mb", "gw", "tps")}
d = {k: 100 * (S[k] / A[k] - 1) for k in ("mb", "gw", "busy", "ld", "on", "tps")}
pools_ok = abs(np.mean([R[r]["pools"] for r in ("S1", "S2")]) / np.mean([R[r]["pools"] for r in ("A1", "A2")]) - 1) <= 0.005
print("A1 vs A2 (run-to-run noise): " + ", ".join(f"{k} {v:.1f} %" for k, v in noise.items()))
if os.environ.get("AB_LINES") == "r38":                  # b41b (R-38 PF24, NewIdeas): bigger pools by design
    # the peak: S's LIFETIME peak phys_footprint on the worst prompt (code prompt 0, spec mode, ram24_probe), passed in
    # as PEAK_GB (NewIdeas 10-28); without it, b41b's per-prompt MLX peak of these runs
    pk = float(os.environ["PEAK_GB"]) if os.environ.get("PEAK_GB") else max(R[r]["peak"] for r in ("S1", "S2"))
    v = ("INVALID (not exact)" if not exact else
         "PASS" if pk <= 23.9 and d["mb"] <= -5 else "KILL" if pk > 24.0 or d["mb"] > -2 else "BETWEEN")
    print(f"lines (R-38): PASS S peak <= 23.9 GB (all shas equal) AND bytes/tok <= -5 %; KILL S peak > 24 GB or bytes > -2 %"
          f"  [S peak {pk:.2f} GB ({'lifetime footprint, ram24_probe' if os.environ.get('PEAK_GB') else 'MLX, these runs'})]")
elif os.environ.get("AB_LINES") == "m01":                # b44 (R-44 a, M-01 exact re-pack; S = BIGMOE_M01 on records3m)
    v = ("KILL (not exact)" if not exact else "INVALID (pool RAM differs)" if not pools_ok else
         "PASS" if d["mb"] <= -2.5 else "BETWEEN")
    print("lines (R-44 a, M-01; NewIdeas 10-28): PASS every sha equal AND rusage SSD bytes/tok <= -2.5 %; KILL any mismatch")
elif os.environ.get("AB_LINES") == "tps":                # b39 (cheap M1a): the lines are on tok/s
    v = ("INVALID (not exact)" if not exact else "INVALID (pool RAM differs)" if not pools_ok else
         "PASS" if d["tps"] >= 3 and d["mb"] <= -5 else "KILL" if d["tps"] < 1 else "BETWEEN")
    print("lines (b39): PASS tok/s >= +3 % AND bytes/tok <= -5 %; KILL tok/s < +1 %")
else:
    v = ("INVALID (not exact)" if not exact else "INVALID (pool RAM differs)" if not pools_ok else
         "PASS" if d["mb"] <= -5 and d["tps"] >= -1 else "KILL" if d["mb"] > -2 else "BETWEEN")
print(f"S vs A: SSD bytes/tok {d['mb']:+.1f} %; gate wait/tok {d['gw']:+.1f} %; SSD busy/tok {d['busy']:+.1f} %; "
      f"late + demand/tok {d['ld']:+.1f} %; on time/tok {d['on']:+.1f} %; tok/s {d['tps']:+.1f} %  -> {v}")
for n in names:
    a, s_ = np.mean([R[r]["tps"][n] for r in ("A1", "A2")]), np.mean([R[r]["tps"][n] for r in ("S1", "S2")])
    print(f"    {n}: tok/s A {a:.2f}  S {s_:.2f}  ({100 * (s_ / a - 1):+.1f} %)")
json.dump(dict(A=A, S=S, d=d, noise=noise, exact=bool(exact), pools_ok=bool(pools_ok), verdict=v),
          open(os.path.join(D, "ab.json"), "w"), indent=1)
