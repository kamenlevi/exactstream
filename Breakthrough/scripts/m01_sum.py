"""R-48 M01-12 summary of the A1 S1 S2 A2 runs (scripts/m01_ab.py rows in DIR/<arm>.jsonl) against the lines written before
the run (SCOREBOARD 4b90156):
  PASS: every hash equal in all 4 runs AND decode SSD bytes per token (rusage; the 3 prompts' decode bytes / decoded
        tokens, S1 + S2 vs A1 + A2) -2.5 % or better AND lifetime peak not higher (max S <= max A + 0.02 GB)
  KILL: any hash differs, or a decode bytes/token saving under 1.5 %. BETWEEN: report; tok/s decides.
  INVALID (re-run): A1 vs A2 decode bytes/token differ by more than 1 %, or the arms' C_m / C_l differ.
Usage: [BigMoE venv] m01_sum.py DIR"""
import json, os, sys

D = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/Breakthrough/data/m01")
ARMS = ("A1", "S1", "S2", "A2")
R = {a: [json.loads(l) for l in open(os.path.join(D, f"{a}.jsonl"))] for a in ARMS}
prompts = [r["prompt"] for r in R["A1"]]
assert all([r["prompt"] for r in R[a]] == prompts for a in ARMS), {a: [r["prompt"] for r in R[a]] for a in ARMS}
assert all(r["m01"] == a.startswith("S") for a in ARMS for r in R[a]), "an arm ran on the wrong engine"
HK = ("logits", "lpre", "sha32", "shaN", "ntok", "kv", "head")
exact, pools = True, True
lines = []
for i, p in enumerate(prompts):
    rows = [R[a][i] for a in ARMS]
    same = all(all(r[k] == rows[0][k] for k in HK) for r in rows)
    pq = all((r["cm"], r["cm_pre"], r["cl"]) == (rows[0]["cm"], rows[0]["cm_pre"], rows[0]["cl"]) for r in rows)
    exact &= same; pools &= pq
    w = rows[0]
    lines.append(f"{p}: hashes {'equal' if same else 'DIFFER'} in 4 runs (logits {w['logits']} sha32 {w['sha32']} sha{w['ntok']} "
                 f"{w['shaN']} kv {w['kv']} head {w['head']}); C_m {w['cm']} C_l {w['cl']} {'equal' if pq else 'DIFFER'}"
                 + (f"; write16 vs b12llfoff first diff {w['w16_first_diff_vs_b12llfoff']}" if p == "write16" else ""))
    if not same:
        for a, r in zip(ARMS, rows):
            lines.append(f"    {a}: " + " ".join(f"{k} {r[k]}" for k in HK))


def tot(a, k):
    return sum(r[k] for r in R[a])


dec = {a: tot(a, "disk_decode") / tot(a, "ntok") for a in ARMS}
pre = {a: tot(a, "disk_prefill") / sum(r["n"] for r in R[a]) for a in ARMS}
tps = {a: tot(a, "ntok") / tot(a, "decode_s") for a in ARMS}
pk = {a: max(r["lifetime_peak"] for r in R[a]) for a in ARMS}
A = (dec["A1"] + dec["A2"]) / 2; S = (dec["S1"] + dec["S2"]) / 2
dB = 100 * (S - A) / A
noise = 100 * abs(dec["A1"] - dec["A2"]) / A
dP = 100 * ((pre["S1"] + pre["S2"]) - (pre["A1"] + pre["A2"])) / (pre["A1"] + pre["A2"])
dT = 100 * ((tps["S1"] + tps["S2"]) - (tps["A1"] + tps["A2"])) / (tps["A1"] + tps["A2"])
peak_ok = max(pk["S1"], pk["S2"]) <= max(pk["A1"], pk["A2"]) + 0.02
print(f"{'run':4} {'decode MB/tok':>14} {'prefill KB/prompt tok':>22} {'decode tok/s':>13} {'lifetime peak GB':>17}")
for a in ARMS:
    print(f"{a:4} {dec[a] / 1e6:14.3f} {pre[a] / 1e3:22.1f} {tps[a]:13.2f} {pk[a]:17.3f}")
for p_i, p in enumerate(prompts):
    print(f"  {p}: decode MB/tok " + "  ".join(f"{a} {R[a][p_i]['dec_bpt'] / 1e6:.2f}" for a in ARMS) + " | tok/s "
          + "  ".join(f"{a} {R[a][p_i]['tps']:.2f}" for a in ARMS) + " | warm " + "  ".join(f"{a} {R[a][p_i]['warm']:.2f}" for a in ARMS))
print("\n".join(lines))
print(f"A1 vs A2 decode bytes/token noise {noise:.2f} %; S vs A: decode bytes/token {dB:+.2f} %, prefill bytes/token {dP:+.2f} %, "
      f"decode tok/s {dT:+.2f} %, lifetime peak max S {max(pk['S1'], pk['S2']):.3f} vs max A {max(pk['A1'], pk['A2']):.3f} GB")
if not pools:                                       # a set-up error: the arms must run equal pools
    v = "INVALID (re-run): the arms' C_m / C_l differ"
elif not exact:
    v = "KILL: a hash differs"
elif noise > 1.0:
    v = f"INVALID (re-run): A1-A2 decode bytes/token noise {noise:.2f} % > 1 %"
elif dB > -1.5:
    v = f"KILL: decode bytes/token saving {-dB:.2f} % < 1.5 %"
elif dB <= -2.5 and peak_ok:
    v = f"PASS: every hash equal, decode bytes/token {dB:+.2f} %, peak not higher"
else:
    v = f"BETWEEN: decode bytes/token {dB:+.2f} %, peak {'ok' if peak_ok else 'HIGHER'}; tok/s {dT:+.2f} % decides"
print("VERDICT " + v)
