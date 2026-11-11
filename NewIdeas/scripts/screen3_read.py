"""SCREEN 3 read-out: the engine vs the replay at 24 GB (NewIdeas, Wed 4 Nov 2026 08:3x; pre-registered QUEUE 04:0x).
BigMoE's W1B 24 GB traced arm (data/w1b24/w24t, Wed 07:54-07:56: tier 4433333333, C_m 147 / C_l 46 per layer = the
replay's 24 GB line; JOINT, LFU 0.99 eviction, prefetch 12; 6 prompts x 256 tokens: HumanEval/68, /91, 4 math).
miss.bin (s10b_trace.py docstring): int32 records [seq, L, n, (e | miss << 16) x n], one per (pass, ext layer) served;
miss = not resident at service (prefetched-not-promoted counts as a miss). L < side0 = base pools (48 = the MTP head's
experts), L >= side0 = the LSB sidecar of layer L - side0. A prompt's decode = the seqs in its pNN npz (verify passes
L 0-47 and MTP draft calls L 48); other seqs = prefill / warm-up. Bytes with the ledger's record sizes (MR 2,162,688,
LR 770,048) so the comparison with the replay is in the same units. Usage (BigMoE venv): python screen3_read.py [dir]
Writes ~/NewIdeas/data/ledger/screen3_read.txt."""
import json, os, sys
import numpy as np
D = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/BigMoE/data/w1b24/w24t")
MR, LR = 2162688, 770048
REF = {"mine": 212.8, "fablec": 245.2, "kept": 7.91}          # replay, code, EXACT, 24 GB, live checks (QUEUE 04:0x)
m = json.load(open(os.path.join(D, "meta.json"))); side0 = m["side0"]
raw = np.fromfile(os.path.join(D, "miss.bin"), np.int32)
seq_of = {}
for p in m["prompts"]:
    z = np.load(os.path.join(D, p["name"] + ".npz"))
    for s, L in zip(z["seq"].tolist(), z["L"].tolist()): seq_of.setdefault(s, set()).add((p["name"], L == 48))
acc = {}                                                        # name -> [base miss trunk, base miss mtp, side miss, uniq]
other = [0, 0, 0]; i = 0; nrec = 0
while i < len(raw):
    s, L, n = int(raw[i]), int(raw[i + 1]), int(raw[i + 2]); v = raw[i + 3:i + 3 + n]; i += 3 + n; nrec += 1
    ms = int(((v >> 16) & 1).sum())
    own = seq_of.get(s)
    if not own: other[0 if L < 48 else 1 if L == 48 else 2] += ms; continue
    name = next(iter(own))[0]; a = acc.setdefault(name, [0, 0, 0, 0])
    a[0 if L < 48 else 1 if L == 48 else 2] += ms; a[3] += n
assert i == len(raw), "miss.bin did not parse to its end"
out = [f"SCREEN 3: W1B 24 GB traced arm ({D}); {nrec} miss.bin records; MB with MR {MR} / LR {LR} B",
       "| prompt | kept per pass | decode MB per kept: trunk base + MTP base + LSB = total | process disk_read MB per token | "
       "engine reads (trace) MB per token | ref x 7.91 / kept (mine / Fable C) | ratio to ref, unscaled (mine) | ratio, scaled |",
       "|---|---|---|---|---|---|---|---|"]
tot = {}
for p in m["prompts"]:
    nm, k = p["name"], p["n_tok"] / p["passes"]; a = acc.get(nm, [0, 0, 0, 0])
    mb = [a[0] * MR / 1e6 / p["n_tok"], a[1] * MR / 1e6 / p["n_tok"], a[2] * LR / 1e6 / p["n_tok"]]; dec = sum(mb)
    rd = p["trace"]["reads"]
    rb = sum(c for L, kind, c in rd if L < side0); rs = sum(c for L, kind, c in rd if L >= side0)
    rmb = (rb * MR + rs * LR) / 1e6 / p["n_tok"]
    sc = REF["kept"] / k
    out.append(f"| {nm} | {k:.2f} | {mb[0]:.1f} + {mb[1]:.1f} + {mb[2]:.1f} = {dec:.1f} | {p['disk_read'] / 1e6 / p['n_tok']:.1f} | "
               f"{rmb:.1f} | {REF['mine'] * sc:.1f} / {REF['fablec'] * sc:.1f} | {dec / REF['mine']:.2f} | {dec / (REF['mine'] * sc):.2f} |")
    d = p["domain"]; t = tot.setdefault(d, [0.0, 0, 0, 0.0]); t[0] += dec * p["n_tok"]; t[1] += p["n_tok"]; t[2] += p["passes"]
    t[3] += p["disk_read"] / 1e6
for d, (b, n, ps, dr) in tot.items():
    k = n / ps; sc = REF["kept"] / k
    out.append(f"{d}: decode {b / n:.1f} MB per kept token at {k:.2f} kept per pass; process {dr / n:.1f}; ratio to the replay "
               f"(code ref, mine) unscaled {b / n / REF['mine']:.2f}, scaled by 7.91 / kept {b / n / (REF['mine'] * sc):.2f}")
out.append(f"misses outside the prompts' decode seqs (prefill, warm-up): trunk {other[0]}, MTP {other[1]}, LSB {other[2]}")
os.makedirs(os.path.expanduser("~/NewIdeas/data/ledger"), exist_ok=True)
open(os.path.expanduser("~/NewIdeas/data/ledger/screen3_read.txt"), "w").write("\n".join(out) + "\n"); print("\n".join(out))
