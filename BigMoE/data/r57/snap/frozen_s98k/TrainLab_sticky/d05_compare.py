"""D-05 like-with-like read-out (MTP Head, Wed 10-28 ~19:3x; NewIdeas' 18:49 asks): the native MTP chain (k = 3, data/
d05_mtp_depth.json, shipped 24 GB v3 stack) and DFlash2 L1-20 (block 16, official 1-step loop, data/d02b_walk_{code,math}
.npz) on the SAME prompts (the 8 code + 8 gsm D-05 prompts are all in Leap's dfl2d sets) and the SAME region: completion
positions 1..L, L = the first end token (<|im_end|> / <|endoftext|>) of the 125B's answer, capped at the MTP decode's
last committed position (300 tokens). Position 0 (the prefill's token) is given on both sides. Per-depth acceptance is
conditional (depth j counted when depths < j were accepted and position j lies inside 1..L), with counts.
MTP pass p drafts positions s_p.., s_1 = 0 (the first pass verifies the prompt's last token, so the head drafts the
first answer token too; that one trial is dropped, DFlash2 gets it from the prefill), s_{p+1} = s_p + n_p + 1; DFlash2 check at completion index rho = a - gen
drafts rho+1..rho+m, keeps rho+1..rho+k. The MTP decode's known committed tokens (g[0..n], the bonus row is not saved
when n = k) are compared with dfl2d's ids to confirm both sides walk the same text.
Usage: taskpolicy -b nice -n 10 [BigMoE venv] python sticky/d05_compare.py -> data/d05_compare.txt"""
import json, os
import numpy as np
from tokenizers import Tokenizer

H = os.path.expanduser
tk = Tokenizer.from_file(H("~/BigMoE/models/bare/tokenizer.json"))
END = {tk.token_to_id(t) for t in ("<|im_end|>", "<|endoftext|>")} - {None}
SRC = os.environ.get("TL_MTP_JSON", "d05_mtp_depth.json")       # 10-28 19:4x: also the exact-mode re-run
D = json.load(open(H("~/TrainLab/data/" + SRC)))
K = D["k"]
walk = {}
for dom in ("code", "math"):
    W = np.load(H(f"~/TrainLab/data/d02b_walk_{dom}.npz"))
    for s, nm in enumerate(W["names"]):
        walk[str(nm)] = (W[f"{s}_ids"].tolist(), int(W[f"{s}_off"]), int(W[f"{s}_gen"]), W[f"{s}_checks"].tolist())

lines = []
def say(s=""):
    print(s, flush=True); lines.append(s)

def mtp_positions(passes):
    """per pass: start position s_p and the known committed tokens {position: token}"""
    s, out, known = 0, [], {}
    for n, m, g in passes:
        out.append(s)
        for j in range(min(n + 1, len(g))):
            known[s + j] = g[j]
        s += n + 1
    return out, known, s - 1

def rates(acc):
    return [h / t if t else float("nan") for h, t in acc]

def per_check(r):
    """tokens per check at width 3 from conditional rates: 1 + a1 + a1 a2 + a1 a2 a3"""
    e, p = 1.0, 1.0
    for a in r:
        p *= a; e += p
    return e

say("D-05 like-with-like: native MTP chain (k = 3) vs DFlash2 L1-20 (block 16, official loop), same prompts, same region")
say("(completion positions 1..L, L = the answer's first end token, capped at the MTP decode's 300 tokens; conditional")
say("per-depth acceptance, hits / trials):")
say(f"MTP side: data/{SRC} (BIGMOE_TIER={D['env'].get('BIGMOE_TIER')}, MTP_STREAM={D['env'].get('BIGMOE_MTP_STREAM')})")
dom_m, dom_d, dom_k, dom_info = {}, {}, {}, {}
for P in D["prompts"]:
    dom = P["domain"]
    starts, known, last = mtp_positions(P["passes"])
    if "out" in P:                                        # the re-run saves the decode itself
        known, last = dict(enumerate(P["out"])), len(P["out"]) - 1
    E_m = min([p for p, t in known.items() if t in END], default=None)
    L = E_m if E_m is not None else last
    nm = (P["key"] or "").replace("/", "_") + ".npz"
    info = dom_info.setdefault(dom, dict(prompts=0, pos=0, cmp=0, diff=0, first_diff=[], dfl=0))
    info["prompts"] += 1
    if nm in walk:
        ids, off, gen, checks = walk[nm]
        comp = ids[gen - off:]
        E_d = next((i for i, t in enumerate(comp) if t in END), len(comp) - 1)
        L = min(L, E_d)
        same = [(p, known[p] == comp[p]) for p in sorted(known) if p <= L and p < len(comp)]
        info["cmp"] += len(same)
        bad = [p for p, ok in same if not ok]
        info["diff"] += len(bad)
        if bad:
            info["first_diff"].append((P["key"], bad[0]))
        info["dfl"] += 1
        acc = dom_d.setdefault(dom, [[0, 0] for _ in range(16)])
        kk = dom_k.setdefault(dom, [0, 0, 0])                # kept tokens, checks, kept capped at 3
        for k, m, a in checks:
            rho = a - gen
            if rho >= L:
                continue
            mm = min(m, L - rho)
            k2 = min(k, mm)
            kk[0] += k2; kk[1] += 1; kk[2] += min(k2, 3)
            for j in range(1, mm + 1):
                if k2 < j - 1:
                    break
                acc[j - 1][0] += int(k2 >= j)
                acc[j - 1][1] += 1
    info["pos"] += L
    acc = dom_m.setdefault(dom, [[0, 0] for _ in range(K)])
    for (n, m, g), s in zip(P["passes"], starts):
        if s > L:
            break
        for j in range(K):
            if s + j > L or not (j == 0 or n >= j):
                break
            if s + j < 1:
                continue
            acc[j][0] += int(m[j] == g[j])
            acc[j][1] += 1

for dom in dom_m:
    i = dom_info[dom]
    rm = rates(dom_m[dom])
    say(f"  {dom:6s} ({i['prompts']} prompts, {i['pos']} positions)")
    say(f"    MTP chain  depth 1 / 2 / 3 = " + " / ".join(f"{x:.3f}" for x in rm) + "   ("
        + ", ".join(f"{h}/{t}" for h, t in dom_m[dom]) + f");  tokens per check at width 3: {per_check(rm):.2f}")
    if dom in dom_d:
        rd = rates(dom_d[dom])
        kk = dom_k[dom]
        say(f"    DFlash2    depth 1 / 2 / 3 = " + " / ".join(f"{x:.3f}" for x in rd[:3]) + "   ("
            + ", ".join(f"{h}/{t}" for h, t in dom_d[dom][:3]) + f");  tokens per check at width 3: "
            f"{per_check(rd[:3]):.2f}")
        say(f"    DFlash2    depths 4-15: " + " ".join(f"{x:.2f}" for x in rd[3:15]) + f";  kept per block-16 check "
            f"(accepted + 1, D-02b's measure) {kk[0] / max(kk[1], 1) + 1:.2f} ({kk[1]} checks)")
        say(f"    same text: {i['cmp']} known MTP-decode positions compared with dfl2d's ids, {i['diff']} differ"
            + (f" (first: {i['first_diff'][:3]})" if i["first_diff"] else ""))
    else:
        say("    DFlash2    no features for these prompts yet (D-02a)")
say("drafting cost (spec125 timers, decode only; draft = t_draft + t_head per pass):")
for f in (SRC, "d05_mtp_cost_k3.json", "d05_mtp_cost_k1.json"):
    fp = H("~/TrainLab/data/" + f)
    if not os.path.exists(fp):
        continue
    R = json.load(open(fp))
    T = [P["timers"] for P in R["prompts"] if "timers" in P]
    if not T:
        continue
    tot = lambda k: sum(t.get(k, 0.0) for t in T)
    ps = max(tot("passes"), 1)
    say(f"  {f} (k = {R['k']}, MTP_STREAM={R['env'].get('BIGMOE_MTP_STREAM')}, {len(T)} prompts): draft "
        f"{1e3 * (tot('t_draft') + tot('t_head')) / ps:.2f} ms per pass (t_draft {1e3 * tot('t_draft') / ps:.2f}, "
        f"t_head {1e3 * tot('t_head') / ps:.2f}), verify {1e3 * tot('t_verify') / ps:.1f} ms, "
        f"{sum(len(P['out']) for P in R['prompts']) / ps:.2f} tokens per pass, "
        f"{1e3 * tot('time') / max(sum(len(P['out']) for P in R['prompts']), 1):.1f} ms per token")
open(H("~/TrainLab/data/" + os.environ.get("TL_OUT", "d05_compare.txt")), "w").write("\n".join(lines) + "\n")
