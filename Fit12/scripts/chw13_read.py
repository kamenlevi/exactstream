"""Fit12 T2 CHAT WIDTH stage 1b (Sun 8 Nov 10:0x; pre-registration: /Users/kamen/Fit12/PLAN_12GB_90_fit12.md 'T2 CHAT WIDTH
stage 1b'): the reader of /Users/kamen/Fit12/scripts/chw13.sh. Arm ex + PR1 + the tail rule on chat12's 12 IFEval prompts,
CAP 128, one process per arm: A' = W1 4 (the reference), P' = cf:0.9:2:1.01:3 (the policy), Q' = rand:2,3,4 (the probe).
Per arm: tokens + kept-row logits hash (lsha) = A' per prompt; whole decode (n / decode wall, geometric mean + pooled); SSD
MB per kept token (the decode phases' own reads); kept per check; first-call widths; PR1's keep-1 replays; the lifetime
peak. A' vs chat12 (no PR1): tokens, information.
Usage: python -I -B chw13_read.py [DIR] [TAG_REF] [TAG ...] [REF_IDS.json]  (stage 2 reuses it: any number of tags)"""
import glob, json, math, os, re, sys

D = sys.argv[1] if len(sys.argv) > 1 else "/Users/kamen/Fit12/data/scr/chw13"
TAGS = [a for a in sys.argv[2:] if not a.endswith(".json")] or ["chw13a", "chw13p", "chw13q"]   # the first = the reference
REF = next((a for a in sys.argv[2:] if a.endswith(".json")), "/Users/kamen/Fit12/data/scr/chat12/chat12_ids.json")


def read(tag):
    p_ = os.path.join(D, f"{tag}.txt")
    out, pr1, life = {}, None, None
    if not os.path.exists(p_):
        return out, pr1, life
    for ln in open(p_):
        m = re.match(rf"^\[{re.escape(tag)}\] (\S+) ex theta None .*?\| tok (\d+) sha (\w+) \| lsha (\w+) .*?\| checks (\d+), kept ([\d.]+) .*?"
                     rf"decode ([\d.]+) s", ln)
        if m:
            out[m.group(1)] = dict(tok=int(m.group(2)), lsha=m.group(4), checks=int(m.group(5)), kept=float(m.group(6)),
                                   dec=float(m.group(7)))
        m = re.search(r"PR1: (\d+) keep-1 commits", ln)
        if m:
            pr1 = int(m.group(1))
        m = re.match(rf"^\[{re.escape(tag)}\] done; lifetime ([\d.]+) GB", ln)
        if m:
            life = float(m.group(1))
    for k, r in out.items():
        kf = k.replace("/", "_")
        base = os.path.join(D, f"{tag}_{kf}_ex")
        r["toks"] = json.load(open(base + "_toks.json")) if os.path.exists(base + "_toks.json") else None
        ph = json.load(open(base + "_ph.json")) if os.path.exists(base + "_ph.json") else []
        r["mb"] = sum(p[8] - p[4] for p in ph) / 1e6
        cl = json.load(open(base + "_calls.json")) if os.path.exists(base + "_calls.json") else []
        r["w1"] = [c[2] for c in cl if c[1] == 0]
    return out, pr1, life


R = {t: read(t) for t in TAGS}
ref = json.load(open(REF)) if os.path.exists(REF) else {}
A = R[TAGS[0]][0]
print(f"T2 CHAT WIDTH stage 1b ({D}): reference A' = {TAGS[0]}; arms {', '.join(TAGS[1:])}")
res = {}
for t in TAGS:
    o, pr1, life = R[t]
    if not o:
        print(f"  {t}: no rows"); continue
    keys = sorted(o)
    tps = [o[k]["tok"] / o[k]["dec"] for k in keys if o[k]["dec"] > 0]
    geo = math.exp(sum(math.log(v) for v in tps) / len(tps))
    pooled = sum(o[k]["tok"] for k in keys) / sum(o[k]["dec"] for k in keys)
    toks_n = sum(o[k]["tok"] for k in keys)
    mbk = sum(o[k]["mb"] for k in keys) / max(1, toks_n)
    kept = sum(o[k]["kept"] * o[k]["checks"] for k in keys) / max(1, sum(o[k]["checks"] for k in keys))
    w1 = [w for k in keys for w in o[k]["w1"]]
    wd = {w: w1.count(w) for w in sorted(set(w1))}
    eq_t = sum(1 for k in keys if k in A and o[k]["toks"] == A[k]["toks"])
    eq_l = sum(1 for k in keys if k in A and o[k]["lsha"] == A[k]["lsha"])
    eq_ref = sum(1 for k in keys if (ref.get(k) or {}).get("ex", [[]])[0][:len(o[k]["toks"] or [])] == o[k]["toks"])
    print(f"  {t}: whole decode geo {geo:.2f} (pooled {pooled:.2f}) tok/s over {len(keys)} prompts; {mbk:.0f} MB per kept token; "
          f"kept {kept:.3f} per check; first-call widths {wd}; PR1 replays {pr1}; lifetime {life} GB")
    print(f"      = {TAGS[0]}: tokens {eq_t} / {len(keys)}, lsha {eq_l} / {len(keys)}; = chat12 (no PR1) tokens {eq_ref} / {len(keys)}")
    res[t] = dict(geo=geo, pooled=pooled, mb_per_tok=mbk, kept=kept, w1=wd, pr1=pr1, life=life, eq_tokens=eq_t, eq_lsha=eq_l,
                  eq_chat12=eq_ref, n=len(keys))
if TAGS[0] in res:
    for t in TAGS[1:]:
        if t in res:
            print(f"  {t} vs {TAGS[0]}: x{res[t]['geo'] / res[TAGS[0]]['geo']:.3f} whole decode geo, x{res[t]['pooled'] / res[TAGS[0]]['pooled']:.3f} "
                  f"pooled; MB per kept token x{res[t]['mb_per_tok'] / res[TAGS[0]]['mb_per_tok']:.3f}")
json.dump(res, open(os.path.join(D, "chw13_report.json"), "w"), indent=1)
