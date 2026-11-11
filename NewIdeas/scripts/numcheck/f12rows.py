"""Independent re-derivation of SMALL_RAM's 12 GB rows from Fit12's run logs (per-prompt lines) and phase logs.
whole decode = sum tok / sum decode s; whole request = sum tok / sum (prefill + decode); first token = prefill + (end of
check 0's verify phase - decode window start), window start taken as the last phase end - decode s (approximation)."""
import json, os, re, statistics as st, glob
S = "/Users/kamen/Fit12/data/scr"
rx = re.compile(r"^\[(\w+)\] (\S+) (\w+) theta (\S+) .*?\| tok (\d+) .*?\| prefill ([\d.]+) s, decode ([\d.]+) s, ([\d.]+) tok/s \| lifetime ([\d.]+) GB")
for run, tag in (("tab12", "tab12e"), ("chat12", "chat12"), ("tab12", "tab12f"), ("c1u16", "c1u16"), ("ex16", "ex16"),
                 ("chw13", "chw13a"), ("chw14", "chw14c2"), ("chw14", "chw14ha"), ("chw14", "chw14h2")):
    rows = []
    for l in open(f"{S}/{run}/{tag}.txt"):
        m = rx.match(l)
        if m:
            t, key, arm, th, tok, pf, dec, tps, life = m.groups()
            rows.append(dict(key=key, arm=arm, tok=int(tok), pf=float(pf), dec=float(dec), tps=float(tps), life=float(life)))
    if not rows:
        print(tag, "no rows"); continue
    n = sum(r["tok"] for r in rows); D = sum(r["dec"] for r in rows); P = sum(r["pf"] for r in rows)
    ft = []
    for r in rows:
        ph = glob.glob(f"{S}/{run}/{tag}_{r['key'].replace('/', '_')}_{r['arm']}_ph.json")
        if not ph:
            continue
        p = json.load(open(ph[0]))
        ends0 = [x[6] for x in p if x[1] == 0 and x[0] != "draft"]
        last = max(x[6] for x in p)
        if ends0:
            ft.append(r["pf"] + (max(ends0) - (last - r["dec"])))
    geo = st.geometric_mean([r["tok"] / r["dec"] for r in rows])
    s = (f"{tag}: prompts {len(rows)}, whole decode {n / D:.2f} (geo {geo:.2f}), whole request {n / (D + P):.2f}, "
         f"peak {max(r['life'] for r in rows):.3f} GB")
    if ft:
        s += f", first token median {st.median(ft):.2f} ({min(ft):.2f}-{max(ft):.2f}) from {len(ft)} phase logs"
    print(s)
