"""After the code arms: the two log checks and the chat decision, from the arm files (CPU, read-only).
Usage: decide.py <out dir> <baseline json>      prints lines and ends with 'CHAT GO' or 'CHAT NO: <why>'
GO needs: four files with 12 code windows each; xs64 identical to the baseline at top-1 on every row; every xs48 / xs40 /
xs32 window carries its xs record (the right c) and no warning; xs48's non-tie code flips within the room (<= 0.66 %)."""
import json, os, sys
out, basep = sys.argv[1], sys.argv[2]
base = json.load(open(basep)); fr = json.load(open(os.path.expanduser("~/FableGod/floor_rows.json")))
CLS = {(r["nm"], int(r["i"])): r["cls"] for r in fr}
why = []; res = {}
for c in (64, 48, 40, 32):
    p = os.path.join(out, f"xs{c}+code.json")
    if not os.path.exists(p): why.append(f"xs{c}+code.json missing"); continue
    A = json.load(open(p)); W = [w for w, v in A.items() if isinstance(v, dict) and "tk" in v]
    if len(W) != 12: why.append(f"xs{c}: {len(W)} windows, want 12")
    fl = n = nt_fl = nt_n = same8 = 0; rows_masked = 0
    for w in W:
        v = A[w]; b = base[w]
        if len(v["tk"]) != len(b["tk"]): why.append(f"xs{c} {w}: {len(v['tk'])} rows vs {len(b['tk'])}"); continue
        if v.get("warn"): why.append(f"xs{c} {w}: {v['warn'][:120]}")
        if c < 64:
            x = v.get("xs")
            if not x or x.get("c") != c: why.append(f"xs{c} {w}: no xs record (c {x})")
            else: rows_masked += x["rows"]
        elif "xs" in v: why.append(f"xs64 {w}: carries an xs record")
        for j, (ta, tb) in enumerate(zip(v["tk"], b["tk"])):
            f = int(ta[0] != tb[0]); fl += f; n += 1; same8 += int(ta == tb)
            if CLS.get((w, int(b["a"]) + j)) in ("sure", "mid"): nt_fl += f; nt_n += 1
    res[c] = dict(flips=fl, rows=n, nontie=100.0 * nt_fl / max(nt_n, 1), allrow=100.0 * fl / max(n, 1), same8=same8, masked=rows_masked)
    print(f"xs{c}+code: {len(W)} windows, {n} rows, top-1 flips {fl} ({res[c]['allrow']:.2f} %), non-tie {res[c]['nontie']:.2f} %, top-8 lists equal on {same8} rows"
          + (f", {rows_masked} masked switch rows recorded" if c < 64 else ""))
if 64 in res and res[64]["flips"] != 0: why.append(f"the identity arm has {res[64]['flips']} flips: the wrapper or the environment is not today's")
if 48 in res and res[48]["nontie"] > 0.66: why.append(f"xs48 non-tie code {res[48]['nontie']:.2f} % is over the room 0.66: code leaves no room")
print("CHAT GO" if not why else "CHAT NO: " + "; ".join(why))
