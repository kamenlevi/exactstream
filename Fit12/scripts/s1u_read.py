"""Fit12 S1u reader (Mon 9 Nov 21:5x; PLAN_12GB_90_fit12.md '12 GB -> 90: WHY <= 4 ROWS'). READ-ONLY on the run folder.
Per arm (tag:arm), from <tag>.txt (the harness's per-prompt lines) and the per-prompt phase logs <tag>_<key>_<arm>_ph.json
(rows: name, check, start mark, end mark; a mark = uptime s, gate seq, process disk bytes, loader misses):
- whole decode tok/s pooled (sum n / sum decode s) and geo (per prompt); checks, kept per check;
- ms per check by phase (draft / p1 / commit / exact) and the rest of the decode window;
- the wide call (phase p1): calls, ms per call, GB per call, and its time OUTSIDE reads = ms - 83.7 ms per GB (the chw16 fit,
  /Users/kamen/Fit12/data/scr/chw16/chw16_prof.txt l21);
- MB per kept token over all decode phases; tokens vs the first arm given (equal prompts, first divergence).
The pre-registered lines (S1u): the one-call route stands if the wide call's time outside reads <= 90 ms AND MB per kept
token <= 150; else it is re-priced before S2. Usage: python -I -B s1u_read.py RUN_DIR tag:arm [tag:arm ...]"""
import glob, json, math, os, re, sys

O, PAIRS = sys.argv[1], sys.argv[2:]
B_MS_GB = 83.7
NAMES = ("draft", "p1", "commit", "exact")
geo = lambda xs: math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float("nan")
res, toks_by = [], []
for ta in PAIRS:
    tag, arm = ta.split(":")
    rx = re.compile(rf"^\[{re.escape(tag)}\] (\S+) {re.escape(arm)} theta \S+ .*?\| tok (\d+) .*?\| checks (\d+), kept ([\d.]+) "
                    rf".*?\| prefill ([\d.]+) s, decode ([\d.]+) s")
    rows, pend, anyp = {}, {}, False                # (Mon 22:5x, NewIdeas' check) PAIR-ok prompts only, as paper_rows.py
    rpair = re.compile(rf"^\[{re.escape(tag)}\] PAIR (\S+?)( \(re-run\))?: .*-> (.*)$")
    for ln in open(os.path.join(O, f"{tag}.txt")):
        m = rx.match(ln)
        if m:
            pend[m.group(1)] = dict(n=int(m.group(2)), checks=int(m.group(3)), pf=float(m.group(5)), dec=float(m.group(6)))
            continue
        m = rpair.match(ln)
        if m:
            anyp = True
            r_ = pend.pop(m.group(1), None)
            if r_ is not None and m.group(3).startswith("ok"):
                rows[m.group(1)] = r_
    if not anyp:
        rows = pend
    ph = {nm: [0.0, 0, 0] for nm in NAMES}         # ms, bytes, phases
    other_ms = 0.0
    tk = {}
    for key, r in rows.items():
        kf = key.replace("/", "_")
        f = os.path.join(O, f"{tag}_{kf}_{arm}_ph.json")
        if os.path.exists(f):
            for p in json.load(open(f)):
                a = ph.setdefault(p[0], [0.0, 0, 0])
                a[0] += (p[6] - p[2]) * 1e3; a[1] += p[8] - p[4]; a[2] += 1
        ft = os.path.join(O, f"{tag}_{kf}_{arm}_toks.json")
        if os.path.exists(ft):
            tk[key] = json.load(open(ft))
    n = sum(r["n"] for r in rows.values()); dec = sum(r["dec"] for r in rows.values())
    ck = sum(r["checks"] for r in rows.values())
    tot_ms = sum(v[0] for v in ph.values())
    o = dict(tag=tag, arm=arm, prompts=len(rows), tokens=n, checks=ck, kept=n / max(1, ck),
             whole_pooled=n / dec if dec else float("nan"), whole_geo=geo([r["n"] / r["dec"] for r in rows.values() if r["dec"]]),
             req_pooled=n / sum(r["pf"] + r["dec"] for r in rows.values()) if rows else float("nan"),
             ms_check={k: v[0] / max(1, ck) for k, v in ph.items()}, rest_ms_check=(dec * 1e3 - tot_ms) / max(1, ck),
             mb_per_tok=sum(v[1] for v in ph.values()) / 1e6 / max(1, n),
             # (Mon 22:3x, NewIdeas 22:28) the whole check: ms per check = decode window / checks; its SSD bytes (all phases);
             # read ms = GB x 83.7 (the fit's marginal read cost, [P]); non-read = the rest
             ck_ms=dec * 1e3 / max(1, ck), ck_gb=sum(v[1] for v in ph.values()) / 1e9 / max(1, ck))
    o["ck_read"] = B_MS_GB * o["ck_gb"]; o["ck_nonread"] = o["ck_ms"] - o["ck_read"]
    c = ph.get("p1", [0.0, 0, 0])
    if c[2]:
        o.update(p1_calls=c[2], p1_ms=c[0] / c[2], p1_gb=c[1] / 1e9 / c[2], p1_nonread=(c[0] - B_MS_GB * c[1] / 1e9) / c[2])
    o["rows"] = rows
    res.append(o); toks_by.append(tk)
L = [f"S1u READ ({O}); speed = whole decode tok/s pooled (geo); ms per check by phase; [P] = split by the chw16 fit, nonread = p1 ms - {B_MS_GB} ms per GB"]
for o in res:
    L.append(f"{o['tag']}:{o['arm']}: prompts {o['prompts']}, tokens {o['tokens']}, checks {o['checks']}, kept {o['kept']:.2f} | "
             f"whole decode {o['whole_pooled']:.2f} ({o['whole_geo']:.2f}), whole request {o['req_pooled']:.2f} | ms per check "
             + ", ".join(f"{k} {v:.1f}" for k, v in o["ms_check"].items() if v) + f", rest {o['rest_ms_check']:.1f} | "
             f"MB per kept token {o['mb_per_tok']:.0f} | whole check {o['ck_ms']:.1f} ms = [P] SSD read {o['ck_read']:.1f} "
             f"({o['ck_gb']:.3f} GB x {B_MS_GB} ms/GB, fit) + non-read {o['ck_nonread']:.1f}"
             + (f" | wide call: {o['p1_calls']} calls, {o['p1_ms']:.1f} ms, {o['p1_gb']:.3f} GB, outside reads [P] {o['p1_nonread']:.1f} ms"
                if "p1_calls" in o else ""))
if len(res) > 1:
    a, b = res[0], res[1]                          # (Mon 22:5x) on the prompts both arms counted (PAIR ok) only
    com = [k for k in a["rows"] if k in b["rows"]]
    pool = lambda o: sum(o["rows"][k]["n"] for k in com) / sum(o["rows"][k]["dec"] for k in com)
    gg = lambda o: geo([o["rows"][k]["n"] / o["rows"][k]["dec"] for k in com])
    same = sum(1 for k in com if k in toks_by[0] and k in toks_by[1] and toks_by[0][k] == toks_by[1][k])
    L.append(f"{a['tag']} vs {b['tag']} on the {len(com)} prompts both counted: whole decode {pool(a):.2f} vs {pool(b):.2f} = "
             f"x{pool(a) / pool(b):.3f} pooled, x{gg(a) / gg(b):.3f} geo; tokens equal on {same} / {len(com)}")
for o in res:
    if "p1_calls" in o:
        ok = o["p1_nonread"] <= 90 and o["mb_per_tok"] <= 150
        L.append(f"LINES ({o['tag']}): outside reads [P] {o['p1_nonread']:.1f} ms (<= 90) and {o['mb_per_tok']:.0f} MB per kept token "
                 f"(<= 150) -> {'THE ONE-CALL ROUTE STANDS' if ok else 'RE-PRICE BEFORE S2'}")
print("\n".join(L))
