"""Fit12 (Sun 8 Nov 00:0x; NewIdeas 00:2x, PROGRAM CHAT 150 / PREFILL 500): the 12 GB PREFILL A/B of
/Users/kamen/Fit12/scripts/chat12.sh: arm ex on BigMoE's PFX prompts (agent/edit10 1.6K, agent/write13 4.7K, agent/write11
7.9K tokens, ascending, one process per arm), CAP 32. A = today's b12_prefill (MoE chunk s_moe 4096: experts read once per
layer per 4096 prompt rows), B = one MoE call per prompt (s_moe 8192: each expert read once per prompt, BigMoE's PFX form).
Per prompt and arm: prefill seconds and tok/s (ids[:-1]), the prefill's own SSD reads, TTFT = prefill + the first check and
the drafter's setup (the first tokens appear at check 0's commit), the process's lifetime peak after the prompt (ascending
lengths: the running max is the prompt's own peak), and the 32 decoded tokens A = B (s_moe changes no arithmetic).
Usage: python -I -B pf12_read.py [DIR] [TAG_A] [TAG_B]  (defaults /Users/kamen/Fit12/data/scr/chat12 pf12a pf12b)"""
import json, os, re, sys

D = sys.argv[1] if len(sys.argv) > 1 else "/Users/kamen/Fit12/data/scr/chat12"
TA = sys.argv[2] if len(sys.argv) > 2 else "pf12a"
TB = sys.argv[3] if len(sys.argv) > 3 else "pf12b"


def read(tag):
    T = re.escape(tag)
    rpf = re.compile(rf"^\[{T}\] (\S+) ex PREFILL n (\d+) smoe (\d+) \| ([\d.]+) s, ([\d.]+) tok/s, SSD (\d+) MB \| lifetime ([\d.]+) GB")
    rdec = re.compile(rf"^\[{T}\] (\S+) ex theta None .*?\| tok (\d+) .*?\| prefill ([\d.]+) s, decode ([\d.]+) s,")
    rpair = re.compile(rf"^\[{T}\] PAIR (\S+?)( \(re-run\))?: .*-> (.*)$")
    pend, cnt, void, done = {}, {}, [], None
    p_ = os.path.join(D, f"{tag}.txt")
    if not os.path.exists(p_):
        return {}, [("(no log)", "", "")], None
    for ln in open(p_):
        s = ln.rstrip("\n")
        m = rpf.match(s)
        if m:
            pend.setdefault(m.group(1), {}).update(n=int(m.group(2)), smoe=int(m.group(3)), pf=float(m.group(4)),
                                                   pftps=float(m.group(5)), pfmb=int(m.group(6)), life=float(m.group(7)))
            continue
        m = rdec.match(s)
        if m:
            pend.setdefault(m.group(1), {}).update(tok=int(m.group(2)), dec=float(m.group(4)))
            continue
        m = rpair.match(s)
        if m:
            r = pend.pop(m.group(1), None)
            if r is not None and m.group(3).startswith("ok"):
                cnt[m.group(1)] = r
            elif r is not None:
                void.append((m.group(1), m.group(3)))
            continue
        if s.startswith(f"[{tag}] done; lifetime "):
            done = float(re.search(r"lifetime ([\d.]+) GB", s).group(1))
    for k, r in cnt.items():
        kf = k.replace("/", "_")
        ph = json.load(open(os.path.join(D, f"{tag}_{kf}_ex_ph.json")))
        e0, eN = max(p[6] for p in ph if p[1] == 0), max(p[6] for p in ph)
        r["first"] = r["dec"] - (eN - e0)                    # check 0 + the drafter's setup, s
        r["ttft"] = r["pf"] + r["first"]
        r["toks"] = json.load(open(os.path.join(D, f"{tag}_{kf}_ex_toks.json")))
    return cnt, void, done


a, va, da = read(TA)
b, vb, db = read(TB)
print(f"12 GB PREFILL A/B ({D}): A = {TA} (s_moe 4096, today's), B = {TB} (s_moe 8192, one MoE call per prompt); arm ex, CAP 32")
print("  prompt          | n    | A prefill s  tok/s   SSD GB  TTFT s  life GB | B prefill s  tok/s   SSD GB  TTFT s  life GB | "
      "B/A tok/s | tokens A = B")
out = {}
for k in sorted(set(a) | set(b), key=lambda k: (a.get(k) or b.get(k))["n"]):
    x, y = a.get(k), b.get(k)
    f = lambda r: (f"{r['pf']:10.2f} {r['pftps']:7.1f} {r['pfmb'] / 1e3:8.1f} {r['ttft']:7.2f} {r['life']:8.3f}" if r
                   else f"{'-':>10s} {'-':>7s} {'-':>8s} {'-':>7s} {'-':>8s}")
    rat = f"x{y['pftps'] / x['pftps']:.3f}" if x and y else "-"
    eq = (x["toks"] == y["toks"]) if x and y else None
    print(f"  {k:15s} | {(x or y)['n']:4d} | {f(x)} | {f(y)} | {rat:9s} | {eq}")
    out[k] = dict(a={kk: v for kk, v in (x or {}).items() if kk != "toks"}, b={kk: v for kk, v in (y or {}).items() if kk != "toks"},
                  ratio=(y["pftps"] / x["pftps"]) if x and y else None, same_tokens=eq)
print(f"lifetime peak per process: A {da} GB, B {db} GB; void / paused: A {va or 'none'}, B {vb or 'none'}")
json.dump(dict(prompts=out, life_a=da, life_b=db, void_a=va, void_b=vb), open(os.path.join(D, "pf12_report.json"), "w"), indent=1)
