"""Re-derive the PF-A short-last-chunk numbers used in v1.1.0 (v110_pfa.py) from the raw runs. Read-only.
usage: python3 -I rd_pfa_tail.py"""
import json, os, re
H = os.path.expanduser("~")
L = H + "/Leap/data/"

def last_chunk(p, c=1024):
    return p - c * ((p - 1) // c)

print("1. PF-A vs the default prefill, exact-32, 64 tokens (Leap/data/pfa_tail_1111/)")
for t in ("t116", "t300"):
    r = {a: json.load(open(f"{L}pfa_tail_1111/{t}_{a}_runs.json"))[0] for a in ("default", "pfa")}
    tok = {a: list(list(json.load(open(f"{L}pfa_tail_1111/{t}_{a}_tokens.json")).values())[0].values())[0][0]
           for a in ("default", "pfa")}
    a, b = r["default"]["lh"], r["pfa"]["lh"]
    eq = sum(x == y for x, y in zip(a, b))
    fd = next((i + 1 for i, (x, y) in enumerate(zip(tok["default"], tok["pfa"])) if x != y), None)
    pk = {k: max(v["mem"][1], v["mem"][-1]) for k, v in r.items()}
    print(f"  P {r['default']['P']:,} (last chunk {last_chunk(r['default']['P'])}): hashes {eq} / {len(a)} "
          f"(PF-A {len(b)} checks); tokens {len(tok['default'])} / {len(tok['pfa'])}, first differing token {fd}; "
          f"first token {r['default']['pfs']:.2f} / {r['pfa']['pfs']:.2f} s; peaks {pk['default']:.2f} / {pk['pfa']:.2f} GB")

print("2. The switch: cache hits whose tail ran chunk-major vs the cold PF-A sweep (chat-32; T1), and T1b (tail via PF-A)")
for run in ("t1_32_1111", "t1b_32_1111"):
    out = []
    for ln in open(f"{L}agent_cc_1111/{run}/convcache.log"):
        m = re.search(r"hit N (\d+) c (\d+) .*prefilled (\d+).*check (PASS|FAIL: (.*))$", ln.strip())
        if m and m.group(2) and int(m.group(1)) % 1024 == int(m.group(3)) % 1024 and int(m.group(2)) % 1024 == 0:
            sigA = bool(m.group(5) and "last logits" in m.group(5))
            out.append((int(m.group(3)), "A" if sigA else ("PASS" if m.group(4) == "PASS" else "other")))
    print(f"  {run}: " + ", ".join(f"{n}:{s}" for n, s in sorted(set(out))))

print("3. Last chunks of the other PF-A comparisons")
rep = open(H + "/BigMoE/data/pfx/pfx_report.txt").read()
ps = sorted({int(x) for x in re.findall(r"P (\d+):", rep)})
print("  45 GB audit prompts (BM:data/pfx/pfx_report.txt):", ", ".join(f"{p:,} -> {last_chunk(p)}" for p in ps))
for p in (29988, 5412, 9988, 15002, 20003, 25003, 6980):
    print(f"  {p:,} -> {last_chunk(p)}")

print("4. 24 GB (chat-24): PF-A vs default at 6,980 tokens (Leap/data/tue_afternoon_1110/pfa24_*.jsonl)")
rows = {a: [json.loads(l) for l in open(f"{L}tue_afternoon_1110/pfa24_{a}.jsonl") if l.strip()] for a in ("default", "pfa")}
for x, y in zip(rows["default"], rows["pfa"]):
    print(f"  P {x['P']:,}: n {x['n']} / {y['n']}, answer text equal {x['answer'] == y['answer']}, "
          f"first token {x['pfs']:.2f} / {y['pfs']:.2f} s, peak {x['peak']:.2f} / {y['peak']:.2f} GB")
