"""Re-derive every number of RELEASE's chat.py / serve.py part (v1.1.0, orchestrator Wed 11 Nov 03:4x) from Leap's raw
records (read only). usage: python3 -I rd_release_chat.py
1. BIGMOE_IDLE_MS: idle CPU without it (data/agent_cc_1111/idle_cpu_1111/idle_cpu.log) and with it
   (data/idle_ms_1111/runner.log); the decode A/B (dec_*.txt, dec_*_tokens.json).
2. PF-A on the chat configuration: 32 GB data/prefill_1109/cli_1110_{000737,000918,001004,001438}_runs.json;
   24 GB data/tue_afternoon_1110/pfa24_{default,pfa}.jsonl (7K, via chat.py) and p24_{10k,30k}_{default,pfa}_runs.json.
3. The checks: data/w24_pkg/candidate_110c/report.txt (chat.py / serve.py, first build); data/chatpy_fixes_1110/
   (pty_test_final.log, chatfix_now.log, filecheck_night_1111.log).
4. What 1.1.0 changes against the build tested on 10 Nov 10:18 (release/out/pkg_w24 vs pkg_110g)."""
import filecmp
import json
import pathlib
import re

L = pathlib.Path.home() / "Leap"
D = L / "data"

print("== 1. BIGMOE_IDLE_MS")
for f, tag in ((D / "agent_cc_1111/idle_cpu_1111/idle_cpu.log", "without"), (D / "idle_ms_1111/runner.log", "with 50")):
    for m in re.finditer(r"(\d\d) GB: idle CPU over 60 s after the request: .*?\(cpu time ([\d.]+) -> ([\d.]+) s\)",
                         f.read_text()):
        g, a, b = m.group(1), float(m.group(2)), float(m.group(3))
        print(f"  {tag:8s} {g} GB: cpu {a} -> {b} s over 60 s = {100 * (b - a) / 60:.2f} % of one core")
for g in ("24", "32"):
    tps, toks, ft = {}, {}, {}
    for r in ("1A", "2B", "3B", "4A"):
        s = (D / f"idle_ms_1111/dec_{g}_{r}.txt").read_text()
        tps[r] = float(re.search(r"whole decode ([\d.]+) tok/s", s).group(1))
        ft[r] = float(re.search(r"first token ([\d.]+) s", s).group(1))
        toks[r] = json.loads((D / f"idle_ms_1111/dec_{g}_{r}_tokens.json").read_text())
    A = [tps["1A"], tps["4A"]]; B = [tps["2B"], tps["3B"]]
    same = all(toks[r] == toks["1A"] for r in toks)
    print(f"  decode {g} GB: A {A} B {B}; B/A {sum(B) / sum(A):.4f}; A-A spread {max(A) / min(A) - 1:.4f}, "
          f"B-B {max(B) / min(B) - 1:.4f}; first token A {[ft['1A'], ft['4A']]} B {[ft['2B'], ft['3B']]}; "
          f"tokens equal in all 4: {same} ({len(toks['1A'])} prompts)")

print("== 2. PF-A on the chat configuration (first token = pfs; lifetime peak = mem[-1])")
def run(p):
    r = json.loads(p.read_text())[0]
    return r["P"], r["pfs"], r["mem"][-1], bool(r.get("pflm"))
rows = {"32": [("10K", "prefill_1109/cli_1110_000737", "prefill_1109/cli_1110_000918"),
               ("30K", "prefill_1109/cli_1110_001004", "prefill_1109/cli_1110_001438")],
        "24": [("10K", "tue_afternoon_1110/p24_10k_default", "tue_afternoon_1110/p24_10k_pfa"),
               ("30K", "tue_afternoon_1110/p24_30k_default", "tue_afternoon_1110/p24_30k_pfa")]}
for g, rs in rows.items():
    for sz, a, b in rs:
        Pa, fa, ma, la = run(D / f"{a}_runs.json"); Pb, fb, mb, lb = run(D / f"{b}_runs.json")
        assert Pa == Pb and not la and lb
        print(f"  {g} GB {sz} (P {Pa}): default {fa:.2f} s / {ma:.2f} GB; PF-A {fb:.2f} s / {mb:.2f} GB; "
              f"x{fa / fb:.3f}; +{mb - ma:.2f} GB")
j = {k: [json.loads(x) for x in (D / f"tue_afternoon_1110/pfa24_{k}.jsonl").read_text().splitlines() if x.strip()]
     for k in ("default", "pfa")}
for a, b in zip(j["default"], j["pfa"]):
    print(f"  24 GB chat.py P {a['P']}: default {a['pfs']:.2f} s / {a['peak']:.2f} GB; PF-A {b['pfs']:.2f} s / "
          f"{b['peak']:.2f} GB; x{a['pfs'] / b['pfs']:.3f}; +{b['peak'] - a['peak']:.2f} GB; answer equal "
          f"{a['answer'] == b['answer']} ({a['n']} / {b['n']} tokens)")

print("== 3. checks")
rep = (D / "w24_pkg/candidate_110c/report.txt").read_text()
for ln in rep.splitlines():
    if re.search(r"chat\.py|serve\.py|stream|CLEAN-CLONE", ln):
        print("  110c:", ln.strip()[:170])
pty = (D / "chatpy_fixes_1110/pty_test_final.log").read_text()
print("  pty:", re.findall(r"PTY TEST: .*", pty)[-1])
for f in ("chatfix_now.log", "filecheck_night_1111.log"):
    for ln in (D / "chatpy_fixes_1110" / f).read_text().splitlines():
        if re.search(r"rc \d|PASS|FAIL|peak", ln):
            print(f"  {f}:", ln.strip()[:170])

print("== 4. pkg_w24 (tested 10 Nov 10:18) vs pkg_110g")
def walk(c, pre=""):
    out = [("differ", pre + x) for x in c.diff_files] + [("only in 110g", pre + x) for x in c.right_only] \
        + [("only in w24", pre + x) for x in c.left_only]
    for n, s in c.subdirs.items():
        out += walk(s, pre + n + "/")
    return out
c = filecmp.dircmp(L / "release/out/pkg_w24", L / "release/out/pkg_110g", ignore=["__pycache__", ".git", ".DS_Store"])
for k, p in walk(c):
    print(f"  {k}: {p}")

print("== 5. the shipped build's checks (data/w24_pkg/candidate_110g/checks_0345/, Wed 03:45-03:57)")
C = D / "w24_pkg/candidate_110g/checks_0345"
import hashlib
ref = hashlib.sha256(json.dumps(json.loads((D / "idle_ms_1111/dec_32_1A_tokens.json").read_text()),
                                sort_keys=True).encode()).hexdigest()[:12]
A, B = [130.0, 131.9], [128.6, 129.7]                     # the 01:5x runs (part 1)
for f in sorted(C.glob("dec_32_*.out")):
    s = f.read_text()
    tps = float(re.search(r"whole decode ([\d.]+) tok/s", s).group(1))
    m = re.search(r"tokens: (\S+_tokens\.json)", s)
    h = hashlib.sha256(json.dumps(json.loads(pathlib.Path(m.group(1)).read_text()), sort_keys=True).encode()).hexdigest()[:12] \
        if m and pathlib.Path(m.group(1)).exists() else "tokens file not here"
    arm = f.stem[-1]
    print(f"  {f.name}: {tps} tok/s, tokens {h} ({'= 01:5x' if h == ref else 'DIFFERS'})")
    if tps > 50:
        (A if arm == "A" else B).append(tps)
print("  heat pause:", " | ".join(l.strip() for l in (C / "fanwatch_0347.txt").read_text().splitlines()
                                  if re.search(r"froze|cool", l))[:200])
print(f"  pooled 32 GB without the frozen run: A {sum(A) / len(A):.1f} (n {len(A)}), B {sum(B) / len(B):.1f} (n {len(B)}), "
      f"B/A {(sum(B) / len(B)) / (sum(A) / len(A)):.4f}")
for l in (C / "runner.log").read_text().splitlines():
    if "idle CPU" in l:
        print("  ", l.strip()[:150])
for l in (C / "clean_clone_run.log").read_text().splitlines():
    if re.search(r"prefix-equal|closure|chat\.py rc|stream|stopped|CLEAN-CLONE", l):
        print("  cc:", l.strip()[:150])
