"""Which runs can hit MLX 0.32.2's NAX sorted-gather bug (Leap 11-08, ~/Leap/TABLE_350.md "MLX NAX SORTED-GATHER BUG"):
one MoE call with > 32,768 sorted rows (> 3,276 tokens at top-10) and a row count that is not a multiple of 64 (T % 32
tokens != 0) leaves output rows unwritten. Per run with rows on long items (agent, agentlong, ruler): the rows that
would have such a call if the MoE ran once per 1,024 tokens, once per 4,096, or once per prompt (<= 16K), from each
row's n_prompt; the exposed keys under the 4,096 rule. docs/AUDIT_nax_gather_2026-11-08.md.
  .venv/bin/python tools/nax_gather_audit.py"""
import glob, json, os, re

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
LONG = ("agent/", "agentlong/", "ruler/")
VER = re.compile(r'"mlx[^"]*"\s*:\s*"([^"]+)"|mlx[ =]+([0-9]+\.[0-9]+\.[0-9]+)')


def hits(n, chunk):
    calls = [chunk] * (n // chunk) + ([n % chunk] if n % chunk else [])
    return [t for t in calls if t * 10 > 32768 and t % 32]


for d in sorted(glob.glob("results/*/decode.jsonl")):
    run = d.split("/")[1]
    rows = []
    for line in open(d, errors="replace"):
        if any(f'"key": "{p}' in line[:200] for p in LONG):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if isinstance(r.get("n_prompt"), int):
                rows.append((r["key"], r["n_prompt"]))
    if not rows:
        continue
    mj = os.path.join("results", run, "decode.json")
    m = VER.search(open(mj, errors="replace").read()) if os.path.exists(mj) else None
    ver = (m.group(1) or m.group(2)) if m else "not recorded"
    k4 = [f"{k} ({n}: {hits(n, 4096)})" for k, n in rows if hits(n, 4096)]
    print(f"{run}: {len(rows)} long rows; exposed per-1024 {sum(bool(hits(n, 1024)) for _, n in rows)}, per-4096 {len(k4)}, "
          f"one call {sum(bool(hits(n, 16384)) for _, n in rows)}; mlx {ver}; per-4096 keys {k4}")
