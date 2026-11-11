#!/opt/homebrew/bin/python3.12
"""DRY RUN ONLY: a stand-in for `strata --serve` that speaks the same line protocol and prints the same `strata metal ...`
lines as the Metal engine, so driver.sh / run_arm.py / report.py can be tested without the model. It answers each prompt
with NAX6-N's AR ids (exact arms, STRATA_HC_Q8=0) or the AR ids with one token changed at position 50 (lossy arms), and
fakes timings (prompt 0.4 ms/token, decode 1.5 ms/token, MTP arms 1.0 ms/token). No GPU, no model."""
import json, os, sys, time
from pathlib import Path

H = Path(__file__).resolve().parent.parent
man = json.loads((H / "prompts" / "manifest.json").read_text())
ar = json.loads((H / "prompts" / "ar_ref.json").read_text())["refs"]
by_ids = {}
for p in man["prompts"]:
    by_ids[(H / p["file"]).read_text().strip()] = p["key"]
args = sys.argv[1:]
mtp = "--mtp" in args
exact = os.environ.get("STRATA_HC_Q8") == "0"
e = sys.stderr
if not exact:
    print("strata metal: HC read in Q8_0 (STRATA_HC_Q8=0: BF16): 192 matrices, 403 MB, made in 812 ms", file=e)
print("strata metal stream: planned resident footprint 41.2 GB = arena 30.9 + 9.8 held before it + 0.5 reserved for the n-gram "
      "table; cap 41.2 GB = the lesser of the GPU working set 51.5 GB and the RAM 68.7 GB less 27.5 GB left to macOS and its file cache", file=e)
print("strata metal stream: expert arena 30.90 GB, 18432 slots in 3 size classes, 18432 experts read from the profile's top "
      "(24576 ranked) in 9.4 s (3.29 GB/s); 8 I/O threads, read-ahead top 4; 12 buffers", file=e)
print("strata metal stream: 30.9 GB of the arena wired (mlock) in 950 ms; macOS cannot compress it while the engine idles "
      "(STRATA_METAL_WIRE_ARENA=0 turns it off)", file=e)
print("strata metal stream: resident footprint after the fill: the process holds 40.6 GB (41.0 GB resident), planned 41.2 GB, "
      "cap 41.2 GB", file=e)
print("strata metal: decode window on Apple M5 Pro: IQ3_XXS streamed (47.0 GB), 32768 cells, PLE direct, ready in 2100 ms", file=e)
e.flush()
time.sleep(0.3)
print("READY 32768 stop", flush=True)
for line in sys.stdin:
    line = line.strip()
    if line == "QUIT":
        break
    if not line.startswith("GEN "):
        print("ERR expected: GEN <max_new> <id,id,...>", flush=True)
        continue
    parts = line.split()
    max_new = int(parts[1])
    ids_txt = [x for x in parts[2:] if "=" not in x][0]
    key = by_ids.get(ids_txt)
    n = len(ids_txt.split(","))
    ref = list(ar.get(key, {}).get("completion_ids", [])) or [1000 + i for i in range(max_new)]
    out = ref[:max_new]
    if not exact and len(out) > 50:
        out[50] = out[50] + 1
    t_ms = 0.4 * n
    time.sleep(t_ms / 1000)
    per = 1.0 if mtp else 1.5
    for t in out:
        print(f"T {t}", flush=True)
        time.sleep(per / 1000)
    fin = "stop" if out and out[-1] in (248044, 248046) else "length"
    acc, off = (len(out) // 2, len(out)) if mtp else (0, 0)
    print(f"DONE {len(out)} {n} {t_ms:.1f} {per * len(out):.1f} {fin} {acc} {off} 0 0 0 0 0 0.0 {n} 0 chunk=0 experts_streamed=0", flush=True)
