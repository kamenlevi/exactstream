"""release/convert_full_record.py: write data/release/convert_full.txt from release/convert_full.sh's logs.

Copies the raw logs to data/release/convert_full/ and summarises: the verdict line, layers converted and the wall, per-layer
step times (convert_store_report.txt), the byte compare, disk and swap. Run with any python3 after convert_full.sh ends."""
import os, re, shutil, statistics as st, subprocess, sys

LEAP = "/Users/kamen/Leap"
T = os.environ.get("CF_T", f"{LEAP}/release/cleanclone/convert_full")      # overridable for a test
D = os.environ.get("CF_D", f"{LEAP}/data/release/convert_full")
REC = os.environ.get("CF_REC", f"{LEAP}/data/release/convert_full.txt")
os.makedirs(D, exist_ok=True)
for f in ("convert_full.log", "convert_store_report.txt", "convert_store.stdout"):
    if os.path.exists(f"{T}/{f}"):
        shutil.copy2(f"{T}/{f}", f"{D}/{f}")
log = open(f"{D}/convert_full.log").read() if os.path.exists(f"{D}/convert_full.log") else ""
rep = open(f"{D}/convert_store_report.txt").read() if os.path.exists(f"{D}/convert_store_report.txt") else ""
L = [ln for ln in log.splitlines() if ln.strip()]
verdict = next((ln for ln in reversed(L) if "CONVERT_FULL" in ln), L[-1] if L else "no log")
rows = [tuple(int(x) for x in m) for m in re.findall(r"L\d+: native\+p11\+pad; repack (\d+) s, split \+ verify (\d+) s, p11 (\d+) s", rep)]
done = re.search(r"STORE DONE: (\d+) layers converted in (\d+) s; (\d+) layer files", rep)
first = re.search(r"^(\d\d-\d\d \d\d:\d\d:\d\d) start:", log, re.M)
last = re.search(r"^(\d\d-\d\d \d\d:\d\d:\d\d) CONVERT_FULL", log, re.M)
commit = subprocess.run(["git", "-C", LEAP, "log", "-1", "--format=%h", "--", "release/convert_full.sh"], capture_output=True, text=True).stdout.strip()

def rng(v):
    return f"{min(v)}-{max(v)} s (median {st.median(v):g})" if v else "-"

out = [f"ENGINE PACKAGE: full expert-store conversion from the model's safetensors, Mon 9 Nov 2026 "
       f"{first.group(1)[6:] if first else '?'} - {last.group(1)[6:] if last else '?'} (the orchestrator's 12:44 request; "
       f"RELEASE section 4, Conversion; runner release/convert_full.sh, Leap {commit})",
       "",
       "WHAT: the tested package clone's convert/convert_store.sh (rel1 242553d, release/cleanclone/final/clone, its venv) on",
       "every MoE layer of ~/BigMoE/models/bare (the script's default layer detection) into a temporary folder",
       "(release/cleanclone/convert_full/records3), then a byte compare (cmp) of every layer file and of index.json with the",
       "measured store ~/BigMoE/models/bare/records3 (index entries parsed when the bytes differ), then the temporary store",
       "deleted. Share lock, after EvalSuite's m1b ended; CPU math libraries at 4 threads; watchdog: hard stop 16:58, swap +2 GB,",
       "free disk < 40 GB. The store the engine reads was never written (the script only reads the model folder).",
       "",
       f"RESULT [K]: {verdict}"]
if rows:
    tot = [a + b + c for a, b, c in rows]
    out += [f"  layers converted: {len(rows)}" + (f" ({done.group(1)} by STORE DONE, {done.group(3)} layer files) in "
                                                  f"{done.group(2)} s = {int(done.group(2)) / 60:.1f} min" if done else " (no STORE DONE line)"),
            f"  per layer: repack {rng([r[0] for r in rows])}, split + verify {rng([r[1] for r in rows])}, "
            f"p11 {rng([r[2] for r in rows])}; the three together {rng(tot)}"]
for key in ("layer files:", "index.json", "layers in reference", "new store size", "convert_store.sh rc", "temporary store deleted",
            "STOPPING", "NOT STARTED", "FAILED"):
    out += [f"  {ln.strip()}" for ln in L + rep.splitlines() if key in ln and f"  {ln.strip()}" not in out]
out += ["", "RAW: data/release/convert_full/convert_full.log (the runner), convert_store_report.txt (convert_store.sh's report,",
        "per layer), convert_store.stdout. The runner's log, verbatim:", ""] + ["  " + ln for ln in L] + [""]
open(REC, "w").write("\n".join(out))
print("\n".join(out[:16]))
