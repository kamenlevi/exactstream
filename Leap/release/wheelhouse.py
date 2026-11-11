#!/usr/bin/env python3
"""wheelhouse.py <site-packages dir> <out dir>: rebuild one wheel per installed distribution from its dist-info RECORD
(the installed files, byte for byte; console scripts and .pyc files are left out, the installer regenerates them), so
a clean venv can be installed OFFLINE with exactly the packages the research venv ran:
  uv pip install --offline --find-links <out dir> -r requirements.txt
Used by the release package's clean-clone test on the research machine (no network, no download)."""
import base64, csv, hashlib, os, re, sys, zipfile

sp, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
n = 0
for d in sorted(os.listdir(sp)):
    if not d.endswith(".dist-info"):
        continue
    meta = dict(l.split(": ", 1) for l in open(os.path.join(sp, d, "METADATA"), encoding="utf-8").read().split("\n\n")[0].splitlines()
                if ": " in l)
    name, ver = re.sub(r"[-_.]+", "_", meta["Name"]), meta["Version"]
    tags = [l.split(": ", 1)[1].strip() for l in open(os.path.join(sp, d, "WHEEL")).read().splitlines() if l.startswith("Tag: ")]
    wh = os.path.join(out, f"{name}-{ver}-{tags[0] if tags else 'py3-none-any'}.whl")
    rec, files = [], []
    for row in csv.reader(open(os.path.join(sp, d, "RECORD"))):
        f = row[0]
        if f.startswith("..") or f.endswith(".pyc") or "__pycache__" in f or f.endswith("/RECORD") or f.endswith("INSTALLER") \
                or f.endswith("REQUESTED") or f.endswith("direct_url.json"):
            continue
        p = os.path.join(sp, f)
        if os.path.isfile(p):
            b = open(p, "rb").read()
            files.append((p, f)); rec.append(f"{f},sha256={base64.urlsafe_b64encode(hashlib.sha256(b).digest()).rstrip(b'=').decode()},{len(b)}")
    with zipfile.ZipFile(wh, "w", zipfile.ZIP_DEFLATED) as z:
        for p, f in files:
            z.write(p, f)
        z.writestr(f"{d}/RECORD", "\n".join(rec + [f"{d}/RECORD,,"]) + "\n")
    n += 1
print(f"{n} wheels -> {out}")
