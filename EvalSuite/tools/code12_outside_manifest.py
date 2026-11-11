# The S1q code gate's outside code (PREREG_code12_s1q): sha256 of every .py file in the trees the frozen harness can load
# from, plus size + mtime of the drafter weights. usage: python3 -I outside_manifest.py OUT  (diff two outputs to compare)
import glob, hashlib, os, sys
H = os.path.expanduser
TREES = [  # (pattern, recursive)
    ("~/Fit12/snap/s1q1110/*.py", False), ("~/Fit12/pinned/8db7/*.py", False),
    ("~/Fit12/scripts/k12sync.py", False), ("~/Fit12/scripts/f12_kports.py", False),
    ("~/Breakthrough/v3p/**/*.py", True), ("~/Breakthrough/scripts/*.py", False),
    ("~/SpecDecode/*.py", False), ("~/Scout/**/*.py", True),
    ("~/BigMoE/patches/**/*.py", True), ("~/BigMoE/bigmoe/**/*.py", True),
    ("~/BigMoE/.venv/lib/python3.12/site-packages/mtplx/**/*.py", True),
    ("~/Leap/scripts/d7b_live.py", False), ("~/Leap/third_party/**/*.py", True),
]
rows = []
for pat, rec in TREES:
    for f in sorted(glob.glob(H(pat), recursive=rec)):
        rows.append(f"{hashlib.sha256(open(f, 'rb').read()).hexdigest()[:16]}  {f.replace(H('~'), '~')}")
for f in (H("~/Leap/data/l120_ft.safetensors"),):
    if os.path.exists(f):
        st = os.stat(f)
        rows.append(f"size {st.st_size} mtime {int(st.st_mtime)}  {f.replace(H('~'), '~')}")
digest = hashlib.sha256("\n".join(rows).encode()).hexdigest()[:16]
open(sys.argv[1], "w").write(f"# S1q code gate outside-code manifest: {len(rows)} entries, digest {digest}\n" + "\n".join(rows) + "\n")
print(len(rows), "entries; digest", digest)
