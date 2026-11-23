"""E1 / E5 training set (MTP Head, Sun 11-01 night): Fable's E1 spec = 20K answer tokens of the 125B's own answers in each of
de / ru / ja. From tonight's DLOG files (data/dlog/n1101, index order = capture order), per language the first files until
>= E15_TOKENS answer tokens (default 20000); symlinked into data/dlog/e15set/ (the trainers read every *.npz in a folder).
Prints the per-language totals; exit 1 if a language has none."""
import json, os, sys

D = os.path.expanduser("~/TrainLab/data/dlog/n1101")
OUT = os.path.expanduser("~/TrainLab/data/dlog/e15set")
N = int(os.environ.get("E15_TOKENS", "20000"))
os.makedirs(OUT, exist_ok=True)
for f in os.listdir(OUT):
    os.remove(os.path.join(OUT, f))
tot, files = {"de": 0, "ru": 0, "ja": 0}, {"de": 0, "ru": 0, "ja": 0}
for l in open(os.path.join(D, "index.jsonl")):
    r = json.loads(l)
    fn = r["key"].replace("/", "__") + ".npz"
    if r["lang"] not in tot or tot[r["lang"]] >= N or not os.path.exists(os.path.join(D, fn)):
        continue
    os.symlink(os.path.join(D, fn), os.path.join(OUT, fn))
    tot[r["lang"]] += r["n"]; files[r["lang"]] += 1
print(f"e15set: " + ", ".join(f"{k} {files[k]} files / {tot[k]} answer tokens" for k in tot) + f" (target {N} each)")
sys.exit(0 if all(tot.values()) else 1)
