"""WEEK98 T30C manifests (MTP Head; REPORT 'WEEK98 T30C'): the n1112 capture files for t30c, in BigMoE's split format
{night, rule, train, held_out} (dd_ttt.py --manifest reads names under its CAP, ~/FableGod/deepdraft/capture/n1112).
Held out in both: the bank's val 40 (the ridge / proxy validation files) and t30b's held-out 113 (CEILING read their
margins). A = unseen only: the 795 files t30b never saw (BigMoE's tapsq/split.json, pinned 11-14 14:13) minus the
bank's val 40 = 755. B = mixed: t30b's 2,164 training files + A's 755. Prints each manifest's one-pass slice count at
192 rows by dd_ttt.py's pass_slices rule (anchors from 2 before the first answer token to len - 3; ceil(n / 192) slices
per file), so the step count is known before the run.
Usage: BigMoE venv python -I -B t30c_manifest.py OUTDIR"""
import os, sys, json
import numpy as np

H = os.path.expanduser("~")
CAP = f"{H}/FableGod/deepdraft/capture/n1112"
OUT = sys.argv[1]
t30b = json.load(open(f"{H}/BigMoE/data/deepdraft/ttt30b/split.json"))
tapsq = json.load(open(f"{H}/BigMoE/data/deepdraft/tapsq/split.json"))
bank = json.load(open(f"{H}/BigMoE/data/deepdraft/bank/split.json"))
seen = set(t30b["train"]) | set(t30b["held_out"])
unseen = [n for n in tapsq["train"]]
assert len(unseen) == 795 and not (set(unseen) & seen), "tapsq's list is not the 795 unseen files"
val40 = [os.path.basename(x) for x in bank["val"]]
assert len(val40) == 40 and set(val40) <= set(unseen), "the bank's val 40 are not among the unseen files"
on_disk = {n for n in os.listdir(CAP) if n.endswith(".npz")}
A = sorted(set(unseen) - set(val40))
B = sorted(set(t30b["train"]) | set(A))
held = sorted(set(val40) | set(t30b["held_out"]))
assert set(A) <= on_disk and set(B) <= on_disk and not (set(B) & set(held))


def slices(names, R=192):
    k = 0
    for n in names:
        z = np.load(os.path.join(CAP, n))
        n_ids = len(z["ids"]); a = np.flatnonzero(np.asarray(z["asst"]) > 0)
        first = max(0, int(a[0]) - 2) if len(a) else 0
        k += -(-((n_ids - 3) - first + 1) // R)
    return k


os.makedirs(OUT, exist_ok=True)
rule_common = ("held out: the bank's val 40 (bank/split.json val) + t30b's held_out 113 (ttt30b/split.json); "
               "MTP Head T30C, built by TrainLab sb98/t30c_manifest.py")
for name, tr, rule in (("A", A, "unseen only: tapsq/split.json's 795 files t30b never saw, minus the bank's val 40"),
                       ("B", B, "mixed: t30b's 2,164 training files + the 755 unseen files of manifest A")):
    k = slices(tr)
    json.dump({"night": "n1112", "rule": f"{rule}; {rule_common}", "train": tr, "held_out": held, "slices_192": k},
              open(os.path.join(OUT, f"split_{name}.json"), "w"), indent=0)
    print(f"manifest {name}: {len(tr)} train files, {len(held)} held out, one pass = {k} slices at 192 rows")
print(f"check: t30b's split gives {slices(t30b['train'])} slices (its run: 4,206 steps)")
