#!/usr/bin/env python3
"""V1 (R57_LMSL) code head lists, built out of sample, and their forced misses on the map's 24 code windows
(Breakthrough, Mon 16 Nov 2026 13:1x; CHECK BYTES, docs/CHECK_BYTES_900.md section 3). CPU only. [post hoc, unregistered]

Finding: the "16K code list" used in the map read-outs and the plan was built from all of BigMoE's bank
(~/datasets/deepdraft/bank: 400 dd__mc + 100 dlog__mbpp + 24 tf_code_HumanEval files). The 24 tf_code_HumanEval files ARE
the map's 24 code windows (same names; the map's baseline top-1 sequence sits inside the bank's tk[:, 0]). The bank has
only 8,775 distinct top-1 tokens, so "most_common(16384)" returned those 8,775: every top-1 token of the evaluated
windows but 2 rows. The shortlist part of the map's V1 reads was in sample (uninformative).

Here: lists from the Magicoder part only (dd__mc, disjoint from HumanEval / MBPP), filled to 16,384 / 32,768 by FR-Spec
rank (~/Leap/data/frspec_rank.npy, SVH's own ranking). Forced miss = the baseline top-1 outside list U session rows
(the prompt + the committed answer prefix, as SVH's screened_head_add feeds them); drafts are not added (a draft equal to
the true token would also be scored, so these are upper bounds). Writes data/m1/v1_lmsl_lists.out and the lists
data/m1/lmsl_code16k.npy / lmsl_code32k.npy (int32, sorted).
Usage: ~/BigMoE/.venv/bin/python -I -B v1_lmsl_lists.py
"""
import collections
import glob
import hashlib
import json
import os

import numpy as np

B = os.path.expanduser("~/datasets/deepdraft/bank")
O = os.path.expanduser("~/FableGod/decision_map/out_code24")
D = os.path.expanduser("~/Breakthrough/data/m1")
lines = []


def say(s=""):
    print(s, flush=True)
    lines.append(s)


base = json.load(open(os.path.join(O, "baseline.json")))
files = sorted(glob.glob(os.path.join(B, "*.npz")))
grp = collections.Counter(os.path.basename(f).rsplit("_", 1)[0] for f in files)
same = sorted(set(os.path.basename(f) for f in files) & set(base))
say(f"bank: {len(files)} files {dict(grp)}; files that ARE map code windows: {len(same)} of {len(base)}")
cnt = {"all": collections.Counter(), "mc": collections.Counter()}
for f in files:
    t = np.load(f)["tk"][:, 0].tolist()
    cnt["all"].update(t)
    if os.path.basename(f).startswith("dd__mc"):
        cnt["mc"].update(t)
rank = np.load(os.path.expanduser("~/Leap/data/frspec_rank.npy"))


def fill(seed, n):
    s = list(dict.fromkeys(seed)); have = set(s)
    for t in rank.tolist():
        if len(s) >= n:
            break
        if t not in have:
            s.append(t); have.add(t)
    return s[:n]


mc_order = [t for t, _ in cnt["mc"].most_common()]
LISTS = {"in-sample all-bank (as used)": sorted(cnt["all"]), "Magicoder only": sorted(cnt["mc"]),
         "Magicoder + FR-Spec fill to 16,384": sorted(fill(mc_order, 16384)),
         "Magicoder + FR-Spec fill to 32,768": sorted(fill(mc_order, 32768)),
         "SVH base (FR-Spec top 98,304)": sorted(set(np.unique(rank[:98304]).tolist()))}
say("forced misses on the 24 code windows' answer rows (baseline top-1 outside the set; no drafts added):")
say("| list | size | outside the list | outside list + session rows | rows |")
say("|---|---|---|---|---|")
res = {}
for nm, L in LISTS.items():
    S = set(L); tot = m0 = m1 = 0
    for w, v in base.items():
        ids = np.load(os.path.join(O, "routing", w))["tokens"].astype(np.int64); a = v["a"]
        for j, row in enumerate(v["tk"]):
            t = row[0]; tot += 1
            if t in S:
                continue
            m0 += 1
            if t not in set(ids[:a + j + 1].tolist()):
                m1 += 1
    res[nm] = (len(S), m0, m1, tot)
    say(f"| {nm} | {len(S)} | {100 * m0 / tot:.2f} % | {100 * m1 / tot:.2f} % ({m1}) | {tot} |")
for k, nm in ((16384, "Magicoder + FR-Spec fill to 16,384"), (32768, "Magicoder + FR-Spec fill to 32,768")):
    arr = np.array(LISTS[nm], np.int32)
    p = os.path.join(D, f"lmsl_code{k // 1024}k.npy")
    np.save(p, arr)
    say(f"wrote {p}: {arr.size} ids (a multiple of 32: {arr.size % 32 == 0}), sha256 {hashlib.sha256(arr.tobytes()).hexdigest()[:12]}")
open(os.path.join(D, "v1_lmsl_lists.out"), "w").write("\n".join(lines) + "\n")
