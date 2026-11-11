"""Where the ~17 ms between verify passes goes (data/pass_profile125.json, frac 0.5; clock shared by the loader trace and
Python): per pass, t_rel[47] (last layer's experts released) -> the MTP draft steps (Python-side timestamps of
mtp.fuse_and_run, seq-tagged) -> the next pass's t_seen[0] (GPU reached layer 0's gate). Splits:
  a  t_rel[47] -> first draft step: layer 47's MoE, final norm + lm_head, sampling/acceptance, MTP history append
  b  first -> last draft step (draft steps 1..2 incl. their syncs)
  c  last draft step -> next t_seen[0]: the last draft step, verify launch, layer 0 attention + router
usage: interpass125.py   (CPU, instant)"""
import json, collections
import numpy as np
d = json.load(open("/Users/kamen/MoonShots/data/pass_profile125.json"))
for run in d["EV"]:
    by = collections.defaultdict(dict)
    for L, seq, kind, ts, tr in run["events"]:
        by[int(seq)][int(L)] = (ts, tr)
    dr = collections.defaultdict(list)
    for seq, t0 in run["drafts"]:
        dr[int(seq)].append(t0)
    seqs = sorted(s for s in by if len(by[s]) == 48)
    rows = []
    for s, s2 in zip(seqs, seqs[1:]):
        t47, t0n = by[s][47][1], by[s2][0][0]
        ds = sorted(t for k in (s, s + 1) for t in dr.get(k, []) if t47 <= t <= t0n)
        if not ds:
            continue
        rows.append((ds[0] - t47, ds[-1] - ds[0], t0n - ds[-1], t0n - t47, len(ds)))
    a = np.array(rows) * 1e3
    print(f"{run['prompt']}: {len(rows)} passes, draft steps/pass {np.mean(a[:, 4] / 1e3):.1f}; inter-pass {a[:, 3].mean():.1f} ms = "
          f"a {a[:, 0].mean():.1f} (verify tail: L47 MoE, lm_head, sampling, acceptance, history) + b {a[:, 1].mean():.1f} "
          f"(between draft steps) + c {a[:, 2].mean():.1f} (last draft, launch, L0 attention); p90 total {np.percentile(a[:, 3], 90):.1f}")
