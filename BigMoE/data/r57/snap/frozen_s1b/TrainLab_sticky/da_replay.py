"""DA-0, the DRAFT-AHEAD replay (MTP Head, Wed 11-04 15:2x; the orchestrator's item 2 on Kamen's 300 tok/s FAST priority;
BREAK_THE_WALL 4.2): how often a next block pre-drafted during check i would be the one actually needed, on the engine's
cut chain. Data: the TOP16 walks data/d02b_walk_kw_{set}.npz (L1-20 at HPD-0's anchors, block 16): per check (k, m, A), own
picks dtok and their cf dcf per depth, the 16 candidates cand and the teacher-forced selector softmax tsm.
Per check (CPU only):
  segment c = the leading drafts whose running cf product stays >= T (0.3: the engine's L1-14 cut);
  the 125B's first miss j* among them (own picks = teacher-forced picks up to it);
  the check ends at the segment end if no miss in 1..c (next anchor = the 125B's token at depth c + 1, guessed by the own
  draft there; at c = 15 no draft exists today, DRAFT17's extra position would supply it), else at the miss depth j* (next
  anchor = the 125B's correction there, guessed by the selector's 2nd-ranked candidate);
  pre-drafted prefixes: the segment end + the two lowest-cf depths inside the segment.
  hit = the check ends at a pre-drafted prefix AND the anchor guess is right (then the pre-drafted block = the serial one).
Output: data/da_replay.txt (appended)."""
import os, time
import numpy as np
TD = os.path.expanduser("~/TrainLab/data")
T = float(os.environ.get("DA_T", "0.3"))
lines = []
def say(s=""):
    print(s, flush=True); lines.append(s)


say(f"DA-0 {time.strftime('%m-%d %H:%M')}: DRAFT-AHEAD hit rate on the cut chain (t {T}), L1-20 at HPD-0's anchors, block 16")
for s in ("code", "ifeval", "multi"):
    Z = np.load(os.path.join(TD, f"d02b_walk_kw_{s}.npz"))
    n = 0
    C = dict(seg_end=0, seg_end_ok=0, seg_full=0, low=0, low_ok=0, other=0)
    kept = []
    for w in range(len(Z["names"])):
        ids, off = Z[f"{w}_ids"], int(Z[f"{w}_off"])
        ck, cand, tsm = Z[f"{w}_checks"], Z[f"{w}_cand"], Z[f"{w}_tsm"].astype(np.float32)
        dtok, dcf = Z[f"{w}_dtok"].reshape(len(ck), -1), Z[f"{w}_dcf"].reshape(len(ck), -1).astype(np.float64)
        for c_, (k, m, A) in enumerate(ck):
            k, m, A = int(k), int(m), int(A)
            D = min(15, m, len(ids) - (A - off) - 1)
            if D < 1:
                continue
            t = ids[A - off + 1:A - off + 1 + D]
            run = np.cumprod(dcf[c_, :D])
            c = int(np.sum(run >= T)) if run[0] >= T else 0       # leading drafts with running product >= T
            right = dtok[c_, :D] == t
            js = int(np.argmin(right)) + 1 if not right.all() else D + 1     # first miss depth (1-based)
            n += 1
            if js > c:                                    # the whole segment accepted: ends at the segment end
                C["seg_end"] += 1
                kept.append(c + 1)
                if c >= 15:
                    C["seg_full"] += 1                    # no draft past the block today (DRAFT17 would add one)
                elif c < D and dtok[c_, c] == t[c]:
                    C["seg_end_ok"] += 1
                continue
            kept.append(js)
            seg_cf = dcf[c_, :c]
            low = set((np.argsort(seg_cf, kind="stable")[:2] + 1).tolist())     # the two lowest-cf depths, 1-based
            if js in low:
                C["low"] += 1
                o = np.argsort(-tsm[c_, js - 1], kind="stable")
                if cand[c_, js - 1, o[1]] == t[js - 1]:   # the 2nd-ranked candidate is the 125B's correction
                    C["low_ok"] += 1
            else:
                C["other"] += 1
    at = C["seg_end"] + C["low"]
    hit = C["seg_end_ok"] + C["low_ok"]
    say(f"  {s}: {n} checks, kept {np.mean(kept):.2f} per check on the cut chain; ends at a pre-drafted prefix {100 * at / n:.1f} % "
        f"(segment end {100 * C['seg_end'] / n:.1f} %, of which a full block {100 * C['seg_full'] / n:.1f} %; at one of the two "
        f"lowest-cf depths {100 * C['low'] / n:.1f} %; elsewhere {100 * C['other'] / n:.1f} %)")
    say(f"    anchor guess right: segment end {100 * C['seg_end_ok'] / max(1, C['seg_end'] - C['seg_full']):.1f} % (draft past the "
        f"cut), lowest-cf misses {100 * C['low_ok'] / max(1, C['low']):.1f} % (2nd-ranked candidate)")
    say(f"    HIT (pre-drafted block usable): {100 * hit / n:.1f} % of checks; with DRAFT17's extra draft at full blocks right "
        f"as often as at other segment ends: {100 * (hit + C['seg_full'] * C['seg_end_ok'] / max(1, C['seg_end'] - C['seg_full'])) / n:.1f} %")
open(os.path.join(TD, "da_replay.txt"), "a").write("\n".join(lines) + "\n\n")
