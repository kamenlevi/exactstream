"""LANDMARK-ANS screen 1 on the 2B (Breakthrough's pre-registration, ~/Breakthrough/docs/IDEAS_1117_landmark.md: "Screens"
item 1 + the ADDENDUM (the orchestrator's audit) + the report-only k split; NewIdeas 07:4x-07:5x: G by match length,
KILL-COPY per k). Eval-only on the FROZEN s30 head. LAB = teacher-forced on the 2B, not real decoding.

The lookup is Breakthrough's (~/Breakthrough/scripts/break98/landmark_source.py, imported read-only): for the drafted
position p = s + d (anchor s, band depth d >= 1, i.e. the INPUT state of chain step d + 1), the landmark is the most
recent ANSWER position q <= s whose last k tokens equal p's (k = 8, 6, 4, 3, 2, longest first). Teacher-forced, the path
equals the true text, which is what a walk-strict path is.
Modes (each its own chain, the head frozen):
  own     the carried stand-in (today)
  true    the true W at every drafted depth (the LAB oracle, evalx's 'true')
  lm1     at covered cells the input state := W[q]                      (w 1.0)
  lm05    at covered cells the input state := 0.5 W[q] + 0.5 carried    (w 0.5)
  stale1  at covered cells the input state := W[p - 1], the TRUE state one position back (the ADDENDUM's upper reference)
  copy    (i) own's chain; at covered cells the prediction of ids[p + 2] := ids[q + 2], the token two after the matched
          occurrence (the same match and the same cell as the landmark; a token on the walk-strict path, never unknown)
  copy2   (ii) re-matched lookup decoding at the same covered cells: the context ending at p + 1 (it holds the true
          ids[p + 1]) is looked up again (q' <= s, answer-only, k 8..2) and ids[p + 2] := ids[q' + 1]; no re-match = the
          head's own call. KILL-COPY uses the STRONGER of (i) / (ii) (Breakthrough 08:05, recorded in its doc first)
  copy1   (info) own's chain; the prediction of ids[p + 1] (band d - 1's target) := ids[q + 1], the occurrence's next token
Read-out: walk-strict P(right at d | right at 0..d-1) counted ONLY over covered cells (each mode on its own chain), bands
1-2 / 3-6 / 7-8 / 1-8, G = (mode - own) / (true - own) with a paired window bootstrap (2000, seed 1116); the same by the
longest matched k (8 / 6 / 4 / 3 / 2, and the groups k >= 4 / k <= 3); all-cell walk-strict and whole-walk kept per
32-row check (the coverage priced as it falls); coverage per band; the angle between W[q] and W[p] (share within 0.40
rad, beside MoonShots' 3-6 %), and between W[p - 1] and W[p].
usage: landmark.py --sets he,mbpp [--nh 60 --nm 100] [--head s30] [--depth 31] --out results/landmark_s1.json"""
import argparse, hashlib, importlib.util, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import mlx.core as mx
from sb import core as C

LMSRC = os.path.expanduser("~/Breakthrough/scripts/break98/landmark_source.py")
BANDS = ("1-2", "3-6", "7-8", "1-8", "9-14", "15-30")
KGROUPS = {"k>=4": (8, 6, 4), "k<=3": (3, 2), "k8": (8,), "k6": (6,), "k4": (4,), "k3": (3,), "k2": (2,)}
MODES = ("own", "true", "lm1", "lm05", "stale1")


def load_lm():
    sp = importlib.util.spec_from_file_location("bt_landmark_source", LMSRC)
    m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m)
    return m, hashlib.sha256(open(LMSRC, "rb").read()).hexdigest()


def tables(LM, ids, gen, a0, Dp):
    """Qt [n, Dp] the landmark q for the input state of band d (p = s + d), -1 = none; Kt the matched k; Q2 [n, Dp] the
    re-matched occurrence q' for the context ending at p + 1 (copy (ii)), -1 = none"""
    S = len(ids); n = S - 1 - a0
    idx = LM.LandmarkIndex(ids, gen, answer_only=True)
    Qt = np.full((n, Dp), -1, np.int64); Kt = np.zeros((n, Dp), np.int8); Q2 = np.full((n, Dp), -1, np.int64)
    for i in range(n):
        s = a0 + i
        for d in range(1, Dp):
            p = s + d
            if p + 2 > S - 1:
                break
            q, k = idx.lookup(ids, p, s)
            if q is not None:
                Qt[i, d] = q; Kt[i, d] = k
                q2, _ = idx.lookup(ids, p + 1, s)
                if q2 is not None:
                    Q2[i, d] = q2
    return Qt, Kt, Q2


def standin_for(mode, Qt, Wf, a0):
    w = {"lm1": 1.0, "lm05": 0.5}.get(mode)

    def fn(j, h, anc):
        d = j - 1
        a = np.array(anc) - a0
        q = Qt[a, d] if d < Qt.shape[1] else np.full(len(a), -1)
        cov = q >= 0
        if not cov.any():
            return h
        m = mx.array(cov.astype(np.float32))[None, :, None]
        if mode == "stale1":
            src = np.where(cov, np.array(anc) + j - 2, 0)          # p - 1 = s + d - 1
            R = Wf[mx.array(src)][None]
            return h * (1 - m) + m * R
        R = Wf[mx.array(np.where(cov, q, 0))][None]
        return h * (1 - w * m) + (w * m) * R
    return fn


def cov_counts(rt, cov):
    """walk-strict counts over the cells in cov [n, Dp] (bool): r[d], v[d] with alive = right at 0..d-1 on rt's chain"""
    r = np.zeros(rt.shape[1], np.int64); v = np.zeros(rt.shape[1], np.int64)
    alive = np.ones(rt.shape[0], bool)
    for d in range(rt.shape[1]):
        ok = alive & (rt[:, d] >= 0) & cov[:, d]
        r[d], v[d] = int((ok & (rt[:, d] == 1)).sum()), int(ok.sum())
        alive &= rt[:, d] == 1
    return r, v


def angle(A, B):
    c = (A * B).sum(-1) / (np.linalg.norm(A, axis=-1) * np.linalg.norm(B, axis=-1) + 1e-9)
    return np.arccos(np.clip(c, -1, 1)), c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", default="he,mbpp"); ap.add_argument("--nh", type=int, default=60); ap.add_argument("--nm", type=int, default=100)
    ap.add_argument("--head", default="s30"); ap.add_argument("--depth", type=int, default=31)
    ap.add_argument("--mem", type=float, default=4.5); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    C.limit_memory(a.mem)
    LM, lmsha = load_lm()
    C.load_target(); C.drop_trunk()
    D = C.load_drafter(a.head)
    t0 = time.time()
    out = {"args": vars(a), "landmark_source_sha256": lmsha, "sets": {}}
    for st in a.sets.split(","):
        fs = C.files(st)[:{"he": a.nh, "mbpp": a.nm}[st]]
        per = {}; ang = {"lm": [], "lm_k": [], "stale1": []}; cover = np.zeros((2, a.depth), np.int64)
        for f in fs:
            C.wait_pause()
            z = np.load(f)
            ids = np.asarray(z["ids"], np.int64); gen = int(z["gen"]); a0 = max(0, gen - 2); S = len(ids)
            Qt, Kt, Q2 = tables(LM, ids, gen, a0, a.depth)
            W = C.load_W(z).astype(mx.float32)[0]; mx.eval(W)
            rts = {}
            for mode in MODES:
                if mode == "own":
                    rt, _, _ = C.chain_table(D, f, a.depth)
                elif mode == "true":
                    rt, _, _ = C.chain_table(D, f, a.depth, R=W[None])
                else:
                    rt, _, _ = C.chain_table(D, f, a.depth, fn=standin_for(mode, Qt, W, a0))
                rts[mode] = rt
                mx.clear_cache()
            n = rts["own"].shape[0]
            cov = Qt[:, :a.depth] >= 0
            valid = rts["own"] >= 0
            # copy (line) and copy1 (info): own's chain, covered predictions replaced
            rc = rts["own"].copy(); rc1 = rts["own"].copy(); rc2 = rts["own"].copy()
            ii, dd = np.nonzero(cov & valid)
            s_ = a0 + ii; p_ = s_ + dd; q_ = Qt[ii, dd]
            rc[ii, dd] = (ids[q_ + 2] == ids[p_ + 2]).astype(np.int8)
            q2 = Q2[ii, dd]; h2 = q2 >= 0                                        # no re-match: the head's own call
            rc2[ii[h2], dd[h2]] = (ids[q2[h2] + 1] == ids[p_[h2] + 2]).astype(np.int8)
            m1 = dd >= 1
            rc1[ii[m1], dd[m1] - 1] = np.where(rts["own"][ii[m1], dd[m1] - 1] >= 0,
                                              (ids[q_[m1] + 1] == ids[p_[m1] + 1]).astype(np.int8), rts["own"][ii[m1], dd[m1] - 1])
            rts["copy"] = rc; rts["copy2"] = rc2; rts["copy1"] = rc1
            key = os.path.basename(f)[:-4]
            per[key] = {"cov": {}, "all": {}, "kept": {}, "copy2_rematch": [int(h2.sum()), int(len(h2))]}
            for g, ks in [("all", None)] + list(KGROUPS.items()):
                cg = cov if ks is None else (cov & np.isin(Kt[:, :a.depth], ks))
                per[key]["cov"][g] = {m: [x.tolist() for x in cov_counts(rt, cg)] for m, rt in rts.items() if m != "copy1"}
            # copy1 acts one band earlier: its cells are the covered cells shifted to d - 1
            sh = np.zeros_like(cov); sh[:, :-1] = cov[:, 1:]
            per[key]["cov"]["all_shift1"] = {m: [x.tolist() for x in cov_counts(rts[m], sh)] for m in ("own", "true", "copy1")}
            for m, rt in rts.items():
                per[key]["all"][m] = [x.tolist() for x in cov_counts(rt, np.ones_like(cov))]
                kk = C.kept_walk(rt, max(0, gen - 1 - a0)); per[key]["kept"][m] = [int(sum(kk)), len(kk)]
            cover[0] += (cov & valid).sum(0); cover[1] += valid.sum(0)
            if len(ii):
                Wn = np.array(W)
                an, _ = angle(Wn[q_], Wn[p_]); ang["lm"] += an.tolist(); ang["lm_k"] += Kt[ii, dd].tolist()
                a1, _ = angle(Wn[p_ - 1], Wn[p_]); ang["stale1"] += a1.tolist()
            print(f"[lm] {st} {key}: n {n}, covered {int((cov & valid).sum())}/{int(valid[:, 1:].sum())} cells (bands >= 1); "
                  f"[{time.time() - t0:.0f}s]", flush=True)
        out["sets"][st] = {"per": per, "coverage_by_band": (cover[0] / np.maximum(cover[1], 1)).tolist(),
                           "angles": {k: (np.array(v).tolist() if k != "lm_k" else v) for k, v in ang.items()}}
    out["wall_s"] = time.time() - t0; out["mlx_peak_gb"] = mx.get_peak_memory() / 1e9
    json.dump(out, open(a.out, "w"))
    print(f"[lm] DONE in {out['wall_s']:.0f}s, peak {out['mlx_peak_gb']:.2f} GB", flush=True)


if __name__ == "__main__":
    main()
