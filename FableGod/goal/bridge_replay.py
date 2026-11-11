"""BRIDGE replay (GOAL, Thu 5 Nov 2026): what does the draft's tail look like after the first miss?
CPU only, numpy, python -I, < 0.5 GB. Data: ~/TrainLab/data/d02b_walk_d23h_dfl_<dom>.npz (DFlash2 L1-20 drafts at EVERY
position, teacher-forced along the 125B's own tokens: dtok = the drafter's own chain picks, dcf their confidences).
Convention (d02b_walk.py): a check at anchor a drafts tokens a+1..a+15; checks = [accepted, checkable, anchor].

Rules replayed on the official loop (next anchor = a + kept):
  base      prefix acceptance (kept = first miss + 1), with the default's 0.3 running-confidence cut
  bridge-m  after a miss the 125B's own token is committed (the bonus) and the SAME check goes on through the draft's
            tail at the same alignment, up to m bridged misses; ceiling = a perfect acceptor (tail token kept iff it
            equals the 125B's token). Output stays the exact text; the lossy part is the context the tail rows saw.
Also: tail anatomy after the first miss (same alignment / shift +1 / shift -1), tail precision by drafter cf.
"""
import sys
import numpy as np

D = "/Users/kamen/TrainLab/data/"
DOMS = sys.argv[1:] or ["code", "mbpp", "math", "ifeval", "multi"]
COST = {"today (62 + 2.8/row)": (62.0, 2.8), "floor engine (30 + 1.5/row)": (30.0, 1.5)}
CUT = 0.3


def load(dom):
    z = np.load(D + f"d02b_walk_d23h_dfl_{dom}.npz", allow_pickle=False)
    out = []
    for s in range(len(z["names"])):
        out.append((z[f"{s}_ids"], int(z[f"{s}_off"]), int(z[f"{s}_gen"]), z[f"{s}_checks"], z[f"{s}_dtok"], z[f"{s}_dcf"]))
    return out


def rows_cut(cf, n, t=CUT):
    """the default's cut: verify drafts while the running confidence (cumprod) stays >= t"""
    run = np.cumprod(cf[:n].astype(np.float64))
    bad = np.nonzero(run < t)[0]
    return int(bad[0]) if len(bad) else n


def main():
    for dom in DOMS:
        try:
            S = load(dom)
        except Exception as e:
            print(dom, "ERR", e)
            continue
        # sanity: my indexing reproduces the stored accepted counts
        okc = tot = 0
        for ids, off, gen, ch, dt, cf in S:
            for (k, m, a), d in zip(ch, dt):
                true = ids[a + 1:a + 1 + m]
                mm = np.nonzero(d[:m] != true)[0]
                kk = int(mm[0]) if len(mm) else int(m)
                okc += (kk == k)
                tot += 1
        print(f"\n=== {dom}: {len(S)} samples, {tot} every-position checks; indexing check {okc}/{tot}")

        # ---- tail anatomy after the first miss (every-position checks with a miss at k <= 11 and 15 checkable)
        same = []; sp = []; sm = []; nxt = []
        prec = {}   # (cf band) -> [right, n] for tail tokens after the first miss, same alignment
        dist = {}   # distance after the miss -> [right, n]
        for ids, off, gen, ch, dt, cf in S:
            for (k, m, a), d, c in zip(ch, dt, cf):
                if m < 15 or k > 11:
                    continue
                true = ids[a + 1:a + 16]
                eq = (d == true)
                # same alignment: run of right tokens right after the miss
                r = 0
                while k + 1 + r < 15 and eq[k + 1 + r]:
                    r += 1
                same.append(r)
                nxt.append(bool(eq[k + 1]))
                # shift +1: the draft dropped a token (truth has one more): d[k+j] == true[k+1+j]
                r = 0
                while k + 1 + r < 15 and d[k + r] == true[k + 1 + r]:
                    r += 1
                sp.append(r)
                # shift -1: the draft has an extra token: d[k+1+j] == true[k+j]
                r = 0
                while k + 1 + r < 15 and d[k + 1 + r] == true[k + r]:
                    r += 1
                sm.append(r)
                for j in range(k + 1, 15):
                    b = ">=.999" if c[j] >= 0.999 else ">=.99" if c[j] >= 0.99 else ">=.9" if c[j] >= 0.9 else ">=.5" if c[j] >= 0.5 else "<.5"
                    x = prec.setdefault(b, [0, 0]); x[0] += int(eq[j]); x[1] += 1
                    x = dist.setdefault(min(j - k, 6), [0, 0]); x[0] += int(eq[j]); x[1] += 1
        same = np.array(same); sp = np.array(sp); sm = np.array(sm)
        n = len(same)
        print(f"  first-miss checks analysed: {n}")
        print(f"  tail right at the SAME alignment: next token right {np.mean(nxt):.3f}; run >=2 {np.mean(same >= 2):.3f}, >=4 {np.mean(same >= 4):.3f}; mean run {same.mean():.2f}")
        print(f"  shift +1 (draft dropped a token): run >=2 {np.mean(sp >= 2):.3f}, >=4 {np.mean(sp >= 4):.3f}; mean {sp.mean():.2f}")
        print(f"  shift -1 (draft has an extra token): run >=2 {np.mean(sm >= 2):.3f}, >=4 {np.mean(sm >= 4):.3f}; mean {sm.mean():.2f}")
        best = np.maximum(same, np.maximum(sp, sm))
        print(f"  best of the three alignments: run >=2 {np.mean(best >= 2):.3f}, >=4 {np.mean(best >= 4):.3f}; mean {best.mean():.2f}")
        print("  tail-token precision after the first miss (same alignment), by distance: " +
              ", ".join(f"+{j}{'+' if j == 6 else ''}: {v[0] / v[1]:.3f}" for j, v in sorted(dist.items())))
        print("  ... by the drafter's cf: " + ", ".join(f"{b}: {v[0] / v[1]:.4f} (n {v[1]}, share {v[1] / sum(x[1] for x in prec.values()):.2f})"
                                                        for b, v in sorted(prec.items(), key=lambda kv: -kv[1][1])))

        # ---- the official loop under each rule
        def loop(rule, m_bridge=0, cfgate=None, block=15):
            kept = rows = checks = flips = toks = 0
            for ids, off, gen, ch, dt, cf in S:
                idx = {int(a): i for i, (k, m, a) in enumerate(ch)}
                a = int(ch[:, 2].min())
                while a in idx:
                    i = idx[a]
                    m = min(int(ch[i, 1]), block)
                    if m <= 0:
                        break
                    d = dt[i]; c = cf[i]
                    true = ids[a + 1:a + 1 + m]
                    R = rows_cut(c, m) if rule != "nocut" else m
                    R = max(R, 1)
                    eq = (d[:R] == true[:R])
                    miss = np.nonzero(~eq)[0]
                    if m_bridge == 0:
                        k = int(miss[0]) if len(miss) else R
                        got = k + 1
                    elif cfgate is None:      # perfect acceptor (ceiling)
                        # a tail token can only be kept if the row BEFORE it had a right input, or was the miss row
                        # itself (its input is the wrong draft, the one lossy step); a miss right after a miss ends
                        # the check with no bonus (that row's input was wrong, its output is not the 125B's token).
                        if len(miss) == 0:
                            got = R + 1
                        else:
                            j = int(miss[0]); b = 0; got = j + 1
                            while b < m_bridge:
                                if j + 1 >= R or not eq[j + 1]:
                                    break             # the token after the miss is wrong too: nothing to keep
                                nx = miss[miss > j]
                                j2 = int(nx[0]) if len(nx) else R
                                got = j2 + 1          # tail kept up to the next miss + that row's (polluted) bonus
                                if j2 >= R:
                                    break
                                j = j2; b += 1
                    else:
                        # draft-cf acceptor: after the first miss keep tail tokens while cf >= gate; a wrong kept token is a flip
                        if len(miss) == 0:
                            got = R + 1
                        else:
                            k = int(miss[0]); j = k + 1; b = 1
                            while j < R:
                                if c[j] < cfgate:
                                    break
                                if not eq[j]:
                                    # the engine's row would have to catch it; here: a false accept is counted, then stop
                                    flips += 1
                                    break
                                j += 1
                            got = j + 1 if j < R else R + 1
                            got = max(got, k + 1)
                    got = min(got, m + 1) if m < 15 else got
                    kept += got; rows += R + 1; checks += 1
                    a += got
                toks += 1
            return kept, rows, checks, flips

        base = loop("cut")
        print(f"  official loop, 0.3 cut: kept per check {base[0] / base[2]:.2f}, rows per check {base[1] / base[2]:.2f} ({base[2]} checks)")
        res = {"base": base,
               "bridge-1 ceiling": loop("cut", 1), "bridge-2 ceiling": loop("cut", 2), "bridge-all ceiling": loop("cut", 99),
               "bridge-1, no cut": loop("nocut", 1), "bridge-all, no cut": loop("nocut", 99), "base, no cut": loop("nocut")}
        for name, (F, r) in COST.items():
            t0 = base[0] / (base[2] * F + base[1] * r) * 1000
            print(f"  cost model {name}: base {t0:.0f} tok/s")
            for k, (kp, rw, ck, fl) in res.items():
                if k == "base":
                    continue
                t = kp / (ck * F + rw * r) * 1000
                extra = f"; false accepts {fl} = {100 * fl / kp:.2f} % of kept tokens" if "acceptor" in k else ""
                print(f"    {k:38s} kept {kp / ck:6.2f} rows {rw / ck:5.2f}  tok/s x{t / t0:.3f}{extra}")


if __name__ == "__main__":
    main()
