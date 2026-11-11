"""l350_grid.py (Fable B, MISSION LOSSY350-B): the 350 identity on the real check distribution. CPU, numpy, < 1 s.
tok/s = sum(kept) / sum(F + c * rows) over the 664 HumanEval+ checks of Breakthrough's DFlash2 lattice (the live t 0.3 cut;
rows = verified drafts + the bonus row), for a grid of the fixed cost per check F (dense + draft + commit, flat in rows
[K C0]) and the cost per verified row c (experts + exposed SSD). Also: the same grid with a relaxed acceptance stand-in
(each first miss forgiven with probability r, the chain then continues to the next miss or the cut), and the price of
skipping the pass on all-sure blocks with the drafter's measured feature-free decay (0.92 -> 0.80 per row at depths 16-20
[K TrainLab d02b_walk_m_base32.txt]).  Run: nice -n 15 python -I l350_grid.py > l350_grid.out"""
import os, numpy as np
z = np.load(os.path.expanduser("~/Breakthrough/data/trees/lattice_code.npz"), allow_pickle=True)
cand, prob, pick, true, nt = z["sel_cand"], z["sel_prob"].astype(np.float64), z["pick"], z["true"], z["ntrue"].astype(int)
B, DEP = pick.shape
idx = (cand == pick[:, :, None]).argmax(2)
cf = np.take_along_axis(prob, idx[:, :, None], 2)[:, :, 0]
valid = np.arange(DEP)[None, :] < nt[:, None]
hit = (pick == true) & valid
V = np.minimum(np.maximum((np.cumprod(cf, 1) >= 0.3).sum(1), 1), np.maximum(nt, 1))
fm = np.where(hit.all(1), DEP, (~hit).argmax(1))
a = np.minimum(np.minimum(fm, nt), V)
kept, rows = a + 1, V + 1
print("code, 664 checks, live cut: kept %.2f, rows %.2f per check; rows per kept token %.3f" % (kept.mean(), rows.mean(), rows.sum() / kept.sum()))
FS = ((39.0, "tonight: dense 28 + draft 6 + commit 5"), (29.5, "realistic ~Nov 14: 22 + 4.5 + 3"), (23.0, "exact floors: 17 + 3.5 + 2.5"),
      (20.3, "FAST floors: 14.3 + 3.5 + 2.5"), (10.0, "half the FAST floor (no mechanism)"))
CS = ((3.46, "tonight: experts 22 + SSD 12 over 9.84 rows"), (2.24, "experts only (SSD hidden)"), (1.05, "split form x0.47, SSD hidden"),
      (0.69, "split form at the group floor (14.5 x 0.47 / 9.84)"), (0.0, "rows free"))
print("\ntok/s on the real check distribution (kept as measured, %.2f per check)" % kept.mean())
print("%-42s | " % "fixed per check F (ms)" + " | ".join("c %.2f" % c for c, _ in CS))
for Fv, fl in FS:
    print("%-42s | " % ("%.1f  %s" % (Fv, fl)) + " | ".join("%6.0f" % (1000 * kept.sum() / (Fv * B + c * rows.sum())) for c, _ in CS))
for c, cl in CS:
    print("   c %.2f = %s" % (c, cl))
# relaxed acceptance stand-in: each wrong pick inside the verified rows is forgiven with probability r (expected value)
print("\nwith a relaxed-acceptance stand-in (each miss forgiven with probability r; expected kept per check, same rows)")
for r in (0.0, 0.25, 0.5):
    ke = np.zeros(B)
    for b in range(B):
        p, tot = 1.0, 0.0
        for d in range(int(V[b])):
            p *= 1.0 if hit[b, d] else r
            tot += p
        ke[b] = tot + 1
    line = "  r %.2f: kept %.2f | " % (r, ke.mean())
    line += " | ".join("F %.1f c %.2f -> %4.0f" % (Fv, c, 1000 * ke.sum() / (Fv * B + c * rows.sum())) for Fv, c in ((39.0, 3.46), (29.5, 2.24), (20.3, 1.05), (20.3, 0.69)))
    print(line)
# pass skipping on all-sure blocks, with the drafter's feature-free decay priced
print("\npass skipping on all-sure blocks (all 15 cf >= tau), decay priced: the NEXT check's chain runs at 0.80 per row instead of 0.92")
full = nt >= DEP
ch = lambda q: sum(q ** k for k in range(1, 16))
lost = ch(0.92) - ch(0.80)                       # accepted drafts lost on the check after a skipped pass
for Fd, Fdr, Fc, c, tag in ((28.0, 6.0, 5.0, 3.46, "tonight"), (14.3, 3.5, 2.5, 1.05, "FAST floors + split")):
    Fv = Fd + Fdr + Fc
    T = Fv + c * rows
    base = kept.sum() / T.sum()
    for tau in (0.9, 0.95):
        s = full & (cf.min(1) >= tau)
        n = s.sum()
        tok = kept.sum() - (kept[s] - 15).sum() - n * lost
        tim = T.sum() - T[s].sum() + n * (Fdr + 0.24 * Fd + 15 * c / 2) - n * c * lost
        print("  %-20s tau %.2f: %4.1f %% of passes skipped; %5.0f -> %5.0f tok/s (x%.3f); flips 0.%s" % (
            tag, tau, 100 * s.mean(), 1000 * base, 1000 * tok / tim, (tok / tim) / base,
            "18-0.39 %" if tau == 0.9 else "02-0.05 %"))
    s = full & hit.all(1)
    n = s.sum()
    tok = kept.sum() - (kept[s] - 15).sum() - n * lost
    tim = T.sum() - T[s].sum() + n * (Fdr + 0.24 * Fd + 15 * c / 2) - n * c * lost
    print("  %-20s oracle (every fully accepted block): %4.1f %% skipped; x%.3f" % (tag, 100 * s.mean(), (tok / tim) / base))
print("  (chain of 15 at 0.92 keeps %.2f drafts, at 0.80 keeps %.2f: %.2f lost per skipped pass)" % (ch(0.92), ch(0.80), lost))
