"""l350_checks.py (Fable B, MISSION LOSSY350-B): where the fixed cost per check goes, and what per-CHECK gating can reach.
CPU, numpy, seconds. Data: Breakthrough's DFlash2 lattices (~/Breakthrough/data/trees/lattice_{code,chat}.npz; one block =
one check of the exact walk: 15 drafted depths, the selector's pick + its probability, the 125B's true tokens; code = 664
checks of the 24 HumanEval+ windows, chat = 6,134 checks of IFEval + multi).
Per check: a = accepted drafts (longest true prefix of the picks, inside the verified rows), kept = a + 1 (the bonus);
V = drafted rows verified under the live cut (rows while the running product of the pick's probability >= 0.3, >= 1).
Cost model per check, FAST code at 45 GB, tonight's engine [FULL.md table, 'tonight + Thu']: fixed F = dense 28 + draft 6 +
commit 5 = 39 ms (flat in rows [K C0]); per verified row c = (experts 22 + SSD 12) / 9.84 rows = 3.46 ms.
Questions: (1) kept histogram and the share of time that is FIXED cost paid by short checks; (2) all-sure checks (every
drafted row's cf >= tau): share, precision, the flips of trusting them; (3) the ceiling of a doubt-triggered pass (an all-sure
block is emitted with no pass; its rows ride the next pass at D32's x1.24 dense [K TRIED 478]); (4) top-16 candidate recall.
Run: nice -n 15 python -I l350_checks.py > l350_checks.out"""
import os, numpy as np
D = os.path.expanduser("~/Breakthrough/data/trees")
F_DENSE, F_DRAFT, F_COMMIT = 28.0, 6.0, 5.0
F = F_DENSE + F_DRAFT + F_COMMIT
C_ROW = (22.0 + 12.0) / 9.84
T_CUT = 0.3


def load(name):
    z = np.load(os.path.join(D, name), allow_pickle=True)
    cand, prob, pick, true, nt, dom, samp = (z[k] for k in ("sel_cand", "sel_prob", "pick", "true", "ntrue", "dom", "sample"))
    idx = (cand == pick[:, :, None]).argmax(2)
    has = (cand == pick[:, :, None]).any(2)
    cf = np.take_along_axis(prob.astype(np.float64), idx[:, :, None], 2)[:, :, 0]
    cf = np.where(has, cf, prob.astype(np.float64).max(2))
    return cand, cf, pick, true, nt.astype(int), dom, samp


def stats(tag, cand, cf, pick, true, nt, sel):
    cand, cf, pick, true, nt = cand[sel], cf[sel], pick[sel], true[sel], nt[sel]
    B, DEP = pick.shape
    valid = np.arange(DEP)[None, :] < nt[:, None]
    hit = (pick == true) & valid
    # live cut: rows verified while cumprod(cf) >= T_CUT (at least one), never past the sample's end
    cp = np.cumprod(cf, 1)
    V = np.maximum((cp >= T_CUT).sum(1), 1)
    V = np.minimum(V, np.maximum(nt, 1))
    first_miss = np.where(hit.all(1), DEP, (~hit).argmax(1))           # first depth that is wrong (or past the end)
    a_full = np.minimum(first_miss, nt)                                 # accepted drafts with all 15 rows verified
    a = np.minimum(a_full, V)                                           # accepted drafts under the live cut
    kept, rows = a + 1, V + 1                                           # + bonus token; + the bonus row
    T = F + C_ROW * rows
    print("\n==== %s: %d checks; rows per check %.2f (verified drafts %.2f + 1), kept %.2f (16-row no-cut kept %.2f); "
          "model cycle %.1f ms -> %.0f tok/s ====" % (tag, B, rows.mean(), V.mean(), kept.mean(), (a_full + 1).mean(), T.mean(), 1000 * kept.sum() / T.sum()))
    print("  per-row pick accuracy by depth 1..15: " + " ".join("%.2f" % (hit[:, d].sum() / max(valid[:, d].sum(), 1)) for d in range(DEP)))
    print("  (1) kept histogram and who pays the fixed cost (F = %.0f ms of a %.1f ms mean check)" % (F, T.mean()))
    print("      %-14s | %7s | %8s | %9s | %12s | %10s" % ("checks with", "share", "tokens", "time", "fixed time", "ms / token"))
    for lo, hi in ((1, 2), (3, 4), (5, 8), (9, 15), (16, 16)):
        m = (kept >= lo) & (kept <= hi)
        if m.sum() == 0:
            continue
        print("      kept %2d-%-6d | %6.1f%% | %7.1f%% | %8.1f%% | %11.1f%% | %10.2f" % (
            lo, hi, 100 * m.mean(), 100 * kept[m].sum() / kept.sum(), 100 * T[m].sum() / T.sum(), 100 * F * m.sum() / T.sum(), T[m].sum() / kept[m].sum()))
    print("      all checks: fixed cost = %.1f %% of the time; a check that keeps <= 4 costs %.1f ms per token, one that keeps 16 costs %.2f"
          % (100 * F * B / T.sum(), T[kept <= 4].sum() / max(kept[kept <= 4].sum(), 1), T[kept == 16].sum() / max(kept[kept == 16].sum(), 1)))
    # (2) all-sure checks: every drafted row's cf >= tau (blocks with all 15 depths inside the sample)
    full = nt >= DEP
    print("  (2) all-sure checks (all 15 drafted rows cf >= tau; %d of %d checks have 15 rows inside the sample)" % (full.sum(), B))
    print("      %-6s | %9s | %12s | %13s | %12s | %15s | %14s" % ("tau", "share", "all 15 right", "row error", "token share", "flips if trusted", "sure-to-the-cut"))
    res = {}
    for tau in (0.5, 0.8, 0.9, 0.95, 0.99, 0.999):
        s = full & (cf.min(1) >= tau)
        ns = s.sum()
        allr = hit[s].all(1).mean() if ns else float("nan")
        err = 1 - hit[s].mean() if ns else float("nan")
        # sure up to the live cut: every verified drafted row has cf >= tau
        vm = np.arange(DEP)[None, :] < V[:, None]
        sc = np.where(vm, cf, 1.0).min(1) >= tau
        res[tau] = s
        print("      %-6g | %8.1f%% | %11.1f%% | %12.2f%% | %11.1f%% | %14.3f%% | %13.1f%%" % (
            tau, 100 * s.mean(), 100 * allr, 100 * err, 100 * (15 * ns) / kept.sum() if ns else 0.0,
            100 * (~hit[s]).sum() / kept.sum() if ns else 0.0, 100 * sc.mean()))
    # (3) ceiling of a doubt-triggered pass: an all-sure block is emitted with no pass (15 tokens, no bonus); its 15 rows
    # ride the NEXT pass (dense x1.24 of 28 [K D32], rows at half the row price [G: sure rows, strict top-2 + members]);
    # the drafter is assumed to draft the next block as well as today WITHOUT the 125B's features of the skipped block
    # (optimistic: DFlash2 reads the 125B's layer taps; DEPTH32 says deep rows decay) -> this is a ceiling.
    print("  (3) doubt-triggered pass, ceiling (no drafter decay; deferred rows at x1.24 dense and half the row price)")
    base = 1000 * kept.sum() / T.sum()
    for tau in (0.9, 0.95, 0.99, 0.999):
        s = res[tau]
        tok = np.where(s, 15, kept)
        tim = np.where(s, F_DRAFT + 0.24 * F_DENSE + 15 * C_ROW / 2, T)
        print("      tau %-6g: skipped %.1f %% of passes; %.0f -> %.0f tok/s (x%.3f); flips %.3f %% of tokens" % (
            tau, 100 * s.mean(), base, 1000 * tok.sum() / tim.sum(), (tok.sum() / tim.sum()) / (kept.sum() / T.sum()),
            100 * (~hit[s]).sum() / tok.sum()))
    # (4) screened verify head: is the 125B's token among the drafter's 16 candidates at that depth?
    in16 = (cand == true[:, :, None]).any(2) & valid
    in1 = hit
    print("  (4) the 125B's token is the pick on %.1f %% of rows, in the 16 candidates on %.1f %% (so a head screened to the candidates alone would flip %.1f %% of rows)"
          % (100 * in1.sum() / valid.sum(), 100 * in16.sum() / valid.sum(), 100 * (1 - in16.sum() / valid.sum())))
    # (5) the width law: tokens per ms if checks were 2x / 4x wider at the same kept per row (dense x1.24 per doubling)
    return kept, T


for name in ("lattice_code.npz", "lattice_chat.npz"):
    cand, cf, pick, true, nt, dom, samp = load(name)
    doms = sorted(set(dom.tolist()))
    groups = [("code (all)", np.ones(len(dom), bool))] if "code" in name else [("chat: " + d, dom == d) for d in doms]
    for tag, sel in groups:
        stats(tag, cand, cf, pick, true, nt, sel)
