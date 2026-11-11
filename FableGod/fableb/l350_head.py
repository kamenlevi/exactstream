"""l350_head.py (Fable B, MISSION LOSSY350-B): what a SCREENED VERIFY HEAD would flip. CPU, numpy, seconds.
The 125B's lm_head restricted to Leap's FR-Spec list (~/Leap/data/frspec_rank.npy: token ids by rank) + the session's own
ids (prompt + everything generated so far), as the MTP chain's ScreenedHead already does [K TRIED 751]. Subset logits are
bitwise equal to the full head's, so the verify's argmax is wrong only when the 125B's token is OUTSIDE the list: that
rate is the head's flip rate. Data: MTP Head's exact walks (~/TrainLab/data/d02b_walk*_{set}.npz: per sample the ids and
the index where the 125B's answer starts).  Run: nice -n 15 python -I l350_head.py > l350_head.out"""
import os, numpy as np
rank = np.load(os.path.expanduser("~/Leap/data/frspec_rank.npy"))
D = os.path.expanduser("~/TrainLab/data")
SETS = (("code (HumanEval+ 24)", "d02b_walk_code.npz"), ("mbpp", "d02b_walk_m_base32_mbpp.npz"), ("IFEval", "d02b_walk_d02a_ifeval.npz"),
        ("multi", "d02b_walk_d02a_multi.npz"), ("math (mgsm)", "d02b_walk_d02a_mgsm.npz"), ("agent", "d02b_walk_d02a_agent.npz"))
print("%-22s | %7s | %6s | " % ("set", "samples", "tokens") + " | ".join("%-22s" % ("top %d: list / + session" % n) for n in (8192, 16384, 32768, 65536)))
for tag, f in SETS:
    p = os.path.join(D, f)
    if not os.path.exists(p):
        print("%-22s | missing %s" % (tag, f)); continue
    z = np.load(p, allow_pickle=True)
    n = len(z["names"])
    out = []
    tot = 0
    res = {N: [0, 0] for N in (8192, 16384, 32768, 65536)}
    for i in range(n):
        ids = z["%d_ids" % i].astype(np.int64); g = int(z["%d_gen" % i])
        ans = ids[g:]
        tot += len(ans)
        for N in res:
            inl = np.zeros(int(max(rank.max(), ids.max())) + 1, bool); inl[rank[:N]] = True
            miss_list = ~inl[ans]
            seen = set(ids[:g].tolist()); miss_sess = 0
            for t, m in zip(ans.tolist(), miss_list.tolist()):
                if m and t not in seen:
                    miss_sess += 1
                seen.add(t)
            res[N][0] += int(miss_list.sum()); res[N][1] += miss_sess
    print("%-22s | %7d | %6d | " % (tag, n, tot) + " | ".join("%8.2f %% / %7.3f %%  " % (100 * res[N][0] / tot, 100 * res[N][1] / tot) for N in res))
print("\n(outside-the-list share of the 125B's own answer tokens = the screened verify head's flips; '+ session' adds the prompt and the tokens already generated)")
