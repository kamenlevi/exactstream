"""SPLIT-THRESHOLD CHAT ROUTE, BYTES SIDE (NewIdeas, Wed 4 Nov 2026 22:4x; CPU only, one process). The cf 0.5 chat route
PASSES at 147 slots (0.71 / 4.30 / 2.73 %, ~88 MB on chat); the cf 0.3 route (49-52 MB) missed chat by +0.14 / +0.17
points. They differ in two ways:
  (a) drafted rows with cf in [0.3, 0.5) are sure at 0.3, exact at 0.5;
  (b) bonus rows whose next draft has cf in [0.3, 0.5) are gated at 0.3, exact at 0.5.
This prices the two hybrids so a quality arm is only spent if one is cheap:
  V1 = drafted rows at 0.3 (= cf 0 on the live map) + the bonus row gated at next cf >= 0.5;
  V2 = drafted rows at 0.5 + the bonus row gated at next cf >= 0.3.
Same replay as chat_route_price.py (24 GB, 11-bit parts, 4-bit sure rows, best RAM split); the bonus row's threshold is
applied by giving it cf 1.0 (sure) if its next draft's cf >= tb, else -1.0 (exact). The replay uses cf only for sure /
unsure, so this is exact for the rule. Checks: (0.5, 0.5) = 87.9 / 87.2 and (0.3, 0.3) = 49.2 / 51.7 on chat.
Pre-registered (22:4x, before the run): a hybrid is worth a map arm only if its chat bytes are <= 0.75x the cf 0.5
route's on both IFEval and multi (<= ~66 MB); otherwise the split dies on bytes.
Writes ~/NewIdeas/data/ledger/split_price.txt."""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ten_x_ledger as T  # noqa: E402
import ledger_lastrow as LL  # noqa: E402
import chat_route_price as CR  # noqa: E402
DOMS = ("code", "ifeval", "multi")
VARS = [("cf 0.5 route (check)", 0.5, 0.5), ("cf 0.3 route (check)", 0.3, 0.3), ("V1 drafted 0.3 + bonus 0.5", 0.3, 0.5),
        ("V2 drafted 0.5 + bonus 0.3", 0.5, 0.3), ("drafted 0.4 + bonus 0.5", 0.4, 0.5), ("drafted 0.3 + bonus 0.4", 0.3, 0.4),
        ("drafted 0.3 + bonus never gated", 0.3, 9.0)]


def blocks_split(w, tb):
    """LL.blocks_conv(w, 'engine') with the last row's cf replaced by 1.0 / -1.0 for its next draft's cf >= tb."""
    ids, g = w["ids"], w["g"]; N = len(ids); bl = []
    for (k0_, m0_, A0_), dc0_ in zip(w["checks"], w["dcf"]):
        for (k, m, A, dc, nxt) in LL.segments_nxt(k0_, m0_, A0_, dc0_):
            last, fm = min(A + m, N - 1), min(A + k, N - 1)
            rows, cfs = [], []
            for p in range(A, last + 1):
                is_last = p - A == m
                if p <= fm:
                    rows.append(p); c = -1.0 if (k == m and is_last) else g[p]
                else:
                    rows.append((p + 97) % N); c = dc[p - A] if p - A < m else -1.0
                if is_last: c = 1.0 if nxt >= tb else -1.0
                cfs.append(c)
            if rows: bl.append((rows, cfs, min(k, fm - A) + 1))
    return bl


def job(dom, cl, thr, tb):
    cm = (CR.B24 - cl * CR.LR11) // CR.MR11
    T.MR, T.LR = CR.MR11, CR.LR11
    T.POOLS["x"] = ([cm] * 48, [cl] * 48)
    T.live_blocks = lambda w: blocks_split(w, tb)
    _, mbk, _, _, _ = T.replay((dom, "x", "live", (thr, False)))
    return mbk


def main():
    t0 = time.time(); T.W = T.windows()
    L = ["SPLIT-THRESHOLD CHAT ROUTE, BYTES: drafted rows sure at cf >= t, the bonus row gated at next cf >= tb; 24 GB, 11-bit "
         "parts, 4-bit sure rows; SSD MB per kept token in decode [P], best split per domain",
         "| variant | t | tb | code | IFEval | multi | chat x the cf 0.5 route (IFEval / multi) |", "|---|---|---|---|---|---|---|"]
    ref = None; res = {}
    for name, thr, tb in VARS:
        b = {d: min((job(d, cl, thr, tb), cl) for cl in CR.CLS) for d in DOMS}; res[name] = b
        if ref is None: ref = b
        L.append(f"| {name} | {thr} | {tb if tb < 2 else 'never'} | " + " | ".join(f"{b[d][0]:.1f} (Cl {b[d][1]})" for d in DOMS) +
                 f" | {b['ifeval'][0] / ref['ifeval'][0]:.2f} / {b['multi'][0] / ref['multi'][0]:.2f} |")
    for name in ("V1 drafted 0.3 + bonus 0.5", "V2 drafted 0.5 + bonus 0.3"):
        x = [res[name][d][0] / ref[d][0] for d in ("ifeval", "multi")]
        L.append(f"pre-registered read, {name}: chat x{x[0]:.2f} / x{x[1]:.2f} -> {'WORTH AN ARM' if max(x) <= 0.75 else 'DIES ON BYTES'}")
    L.append(f"({time.time() - t0:.0f} s)")
    open(os.path.join(T.DST, "split_price.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L))


if __name__ == "__main__":
    main()
