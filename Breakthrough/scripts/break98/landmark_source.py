"""landmark_source.py (Breakthrough, Tue 17 Nov 2026 07:5x; BREAK98 LANDMARK-ANS, docs/IDEAS_1117_landmark.md): the
landmark lookup as a drop-in state source for a head screen (e.g. Sandbox98's SIB harness). Pure Python + numpy.

For a drafted position p at depth d (true states exist only up to the anchor a = p - d), the landmark is the most
recent verified position q <= a whose last k tokens equal p's last k tokens, longest k first (8, 6, 4, 3, 2).
answer_only=True (LANDMARK-ANS) searches answer positions only (q >= gen): the model's state while restating the
prompt differs from its state while reading it. Tokens are the drafted path's, which equal the true text along a
walk-strict path.

    idx = LandmarkIndex(ids, gen, answer_only=True)   # ids: the token ids of the window (prompt + answer)
    q, k = idx.lookup(path_ids, p, anchor)           # path_ids: tokens up to p on the drafted path (len >= p + 1)
    if q is not None: state_in = true_state[q]       # (or a w-mix with the carried state)

Self-test: python -I -B landmark_source.py (checks against a brute-force search on random token streams)."""
import bisect

KS = (8, 6, 4, 3, 2)


class LandmarkIndex:
    def __init__(self, ids, gen, answer_only=True, ks=KS):
        self.ks = tuple(ks)
        self.lo = int(gen) if answer_only else 0
        ids = [int(t) for t in ids]
        self.tab = {k: {} for k in self.ks}
        for k in self.ks:
            for q in range(max(k - 1, self.lo), len(ids)):
                self.tab[k].setdefault(tuple(ids[q - k + 1:q + 1]), []).append(q)

    def lookup(self, path_ids, p, anchor):
        """(q, k) for the most recent q <= anchor whose last k tokens equal path_ids[p-k+1 .. p], longest k first; or
        (None, 0). Positions in the index come from the window's TRUE text, which is the verified text up to the
        anchor."""
        for k in self.ks:
            if p - k + 1 < 0:
                continue
            L = self.tab[k].get(tuple(int(t) for t in path_ids[p - k + 1:p + 1]))
            if not L:
                continue
            j = bisect.bisect_right(L, anchor) - 1
            if j >= 0:
                return L[j], k
        return None, 0


def _selftest():
    import random
    rnd = random.Random(20261117)
    for trial in range(200):
        n, gen = rnd.randint(30, 120), rnd.randint(0, 20)
        ids = [rnd.randint(0, 6) for _ in range(n)]
        for ans in (True, False):
            idx = LandmarkIndex(ids, gen, answer_only=ans)
            for p in range(n):
                for d in (1, 4, 8):
                    a = p - d
                    got = idx.lookup(ids, p, a)
                    want = (None, 0)
                    for k in KS:
                        if p - k + 1 < 0:
                            continue
                        c = [q for q in range(max(k - 1, gen if ans else 0), a + 1) if ids[q - k + 1:q + 1] == ids[p - k + 1:p + 1]]
                        if c:
                            want = (max(c), k)
                            break
                    assert got == want, (trial, ans, p, d, got, want)
    print("landmark_source self-test: PASS (200 random streams, both modes, d = 1 / 4 / 8)")


if __name__ == "__main__":
    _selftest()
