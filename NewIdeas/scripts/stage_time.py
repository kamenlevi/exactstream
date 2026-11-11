"""Timing model for frugal + d=1 prefetch with SSD contention (NewIdeas 2026-10-22, CPU, 125B traces, M=1 decode).
SSD: each read = 0.191 ms latency floor, then its 2.77 MB record transfers at 13.1 GB/s SHARED equally by all transfers
in flight (fits MoonShots' QD1/4/6/10 = 0.30/0.80/1.13/1.81 ms for 2.16 MB). GPU per layer: attention half A, router
point, wait for the layer's cold ranks 0-1, MoE half B. Prefetch: the d=1 look-ahead top-k for layer l+1 (known at layer
l's router point) read into a global staging ring if not resident; issued 'now' (with the demand reads) or 'after'
(when layer l's demand reads are done). A staged expert routed at ranks 0-1 is waited for only until it lands; staged
experts routed at ranks 2-9 are computed if they have landed by the router point."""
import glob, sys
import numpy as np
from collections import OrderedDict

import os
NL, REC = 48, 2.77e6
if os.environ.get("SSDV", "2") == "1":
    LAT, BW, B1 = 0.191, 13.1e6, 1e18                 # v1: floor + processor sharing of 13.1 GB/s
else:
    LAT, BW, B1 = 0.06, 14.0e6, 8.1e6                 # v2: floor + min(8.1 GB/s per read, 14.0 GB/s shared)
PF = sorted(glob.glob("/Users/kamen/MoEPrefetch/data/traces125/*_400_pf.npz"))
GG = {f.split("/")[-1].split("_")[0]: f for f in glob.glob("/Users/kamen/Breakthrough/data/expA/traces125/*_800_g.npz")}
DATA = []
for f in PF:
    name = f.split("/")[-1].split("_")[0]
    p, g = np.load(f), np.load(GG[name])
    L = np.stack([p[f"L{l}"][:, :10] for l in range(NL)], 1).astype(np.int64)
    G = np.stack([g[f"G{l}"][:400, :10] for l in range(NL)], 1).astype(np.float64)
    DATA.append((L, G / G.sum(-1, keepdims=True), np.stack([p[f"P1_{l}"] for l in range(NL - 1)], 1).astype(np.int64)))


class SSD:
    """Processor-sharing transfers after a fixed latency floor; exact event stepping (no overshoot)."""
    def __init__(self):
        self.t = 0.0; self.act = {}; self.done = {}; self.nid = 0   # act: id -> [transfer start, remaining bytes]
    def _live(self):
        return [i for i, (s, r) in self.act.items() if s <= self.t + 1e-12]
    def _next(self):
        live = self._live(); cand = []
        if live: cand.append(self.t + min(self.act[i][1] for i in live) / min(B1, BW / len(live)))
        pend = [s for s, r in self.act.values() if s > self.t + 1e-12]
        if pend: cand.append(min(pend))
        return min(cand) if cand else None
    def _step(self, T):
        live = self._live()
        if live and T > self.t:
            rate = min(B1, BW / len(live))
            for i in live: self.act[i][1] -= rate * (T - self.t)
        self.t = max(self.t, T)
        for i in live:
            if self.act[i][1] <= 1e-3: self.done[i] = self.t; del self.act[i]
    def adv(self, T):
        while True:
            ne = self._next()
            if ne is None or ne > T: self._step(T); return
            self._step(ne)
    def issue(self, T, nbytes=REC):
        T = max(T, self.t); self.adv(T); i = self.nid; self.nid += 1
        self.act[i] = [T + LAT, nbytes]; return i
    def when(self, i):
        while i not in self.done: self._step(self._next())
        return self.done[i]
    def landed_by(self, i, T):
        if T >= self.t: self.adv(T)
        return i in self.done and self.done[i] <= T


def run(C, k, how, A, B, oracle=False):
    tot_t = 0.0; ntok = 0; waited = 0; hidden = 0; reads = 0; me = 0.0
    for L, W, P1 in DATA:
        caches = [OrderedDict() for _ in range(NL)]; ssd = SSD(); t = 0.0
        for tt in range(L.shape[0]):
            t0 = t; ntok += 1; staged = {}
            for l in range(NL):
                c = caches[l]; t += A                              # attention half, then the router point
                cur = staged; staged = {}
                need = []
                for r in range(10):
                    e = int(L[tt, l, r])
                    if e in c: c.move_to_end(e); continue
                    if r < 2:
                        if e in cur: need.append(cur[e]); hidden += 1
                        else: need.append(ssd.issue(t)); waited += 1; reads += 1
                        c[e] = 1
                        if len(c) > C: c.popitem(last=False)
                    elif e in cur and ssd.landed_by(cur[e], t): pass  # landed staged lower rank: computed
                    else: me += W[tt, l, r] ** 2
                pre = []
                if k and l + 1 < NL:
                    cand = L[tt, l + 1, :2] if oracle else P1[tt, l, :k]
                    pre = [int(e) for e in cand if int(e) not in caches[l + 1]]
                if how == "now":
                    for e in pre: staged[e] = ssd.issue(t); reads += 1
                ready = max([t] + [ssd.when(i) for i in need])
                if how == "after":
                    for e in pre: staged[e] = ssd.issue(ready); reads += 1
                t = ready + B
            t += 0.6                                                # lm_head, sampling, embedding
            tot_t += t - t0
    return 1000 * ntok / tot_t, waited / ntok, hidden / ntok, reads / ntok, me / ntok


if __name__ == "__main__" and len(sys.argv) == 1:
    for eng, A, B in (("fast engine M=1 (0.20+0.21 ms/layer)", 0.20, 0.21), ("eager today (~0.30+0.33)", 0.30, 0.33)):
        print(eng, flush=True)
        for C in (51, 26):
            for k, how, orc in ((0, "now", False), (1, "now", False), (2, "now", False), (3, "now", False),
                                (1, "after", False), (2, "after", False), (3, "after", False), (2, "now", True)):
                tps, w, h, rd, e = run(C, k, how, A, B, orc)
                lab = "oracle top-2" if orc else (f"k={k} {how}" if k else "no prefetch")
                print(f"  C={C:2d} {lab:13s}: {tps:5.1f} tok/s  waited {w:4.1f} hidden {h:4.1f} reads {rd:5.1f}/tok"
                      f"  missed E {e / 48:.4f}/layer", flush=True)


def boosted(L, P1, k, q, seed=0):
    """d=1 top-k with each missed true rank-0/1 expert swapped in with probability q (stand-in for a corrected forecast)."""
    rng = np.random.default_rng(seed); T = L.shape[0]
    out = P1[:, :, :k].copy()
    for t in range(T):
        for l in range(NL - 1):
            pr = list(out[t, l]); tru = [int(e) for e in L[t, l + 1, :2]]
            for e in tru:
                if e in pr or rng.random() >= q: continue
                for j in range(k - 1, -1, -1):
                    if pr[j] not in tru: pr[j] = e; break
            out[t, l] = pr
    return out


def run_mtp(C, k, how, A, B, rows=4, adv=3, over=3.0, oracle=False, q=0.0, msb_only=False, same=0.0):
    """4-row verify passes (rows = trace tokens p..p+3, the next pass starts at p+adv): per layer the union of the rows'
    cold ranks 0-1 is read; prefetch = union of the rows' d=1 top-k. 'over' = drafts + lm_head + sampling per pass (ms)."""
    tot_t = 0.0; npass = 0; ntok = 0; waited = 0; reads = 0
    for L, W, P1 in DATA:
        if q and k: P1 = boosted(L, P1, k, q)
        caches = [OrderedDict() for _ in range(NL)]; ssd = SSD(); t = 0.0
        for p0 in range(0, L.shape[0] - rows, adv):
            t0 = t; npass += 1; ntok += adv; staged = {}
            R = range(p0, p0 + rows)
            for l in range(NL):
                c = caches[l]
                cur = staged; staged = {}; need = {}
                if same:                                   # same-layer forecast at the layer's start (lead = A)
                    rs = np.random.default_rng(l * 7919 + p0)
                    for tt in R:
                        for e in L[tt, l, :2]:
                            e = int(e)
                            if e not in c and e not in cur and rs.random() < same:
                                cur[e] = ssd.issue(t); reads += 1
                t += A
                for tt in R:
                    for r in range(10):
                        e = int(L[tt, l, r])
                        if e in c: c.move_to_end(e); continue
                        if e in need: continue
                        if r < 2:
                            if e in cur:
                                need[e] = cur[e]
                                if msb_only: need[(e, "lsb")] = ssd.issue(t, 0.77e6); reads += 1
                            else: need[e] = ssd.issue(t); waited += 1; reads += 1
                            c[e] = 1
                            if len(c) > C: c.popitem(last=False)
                pre = []
                if k and l + 1 < NL:
                    cand = set()
                    for tt in R:
                        cand |= set(int(e) for e in (L[tt, l + 1, :2] if oracle else P1[tt, l, :k]))
                    pre = [e for e in cand if e not in caches[l + 1]]
                pb = 2.15e6 if msb_only else REC
                if how == "now":
                    for e in pre: staged[e] = ssd.issue(t, pb); reads += 1
                ready = max([t] + [ssd.when(i) for i in need.values()])
                if how == "after":
                    for e in pre: staged[e] = ssd.issue(ready, pb); reads += 1
                t = ready + B
            t += over
            tot_t += t - t0
    return 1000 * ntok / tot_t, waited / npass, reads / npass, tot_t / npass


if __name__ == "__main__" and "same" in sys.argv:
    print("Same-layer forecast (issued at the layer's start, lead = attention half 0.30 ms; recall s of the true cold"
          " ranks 0-1, no false positives) alone and on top of d=1 'after' k=2; MTP 4-row passes, fast engine", flush=True)
    for C in (51,):
        for k, q, sm in ((0, 0, 0.0), (0, 0, 0.8), (0, 0, 1.0), (2, 0.3, 0.0), (2, 0.3, 0.8), (2, 0.3, 1.0), (2, 0.0, 0.8)):
            tps, w, rd, ms = run_mtp(C, k, "after", 0.30, 0.33, q=q, same=sm)
            print(f"  C={C} d=1 k={k} q={q:.1f} + same-layer recall {sm:.1f}: {tps:5.1f} tok/s  {ms:5.1f} ms/pass"
                  f"  waited {w:5.1f} reads {rd:6.1f}/pass", flush=True)


if __name__ == "__main__" and "msb" in sys.argv:
    print("v3 records: prefetch MSB-3 only (2.15 MB), staged ranks 0-1 read their 0.77 MB LSB on demand; MTP, after", flush=True)
    for C in (51, 26):
        for q in (0.0, 0.3):
            for mo in (False, True):
                tps, w, rd, ms = run_mtp(C, 2, "after", 0.30, 0.33, q=q, msb_only=mo)
                print(f"  C={C:2d} k=2 q={q:.1f} {'MSB-only + LSB demand' if mo else 'whole record        '}: {tps:5.1f} tok/s"
                      f"  {ms:5.1f} ms/pass  reads {rd:6.1f}/pass", flush=True)


if __name__ == "__main__" and "boost" in sys.argv:
    print("MTP 4-row passes, fast engine, 'after' policy, d=1 forecast with missed true ranks 0-1 swapped in w.p. q", flush=True)
    for C in (51, 26):
        for k in (1, 2):
            for q in (0.0, 0.3, 0.5):
                rec = []
                for L, W, P1 in DATA:
                    Pb = boosted(L, P1, k, q) if q else P1[:, :, :k]
                    rec.append((L[:, 1:, :2, None] == Pb[:, :, None, :]).any(-1).mean())
                tps, w, rd, ms = run_mtp(C, k, "after", 0.30, 0.33, q=q)
                print(f"  C={C:2d} k={k} q={q:.1f} (recall of ranks 0-1 {np.mean(rec):.3f}): {tps:5.1f} tok/s  {ms:5.1f} ms/pass"
                      f"  waited {w:5.1f} reads {rd:6.1f}/pass", flush=True)


if __name__ == "__main__" and "mtp" in sys.argv:
    print("MTP 4-row verify passes, 3 tokens/pass, fast engine M=4 (0.30+0.33 ms/layer) + 3 ms/pass", flush=True)
    for C in (51, 26):
        for k, how, orc in ((0, "now", False), (1, "after", False), (2, "after", False), (2, "now", False), (2, "after", True)):
            tps, w, rd, ms = run_mtp(C, k, how, 0.30, 0.33, oracle=orc)
            lab = "oracle top-2" if orc else (f"k={k} {how}" if k else "no prefetch")
            print(f"  C={C:2d} {lab:13s}: {tps:5.1f} tok/s  {ms:5.1f} ms/pass  waited {w:5.1f} reads {rd:6.1f}/pass", flush=True)


def run_exact(C, k, how, A, B, rows=4, adv=3, over=5.0, q=0.0, nbytes=REC, same=0.0):
    """EXACT streaming (all top-10 of every row must be resident before the MoE): 4-row verify passes; prefetch = union
    of the rows' d=1 top-k for layer l+1 (cold ones), issued 'now' (at layer l's router point, with the demand reads)
    or 'after' (once layer l's reads have landed); a needed expert that is in flight is waited for, not re-read."""
    tot_t = 0.0; npass = 0; ntok = 0; dem = 0; reads = 0; usedpf = 0
    for L, W, P1 in DATA:
        if q and k: P1 = boosted(L, P1, min(k, 2), q)[:, :, :min(k, 2)] if k <= 2 else P1
        caches = [OrderedDict() for _ in range(NL)]; ssd = SSD(); t = 0.0
        for p0 in range(0, L.shape[0] - rows, adv):
            t0 = t; npass += 1; ntok += adv; staged = {}
            R = range(p0, p0 + rows)
            for l in range(NL):
                c = caches[l]
                cur = staged; staged = {}; need = {}
                if same:                                   # same-layer forecast of ALL cold top-10 at the layer's start
                    rs = np.random.default_rng(l * 7919 + p0)
                    for tt in R:
                        for e in L[tt, l, :10]:
                            e = int(e)
                            if e not in c and e not in cur and rs.random() < same:
                                cur[e] = ssd.issue(t, nbytes); reads += 1
                t += A
                for tt in R:
                    for r in range(10):
                        e = int(L[tt, l, r])
                        if e in need: continue
                        if e in c: c.move_to_end(e); continue
                        if e in cur: need[e] = cur[e]; usedpf += 1
                        else: need[e] = ssd.issue(t, nbytes); dem += 1; reads += 1
                        c[e] = 1
                        if len(c) > C: c.popitem(last=False)
                pre = []
                if k and l + 1 < NL:
                    cand = []
                    for tt in R:
                        for e in P1[tt, l, :k]:
                            e = int(e)
                            if e not in cand and e not in caches[l + 1]: cand.append(e)
                    pre = cand
                if how == "now":
                    for e in pre: staged[e] = ssd.issue(t, nbytes); reads += 1
                ready = max([t] + [ssd.when(i) for i in need.values()])
                if how == "after":
                    for e in pre: staged[e] = ssd.issue(ready, nbytes); reads += 1
                t = ready + B
            t += over
            tot_t += t - t0
    return 1000 * ntok / tot_t, dem / npass, reads / npass, usedpf / npass, tot_t / npass


if __name__ == "__main__" and "exactsame" in sys.argv:
    print("EXACT track, C=128, 4-row passes (0.45+0.50 ms/layer + 5 ms): same-layer forecast of all cold top-10 (recall s,"
          " no false positives) alone and with d=1 top-4 'after'", flush=True)
    for k, sm in ((0, 0.0), (0, 0.8), (0, 1.0), (4, 0.0), (4, 0.8)):
        tps, d, rd, up, ms = run_exact(128, k, "after", 0.45, 0.50, same=sm)
        print(f"  d=1 k={k} same-layer {sm:.1f}: {tps:5.1f} tok/s  {ms:5.1f} ms/pass  demand {d:5.1f}  reads {rd:6.1f}/pass", flush=True)


if __name__ == "__main__" and "exact" in sys.argv and "exactsame" not in sys.argv:
    print("EXACT streaming, 4-row verify passes (3 tokens/pass), eager-v3-like 0.45+0.50 ms/layer + 5 ms/pass,"
          " 2.77 MB reads; d=1 top-k per row (plain forecast)", flush=True)
    for C in (128, 256):
        for k, how in ((0, "now"), (4, "now"), (4, "after"), (8, "now"), (8, "after"), (12, "now"), (12, "after")):
            tps, d, rd, up, ms = run_exact(C, k, how, 0.45, 0.50)
            lab = f"k={k:2d} {how:5s}" if k else "no prefetch"
            print(f"  C={C:3d} {lab}: {tps:5.1f} tok/s  {ms:5.1f} ms/pass  demand {d:5.1f} prefetch-used {up:5.1f}"
                  f" reads {rd:6.1f}/pass", flush=True)
