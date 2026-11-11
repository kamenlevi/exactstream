#!/usr/bin/env python3
"""12 GB -> 90 on code: realistic eviction policies against LRU and the OPT bound, at the 12 GB pools (C_m 51 base +
C_l 32 sidecar slots per layer), on the decision map's CODE windows, live-cut checks. Breakthrough, Mon 9 Nov 2026 21:2x.
CPU only, one core, < 1 GB.

Why: code12_union.py (148ad96) found QOM0's bytes at LRU 1.29 GB per check and the Belady bound 0.74 (-43 %): the eviction
policy is the lossless byte lever left on the 12 GB loop (Fit12 prices 90 = one call + a union cut + overlap + fewer bytes).
Core: bytes10_core.py as a library (block_sets, warm_sets, simulate for LRU / PIN / OPT; its own main() is NOT called).
Policies (each pool per layer; an expert the current check needs is never evicted; the sidecar pool runs the same policy):
- LRU: least recently used (the replay's and, within 1.13x per check, the engine's).
- LIP: LRU, but an expert the check needs ONLY at ranks 2-9 (for QOM0: the bonus row's ranks 2-9 that no row has at ranks
  0-1) is inserted at the LRU end, so it goes first (adaptive insertion, Qureshi et al., ISCA 2007).
- LRU2: evict the expert whose second-last use is oldest (LRU-K with K 2, O'Neil et al., SIGMOD 1993); ties by last use.
- LFUd<g>: evict the lowest decayed use count, sum of g^(age in checks) over its uses; ties by last use.
- PIN: bytes10_core's PIN, half the base pool pinned to the domain's most popular experts (IN-SAMPLE popularity over all 12
  code windows: optimistic), LRU for the rest and for the sidecar.
- OPT: Belady (bytes10_core), the bound.
Output: ~/Breakthrough/data/m1/code12_policy.out and .json, rewritten after each rule (QOM0 first: tonight the replay runs
beside the map arm and is stopped when the arm ends, so a stop keeps the rules done; a partial file says so).
Usage: nice -n 15 ~/BigMoE/.venv/bin/python -I -B code12_policy.py
"""
import json, os, sys, time
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import bytes10_core as B  # noqa: E402

DST = os.path.expanduser("~/Breakthrough/data/m1")
B.RULES.update({
    "QOM0": (0.0, 2, 10, False, None, False, None),
    "QOM0+ball": (0.0, 2, 10, False, None, False, "all"),
})
RULES = ["QOM0", "QOM0+ball", "EXACT"]   # S1u's / QOM0's rows first
POLICIES = ["LRU", "LIP", "LRU2", "LFUd0.5", "LFUd0.8", "LFUd0.9", "PIN", "OPT"]


class Pool:
    """one layer's pool: capacity C; policy in LRU / LIP / LRU2 / LFUd<g>; returns 1 per miss"""
    def __init__(self, C, policy):
        self.C, self.p = max(C, 1), policy
        self.last, self.prev, self.score, self.stamp, self.order = {}, {}, {}, {}, {}
        self.g = float(policy[4:]) if policy.startswith("LFUd") else None
        self.seq = 0

    def _victim(self, cur):
        cand = [e for e in self.last if e not in cur]
        if not cand: return None
        if self.p in ("LRU", "LIP"):
            return min(cand, key=lambda e: self.order[e])
        if self.p == "LRU2":
            return min(cand, key=lambda e: (self.prev.get(e, -1), self.order[e]))
        t = self.now
        return min(cand, key=lambda e: (self.score[e] * self.g ** (t - self.stamp[e]), self.order[e]))

    def touch(self, e, t, cur, low=False):
        self.now = t; self.seq += 1
        miss = e not in self.last
        if miss:
            while len(self.last) >= self.C:
                v = self._victim(cur)
                if v is None:                    # every resident is needed now (a union over C: EXACT only): stream, as
                    v = min(self.last, key=lambda x: self.order[x])   # bytes10_core's LRU does
                for d in (self.last, self.prev, self.score, self.stamp, self.order): d.pop(v, None)
        else:
            if self.last[e] != t: self.prev[e] = self.last[e]
        if self.g is not None:
            s = self.score.get(e, 0.0) * self.g ** (t - self.stamp.get(e, t))
            if self.last.get(e) != t: s += 1.0
            self.score[e], self.stamp[e] = s, t
        self.last[e] = t
        # LIP: a ranks-2-9-only insert sits at the LRU end (order below every resident) until it is used again
        self.order[e] = (min(self.order.values(), default=0) - 1) if (miss and low and self.p == "LIP") else self.seq
        return int(miss)


def simulate(seqs, Cm, Cl, policy):
    r = dict(mb=0, ml=0, kept=0, nchk=0)
    bp = [Pool(Cm[l], policy) for l in range(48)]; lp = [Pool(Cl[l], policy) for l in range(48)]
    t = 0
    for warm, seq in seqs:
        for l in range(48):
            for e in warm[l][0]: bp[l].touch(e, t, set())
            for e in warm[l][1]: lp[l].touch(e, t, set())
        t += 1
        for Ub, Ul, Ua, kp, nr in seq:
            for l in range(48):
                cur = set(Ub[l]); curl = set(Ul[l])
                for e in Ub[l]: r["mb"] += bp[l].touch(e, t, cur, low=e not in Ua[l])
                for e in Ul[l]: r["ml"] += lp[l].touch(e, t, curl)
            r["kept"] += kp; r["nchk"] += 1; t += 1
    return r


def main():
    t0 = time.time(); lines = []; res = {}
    def say(s=""): print(s, flush=True); lines.append(s)
    def save(final):
        os.makedirs(DST, exist_ok=True)
        tail = [] if final else [f"(partial: {len(res) // len(POLICIES)} of {len(RULES)} rules so far, {time.time() - t0:.0f} s)"]
        open(os.path.join(DST, "code12_policy.out"), "w").write("\n".join(lines + tail) + "\n")
        json.dump(res, open(os.path.join(DST, "code12_policy.json"), "w"), indent=0)
    W = B.windows(); Wc = W["code"]
    Cm0, Cl0 = B.POOLS["12"]
    pop = B.popularity({"code": Wc}, "code", None)
    say(f"code12 eviction policies (12 GB pools: C_m {Cm0[0]} base / C_l {Cl0[0]} sidecar per layer; code windows {len(Wc)}; "
        f"live checks; MB per kept token (GB per check), vs LRU)")
    say("| rule | " + " | ".join(POLICIES) + " |")
    say("|---|" + "---|" * len(POLICIES))
    for rule in RULES:
        bm = B.RULES[rule][6]
        seqs = [(B.warm_sets(w), B.block_sets(w, B.live_blocks(w, bm), rule, None)) for w in Wc]
        cells = []; base = None
        for pol in POLICIES:
            if pol in ("PIN", "OPT"):
                rr = B.simulate(seqs, Cm0, Cl0, pol, pop=pop)
                mb, ml, kept, nchk = rr["mb"], rr["ml"], rr["kept"], rr["nchk"]
            else:
                rr = simulate(seqs, Cm0, Cl0, pol); mb, ml, kept, nchk = rr["mb"], rr["ml"], rr["kept"], rr["nchk"]
            mbk = (mb * B.MR + ml * B.LR) / kept / 1e6; gbc = (mb * B.MR + ml * B.LR) / nchk / 1e9
            if pol == "LRU": base = mbk
            res[f"{rule}|{pol}"] = dict(mbk=mbk, gbc=gbc, mb=mb, ml=ml, kept=kept, nchk=nchk)
            cells.append(f"{mbk:.1f} ({gbc:.2f}) {mbk / base - 1:+.0%}")
            print(f"  ({rule} {pol}: {time.time() - t0:.0f} s)", flush=True)
        say(f"| {rule} | " + " | ".join(cells) + " |")
        save(False)
    say("check: LRU here vs bytes10_core.simulate LRU (code12_replay.out): QOM0 162.5, EXACT 636.8, QOM0+ball 101.8")
    say(f"done in {time.time() - t0:.0f} s")
    save(True)


if __name__ == "__main__":
    main()
