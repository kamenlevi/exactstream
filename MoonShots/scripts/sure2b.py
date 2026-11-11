"""M-34 SURE2B (MoonShots, Wed 2026-11-04 05:5x; BYTES10 'bytes per expert', TEN_X_LEDGER lever 6; pre-registered in
~/NewIdeas/QUEUE.md M-34 before any run): the bytes side of 2-bit records for drafter-sure rows (FAST).
Record layout [P2 codes | scales | B3 plane]: a 2-bit read is a prefix of the 3-bit MSB part (one read either way).
Replay = NewIdeas' ten_x_ledger.py part C, reused read-only (windows, live checks, pools, plain LRU per layer, prompt
warm-up, carried across a domain's windows). Base = QOM + 3-bit sure rows: sure rows (cf >= thr) load ranks 0-1 and no
LSB part; unsure rows load all 10 ranks plus the LSB part of ranks 0-1. Variant 2b: a sure row's base miss reads the 2-bit
prefix (B2 bytes); an unsure row needing an expert the pool holds at 2 bits reads the B3 plane (B3 bytes) and the slot
becomes 3-bit; a sure row's hit on either form costs nothing. Slots are fixed-size (no capacity gain); variant 2b-cap
spends the pool by bytes (a 2-bit slot costs B2 / MR of a slot): the upper bound with variable-size slots.
Line (bytes side): code SSD MB per kept token at 24 GB, cf 0, fixed slots, vs the base: <= x0.80 PASS, > x0.92 KILL.
usage: [BigMoE venv] taskpolicy -b python sure2b.py -> data/sure2b.txt"""
import os, sys, time
from collections import OrderedDict
import multiprocessing as mp
sys.path.insert(0, os.path.expanduser("~/NewIdeas/scripts"))
import ten_x_ledger as TX                                         # read-only reuse

MR, LR = TX.MR, TX.LR
B2 = 1540096                                                      # 2-bit codes 1,228,800 + bf16 scales 307,200, 16 KB-padded
B3 = 622592                                                       # the third code bit, 614,400 B, 16 KB-padded
lines = []


def say(s):
    print(s, flush=True); lines.append(s)


def replay(task):
    dom, pool, thr, mode = task[:4]                               # mode: base / 2b / 2bcap
    gate_bonus = len(task) > 4 and task[4]                        # exploratory: ledger lever 1 (the bonus row gated like a sure row)
    Cm, Cl = TX.POOLS[pool]
    bp = [OrderedDict() for _ in range(48)]                       # expert -> bits held (2 or 3)
    lp = [OrderedDict() for _ in range(48)]
    used = [0.0] * 48                                             # 2bcap: slots used, a 2-bit slot counting B2 / MR
    byt = kept = 0.0
    up = n2 = 0

    def put(l, e, bits):
        nonlocal used
        c = bp[l]
        if e in c:
            if mode == "2bcap" and bits == 3 and c[e] == 2:           # an upgrade grows the slot; nothing ever shrinks one
                used[l] += (MR - B2) / MR
            c[e] = max(c[e], bits) if mode != "base" else 3
            c.move_to_end(e)
        else:
            c[e] = bits if mode != "base" else 3
            used[l] += (B2 / MR if (mode == "2bcap" and c[e] == 2) else 1.0)
        cap = Cm[l]
        while (used[l] if mode == "2bcap" else len(c)) > cap:
            k, b = c.popitem(last=False)
            used[l] -= B2 / MR if (mode == "2bcap" and b == 2) else 1.0
        if mode != "2bcap":
            used[l] = len(c)

    for w in TX.W[dom]:
        ids = w["ids"]
        for p in range(w["a"]):                                   # the prompt's prefill warms both pools at 3 bits
            for l in range(48):
                for e in ids[p, l].tolist(): put(l, e, 3)
                for e in ids[p, l, :2].tolist(): TX.touch(lp[l], e, Cl[l])
        for rows, cfs, kp in TX.live_blocks(w):
            sure = [c >= thr or (gate_bonus and c == -1.0) for c in cfs]
            for l in range(48):
                need3, need2, Ul = set(), set(), set()
                for r, s_ in zip(rows, sure):
                    if s_:
                        need2.update(ids[r, l, :2].tolist())
                    else:
                        need3.update(ids[r, l, :10].tolist()); Ul.update(ids[r, l, :2].tolist())
                need2 -= need3
                c = bp[l]
                for e in need3:
                    if e not in c: byt += MR
                    elif mode != "base" and c[e] == 2: byt += B3; up += 1
                for e in need2:
                    if e not in c:
                        byt += MR if mode == "base" else B2
                        n2 += mode != "base"
                for e in need3: put(l, e, 3)
                for e in need2: put(l, e, 2)
                byt += sum(LR for e in Ul if e not in lp[l])
                for e in Ul: TX.touch(lp[l], e, Cl[l])
            kept += kp
    return task, byt / max(kept, 1) / 1e6, up, n2


def main():
    t0 = time.time()
    TX.W = TX.windows()
    if sys.argv[1:] == ["bonus"]:
        return bonus()
    if sys.argv[1:] == ["engine"]:
        return engine()
    tasks = [(d, p, thr, m) for d in ("code", "ifeval", "multi") for p in ("24 GB", "12 GB") for thr in (0.0, 0.5)
             for m in ("base", "2b", "2bcap")]
    with mp.get_context("fork").Pool(4) as pool:
        res = {t[:4]: r for t, *r in pool.map(replay, tasks)}
    say(f"== M-34 SURE2B ({time.strftime('%a %H:%M')}): SSD MB per kept token, QOM + 3-bit sure rows (base) vs 2-bit prefix "
        f"reads for sure rows (2b fixed slots; 2bcap pool spent by bytes); B2 {B2} / B3 {B3} / MR {MR} B")
    say("  domain | pool | cf | base | 2b (x base; upgrade reads, 2-bit reads) | 2bcap (x base)")
    for d in ("code", "ifeval", "multi"):
        for p in ("24 GB", "12 GB"):
            for thr in (0.0, 0.5):
                b = res[(d, p, thr, "base")][0]
                v, up, n2 = res[(d, p, thr, "2b")]
                c = res[(d, p, thr, "2bcap")][0]
                say(f"  {d:6s} | {p} | {thr:.1f} | {b:6.1f} | {v:6.1f} (x{v / b:.3f}; {up} upgrades, {n2} 2-bit reads) | "
                    f"{c:6.1f} (x{c / b:.3f})")
    r = res[("code", "24 GB", 0.0, "2b")][0] / res[("code", "24 GB", 0.0, "base")][0]
    say(f"  -> code 24 GB cf 0, fixed slots: x{r:.3f} -> " + ("PASS" if r <= 0.80 else "KILL" if r > 0.92 else "BETWEEN")
        + f" (line x0.80 / x0.92)  ({time.time() - t0:.0f} s)")
    open(os.path.expanduser("~/MoonShots/data/sure2b.txt"), "a").write("\n".join(lines) + "\n")


def bonus():
    """exploratory (not pre-registered), after the KILL: with ledger lever 1 (the gated bonus row) on, every row is sure at
    cf 0, so the 2-bit prefix would cover nearly all reads; what is it worth there?"""
    t0 = time.time()
    TX.W = TX.windows()
    tasks = [(d, p, 0.0, m, True) for d in ("code", "ifeval", "multi") for p in ("24 GB", "12 GB") for m in ("base", "2b", "2bcap")]
    with mp.get_context("fork").Pool(4) as pool:
        res = {t[:4]: r for t, *r in pool.map(replay, tasks)}
    say(f"== M-34 exploratory ({time.strftime('%a %H:%M')}): with the bonus row gated (ledger lever 1) at cf 0; MB per kept token")
    for d in ("code", "ifeval", "multi"):
        for p in ("24 GB", "12 GB"):
            b = res[(d, p, 0.0, "base")][0]
            v, up, n2 = res[(d, p, 0.0, "2b")]
            c = res[(d, p, 0.0, "2bcap")][0]
            say(f"  {d:6s} | {p} | base {b:6.1f} | 2b {v:6.1f} (x{v / b:.3f}; {up} upgrades) | 2bcap {c:6.1f} (x{c / b:.3f})")
    say(f"  ({time.time() - t0:.0f} s)")
    open(os.path.expanduser("~/MoonShots/data/sure2b.txt"), "a").write("\n".join(lines) + "\n")


def engine():
    """NewIdeas 05:51: lever 1 as the map tested it (qom_0_live_bq): the 'engine' convention of ledger_lastrow.py gates a
    segment's last row by the draft it verifies, and keeps it exact (10 ranks + LSB) where no next draft exists."""
    import ledger_lastrow as LL                                   # read-only, as its own run() does
    t0 = time.time()
    TX.W = TX.windows()
    TX.live_blocks = lambda w: LL.blocks_conv(w, "engine")
    tasks = [(d, p, 0.0, m) for d in ("code", "ifeval", "multi") for p in ("24 GB", "12 GB") for m in ("base", "2b", "2bcap")]
    with mp.get_context("fork").Pool(4) as pool:
        res = {t[:4]: r for t, *r in pool.map(replay, tasks)}
    say(f"== M-34 on the engine convention ({time.strftime('%a %H:%M')}; NewIdeas' ask): cf 0, MB per kept token")
    for d in ("code", "ifeval", "multi"):
        for p in ("24 GB", "12 GB"):
            b = res[(d, p, 0.0, "base")][0]
            v, up, n2 = res[(d, p, 0.0, "2b")]
            c = res[(d, p, 0.0, "2bcap")][0]
            say(f"  {d:6s} | {p} | base {b:6.1f} | 2b {v:6.1f} (x{v / b:.3f}; {up} upgrades, {n2} 2-bit reads) | 2bcap {c:6.1f} "
                f"(x{c / b:.3f})")
    say(f"  ({time.time() - t0:.0f} s)")
    open(os.path.expanduser("~/MoonShots/data/sure2b.txt"), "a").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
