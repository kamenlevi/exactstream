"""CERTIFY98's engine count check (docs/PREREG_certify98_2026-11-15.md, amendments 9 and 10a): the engine's tokens per
check on chain-16 against the replay's kept_walk at cap 15 (certify98.py's kept_per_16) on the same windows. CPU only.
  .venv/bin/python tools/certify98_countcheck.py <engine prefix> <cells.npz> <capture dir> [--keys <file>] [--first 24]
      [--min 20] [--line 3] [--identity 99] [--label L] [--topk <dd_topk dump.npz>]
  .venv/bin/python tools/certify98_countcheck.py selftest
- The engine: BigMoE's r57 outputs <prefix>_runs.json (n, cut), <prefix>_checks.json (per check 'kept' = accepted drafts
  + 1, uncut past the answer's end) and <prefix>_tokens.json (each key's answer ids, arm 'gate+t').
- The replay: the cells file (a dd_topk dump or dd_ceiling cells) with 'files', 'fi' and 'row'; each window's rt as
  certify98.windows() builds it, walked by certify98.kept_walk from row 1 at cap 15.
- The windows: --keys lists them in order (default: the cells' file order). The first --first are read; a window counts
  only if the engine's answer equals the capture's (ids[row0 + 1:][:n0], row0 = its first depth-0 row; the capture file
  is <capture dir>/<the cells' file name>). If fewer than --min are identical, the next keys in order are added until
  --min are (amendment 9). A capture that stopped at its own cap inside a longer engine answer (the dev bank's 400
  against the engine's 512) counts when identical up to its end; both walks are then read to that end.
- The engine's count on kept_walk's own convention: each check's kept is clipped at the answer's end, and a check whose
  bonus token is the answer's last one is followed by the replay's terminal check of 1. Both sides then hold n tokens
  per window, so the line compares checks.
- Line (amendment 10a): the engine's pooled tokens per check within +-<line> % of the replay's.
- Draft identity (the dev rehearsal's line, --identity): the share of the engine's checks whose kept equals the replay's
  from the same position; each mismatch is listed with the target's margin at the first cell where they part, and with
  --topk the drafter's top-1 / top-2 there. The registered line counts draft tokens per position, which needs the
  engine's drafts; r57's records hold only each check's kept, so this is the acceptance-level reading.
Last line `COUNT CHECK <label>: PASS | FAIL (...)`; exit 0 PASS, 1 FAIL, 2 bad input."""
import json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import certify98 as C98

CAP = 15


def engine(prefix):
    runs = {r["key"]: r for r in json.load(open(prefix + "_runs.json"))}
    chk = {c["key"]: c for c in json.load(open(prefix + "_checks.json"))}
    tok = {}
    for k, v in json.load(open(prefix + "_tokens.json")).items():
        t = v["gate+t"] if isinstance(v, dict) else v
        tok[k] = t[0] if t and isinstance(t[0], list) else t
    return runs, chk, tok


def walk_equiv(kept, n, cut=False):
    """the engine's checks on kept_walk's convention: clipped at the answer's n tokens, plus the terminal check of 1 when a
    check's bonus is the last token. cut: the capture stops at n inside a longer engine answer, so the engine's checks past
    n are dropped (a check from n - 1 then gives the terminal 1). -> (list, positions) or raises on an inconsistent record"""
    out, pos, q = [], [], 0
    for k in kept:
        if q >= n:
            if cut:
                break
            raise ValueError(f"a check after the answer's end (check {len(out) + 1} of {len(kept)})")
        t = min(int(k), n - q)
        out.append(t); pos.append(q); q += t
    if q == n - 1:
        out.append(1); pos.append(q); q = n
    if q != n:
        raise ValueError(f"the checks cover {q} of {n} tokens")
    return out, pos


def replay_at(rt, q, cap=CAP):
    """the replay's kept from walk position q (rt row q + 1): consecutive right cells up to cap, + 1"""
    k = 0
    while k < cap and rt[q + 1, k] == 1:
        k += 1
    return k + 1


def run(prefix, cells, capdir, keys=None, first=24, mn=20, line=3.0, ident=None, label=None, topk=None):
    runs, chk, tok = engine(prefix)
    T = np.load(topk) if topk else None                     # dd_topk's dump: the drafter's top-k at (fi, row, depth)
    d = np.load(cells)
    for f in ("files", "fi", "row"):
        if f not in d.files:
            raise ValueError(f"{cells} has no '{f}'")
    W = C98.windows(cells)
    dep, fi = d["depth"].astype(int), d["fi"]
    st = np.flatnonzero((dep == 0) & np.append(True, dep[:-1] != 0))
    files = [str(x) for x in d["files"]]
    byname = {}
    for (rt, mt), s in zip(W, st):
        f = files[int(fi[s])]
        byname[os.path.basename(f)[:-4]] = (rt, mt, f, int(d["row"][s]))
    order = keys or [f"{os.path.basename(f)[:-4]}" for f in files]
    eng = {k.split("/")[-1]: k for k in runs}
    out = [f"engine {prefix} ({len(runs)} keys); replay {cells} ({len(W)} windows); cap {CAP}"]
    used, skipped, per = [], [], []
    for i, k in enumerate(order):
        b = k.split("/")[-1]
        if i >= first and len(used) >= mn:
            break
        if b not in eng or b not in byname:
            skipped.append(f"{b}: {'no engine record' if b not in eng else 'no replay window'}"); continue
        r, ek = runs[eng[b]], eng[b]
        rt, mt, f, row0 = byname[b]
        n0 = rt.shape[0] - 1
        ids = np.load(os.path.join(capdir, f))["ids"].tolist()
        ans = ids[row0 + 1:row0 + 1 + n0]
        if r.get("cut"):
            skipped.append(f"{b}: the engine record is cut"); continue
        e = tok.get(ek) or []
        cut = len(e) > n0 and e[:n0] == ans and len(ans) == n0     # the capture stopped at its cap inside this answer
        if (e != ans or int(r["n"]) != n0) and not cut:
            fd = next((j for j, (x, y) in enumerate(zip(e, ans)) if x != y), min(len(e), len(ans)))
            skipped.append(f"{b}: answer differs from the capture at index {fd} (n {r['n']} vs {n0})"); continue
        if cut:
            out.append(f"  {b}: the capture stops at {n0} of the engine's {len(e)} tokens, identical to there: read to {n0}")
        ew, pos = walk_equiv(chk[ek]["kept"], n0, cut)
        rw = C98.kept_walk(rt, C98.R0, CAP)
        assert sum(rw) == n0, f"{b}: the replay walk covers {sum(rw)} of {n0}"
        same = [ew[j] == replay_at(rt, q) for j, q in enumerate(pos)]
        miss = []
        for j, q in enumerate(pos):
            if not same[j]:
                kr = replay_at(rt, q); dd = min(ew[j], kr) - 1        # the first depth where one accepts and one does not
                m = mt[q + 1, dd] if dd < mt.shape[1] else np.nan
                s = f"{b} check {j + 1} at {q}: engine {ew[j]}, replay {kr}, target margin {m:.3f}"
                if T is not None:      # dd_topk's row is target-aligned (y == ids[row + 1]): the anchor's row + the depth
                    w_ = files.index(f)
                    ix = np.flatnonzero((T["fi"] == w_) & (T["row"] == row0 + q + 1 + dd) & (T["depth"] == dd))
                    if len(ix) == 1:
                        assert bool(T["right"][ix[0]]) == (rt[q + 1, dd] == 1), f"{b}: the TOPK cell disagrees with the replay"
                        p1, p2 = (float(x) for x in T["top_p"][ix[0], :2])
                        s += (f"; the replay's draft there {'right' if T['right'][ix[0]] else 'wrong'}, drafter top-1 {p1:.4f} / "
                              f"top-2 {p2:.4f} (log margin {np.log(p1 / p2):.3f}), the target's token at rank "
                              f"{int(T['y_rank'][ix[0]])}")
                    else:
                        s += f"; no single TOPK cell at depth {dd} ({len(ix)} found)"
                miss.append(s)
        used.append(b); per.append((b, n0, len(ew), len(rw), sum(same), len(same), miss))
    if len(used) < mn:
        return out + skipped + [f"COUNT CHECK {label}: FAIL (only {len(used)} identical windows; {mn} needed)"], 1
    N = sum(p[1] for p in per); ce = sum(p[2] for p in per); cr = sum(p[3] for p in per)
    E, R = N / ce, N / cr
    y, x = np.array([p[1] for p in per], float), np.array([p[3] for p in per], float)
    rp, rse = C98.cr1(y, x)
    assert abs(rp - R) < 1e-9
    sm, tot = sum(p[4] for p in per), sum(p[5] for p in per)
    out += [f"  {b}: {n} tokens; checks engine {a} / replay {c}; kept {n / a:.3f} / {n / c:.3f}; same kept on {s} of {t}"
            for b, n, a, c, s, t, _ in per]
    out += [f"  not read: {s}" for s in skipped]
    mis = [m for p in per for m in p[-1]]
    out += [f"  mismatch: {m}" for m in mis]
    ok = abs(E / R - 1) <= line / 100
    idl = 100 * sm / tot
    iok = ident is None or idl >= ident
    out.append(f"  draft identity (kept equal from the same position): {sm} of {tot} checks = {idl:.2f} %"
               + (f" (line {ident} %: {'PASS' if iok else 'FAIL'})" if ident is not None else ""))
    out.append(f"COUNT CHECK {label}: {'PASS' if ok and iok else 'FAIL'} (engine {E:.4f} vs replay {R:.4f} tokens per check "
               f"at cap {CAP}, ratio {E / R:.4f}, line +-{line:g} %; {len(used)} identical windows, {N} tokens, checks "
               f"{ce} / {cr}{'' if iok else '; draft identity under its line'})")
    return out, 0 if ok and iok else 1


def selftest():
    import tempfile, shutil
    t = tempfile.mkdtemp()
    try:
        def cells(path, wins):
            dep, right, m, fi, an, row, files = [], [], [], [], [], [], []
            for w, (name, n, acc) in enumerate(wins):      # acc[r] = how many drafts from walk row r are right
                files.append(name + ".npz")
                for j in range(31):
                    for r in range(n - j):                    # rt row r has a depth-j target while r + j < n
                        dep.append(j); right.append(j < acc[r]); m.append(5.0); fi.append(w); an.append(r); row.append(10 + r)
                ids = list(range(100, 111 + n)); np.savez(os.path.join(t, name + ".npz"), ids=np.array(ids))
            np.savez(path, depth=np.array(dep), right=np.array(right), m=np.array(m, np.float32), fi=np.array(fi),
                     an=np.array(an), row=np.array(row), files=np.array(files))
        def eng(prefix, recs):
            json.dump([{"key": f"s/{k}", "n": n, "cut": False} for k, n, _, _ in recs], open(prefix + "_runs.json", "w"))
            json.dump([{"key": f"s/{k}", "kept": kp} for k, _, kp, _ in recs], open(prefix + "_checks.json", "w"))
            json.dump({f"s/{k}": {"gate+t": [tk]} for k, _, _, tk in recs}, open(prefix + "_tokens.json", "w"))
        n = 40
        acc = [15] * (n + 1)                                  # every draft right: the walk takes 16 per check
        cp = os.path.join(t, "c.npz"); cells(cp, [("a", n, acc), ("b", n, acc)])
        ans = list(range(111, 111 + n))
        # replay walk from row 1 over 40 rows at cap 15: 16, 16, then the end clips at 8 -> 3 checks
        assert C98.kept_walk(C98.windows(cp)[0][0], 1, 15) == [16, 16, 8]
        eng(os.path.join(t, "e"), [("a", n, [16, 16, 16], ans), ("b", n, [16, 16, 16], ans)])   # the last check overshoots
        o, rc = run(os.path.join(t, "e"), cp, t, first=2, mn=2, ident=99, label="T"); assert rc == 0, o
        eng(os.path.join(t, "x"), [("a", n, [8, 8, 8, 8, 16], ans), ("b", n, [16, 16, 16], ans)])   # 5 checks: ratio off
        o, rc = run(os.path.join(t, "x"), cp, t, first=2, mn=2, label="X"); assert rc == 1 and "FAIL" in o[-1], o
        eng(os.path.join(t, "d"), [("a", n, [16, 16, 16], ans[:-1] + [7]), ("b", n, [16, 16, 16], ans)])  # a differs
        o, rc = run(os.path.join(t, "d"), cp, t, first=2, mn=2, label="D"); assert rc == 1 and "only 1" in o[-1], o
        eng(os.path.join(t, "c"), [("a", n + 9, [16, 16, 16, 2], ans + [7] * 9), ("b", n, [16, 16, 16], ans)])
        o, rc = run(os.path.join(t, "c"), cp, t, first=2, mn=2, ident=99, label="C")    # a's capture stops at 40 of 49
        assert rc == 0 and "stops at 40 of the engine's 49" in "\n".join(o), o
        assert walk_equiv([16, 16, 7, 9], 40, cut=True) == ([16, 16, 7, 1], [0, 16, 32, 39])
        assert walk_equiv([16, 16, 7], 40) == ([16, 16, 7, 1], [0, 16, 32, 39])   # the bonus is the last token: + 1
        try:
            walk_equiv([16, 16, 8, 1], 40); raise AssertionError("a check past the end was accepted")
        except ValueError:
            pass
        print("COUNT CHECK selftest: PASS (the end clip, the terminal check, the ratio line, the identity drop, a "
              "check past the end)")
        return 0
    finally:
        shutil.rmtree(t)


def main(argv):
    if argv[:1] == ["selftest"]:
        return selftest()
    try:
        opt = {a: argv[i + 1] for i, a in enumerate(argv) if a.startswith("--")}
        pos = [a for i, a in enumerate(argv) if not a.startswith("--") and (i == 0 or not argv[i - 1].startswith("--"))]
        prefix, cells, capdir = pos[:3]
        keys = [l.strip() for l in open(opt["--keys"]) if l.strip()] if "--keys" in opt else None
        o, rc = run(prefix, cells, capdir, keys, int(opt.get("--first", 24)), int(opt.get("--min", 20)),
                    float(opt.get("--line", 3)), float(opt["--identity"]) if "--identity" in opt else None,
                    opt.get("--label", os.path.basename(prefix)), opt.get("--topk"))
    except (ValueError, IndexError, KeyError, FileNotFoundError) as e:
        print(f"{__doc__}\nbad input: {e}"); return 2
    print("\n".join(o)); return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
