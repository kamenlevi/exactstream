"""Apply R57_RELAX_DELTA to BigMoE's r57_screen.py (MoonShots, Wed 2026-11-04; written against BigMoE 155d5c7). Exact,
asserted string edits: every anchor must occur once, or nothing is written. BigMoE runs it (MoonShots never edits ~/BigMoE):
    cp ~/MoonShots/patches/r57_relax/relax_delta.py ~/BigMoE/scripts/
    python3 ~/MoonShots/patches/r57_relax/apply.py [--check] [path, default ~/BigMoE/scripts/r57_screen.py] [out]
Unset R57_RELAX_DELTA keeps every code path as it was (each new block is behind RLX_D > 0)."""
import os, sys

EDITS = [
    ("flags", 'FAST = os.environ.get("R57_FAST", "")\n', 'after', '''\
RLX_D = float(os.environ.get("R57_RELAX_DELTA", "0") or 0)   # R57_RELAX_DELTA=<nats> (MoonShots' relax_delta.py): relaxed
RLX_THR = RLX_COPY = None                                    # acceptance at any delta (EvalSuite's R1); R57_RELAX_GATE
if RLX_D > 0:                                                # (cf:<thr> / copy / both) picks the rows that may relax;
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # R57_RELAX_LOG logs each relaxed accept. 0 / unset =
    import relax_delta as RLD                                # exact: none of the RLX blocks below runs
    RLX_THR, RLX_COPY = RLD.parse_gate(os.environ.get("R57_RELAX_GATE", ""))
    if not FAST:                                             # the gate's relax_core (R57_FAST loads it below otherwise)
        import importlib.util as _ilu3
        _sp3 = _ilu3.spec_from_file_location("relax_core", os.path.expanduser("~/BigMoE/scripts/relax_core.py"))
        RXC = _ilu3.module_from_spec(_sp3); _sp3.loader.exec_module(RXC)
'''),
    ("run state", '    scan = RXC.CodeScan() if fz and fz["loop"] else None\n', 'after', '''\
    rgrams = RXC.prompt_grams(ids) if RLX_D > 0 and RLX_COPY else None   # R57_RELAX_GATE copy: its own grams (FAST's stay)
    rst = dict(rows=0, kx=0, k=0, relaxed=0)              # R57_RELAX_DELTA per run: rows checked, kept exact / in all, relaxed
    rlog = RLD.Log(os.environ.get("R57_RELAX_LOG", "")) if RLX_D > 0 else None
'''),
    ("check reset", '        fk_ = sq_ = None                                 # R57_FAST: this check\'s keep mask / U3\'s sure rows (DFlash2 only)\n',
     'after', '''\
        rg_ = None                                       # R57_RELAX_DELTA: the rows that may relax (DFlash2 only)
'''),
    ("row gate", '            if arm in ("gate+t", "gate+t+pf") and not wsp:   # Leap\'s S45c: L1-14 running-confidence truncation\n',
     'before', '''\
            if RLX_D > 0 and (RLX_THR is not None or RLX_COPY):   # R57_RELAX_DELTA: the row gate, on the block before the cut
                tlr = out[-RXC.COPY_N:] if len(out) >= RXC.COPY_N else ids[len(out) - RXC.COPY_N:] + out
                cgr = (cf_raw if cg is None else [c_ if d_ == x_ else 0.0 for c_, d_, x_ in zip(cf_raw, d0_, dr)])
                rg_ = RLD.gate_flags(RXC, RLX_THR, RLX_COPY, cgr, tlr, dr, rgrams)
'''),
    ("exact count", '''\
        while k < len(dr) and dr[k] == am[k]:
            k += 1
        if fz is not None:
            fst["rows"] += len(dr) + 1; fst["kx"] += k + 1
''', 'replace', '''\
        while k < len(dr) and dr[k] == am[k]:
            k += 1
        kx0 = k                                          # R57_RELAX_DELTA: the exact rule's kept drafts
        if fz is not None:
            fst["rows"] += len(dr) + 1; fst["kx"] += k + 1
'''),
    ("accept", '''\
        if fz is not None:
            fst["k"] += k + 1
''', 'before', '''\
        if RLX_D > 0 and dr and (rg_ is not None or (RLX_THR is None and not RLX_COPY)):   # R57_RELAX_DELTA
            if fz is None or not (fz["rules"] or fz["loop"]):     # dlm / rank (the R57_FAST block computed them otherwise)
                l0 = lg[0, :len(dr)].astype(mx.float32)
                ld = mx.take_along_axis(l0, mx.array(dr, dtype=mx.int32)[:, None], axis=-1)
                dl_, rk_ = ld[:, 0] - l0.max(axis=-1), (l0 > ld).sum(axis=-1)
                mx.eval(dl_, rk_)
                dlm, drk = dl_.tolist(), rk_.tolist()
                cm_ = set()
            fok = None if fz is None or not (fz["rules"] or fz["loop"]) else (
                lambda j: any(RXC.rule_ok(r_, dlm[j], drk[j]) for r_ in fz["rules"] + ((fz["loop"],) if j in cm_ else ())))
            k, rl_ = RLD.relaxed_k(dr, am, dlm, RLX_D, rg_, fok)  # the exact rule, the arm's rules, or R1(delta) where gated
            rst["rows"] += len(dr) + 1; rst["kx"] += kx0 + 1; rst["k"] += k + 1; rst["relaxed"] += len(rl_)
            for j in rl_:
                rlog.write(prompt=name, check=FG["chk"], row=j, draft=int(dr[j]), argmax=int(am[j]), dlm=round(dlm[j], 4),
                           rank=int(drk[j]), cf=None if mtp else round(float(cf_raw[j]), 4),
                           gate="all" if rg_ is None else "+".join(x for x, on in (("cf", RLX_THR is not None), ("copy", RLX_COPY)) if on))
'''),
    ("result", '''\
    return dict(lh=lh_, out=out[:CAP], n=min(len(out), CAP), tps=min(len(out), CAP) / dec, drafter=drafter, hist=hist, a6=a6,
''', 'after', '''\
                rlx=rst if RLX_D > 0 else None,      # R57_RELAX_DELTA: rows checked, kept exact / in all, relaxed accepts
'''),
]


def apply(src):
    for name, anchor, how, text in EDITS:
        n = src.count(anchor)
        if n != 1:
            raise SystemExit(f"anchor '{name}' found {n} times (expected 1): r57_screen.py changed; nothing written")
        if how == "after":
            src = src.replace(anchor, anchor + text)
        elif how == "before":
            src = src.replace(anchor, text + anchor)
        else:
            src = src.replace(anchor, text)
    return src


def main():
    args = [a for a in sys.argv[1:] if a != "--check"]
    path = os.path.expanduser(args[0] if args else "~/BigMoE/scripts/r57_screen.py")
    out = os.path.expanduser(args[1]) if len(args) > 1 else path
    src = open(path).read()
    if "R57_RELAX_DELTA" in src:
        raise SystemExit("already patched")
    new = apply(src)
    compile(new, path, "exec")
    if "--check" in sys.argv:
        print(f"OK: {len(EDITS)} edits apply cleanly to {path} (nothing written)")
        return
    open(out, "w").write(new)
    print(f"patched {out} ({len(EDITS)} edits)")


if __name__ == "__main__":
    main()
