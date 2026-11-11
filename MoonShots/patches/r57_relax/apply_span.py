"""Apply R57_RELAX_SPAN (FULL's rule C) to BigMoE's relaxed screen copy r57_screen_rlx.py (MoonShots, Wed 2026-11-04;
written against ~/BigMoE/scripts/r57_screen_rlx.py as of f19aafb, which already carries R57_RELAX_DELTA from apply.py).
Exact, asserted string edits: every anchor must occur once, or nothing is written. BigMoE runs it (MoonShots never edits
~/BigMoE):
    cp ~/MoonShots/patches/r57_relax/relax_span.py ~/BigMoE/scripts/
    python3 ~/MoonShots/patches/r57_relax/apply_span.py [--check] [path, default ~/BigMoE/scripts/r57_screen_rlx.py] [out]
R57_RELAX_SPAN unset (or 0) keeps every code path as it was: each new block is behind SPAN / sscan, and the result gains
spn=None. R57_RELAX_SPAN=1: rule C on code-comment rows (relax_span.py's docstring); R57_RELAX_SPAN_RULE overrides its
acceptance (default R2:3:0.1); R57_RELAX_LOG logs every span accept. Not with R57_RELAX_DELTA (a separate arm) or with a
FAST arm whose comment loop is on (the old rule); FAST rules without the loop (e.g. QOM, E's chat R1) stack."""
import os, sys

EDITS = [
    ("flags", '        RXC = _ilu3.module_from_spec(_sp3); _sp3.loader.exec_module(RXC)\n', 'after', '''\
SPAN = os.environ.get("R57_RELAX_SPAN", "0") not in ("", "0")   # R57_RELAX_SPAN=1 (MoonShots' relax_span.py): FULL's rule C,
if SPAN:                                                 # relaxed acceptance on code-comment rows only (the causal tagger,
    assert not RLX_D > 0, "R57_RELAX_SPAN and R57_RELAX_DELTA are separate arms"   # the line-end guard, R2 top-3 at 0.1 p1);
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # unset / 0 = exact: no SPAN block below runs
    import relax_span as RSP
    SPAN_RULE = RSP.parse_rule(os.environ.get("R57_RELAX_SPAN_RULE", ""))
    if not FAST:                                         # the tagger's relax_core (R57_FAST loads it below otherwise)
        import importlib.util as _ilu4
        _sp4 = _ilu4.spec_from_file_location("relax_core", os.path.expanduser("~/BigMoE/scripts/relax_core.py"))
        RXC = _ilu4.module_from_spec(_sp4); _sp4.loader.exec_module(RXC)
'''),
    ("run state", '    rlog = RLD.Log(os.environ.get("R57_RELAX_LOG", "")) if RLX_D > 0 else None\n', 'after', '''\
    assert not (SPAN and fz and fz["loop"]), "R57_RELAX_SPAN replaces the FAST arm's comment loop: use an arm without it"
    sscan = RXC.CodeScan() if SPAN else None             # R57_RELAX_SPAN: the causal comment tagger, fed the answer's text
    sst = dict(rows=0, kx=0, k=0, relaxed=0, span_rows=0)   # per run: rows checked, kept exact / in all, relaxed, rows open
    slog = RSP.Log(os.environ.get("R57_RELAX_LOG", "")) if SPAN else None
    if sscan is not None:
        sscan.feed(_TKZ.decode([tok]))
'''),
    ("accept", '''\
        if fz is not None:
            fst["k"] += k + 1
''', 'before', '''\
        if SPAN and dr:                                  # R57_RELAX_SPAN: rule C on the comment rows (FULL R2.1)
            l0s = lg[0, :len(dr)].astype(mx.float32)
            lds = mx.take_along_axis(l0s, mx.array(dr, dtype=mx.int32)[:, None], axis=-1)
            sd_, sr_ = lds[:, 0] - l0s.max(axis=-1), (l0s > lds).sum(axis=-1)   # logit(d) - max logit; the draft's rank
            mx.eval(sd_, sr_)
            sdlm, sdrk = sd_.tolist(), sr_.tolist()
            dtx = [_TKZ.decode([int(x)]) for x in dr]
            atx = [_TKZ.decode([int(x)]) for x in am[:len(dr)]]
            srows, spre = RSP.span_rows(RXC, sscan, dtx, atx)   # conditions 1-4 (a copy of the tagger is fed the drafts)
            fok_s = None if fz is None or not fz["rules"] else (
                lambda j: any(RXC.rule_ok(r_, sdlm[j], sdrk[j]) for r_ in fz["rules"]))
            k, sl_ = RSP.span_k(dr, am, sdlm, sdrk, srows, SPAN_RULE, fok_s)   # exact, the arm's rules, or rule C
            sst["rows"] += len(dr) + 1; sst["kx"] += kx0 + 1; sst["k"] += k + 1; sst["relaxed"] += len(sl_)
            sst["span_rows"] += len(srows)
            for j in sl_:
                slog.write(prompt=name, check=FG["chk"], row=j, region="comment", ctext=spre[j][1][-80:], draft=int(dr[j]),
                           argmax=int(am[j]), dtext=dtx[j], atext=atx[j], rank=int(sdrk[j]), dlm=round(sdlm[j], 4),
                           cf=None if mtp else round(float(cf_raw[j]), 4))
'''),
    ("tagger feed", '            scan.feed("".join(_TKZ.decode([int(x)]) for x in new))\n', 'after', '''\
        if sscan is not None:                            # R57_RELAX_SPAN: the committed tokens join the tagger's text
            sscan.feed("".join(_TKZ.decode([int(x)]) for x in new))
'''),
    ("result", '                rlx=rst if RLX_D > 0 else None,      # R57_RELAX_DELTA: rows checked, kept exact / in all, relaxed accepts\n',
     'after', '''\
                spn=sst if SPAN else None,           # R57_RELAX_SPAN: rows checked, kept exact / in all, relaxed, span rows
'''),
]


def apply(src):
    for name, anchor, how, text in EDITS:
        n = src.count(anchor)
        if n != 1:
            raise SystemExit(f"anchor '{name}' found {n} times (expected 1): r57_screen_rlx.py changed; nothing written")
        if how == "after":
            src = src.replace(anchor, anchor + text)
        elif how == "before":
            src = src.replace(anchor, text + anchor)
        else:
            src = src.replace(anchor, text)
    return src


def main():
    args = [a for a in sys.argv[1:] if a != "--check"]
    path = os.path.expanduser(args[0] if args else "~/BigMoE/scripts/r57_screen_rlx.py")
    out = os.path.expanduser(args[1]) if len(args) > 1 else path
    src = open(path).read()
    if "R57_RELAX_SPAN" in src:
        raise SystemExit("already patched")
    if "R57_RELAX_DELTA" not in src:
        raise SystemExit("needs the R57_RELAX_DELTA copy first (apply.py): r57_screen_rlx.py")
    new = apply(src)
    compile(new, path, "exec")
    if "--check" in sys.argv:
        print(f"OK: {len(EDITS)} edits apply cleanly to {path} (nothing written)")
        return
    open(out, "w").write(new)
    print(f"patched {out} ({len(EDITS)} edits)")


if __name__ == "__main__":
    main()
