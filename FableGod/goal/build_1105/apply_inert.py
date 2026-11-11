"""Apply R57_INERT (+ R57_COPYRUN=64) to BigMoE's relaxed screen copy r57_screen_rlx.py (GOAL session, Thu 2026-11-05;
written against ~/BigMoE/scripts/r57_screen_rlx.py at fef5586, sha1 df5d6efea608, which carries R57_RELAX_DELTA and
R57_RELAX_SPAN). Exact, asserted string edits in MoonShots' apply_span.py form: every anchor must occur once, or nothing is
written. BigMoE runs it (this session never edits ~/BigMoE):
    cp ~/FableGod/goal/build_1105/inert_span.py ~/BigMoE/scripts/
    python3 ~/FableGod/goal/build_1105/apply_inert.py [--check] [path, default ~/BigMoE/scripts/r57_screen_rlx.py] [out]
R57_INERT and R57_COPYRUN unset (or 0) keep every code path as it was: each new block is behind INERT / iscan / CRUN, the
three replaced expressions are equal to the old ones for R57_ROWS 16 and 32, and the result gains inert=None.
  R57_INERT=CD16 | D16 | G16 | C16        16-row arms: stack on any R57_FAST arm without the comment loop (QOM, U3C, E's R1)
  R57_INERT=CD32 | D32 | G32              need R57_ROWS=32, which R57_FAST refuses today (its row gate is 16 rows): these
                                          run on the exact engine only, as rule numbers (kept, length, verdicts), not tok/s
  R57_INERT_DOCTEST=0                     lifts the doctest-line guard;  R57_INERT_LOG=<path>  logs every relaxed accept
  R57_COPYRUN=64 with R57_COPY=g2 R57_ROWS=64   exact: a long verbatim copy run goes on past the drafter's 31 picks, up to
                                          63 drafts; never run on a GPU (checks over 32 rows are new to the engine)
Not with R57_RELAX_SPAN or R57_RELAX_DELTA (separate arms), nor with a FAST arm whose comment loop is on."""
import os
import sys

EDITS = [
    ("rows", 'assert R in (16, 32), R\n', 'replace',
     'assert R in (16, 32) or (R == 64 and os.environ.get("R57_COPYRUN") == "64"), R   # 64: R57_COPYRUN only (inert_span.py)\n'),
    ("flags", '        RXC = _ilu4.module_from_spec(_sp4); _sp4.loader.exec_module(RXC)\n', 'after', '''\
INERT, CRUN = None, int(os.environ.get("R57_COPYRUN", "0") or 0)   # R57_INERT=<arm> / R57_COPYRUN=64 (GOAL's inert_span.py,
if os.environ.get("R57_INERT", "0") not in ("", "0") or CRUN:     # ~/FableGod/goal/LOSSY350_GOAL.md). INERT: a drafted row
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # inside a '#' comment or a docstring of a python fence
    import inert_span as INM                             # is kept whatever the 125B's argmax (span ends exact, code tokens
    INERT = INM.parse_arm(os.environ.get("R57_INERT", ""), os.environ.get("R57_INERT_DOCTEST", "1"))   # exact). Unset / 0 =
    if INERT is not None:                                # exact: no INERT block below runs
        assert not SPAN and not RLX_D > 0, "R57_INERT, R57_RELAX_SPAN and R57_RELAX_DELTA are separate arms"
        assert R >= INERT["rows"], f"R57_INERT={INERT['name']} needs R57_ROWS={INERT['rows']}"
        if not FAST:                                     # the tagger's relax_core (R57_FAST loads it below otherwise)
            import importlib.util as _ilu5
            _sp5 = _ilu5.spec_from_file_location("relax_core", os.path.expanduser("~/BigMoE/scripts/relax_core.py"))
            RXC = _ilu5.module_from_spec(_sp5); _sp5.loader.exec_module(RXC)
    assert CRUN in (0, 64) and (not CRUN or (COPY and R == 64)), "R57_COPYRUN=64 needs R57_COPY and R57_ROWS=64"
'''),
    ("dwin masks", '[MASK] * (R - 1)', 'replace', '[MASK] * (min(R, 32) - 1)'),
    ("block rows", '    R_ = R if cg is not None else 16', 'replace',
     '    R_ = R if (cg is not None or (INERT is not None and INERT["rows"] > 16)) else 16'),
    ("run state", '''\
    if sscan is not None:
        sscan.feed(_TKZ.decode([tok]))
''', 'after', '''\
    assert not (INERT is not None and fz and fz["loop"]), "R57_INERT replaces the FAST arm's comment loop: use an arm without it"
    iscan = INM.InertScan(RXC) if INERT is not None and not mtp else None   # R57_INERT: the engine's tagger + the docstring
    ist = dict(rows=0, kx=0, k=0, whole=0, ruled=0, com=0, doc=0, open=0)   # state; per run: rows checked, kept exact / in all,
    ilog = INM.Log(os.environ.get("R57_INERT_LOG", "")) if INERT is not None else None   # kept whole / by a rule, per region,
    if iscan is not None:                                # rows open to INERT
        iscan.feed(_TKZ.decode([tok]))
'''),
    ("propose masks", '[MASK] * (R_ if _d17 else R_ - 1)', 'replace', '[MASK] * (R_ if _d17 else min(R_, 32) - 1)'),
    ("match rule", '                if R_ == 32 and sum(cu_) < 24:', 'replace',
     '                if R_ >= 32 and sum(cu_[:31]) < 24 and not (INERT is not None and INERT["rows"] > 16):'),
    ("copyrun", '                    dr, cfl = dr[:15], cfl[:15]\n', 'after', '''\
                elif CRUN == 64 and R_ == 64:            # R57_COPYRUN=64 (exact): a long verbatim run goes on past the
                    d0_, dr, cu_, ci_, cfl = INM.copyrun_extend(cg, d0_, dr, cu_, ci_, cfl, R_)   # drafter's 31 picks
'''),
    ("pre block", '            if fz is not None and (fz["thr"] is not None or fz["copy"]):', 'before', '''\
            if iscan is not None:                        # R57_INERT: rows in a span are sure in the cut; a longer block
                idx_ = [_TKZ.decode([int(x)]) for x in dr]   # keeps its drafts past the 15th only inside a span (or a copy)
                in_, cfl, _ = INM.pre_block(RXC, iscan, dr, cfl, idx_, INERT, cu_ if cg is not None else None)
                dr, cfl = dr[:in_], cfl[:in_]
'''),
    ("accept", '''\
        if fz is not None:
            fst["k"] += k + 1
''', 'before', '''\
        if iscan is not None and dr:                     # R57_INERT: the rows in comments / docstrings (inert_span.py)
            l0i = lg[0, :len(dr)].astype(mx.float32)
            ldi = mx.take_along_axis(l0i, mx.array(dr, dtype=mx.int32)[:, None], axis=-1)
            id_, ir_ = ldi[:, 0] - l0i.max(axis=-1), (l0i > ldi).sum(axis=-1)   # logit(d) - max logit; the draft's rank
            mx.eval(id_, ir_)
            idlm, idrk = id_.tolist(), ir_.tolist()
            dtx = [_TKZ.decode([int(x)]) for x in dr]
            atx = [_TKZ.decode([int(x)]) for x in am[:len(dr)]]
            icls = INM.classify(RXC, iscan, dtx, atx, INERT)   # the scan before each draft (a copy is fed the drafts)
            fok_i = None if fz is None or not fz["rules"] else (
                lambda j: any(RXC.rule_ok(r_, idlm[j], idrk[j]) for r_ in fz["rules"]))
            k, il_ = INM.inert_k(dr, am, idlm, idrk, icls, fok_i)   # exact, the arm's rules, or INERT
            ist["rows"] += len(dr) + 1; ist["kx"] += kx0 + 1; ist["k"] += k + 1
            ist["open"] += sum(1 for c_ in icls if c_[1] is not None)
            for j, kind_, how_ in il_:
                ist["whole" if how_ == "whole" else "ruled"] += 1; ist["com" if kind_ == "comment" else "doc"] += 1
                ilog.write(prompt=name, check=FG["chk"], row=j, region=kind_, mode=how_, ctext=icls[j][2][-80:],
                           draft=int(dr[j]), argmax=int(am[j]), dtext=dtx[j], atext=atx[j], rank=int(idrk[j]),
                           dlm=round(idlm[j], 4), cf=round(float(cf_raw[j]), 4) if j < len(cf_raw) else None)
'''),
    ("tagger feed", '''\
        if sscan is not None:                            # R57_RELAX_SPAN: the committed tokens join the tagger's text
            sscan.feed("".join(_TKZ.decode([int(x)]) for x in new))
''', 'after', '''\
        if iscan is not None:                            # R57_INERT: the committed tokens join the tagger's text
            iscan.feed("".join(_TKZ.decode([int(x)]) for x in new))
'''),
    ("result", '                spn=sst if SPAN else None,           # R57_RELAX_SPAN: rows checked, kept exact / in all, relaxed, span rows\n',
     'after', '''\
                inert=dict(ist, arm=INERT["name"]) if INERT is not None else None,   # R57_INERT: rows checked, kept exact / in
                                                     # all, kept whole / by a rule, per region, rows open
'''),
    ("runs record", '"copy", "fast", "pf", "lh")},', 'replace', '"copy", "fast", "pf", "lh", "inert")},'),
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
    if "R57_INERT" in src:
        raise SystemExit("already patched")
    if "R57_RELAX_SPAN" not in src:
        raise SystemExit("needs the relaxed copy with R57_RELAX_SPAN (BigMoE fef5586 or later): r57_screen_rlx.py")
    new = apply(src)
    compile(new, path, "exec")
    if "--check" in sys.argv:
        print(f"OK: {len(EDITS)} edits apply cleanly to {path} (nothing written)")
        return
    open(out, "w").write(new)
    print(f"patched {out} ({len(EDITS)} edits)")


if __name__ == "__main__":
    main()
