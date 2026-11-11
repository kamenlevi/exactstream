"""Apply the host-light v2 of both relaxed rules (R57_RELAX_DELTA and R57_RELAX_SPAN) to BigMoE's relaxed screen copy
r57_screen_rlx.py (MoonShots, Fri 2026-11-06; the orchestrator 01:4x after the Fri run's KILL-in-engine: kept +10.4 % /
+9.9 %, tok/s -9.5 % / +1.3 %, eaten by per-check host work). Written against ~/BigMoE/scripts/r57_screen_rlx.py at 462be24
(v1 of both already applied). Exact, asserted string edits; nothing is written unless every anchor occurs once.
v2 (R57_RELAX_V2, default on; 0 = v1's code paths exactly):
  1. one readback: the drafts' dlm (logit(d) - max logit) and rank are built beside the argmax and evaluated in the SAME
     mx.eval as am and the taps, so the relaxed rules add no second GPU sync per check;
  2. a token-text cache: every _TKZ.decode([id]) of the span tagger becomes a dict lookup (same strings);
  3. the tagger runs only on blocks that can touch a comment (relax_span.may_touch: the tagger is in one, or a draft
     holds a '#'); otherwise no row can relax (exactly equivalent);
  4. the combined arm: R57_RELAX_SPAN=1 with R57_RELAX_DELTA=<d> keeps a row by the exact rule, the arm's FAST rules,
     R1(d) where gated, or rule C (v1 refused the pair).
BigMoE runs it (MoonShots never edits ~/BigMoE):
    cp ~/MoonShots/patches/r57_relax/relax_span.py ~/BigMoE/scripts/
    python3 ~/MoonShots/patches/r57_relax/apply_v2.py [--check] [path, default ~/BigMoE/scripts/r57_screen_rlx.py] [out]"""
import os, sys

EDITS = [
    ("decode cache", '_TKZ = Tokenizer.from_file(os.path.expanduser("~/BigMoE/models/bare/tokenizer.json"))', 'line-after', '''\
_TDC = {}                                                # R57_RELAX_V2: token-text cache for the span tagger (same strings)


def _tdec(x):
    x = int(x)
    t = _TDC.get(x)
    if t is None:
        t = _TDC[x] = _TKZ.decode([x])
    return t
'''),
    ("span flag", '    assert not RLX_D > 0, "R57_RELAX_SPAN and R57_RELAX_DELTA are separate arms"   # the line-end guard, R2 top-3 at 0.1 p1);\n',
     'replace', '''\
    assert not RLX_D > 0 or os.environ.get("R57_RELAX_V2", "1") != "0", "v1: R57_RELAX_SPAN and R57_RELAX_DELTA are separate arms"
'''),
    ("v2 flag", 'SPAN = os.environ.get("R57_RELAX_SPAN", "0") not in ("", "0")', 'line-before', '''\
RLXV2 = os.environ.get("R57_RELAX_V2", "1") != "0"     # R57_RELAX_V2 (MoonShots' apply_v2.py, default on): one readback for the
                                                     # relaxed rules' dlm / rank, cached token texts, the tagger only on blocks
                                                     # that can touch a comment, rule C + R1 together; 0 = v1 exactly
'''),
    ("one readback", '''\
        am = mx.argmax(lg[0], axis=-1)
        if mtp:                                          # the MTP head's history needs the verified rows' widened stream
            wv = TXT._last_widened
            mx.eval(am, wv[:, :len(dr) + 1])
        else:
            tp = taps()
            mx.eval(am, tp)
''', 'replace', '''\
        am = mx.argmax(lg[0], axis=-1)
        rx2 = bool(RLXV2 and (RLX_D > 0 or SPAN) and dr)    # R57_RELAX_V2: dlm / rank ride in the argmax readback
        if rx2:
            l0v = lg[0, :len(dr)].astype(mx.float32)
            ldv = mx.take_along_axis(l0v, mx.array(dr, dtype=mx.int32)[:, None], axis=-1)
            dlv_, rkv_ = ldv[:, 0] - l0v.max(axis=-1), (l0v > ldv).sum(axis=-1)
        if mtp:                                          # the MTP head's history needs the verified rows' widened stream
            wv = TXT._last_widened
            mx.eval(am, wv[:, :len(dr) + 1], *((dlv_, rkv_) if rx2 else ()))
        else:
            tp = taps()
            mx.eval(am, tp, *((dlv_, rkv_) if rx2 else ()))
'''),
    ("readback lists", '''\
        am = am.tolist()
        if PREROUTE and EM.STATE.capture is not None:   # R57_PREROUTE: score this check's pre-routes
''', 'replace', '''\
        am = am.tolist()
        if rx2:
            dlm2, drk2 = dlv_.tolist(), rkv_.tolist()
        if PREROUTE and EM.STATE.capture is not None:   # R57_PREROUTE: score this check's pre-routes
'''),
    ("rlx dlm", '''\
            if fz is None or not (fz["rules"] or fz["loop"]):     # dlm / rank (the R57_FAST block computed them otherwise)
                l0 = lg[0, :len(dr)].astype(mx.float32)
''', 'replace', '''\
            if rx2 and (fz is None or not (fz["rules"] or fz["loop"])):   # R57_RELAX_V2: from the one readback
                dlm, drk, cm_ = dlm2, drk2, set()
            elif fz is None or not (fz["rules"] or fz["loop"]):     # dlm / rank (the R57_FAST block computed them otherwise)
                l0 = lg[0, :len(dr)].astype(mx.float32)
'''),
    ("span block", '''\
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
''', 'replace', '''\
            if rx2:                                      # R57_RELAX_V2: dlm / rank from the one readback, cached texts,
                sdlm, sdrk = dlm2, drk2                  # the tagger only where a comment is possible
                dtx = [_tdec(x) for x in dr]
                if RSP.may_touch(sscan, dtx):
                    atx = [_tdec(x) for x in am[:len(dr)]]
                    srows, spre = RSP.span_rows(RXC, sscan, dtx, atx)
                else:
                    srows, spre, atx = set(), None, None
            else:
                l0s = lg[0, :len(dr)].astype(mx.float32)
                lds = mx.take_along_axis(l0s, mx.array(dr, dtype=mx.int32)[:, None], axis=-1)
                sd_, sr_ = lds[:, 0] - l0s.max(axis=-1), (l0s > lds).sum(axis=-1)   # logit(d) - max logit; the draft's rank
                mx.eval(sd_, sr_)
                sdlm, sdrk = sd_.tolist(), sr_.tolist()
                dtx = [_TKZ.decode([int(x)]) for x in dr]
                atx = [_TKZ.decode([int(x)]) for x in am[:len(dr)]]
                srows, spre = RSP.span_rows(RXC, sscan, dtx, atx)   # conditions 1-4 (a copy of the tagger is fed the drafts)
            rlx_on = RLX_D > 0 and (rg_ is not None or (RLX_THR is None and not RLX_COPY))   # the combined arm: v1's R1 gate
            rlx_ok = (lambda j: (rg_ is None or bool(rg_[j])) and sdlm[j] >= -RLX_D) if rlx_on else None
            fok_s = None if (fz is None or not fz["rules"]) and rlx_ok is None else (
                lambda j: (fz is not None and any(RXC.rule_ok(r_, sdlm[j], sdrk[j]) for r_ in fz["rules"]))
                or (rlx_ok is not None and rlx_ok(j)))
'''),
    ("span feed", '''\
            sscan.feed("".join(_TKZ.decode([int(x)]) for x in new))
''', 'replace', '''\
            sscan.feed("".join((_tdec(x) if RLXV2 else _TKZ.decode([int(x)])) for x in new))
'''),
]


def apply(src):
    for name, anchor, how, text in EDITS:
        n = src.count(anchor)
        if n != 1:
            raise SystemExit(f"anchor '{name}' found {n} times (expected 1): r57_screen_rlx.py changed; nothing written")
        if how == "replace":
            src = src.replace(anchor, text)
        elif how == "line-after":                        # after the anchor's whole line
            i = src.index(anchor); j = src.index("\n", i) + 1
            src = src[:j] + text + src[j:]
        elif how == "line-before":
            i = src.index(anchor); i0 = src.rindex("\n", 0, i) + 1
            src = src[:i0] + text + src[i0:]
    return src


def main():
    args = [a for a in sys.argv[1:] if a != "--check"]
    path = os.path.expanduser(args[0] if args else "~/BigMoE/scripts/r57_screen_rlx.py")
    out = os.path.expanduser(args[1]) if len(args) > 1 else path
    src = open(path).read()
    if "R57_RELAX_V2" in src:
        raise SystemExit("already patched")
    if "R57_RELAX_SPAN" not in src or "R57_RELAX_DELTA" not in src:
        raise SystemExit("needs v1 of both rules first (apply.py, apply_span.py)")
    new = apply(src)
    compile(new, path, "exec")
    if "--check" in sys.argv:
        print(f"OK: {len(EDITS)} edits apply cleanly to {path} (nothing written)")
        return
    open(out, "w").write(new)
    print(f"patched {out} ({len(EDITS)} edits)")


if __name__ == "__main__":
    main()
