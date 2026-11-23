"""patch_c25_wed3.py (Leap, Tue 17 Nov 2026 09:5x; slot #3 = Wed morning, the orchestrator 09:4x (b)): adds the 16-row and
the 32-row module arms to c25_engine.py (+ c25_passk.py's row rule, c25_report.py's pairs). APPLY ONLY AFTER tue1 ENDS (c25
files are live-imported while it runs). Anchored replacements; any anchor miss aborts before a file is written.
New arms:
  F16g    F16 at lambda 1.8e-2: G1's 16-row check (int8-free, 8-bit HC mixers; Saturday's TREE-16 path)
  F16gd   F16g + GDN16 (m16: gdn_fast at 2-16 rows; bitwise, so == F16g in every row)
  F32c6p  F32c6i + HC32P (m16: the 32-row one-launch HC read, 4-bit V3 here; bitwise, so == F32c6i in every row)
  F32c6x  F32c6i + the bitwise stack: FASTGDN + HC32P + seams at 32 rows (ref2 seam_hi 32) + int8fold (MODE bit)
usage: python3 patch_c25_wed3.py dry | apply   (dry: patched copies in the scratch dir + py_compile; apply: in place)"""
import os
import py_compile
import sys

D = os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_s98a/Leap_scripts/week98/c25")
SCR = "/private/tmp/claude-501/-Users-kamen-OjoX/f2971937-6c95-4e86-a049-05beff6a8e32/scratchpad/c25_wed3"

ENG = [
    ("  F32c6g   F32c6i + FASTGDN (C25_ARMS selects these two; the default stays the 11-16 nine)\n",
     "  F32c6g   F32c6i + FASTGDN (C25_ARMS selects these two; the default stays the 11-16 nine)\n"
     "  F16g     F16 at lambda 1.8e-2: G1's 16-row check (int8-free, 8-bit HC mixers; Saturday's TREE-16 path; 11-17)\n"
     "  F16gd    F16g + GDN16 (m16.py: gdn_fast at 2-16 rows; bitwise: must equal F16g in every row)\n"
     "  F32c6p   F32c6i + HC32P (m16.py: the 32-row HC read in one launch, per-call barrier slots; bitwise == F32c6i)\n"
     "  F32c6x   F32c6i + the bitwise stack: FASTGDN + HC32P + seams at 32 rows (ref2 seam_hi 32) + int8fold (MODE bit)\n"),
    ('ALL = ["F16", "F32s", "F32o", "F32d", "F32df", "F32if", "F32if6", "F32c5", "F32c6i", "F32g", "F32c6g"]\n',
     'ALL = ["F16", "F32s", "F32o", "F32d", "F32df", "F32if", "F32if6", "F32c5", "F32c6i", "F32g", "F32c6g", "F16g", "F16gd",\n'
     '       "F32c6p", "F32c6x"]\n'),
    ('       "F32c6g": dict(one=True, form="dn32", t1cf=True, hc4=True, v1=True, int8=True, lam=0.018, fastgdn=True)}\n',
     '       "F32c6g": dict(one=True, form="dn32", t1cf=True, hc4=True, v1=True, int8=True, lam=0.018, fastgdn=True),\n'
     '       "F16g": dict(lam=0.018), "F16gd": dict(lam=0.018, gdn16=True),\n'
     '       "F32c6p": dict(one=True, form="dn32", t1cf=True, hc4=True, v1=True, int8=True, lam=0.018, hc32p=True),\n'
     '       "F32c6x": dict(one=True, form="dn32", t1cf=True, hc4=True, v1=True, int8=True, lam=0.018, fastgdn=True, hc32p=True,\n'
     '                     seam=True, fold=True)}\n'),
    ('         ("F32c6g", "F32c6i", "diff"), ("F32c6g", "F16", "ratio")]\n',
     '         ("F32c6g", "F32c6i", "diff"), ("F32c6g", "F16", "ratio"), ("F16g", "F16", "diff"), ("F16gd", "F16g", "diff"),\n'
     '         ("F32c6p", "F32c6i", "diff"), ("F32c6x", "F32c6i", "diff"), ("F32c6x", "F16", "ratio"), ("F16gd", "F16", "diff")]\n'),
    ("    int8d.install()\n    f32one.ON[0] = False\n",
     "    int8d.install()\n"
     "    import int8fold                                    # 11-17 (slot #3): after int8d; then m16 (wraps whatever is installed)\n"
     "    int8fold.install()\n"
     "    int8fold.ON[0] = False\n"
     "    import m16\n"
     "    m16.install()\n"
     "    f32one.ON[0] = False\n"),
    ('    S.update(f32one=f32one, int8d=int8d, lam0=float(G["TIERS_LAM"]), G=G,\n',
     '    S.update(int8fold=int8fold, m16=m16, seam0=int(ref2.CFG["seam_hi"]))\n'
     '    S.update(f32one=f32one, int8d=int8d, lam0=float(G["TIERS_LAM"]), G=G,\n'),
    ("    i.ON[0] = bool(c.get(\"int8\"))\n    G[\"TIERS_LAM\"] = c.get(\"lam\", S[\"lam0\"])\n",
     "    i.ON[0] = bool(c.get(\"int8\"))\n"
     "    S[\"m16\"].GDN16[0] = bool(c.get(\"gdn16\"))\n"
     "    S[\"m16\"].HC32P[0] = bool(c.get(\"hc32p\"))\n"
     "    S[\"int8fold\"].ON[0] = bool(c.get(\"fold\"))\n"
     "    G[\"ref2\"].CFG[\"seam_hi\"] = 32 if c.get(\"seam\") else S[\"seam0\"]\n"
     "    G[\"TIERS_LAM\"] = c.get(\"lam\", S[\"lam0\"])\n"),
    ("    S[\"int8d\"].ON[0] = False\n    G[\"TIERS_LAM\"] = S[\"lam0\"]\n",
     "    S[\"int8d\"].ON[0] = False\n"
     "    S[\"m16\"].GDN16[0] = False\n"
     "    S[\"m16\"].HC32P[0] = False\n"
     "    S[\"int8fold\"].ON[0] = False\n"
     "    G[\"ref2\"].CFG[\"seam_hi\"] = S[\"seam0\"]\n"
     "    G[\"TIERS_LAM\"] = S[\"lam0\"]\n"),
    ('            n = 16 if a == "F16" else 32\n', '            n = 16 if a.startswith("F16") else 32\n'),
    ('                 ("F32c5", "F32df"), ("F32c6i", "F32c5"), ("F32s", "F16"), ("F32g", "F32o"), ("F32c6g", "F32c6i")):\n',
     '                 ("F32c5", "F32df"), ("F32c6i", "F32c5"), ("F32s", "F16"), ("F32g", "F32o"), ("F32c6g", "F32c6i"),\n'
     '                 ("F16g", "F16"), ("F16gd", "F16g"), ("F32c6p", "F32c6i"), ("F32c6x", "F32c6i")):\n'),
    ('    _say(G, f"C25 report: {len(W)} (window, round) samples; arms {ARMS}")\n',
     '    _say(G, f"C25 report: {len(W)} (window, round) samples; arms {ARMS}")\n'
     '    if S.get("m16") is not None:                         # 11-17: engagement + the barrier check (must be 0)\n'
     '        _say(G, f"C25: m16 COUNTS {S[\'m16\'].COUNTS}; one-launch HC spin-cap fails {S[\'m16\'].fails()} (must be 0); "\n'
     '                f"int8fold COUNTS {S[\'int8fold\'].COUNTS}")\n'),
    ('        for a in ("F16", "F32s", "F32o", "F32df", "F32if", "F32c6i", "F32c6g"):\n',
     '        for a in ("F16", "F32s", "F32o", "F32df", "F32if", "F32c6i", "F32c6g", "F16g", "F16gd", "F32c6p", "F32c6x"):\n'),
]
# 11-17 10:3x: Breakthrough's expert one-call (frozen release ~/Breakthrough/data/xrw/knob_v1, imported read-only; xknob
# wraps ooo1cq.tier_glu_nax6n_1cq, COH on and off): MODE tgl16e = EXACT (== ooo1cq bit for bit), tgl16eu = + the TIER-4
# UPGRADE (FAST, batch-dependent like the 1c tiers themselves; its own EvalSuite gate). Applied after ENG (its anchors).
ENG_XK = [
    ("  F32c6x   F32c6i + the bitwise stack: FASTGDN + HC32P + seams at 32 rows (ref2 seam_hi 32) + int8fold (MODE bit)\n",
     "  F32c6x   F32c6i + the bitwise stack: FASTGDN + HC32P + seams at 32 rows (ref2 seam_hi 32) + int8fold (MODE bit)\n"
     "  F16gk / F32c6k   F16g / F32c6i + Breakthrough's expert one-call tgl16e (knob_v1 xknob; EXACT: == ooo1cq bit for bit)\n"
     "  F16gu / F32c6u   F16g / F32c6i + tgl16eu (tgl16e + the TIER-4 UPGRADE: FAST, batch-dependent; its own gate)\n"),
    ('       "F32c6p", "F32c6x"]\n', '       "F32c6p", "F32c6x", "F16gk", "F16gu", "F32c6k", "F32c6u"]\n'),
    ('                     seam=True, fold=True)}\n',
     '                     seam=True, fold=True),\n'
     '       "F16gk": dict(lam=0.018, xk="tgl16e"), "F16gu": dict(lam=0.018, xk="tgl16eu"),\n'
     '       "F32c6k": dict(one=True, form="dn32", t1cf=True, hc4=True, v1=True, int8=True, lam=0.018, xk="tgl16e"),\n'
     '       "F32c6u": dict(one=True, form="dn32", t1cf=True, hc4=True, v1=True, int8=True, lam=0.018, xk="tgl16eu")}\n'),
    ('("F32c6x", "F16", "ratio"), ("F16gd", "F16", "diff")]\n',
     '("F32c6x", "F16", "ratio"), ("F16gd", "F16", "diff"),\n'
     '         ("F16gk", "F16g", "diff"), ("F16gu", "F16g", "diff"), ("F32c6k", "F32c6i", "diff"), ("F32c6u", "F32c6i", "diff")]\n'),
    ("    import m16\n    m16.install()\n",
     "    import m16\n    m16.install()\n"
     "    xk = None\n"
     "    if any(CFG[a].get(\"xk\") for a in ARMS):              # Breakthrough's expert one-call (frozen knob_v1, read-only)\n"
     "        sys.path.insert(0, os.path.expanduser(\"~/Breakthrough/data/xrw/knob_v1\"))\n"
     "        import xknob as xk\n"
     "        assert \"/knob_v1/\" in xk.__file__, xk.__file__\n"
     "        xk.install()\n"
     "        xk.ON[0] = False\n"),
    ('    S.update(int8fold=int8fold, m16=m16, seam0=int(ref2.CFG["seam_hi"]))\n',
     '    S.update(int8fold=int8fold, m16=m16, seam0=int(ref2.CFG["seam_hi"]), xk=xk)\n'),
    ("    S[\"int8fold\"].ON[0] = bool(c.get(\"fold\"))\n    G[\"ref2\"].CFG[\"seam_hi\"] = 32 if c.get(\"seam\") else S[\"seam0\"]\n",
     "    S[\"int8fold\"].ON[0] = bool(c.get(\"fold\"))\n    G[\"ref2\"].CFG[\"seam_hi\"] = 32 if c.get(\"seam\") else S[\"seam0\"]\n"
     "    if S[\"xk\"] is not None:\n"
     "        S[\"xk\"].ON[0] = bool(c.get(\"xk\"))\n"
     "        if c.get(\"xk\"):\n"
     "            S[\"xk\"].MODE[0] = c[\"xk\"]\n"),
    ("    S[\"int8fold\"].ON[0] = False\n    G[\"ref2\"].CFG[\"seam_hi\"] = S[\"seam0\"]\n",
     "    S[\"int8fold\"].ON[0] = False\n    G[\"ref2\"].CFG[\"seam_hi\"] = S[\"seam0\"]\n"
     "    if S.get(\"xk\") is not None:\n"
     "        S[\"xk\"].ON[0] = False\n"),
    ('                 ("F16g", "F16"), ("F16gd", "F16g"), ("F32c6p", "F32c6i"), ("F32c6x", "F32c6i")):\n',
     '                 ("F16g", "F16"), ("F16gd", "F16g"), ("F32c6p", "F32c6i"), ("F32c6x", "F32c6i"), ("F16gk", "F16g"),\n'
     '                 ("F16gu", "F16g"), ("F32c6k", "F32c6i"), ("F32c6u", "F32c6i")):\n'),
    ('                f"int8fold COUNTS {S[\'int8fold\'].COUNTS}")\n',
     '                f"int8fold COUNTS {S[\'int8fold\'].COUNTS}; xknob COUNTS "\n'
     '                f"{S[\'xk\'].COUNTS if S.get(\'xk\') is not None else None}")\n'),
    ('"F16g", "F16gd", "F32c6p", "F32c6x"):\n',
     '"F16g", "F16gd", "F32c6p", "F32c6x", "F16gk",\n                  "F16gu", "F32c6k", "F32c6u"):\n'),
]
PASSK = [('            n = 16 if a == "F16" else 32\n', '            n = 16 if a.startswith("F16") else 32\n')]


def patch(name, reps, out_dir):
    src = open(os.path.join(D, name)).read()
    for a, z in reps:
        assert src.count(a) == 1, (name, a[:90], src.count(a))
        src = src.replace(a, z)
    os.makedirs(out_dir, exist_ok=True)
    p = os.path.join(out_dir, name)
    tmp = p + ".tmp"
    open(tmp, "w").write(src)
    py_compile.compile(tmp, doraise=True)
    return src, p, tmp


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "dry"
    assert mode in ("dry", "apply")
    out = D if mode == "apply" else SCR
    if mode == "apply":
        assert not os.popen("pgrep -f run_c25_tue1").read().strip(), "tue1's runner is still up: do not patch live c25 files"
    done = [patch("c25_engine.py", ENG + ENG_XK, out), patch("c25_passk.py", PASSK, out)]
    for src, p, tmp in done:                    # all compiled: now move into place
        os.replace(tmp, p)
        print(f"{mode}: wrote {p} ({len(src)} bytes)")


if __name__ == "__main__":
    main()
