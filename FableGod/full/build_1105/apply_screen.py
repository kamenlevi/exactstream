"""apply_screen.py (Fable FULL, LOSSY350 build_1105): write a COPY of BigMoE's scripts/r57_screen.py that carries two
flags, both off by default (flag off = the screen, line for line outside the inserted blocks):
  R57_NOCOM=1          T-B1, the comment-free prompt form: on the humaneval set one sentence joins the end of the user turn.
  R57_COPYPROBE=g+x+s  T-A1, rolled-back timing passes over copied rows (copyprobe.py beside this file, loaded by path).
Anchor-based: every anchor must occur exactly once in the source, else nothing is written and the exit code is 2 (the
caller then skips the two arms). The source is never modified.
usage: python -I apply_screen.py <scripts/r57_screen.py> <out.py> [--check]     (--check: verify only, write nothing)
Built against BigMoE 6adad93 (r57_screen.py md5 9a633d1cb43013f9c15c6ed0704cacda); a different md5 is reported, not refused."""
import hashlib, os, sys

MD5 = "9a633d1cb43013f9c15c6ed0704cacda"
HERE = os.path.dirname(os.path.abspath(__file__))

FLAGS = '''NOCOM = os.environ.get("R57_NOCOM") == "1"              # R57_NOCOM (Fable FULL's LOSSY350 T-B1; ~/FableGod/full/build_1105/
NOCOM_END = _TKZ.token_to_id("<|im_end|>")              # INSTALL.md): the comment-free code mode's PROMPT form. On the humaneval
NOCOM_IDS = _TKZ.encode("\\n\\nDo not write comments in the code.", add_special_tokens=False).ids if NOCOM else []   # set one
assert not NOCOM or (NOCOM_END is not None and NOCOM_IDS), "R57_NOCOM: the tokenizer"   # sentence joins the end of the user turn
'''

PROBE = '''CPROBE = None                                            # R57_COPYPROBE (Fable FULL's LOSSY350 T-A1; ~/FableGod/full/build_1105/
if os.environ.get("R57_COPYPROBE"):                      # INSTALL.md): rolled-back timing passes over copied rows (copyprobe.py,
    import importlib.util as _ilu_cp                     # loaded by path; it wraps the MoE block for calls > 16 rows while a probe
    _sp_cp = _ilu_cp.spec_from_file_location("copyprobe", os.path.expanduser(   # pass is being built, and only then)
        os.environ.get("R57_COPYPROBE_PY", "%s")))
    CPROBE = _ilu_cp.module_from_spec(_sp_cp); _sp_cp.loader.exec_module(CPROBE)
    CPROBE.install(globals())
''' % os.path.join(HERE, "copyprobe.py").replace(os.path.expanduser("~"), "~")

SPLICE = '''    if NOCOM and s == "humaneval":                       # R57_NOCOM: before the user turn's <|im_end|> (the prompt's last one)
        _j = len(ids) - 1 - ids[::-1].index(NOCOM_END)
        ids, want = ids[:_j] + NOCOM_IDS + ids[_j:], []  # the stored REF2 row belongs to the unchanged prompt
'''

HOOK = '''        if CPROBE is not None and not mtp:               # R57_COPYPROBE: nothing is committed (every pass is rolled back). The
            _cpd = CPROBE.maybe(globals(), locals())     # probe's time leaves the cycle / decode clocks; the real check's
            c0 += _cpd; c_prev += _cpd; td += _cpd       # counters start after it
            s0, b0_ = stats(), disk_read_bytes()
            c1 = time.perf_counter()
'''

# (anchor line prefix, where, block): 'after' = after the anchor's line, 'before' = before it
EDITS = [
    ('_TOOLS = re.compile(', "after", FLAGS),
    ('def set_arm(arm):', "before", PROBE),
    ('    ids, want = list(it["prompt_ids"]), REFROWS.get(key, {}).get("completion_ids", [])', "after", SPLICE),
    ('        wsf_.append(int(bool(wsp)))', "before", HOOK),
]


def patch(src):
    lines = src.split("\n")
    for anchor, where, block in EDITS:
        hits = [i for i, l in enumerate(lines) if l.startswith(anchor)]
        if len(hits) != 1:
            raise SystemExit(f"apply_screen: anchor {anchor!r} occurs {len(hits)} times (need 1): nothing written")
        i = hits[0] + (1 if where == "after" else 0)
        lines[i:i] = block.rstrip("\n").split("\n")
    out = "\n".join(lines)
    for need in ("R57_NOCOM", "R57_COPYPROBE", "CPROBE.maybe(globals(), locals())", "NOCOM_IDS + ids[_j:]"):
        assert need in out, need
    compile(out, "<patched r57_screen>", "exec")             # a syntax check only: nothing is run
    return out


def main(argv):
    if len(argv) < 3:
        raise SystemExit(__doc__)
    src = open(argv[1]).read()
    if "R57_COPYPROBE" in src or "R57_NOCOM" in src:
        raise SystemExit("apply_screen: the source already carries the flags: nothing written")
    md5 = hashlib.md5(src.encode()).hexdigest()
    out = patch(src)
    note = "" if md5 == MD5 else f" (source md5 {md5} != the build's {MD5}: anchors matched, review the diff)"
    if "--check" in argv:
        print(f"apply_screen: OK, 4 blocks would be inserted{note}")
        return 0
    try:
        with open(argv[2], "w") as f:
            f.write(out)
    except OSError as e:
        print(f"apply_screen: cannot write {argv[2]}: {e}"); return 2
    print(f"apply_screen: wrote {argv[2]} ({len(out.splitlines()) - len(src.splitlines())} lines added){note}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv))
    except SystemExit as e:
        if isinstance(e.code, str):
            print(e.code); sys.exit(2)
        raise
