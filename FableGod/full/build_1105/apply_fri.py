"""apply_fri.py (Fable FULL, LOSSY350 build_1105): write a COPY of BigMoE's scripts/fri_night.sh with two additions:
  T-B1  the comment-free prompt arm, inside the palindrome next to qall, both passes (pair means cancel drift as for the
        other arms):  u3c / qall / QALLP / s99 / s90 / qallr / span | span / qallr / s90 / s99 / QALLP / qall / u3c
        fqallp / fqallpb = qall's env + R57_NOCOM=1 on XS (the frozen screen + the two flags, apply_screen.py).
  T-A1  the copy probe, after every timed arm (after fexr): fcprobe = qall's env + the 0.9 split switch (when set) +
        R57_COPYPROBE=g+x+s; with FRI_CPROBE_T=1 also fcprobet (g+t, the experimental narrow gate call), last.
  and the two CPU reports at the end (data/fri/nocom_report.txt, data/fri/copyprobe_report.txt).
If apply_screen.py cannot patch the frozen screen (an anchor moved), XS is empty and the added arms are skipped by arm()'s
own "no switch" rule; FRI_FULL=0 skips them too. Nothing else in the script changes.
Anchor-based: each anchor must occur exactly once, else nothing is written (exit 2). The source is never modified.
usage: python -I apply_fri.py <scripts/fri_night.sh> <out.sh> [--check]
Built against BigMoE 6adad93 (fri_night.sh md5 5a92d239ecc8c0f243062cd16afe687f); another md5 is reported, not refused."""
import hashlib, os, sys

MD5 = "5a92d239ecc8c0f243062cd16afe687f"
HERE = os.path.dirname(os.path.abspath(__file__)).replace(os.path.expanduser("~"), "$HOME")
PY = '"${FRI_PY:-$HOME/BigMoE/.venv/bin/python}"'

XS = f'''# Fable FULL's build_1105 (LOSSY350 T-B1 / T-A1, {HERE}/INSTALL.md): XS = the frozen screen + two flags that are off
# by default (R57_NOCOM, R57_COPYPROBE). If the applier's anchors do not match, XS stays empty and its arms are skipped by
# arm()'s "no switch" rule; FRI_FULL=0 skips them too.
XS=; B1105="${{FRI_B1105:-{HERE}}}"
if [ "${{FRI_FULL:-1}}" = 1 ]; then
  XS=data/r57/snap/r57_screen_frix.py
  {PY} -I "$B1105/apply_screen.py" "$FS" "$XS" >> "$LOG" 2>&1 || XS=
fi
'''
P1 = 'arm fqallp "$XS" "${XS:+$QA,R57_NOCOM=1}";   '
PROBE = '''# T-A1 (Fable FULL): the copy probe, after every timed arm (it warms the pools; its own tok/s is not a result)
C45_DEADLINE=${FRI_DEADLINE3:-$(date -v+8M +%H%M)} arm fcprobe "$XS" "${XS:+$QA${S90:+,$S90},R57_COPYPROBE=g+x+s}"
[ -n "$FRI_CPROBE_T" ] && C45_DEADLINE=${FRI_DEADLINE4:-$(date -v+6M +%H%M)} arm fcprobet "$XS" "${XS:+$QA${S90:+,$S90},R57_COPYPROBE=g+t}"
'''
REPORT = f'''if [ -n "$XS" ]; then   # Fable FULL's two CPU reads: data/fri/nocom_report.txt, data/fri/copyprobe_report.txt
  {PY} -I "$B1105/nocom_report.py" "$CC" data/fri/nocom_report.txt >> "$LOG" 2>&1; say "nocom report rc=$?"
  {PY} -I "$B1105/copyprobe_report.py" "data/r57/fcprobe_${{CC}}_copyprobe.jsonl" data/fri/copyprobe_report.txt >> "$LOG" 2>&1
  say "copy probe report rc=$?"
fi
'''


def one(text, anchor):
    n = text.count(anchor)
    if n != 1:
        raise SystemExit(f"apply_fri: anchor {anchor!r} occurs {n} times (need 1): nothing written")


def patch(src):
    a1 = "FS=data/r57/snap/r57_screen_fri.py; RS=data/r57/snap/r57_screen_frir.py\n"
    a2 = 'arm fqall $FS "$QA";   arm fs99 $FS'
    a3 = 'arm fqallb $FS "$QA";   arm fu3cb $FS "$U"'
    a4 = 'arm fexr $FS "R57_VTIER="\n'
    a5 = 'say "Fri run end"'
    for a in (a1, a2, a3, a4, a5):
        one(src, a)
    out = src.replace(a1, a1 + XS)
    out = out.replace(a2, 'arm fqall $FS "$QA";   ' + P1.replace("fqallp", "fqallp") + "arm fs99 $FS")
    out = out.replace(a3, P1.replace("fqallp", "fqallpb") + a3)
    out = out.replace(a4, a4 + PROBE)
    out = out.replace(a5, REPORT + a5)
    for need in ("arm fqallp ", "arm fqallpb ", "arm fcprobe ", "apply_screen.py", "nocom_report.py", "copyprobe_report.py"):
        assert out.count(need) == 1, need
    return out


def main(argv):
    if len(argv) < 3:
        raise SystemExit(__doc__)
    src = open(argv[1]).read()
    if "fqallp" in src or "R57_COPYPROBE" in src:
        raise SystemExit("apply_fri: the source already carries the arms: nothing written")
    md5 = hashlib.md5(src.encode()).hexdigest()
    out = patch(src)
    note = "" if md5 == MD5 else f" (source md5 {md5} != the build's {MD5}: anchors matched, review the diff)"
    if "--check" in argv:
        print(f"apply_fri: OK, 5 blocks would be inserted{note}"); return 0
    with open(argv[2], "w") as f:
        f.write(out)
    print(f"apply_fri: wrote {argv[2]} ({len(out.splitlines()) - len(src.splitlines())} lines added){note}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv))
    except SystemExit as e:
        if isinstance(e.code, str):
            print(e.code); sys.exit(2)
        raise
