"""Decision 4's same-scores rule on a decode report (`evalsuite report --variant V --decode --ref R`): in each of math,
IFEval and HumanEval+ the paired delta V - R is above -5 points AND the exact McNemar p is at least 0.05.
  .venv/bin/python tools/d4.py results/report_<V>_vs_<R>.txt   -> one line per set, then `DECISION4: PASS|FAIL`
Exit 0 on PASS, 1 on FAIL, 2 if a set's line is missing."""
import re, sys

LINE = re.compile(r"^(math|ifeval|humaneval) \(n=(\d+)\)\s+([\d.]+)% vs ref ([\d.]+)%.*?([+-]\d+\.\d) \[([^\]]*)\] pt, "
                  r"McNemar p=([\d.]+) \((\d+)/(\d+)\)")
rows = {m.group(1): m for m in map(LINE.match, open(sys.argv[1])) if m}
ok = True
for s in ("math", "ifeval", "humaneval"):
    m = rows.get(s)
    if m is None:
        print(f"{s}: no line in {sys.argv[1]}")
        sys.exit(2)
    d, p = float(m.group(5)), float(m.group(7))
    good = d > -5 and p >= 0.05
    ok &= good
    print(f"{s} (n {m.group(2)}): {m.group(3)} % vs {m.group(4)} %, delta {m.group(5)} [{m.group(6)}] pt, McNemar p "
          f"{m.group(7)} ({m.group(8)}/{m.group(9)}): {'PASS' if good else 'FAIL'}")
print(f"DECISION4: {'PASS' if ok else 'FAIL'}")
sys.exit(0 if ok else 1)
