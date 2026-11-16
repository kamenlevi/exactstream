# Amendment 8's power, re-run with the pinned scorer's own code (tools/certify98_lines.py, sha256 579a60ac9ed4e329...);
# the only change: the T32/16 loop also reads E2's measured R32/R16 (1.433 paired median, 1.437 ratio of means).
# The R loop draws nothing, so every other row uses the registered run's draws (seed 1116) and must reproduce them.
import hashlib, os, sys
p = os.path.join(os.getcwd(), "tools", "certify98_lines.py")
src = open(p).read()
print("scorer sha256", hashlib.sha256(src.encode()).hexdigest()[:16])
old = "for r3216 in (1.40, 1.43, 1.445, 1.480):"
assert src.count(old) == 1, "the R loop literal is not unique"
src = src.replace(old, "for r3216 in (1.40, 1.43, 1.433, 1.437, 1.445, 1.480):")
g = {"__name__": "certify98_lines_e14", "__file__": p}
exec(compile(src, p, "exec"), g)
sys.exit(g["power"](2000))
