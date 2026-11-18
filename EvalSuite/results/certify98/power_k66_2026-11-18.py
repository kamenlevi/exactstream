# The scored sets' power after the contamination drops (information; no line changes): amendment 8's power, re-run with
# the pinned scorer's own code (tools/certify98_lines.py, sha256 579a60ac9ed4e329...) at the counts left after amendment
# 9's contamination check (results/certify98/scored_lists_2026-11-18.json): HE+ K 66 (the gated 68 minus HumanEval 94
# and 143) and MBPP K 297 (the 300 minus 152, 180, 367). As power_k68_2026-11-18.py otherwise: the R loop also reads E2's
# measured 1.433 / 1.437; seed 1116, 2000 draws, as registered.
import hashlib, os, sys
p = os.path.join(os.getcwd(), "tools", "certify98_lines.py")
src = open(p).read()
print("scorer sha256", hashlib.sha256(src.encode()).hexdigest()[:16])
for old, new, n in (('Kf = {"HE+": (140, 80), "MBPP": (300,)}', 'Kf = {"HE+": (66,), "MBPP": (297,)}', 1),
                    ('T32[("HE+", 140)]', 'T32[("HE+", 66)]', 2),
                    ('T32[("MBPP", 300)]', 'T32[("MBPP", 297)]', 1),
                    ('HE+ K 140', 'HE+ K 66', 2),
                    ('MBPP K 300', 'MBPP K 297', 1),
                    ('(HE+ 140, its strict 80; "', '(HE+ 66, the gated set after the contamination drops; "', 1),
                    ('MBPP 300)', 'MBPP 297)', 1),
                    ('for r3216 in (1.40, 1.43, 1.445, 1.480):', 'for r3216 in (1.40, 1.43, 1.433, 1.437, 1.445, 1.480):', 1)):
    assert src.count(old) == n, (old, src.count(old))
    src = src.replace(old, new)
g = {"__name__": "certify98_lines_k66", "__file__": p}
exec(compile(src, p, "exec"), g)
sys.exit(g["power"](2000))
