# The lineage amendment's power (the orchestrator, received Wed 11:1x: "State the 68's power in the amendment"):
# amendment 8's power, re-run with the pinned scorer's own code (tools/certify98_lines.py, sha256 579a60ac9ed4e329...)
# at the gated HE+ count after the lineage exclusion, K 68 (the 140 minus the 72 problems in b5's training data).
# Changes, all literal and asserted: HE+'s fresh counts (140, 80) -> (68,); the T32 rows read HE+ K 68; the R loop
# also reads E2's measured 1.433 / 1.437 (as power_t3216_e2_2026-11-17.py). MBPP 300 rows use the same code; their
# draws differ from the registered run's, since the HE+ draws before them change. Seed 1116, 2000 draws, as registered.
import hashlib, os, sys
p = os.path.join(os.getcwd(), "tools", "certify98_lines.py")
src = open(p).read()
print("scorer sha256", hashlib.sha256(src.encode()).hexdigest()[:16])
for old, new, n in (('Kf = {"HE+": (140, 80), "MBPP": (300,)}', 'Kf = {"HE+": (68,), "MBPP": (300,)}', 1),
                    ('T32[("HE+", 140)]', 'T32[("HE+", 68)]', 2),
                    ('HE+ K 140', 'HE+ K 68', 2),
                    ('(HE+ 140, its strict 80; "', '(HE+ 68, the gated set after the lineage amendment; "', 1),
                    ('for r3216 in (1.40, 1.43, 1.445, 1.480):', 'for r3216 in (1.40, 1.43, 1.433, 1.437, 1.445, 1.480):', 1)):
    assert src.count(old) == n, (old, src.count(old))
    src = src.replace(old, new)
g = {"__name__": "certify98_lines_k68", "__file__": p}
exec(compile(src, p, "exec"), g)
sys.exit(g["power"](2000))
