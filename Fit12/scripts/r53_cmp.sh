#!/bin/sh
# Fit12 R-53 (Fri 30 Oct 2026): compare the A1 B1 B2 A2 outputs of scr_r53 (data/scr/r53/{a1,b1,b2,a2}.txt) against the lines in
# PLAN.md 'R-53 A/B pre-registration' (+ amendments): hashes per prompt equal across all arms, prefill mean B vs mean A per
# prompt, phase peaks, footprint after decode, serving / lifetime peaks, and the patch's counters (B arms only).
# Usage: sh /Users/kamen/Fit12/scripts/r53_cmp.sh [DIR]
D=${1:-/Users/kamen/Fit12/data/scr/r53}
for a in a1 b1 b2 a2; do [ -f "$D/$a.txt" ] || { echo "missing $D/$a.txt"; exit 1; }; done
echo "== engines / patch"
for a in a1 b1 b2 a2; do grep -m1 " bt_b12 " "$D/$a.txt" | sed 's/, loaded in.*//' | cut -c1-200; done
echo "== per prompt"
awk '
FNR == 1 { arm = substr(FILENAME, length(FILENAME) - 5, 2) }
/ RESULT kv / {
  p = $2; k = $5 " " $7 " " $9 " " $11; H[p, arm] = k; T[p, arm] = $13; AF[p, arm] = $16; PZ[p, arm] = ($18 == "paused" ? $19 : "?"); P[p] = 1
}
/\] phase .*: peak footprint/ {
  split($0, x, "phase "); split(x[2], y, ": peak footprint "); ph = y[1]; split(y[2], z, " "); PK[ph, arm] = z[1]; PH[ph] = 1
}
/ RESULT serving peak / { SP[arm] = $5; LP[arm] = $9 }
/ R-53 counts / { split($0, x, "R-53 counts "); split(x[2], y, " \\| PF"); split($2, q, ":"); C[q[1], arm] = y[1] }
END {
  n = split("p2k p8k p16k", ps, " ")
  for (i = 1; i <= n; i++) {
    p = ps[i]; if (!(p in P)) continue
    eq = (H[p, "a1"] == H[p, "b1"] && H[p, "a1"] == H[p, "b2"] && H[p, "a1"] == H[p, "a2"]) ? "EQUAL" : "DIFFER"
    ma = (T[p, "a1"] + T[p, "a2"]) / 2; mb = (T[p, "b1"] + T[p, "b2"]) / 2
    printf "%s: hashes %s (a1 %s)\n", p, eq, H[p, "a1"]
    if (eq == "DIFFER") printf "   a1 %s | b1 %s | b2 %s | a2 %s\n", H[p, "a1"], H[p, "b1"], H[p, "b2"], H[p, "a2"]
    printf "   prefill s: a1 %s a2 %s (mean %.2f) | b1 %s b2 %s (mean %.2f) | B vs A %+.1f %%\n", T[p, "a1"], T[p, "a2"], ma, T[p, "b1"], T[p, "b2"], mb, (mb / ma - 1) * 100
    printf "   after decode GB: a1 %s a2 %s | b1 %s b2 %s; paused a1 %s a2 %s b1 %s b2 %s\n", AF[p, "a1"], AF[p, "a2"], AF[p, "b1"], AF[p, "b2"], PZ[p, "a1"], PZ[p, "a2"], PZ[p, "b1"], PZ[p, "b2"]
    for (s = 1; s <= 2; s++) {
      ph = p " " (s == 1 ? "prefill" : "decode")
      printf "   phase %-12s peak GB: a1 %s a2 %s | b1 %s b2 %s | max B - max A %+.2f\n", ph, PK[ph, "a1"], PK[ph, "a2"], PK[ph, "b1"], PK[ph, "b2"], \
        (PK[ph, "b1"] > PK[ph, "b2"] ? PK[ph, "b1"] : PK[ph, "b2"]) - (PK[ph, "a1"] > PK[ph, "a2"] ? PK[ph, "a1"] : PK[ph, "a2"])
    }
    printf "   counts: b1 %s | b2 %s\n", C[p, "b1"], C[p, "b2"]
  }
  printf "serving peak GB: a1 %s a2 %s | b1 %s b2 %s; lifetime peak GB: a1 %s a2 %s | b1 %s b2 %s\n", SP["a1"], SP["a2"], SP["b1"], SP["b2"], LP["a1"], LP["a2"], LP["b1"], LP["b2"]
}' "$D/a1.txt" "$D/b1.txt" "$D/b2.txt" "$D/a2.txt"
echo "== errors"
grep -l "Traceback\|Error" "$D"/a1.txt "$D"/b1.txt "$D"/b2.txt "$D"/a2.txt 2>/dev/null || echo "none"
