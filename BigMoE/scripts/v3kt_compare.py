"""v3kt_compare.py (BigMoE, Tue 17 Nov 2026): the V3 KEY TEST verdict (scripts/v3_keytest.sh). PASS only if
  - both runs recorded every key: tokens ({TAG}_tokens.json) and runs ({TAG}_runs.json, each with checks' lh);
  - per key the tokens are identical OFF vs ON, and the run records (key, arm, type, n, lh = the sha1 of every check's
    verify logits, R57_LOGHASH) are identical in order, with lh non-empty;
  - the ON run engaged: ref2.CFG bof + hc_b true, bofi ON, bofi.CNT bof > 0 and drift 0, ref2.COUNTS hc_b > 0;
  - the OFF run did not: bof / hc_b false, hc_b count 0, no BOF fire.
usage: python3 scripts/v3kt_compare.py <dir> <off tag> <on tag> <keys, comma list>   (prints one line; rc 0 PASS, 1 FAIL, 2 ERROR)"""
import json
import sys

D, OFF, ON, KEYS = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4].split(",")


def load(tag):
    R = json.load(open(f"{D}/{tag}_runs.json"))
    T = json.load(open(f"{D}/{tag}_tokens.json"))
    V = [ln for ln in open(f"{D}/{tag}.log") if ln.startswith("V3KT ")]
    return [(r.get("key"), r.get("arm"), r.get("type"), r.get("n"), r.get("lh")) for r in R], T, (json.loads(V[-1][5:]) if V else None)


try:
    (ro, to, vo), (rn, tn, vn) = load(OFF), load(ON)
except Exception as e:
    print(f"ERROR: a record is missing ({type(e).__name__}: {e})")
    sys.exit(2)
why = []
miss = [k for k in KEYS if k not in to or k not in tn or not any(r[0] == k for r in ro) or not any(r[0] == k for r in rn)]
if miss:
    why.append("not decoded: " + ", ".join(miss))
dt = [k for k in KEYS if k in to and k in tn and to[k] != tn[k]]
if dt:
    why.append("TOKENS differ: " + ", ".join(dt))
if len(ro) != len(rn):
    why.append(f"run records {len(ro)} OFF vs {len(rn)} ON")
else:
    dl = [f"{a[0]} {a[1]} {a[2]}" for a, b in zip(ro, rn) if a != b]
    if dl:
        why.append("LOGITS / records differ: " + "; ".join(dl[:4]))
empty = [r[0] for r in ro + rn if not r[4]]
if empty:
    why.append("no logit hashes: " + ", ".join(sorted(set(map(str, empty)))))
cn = (vn or {}).get("bofi_cnt") or {}
if not vn or not (vn.get("bof_cfg") is True and vn.get("hc_b_cfg") is True and vn.get("bofi_on") is True
                  and cn.get("bof", 0) > 0 and cn.get("drift", 1) == 0 and (vn.get("hc_b_count") or 0) > 0):
    why.append(f"ON not engaged as required: {vn}")
co = (vo or {}).get("bofi_cnt") or {}
if not vo or vo.get("bof_cfg") or vo.get("hc_b_cfg") or (vo.get("hc_b_count") or 0) != 0 or co.get("bof", 0) or co.get("bof2", 0):
    why.append(f"OFF not clean: {vo}")
ntok = sum(len(x) for k in KEYS if k in to for arm in to[k].values() for x in arm)
nchk = sum(len(r[4] or []) for r in ro)
if why:
    print("FAIL: " + " | ".join(why))
    sys.exit(1)
print(f"PASS: {len(KEYS)} keys, {ntok} tokens, {nchk} checks: tokens + every check's verify logits bitwise identical OFF vs ON; "
      f"ON engaged (bof {cn.get('bof')}, bof2 {cn.get('bof2')}, drift 0, hc_b {vn.get('hc_b_count')}); OFF clean")
