"""The lossy gate's speed read-out (PREREG_lossygate amendment 3a; MoonShots' J(CC): on the 2B, part of close-call's
speed lift was runaway answers). Kept tokens per check, arm vs base, on all answers and on the answers uncapped in BOTH
arms, so capped or looping answers cannot inflate a speed claim. Information beside the verdict, never a line. CPU only.
  .venv/bin/python tools/lossy_speed.py <arm tag> <base tag>      (runner 64's TG: e.g. ccx45cc1 / lx45cc1, or *_rev)
Reads runner 64's per-process runs.json (results/card/es_cd<TG>_<ddHHMM>_<lab><i>_runs.json; the REPLAY r0 processes
are left out) for n (tokens) and checks, and the run's decode rows (results/card_<tag>_t1code[_rev]/decode.jsonl) for
finish; the last uncut entry per key counts. Pooled kept/check = sum n / sum checks."""
import glob, json, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, ROOT)
from evalsuite.decode_report import looped           # noqa: E402

SETS = ("humaneval", "humanevalx", "mbpp", "ifeval")


def runs(tg):
    out = {}
    pat = re.compile(rf"es_cd{re.escape(tg)}_\d{{6}}_([a-z]+)\d+_runs\.json$")
    for f in sorted(glob.glob(os.path.join(ROOT, "results", "card", f"es_cd{tg}_*_runs.json")), key=os.path.getmtime):
        m = pat.search(os.path.basename(f))
        if not m or m.group(1) == "r":                # r0: REPLAY passes commit the reference, not the arm's own tokens
            continue
        for r in json.load(open(f)):
            if not r.get("cut") and r.get("checks"):
                out[r["key"]] = r
    return out


def rows(tg):
    base, rev = (tg[:-4], "_rev") if tg.endswith("_rev") else (tg, "")
    p = os.path.join(ROOT, "results", f"card_{base}_t1code{rev}", "decode.jsonl")
    return {json.loads(l)["key"]: json.loads(l) for l in open(p)} if os.path.exists(p) else {}


def main(arm, base):
    A, B, DA, DB = runs(arm), runs(base), rows(arm), rows(base)
    keys = sorted(set(A) & set(B))
    if not keys:
        print(f"no paired runs for {arm} / {base}"); return 2
    run_ = lambda d, k: d.get(k, {}).get("finish") == "length" or looped(d.get(k, {}).get("completion_ids") or [])
    print(f"kept tokens per check, {arm} vs {base} ({len(keys)} paired keys; decode rows {len(DA)} / {len(DB)})")
    for s in SETS + ("code",):
        ks = [k for k in keys if (k.split("/")[0] in ("humaneval", "humanevalx", "mbpp") if s == "code" else k.split("/")[0] == s)]
        if not ks:
            continue
        clean = [k for k in ks if k in DA and k in DB and not run_(DA, k) and not run_(DB, k)]
        line = []
        for lab, kk in (("all", ks), ("uncapped in both", clean)):
            if not kk:
                line.append(f"{lab}: none"); continue
            ka = sum(A[k]["n"] for k in kk) / sum(A[k]["checks"] for k in kk)
            kb = sum(B[k]["n"] for k in kk) / sum(B[k]["checks"] for k in kk)
            line.append(f"{lab} (n {len(kk)}): {ka:.3f} vs {kb:.3f} ({100 * (ka / kb - 1):+.1f} %)")
        print(f"  {s}: " + "; ".join(line))
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__); sys.exit(2)
    sys.exit(main(*sys.argv[1:3]))
