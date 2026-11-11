"""check_build.py (Fable C, Thu 5 Nov 2026): checks of build_1105's masks. CPU, numpy, read-only outside this folder.
1. BIT-CHECK: build_wall_masks.py's rebuilds of the three installed arms (rebuild_check/) equal the installed masks
   (~/FableGod/decision_map/out/ni/<arm>) and NewIdeas' source (~/NewIdeas/data/dm_masks/<arm>) array for array on 36 / 36 windows.
2. Every new arm (masks/<arm>): 36 windows; keep is bool with the routing capture's shape; rows that no live segment
   realizes (prompt rows, rows outside the walk) untouched; ranks 0-1 kept on every row-layer (the _z arm: except its skipped
   rows, which keep nothing in ALL of layers 8-35 and ranks 0-1 everywhere else, and are realized rows);
   how it differs from the rebuilt qom_0_split0.99_live_ball (kept experts on the answer rows; positions kept here but not
   there, which only the pool's changed history can produce); K6 within K8 within the split.
3. Tiles per kept token as the builder itself counts them (build_*.log), vs the twin qom_0_live_ball.
4. MANIFEST.md5 / MANIFEST.sha256 for masks/ (one line per file), and one md5 per arm (of its sorted per-file md5 lines).
-> check_build.out. Exit 1 if a bit-check or a structural check fails."""
import hashlib, json, os, re, sys
import numpy as np

H = os.path.dirname(os.path.abspath(__file__)); DM = os.path.expanduser("~/FableGod/decision_map")
INST = os.path.join(DM, "out", "ni"); SRCN = os.path.expanduser("~/NewIdeas/data/dm_masks"); ROUT = os.path.join(DM, "out", "routing")
base = json.load(open(os.path.join(DM, "out", "baseline.json")))
BASE_ARMS = ["qom_0_split0.99_live_ball", "qom_0_split0.9_live_ball", "qom_0_live_ball"]
NEW = [("K8", "qom_0_split0.99_k8_live_ball"), ("K6", "qom_0_split0.99_k6_live_ball"), ("K4 (probe)", "qom_0_split0.99_k4_live_ball"),
       ("A", "qom_0_split0.99_r0.7_live_ball"), ("A'", "qom_0_split0.99_m0.9_live_ball"), ("B", "qom_0_split0.99_r0.7_z0.999_live_ball")]
lines, bad = [], []
sys.path.insert(0, H)
import build_wall_masks as BW  # noqa: E402  (the copied builder: its own live segments define the realized rows)
import dm_gate  # noqa: E402
REAL = {}
for dom, wf in dm_gate.WALK.items():
    Z = np.load(os.path.join(dm_gate.TD, wf))
    for i, nm in enumerate(str(x) for x in Z["names"]):
        if nm not in base: continue
        N = len(np.load(os.path.join(ROUT, nm))["ids"]); r = np.zeros(N, bool)
        for (k0_, m0_, A0_), dc0_ in zip(Z[f"{i}_checks"].tolist(), Z[f"{i}_dcf"].tolist()):
            for (k, m, A, dc, nxt) in BW.live_segments_nxt(k0_, m0_, A0_, dc0_):
                if A <= N - 1: r[A:min(A + k, N - 1) + 1] = True
        REAL[nm] = r


def say(s=""):
    print(s, flush=True); lines.append(s)


def ok(c, m):
    say(("  ok   " if c else "  FAIL ") + m)
    if not c: bad.append(m)


def kp(d, n): return np.load(os.path.join(d, n))["keep"]


say("1. BIT-CHECK of the copied builder (rebuild vs installed vs NewIdeas' source)")
for arm in BASE_ARMS:
    rb = os.path.join(H, "rebuild_check", arm); names = sorted(f for f in os.listdir(rb) if f.endswith(".npz"))
    e1 = sum(np.array_equal(kp(rb, n), kp(os.path.join(INST, arm), n)) for n in names)
    e2 = sum(np.array_equal(kp(rb, n), kp(os.path.join(SRCN, arm), n)) for n in names)
    ni = len([f for f in os.listdir(os.path.join(INST, arm)) if f.endswith(".npz")])
    ok(len(names) == 36 and ni == 36 and e1 == 36 and e2 == 36, f"{arm}: {len(names)} windows rebuilt; {e1}/36 equal to the installed masks, {e2}/36 equal to NewIdeas' source")

for arm in BASE_ARMS:
    rb = os.path.join(H, "rebuild_check", arm)
    u = sum(bool(kp(rb, n)[~REAL[n]].all()) for n in sorted(os.listdir(rb)) if n.endswith(".npz"))
    ok(u == 36, f"{arm}: every unrealized row untouched in {u}/36 windows (the realized-row test itself, on an installed arm)")
say("")
say("2. The new arms (masks/<arm>)")
ref = os.path.join(H, "rebuild_check", "qom_0_split0.99_live_ball"); K = {}
for tag, arm in NEW:
    d = os.path.join(H, "masks", arm); names = sorted(f for f in os.listdir(d) if f.endswith(".npz"))
    shp = pr = r01 = 0; kept = kept_ref = extra = rows_changed = zero_rl = zero_bad = 0; n_rl = 0
    for n in names:
        k = kp(d, n); r = kp(ref, n); a = int(base[n]["a"])
        shp += (k.dtype == bool and k.shape == np.load(os.path.join(ROUT, n))["ids"].shape); pr += bool(k[~REAL[n]].all())
        nothing = ~k.any(axis=2)                                   # [rows, 48]: a row-layer that keeps no routed expert
        zr = nothing.any(axis=1)                                   # rows with a skipped layer: all of 8-35 and nothing else, realized
        zero_rl += int(nothing.sum()); zero_bad += int(nothing[:, :8].sum() + nothing[:, 36:].sum() + (zr & ~REAL[n]).sum()
                                                       + (zr & ~nothing[:, 8:36].all(axis=1)).sum())
        r01 += bool((k[..., :2].all(axis=2) | nothing).all())      # ranks 0-1 kept wherever anything is kept
        R_ = REAL[n]; kept += int(k[R_].sum()); kept_ref += int(r[R_].sum()); extra += int((k[R_] & ~r[R_]).sum())
        rows_changed += int((k[R_] != r[R_]).any(axis=(1, 2)).sum()); n_rl += int(R_.sum()) * 48
        K[(tag, n)] = k
    isz = "_z" in arm
    ok(len(names) == 36 and shp == 36 and pr == 36, f"{tag} {arm}: {len(names)} windows, {shp}/36 bool with the routing shape, {pr}/36 with every unrealized row (prompt, outside the walk) untouched")
    ok(r01 == 36 and (zero_rl > 0) == isz and zero_bad == 0,
       f"{tag}: ranks 0-1 kept on every row-layer that keeps anything in {r01}/36 windows; row-layers with nothing kept {zero_rl} "
       f"({'whole rows: all of layers 8-35, realized rows only' if zero_bad == 0 else str(zero_bad) + ' violations'}; expected {'> 0' if isz else '0'})")
    say(f"       kept experts on the realized rows {kept} vs the split's {kept_ref} (x{kept / kept_ref:.3f}; {kept / n_rl:.2f} per row-layer); "
        f"kept here but not in the split {extra} ({100 * extra / max(kept, 1):.2f} % of this arm's kept: the pool's changed history); realized rows changed {rows_changed} of {n_rl // 48}")
v = sum(int((K[("K6", n)] & ~K[("K8", n)]).sum()) for (t, n) in K if t == "K6"); w = sum(int((K[("K4 (probe)", n)] & ~K[("K6", n)]).sum()) for (t, n) in K if t == "K6")
say(f"  info K6 within K8: {v} positions kept by K6 only; K4 within K6: {w} (pool history; 0 would be strict nesting)")

say("")
say("3. Tiles per kept token, the builder's own count (code / ifeval / multi), and the ratio to the twin qom_0_live_ball")
T = {}
for f in ("build_rebuild.log", "build_arms.log"):
    p = os.path.join(H, f)
    if not os.path.exists(p): continue
    for ln in open(p):
        m = re.match(r"\s+(\S+) (code|ifeval|multi): tiles per kept token ([0-9.]+)", ln)
        if m: T.setdefault(m.group(1), {})[m.group(2)] = float(m.group(3))
tw = T.get("qom_0_live_ball", {})
for tag, arm in [("twin", "qom_0_live_ball"), ("split 0.99", "qom_0_split0.99_live_ball"), ("split 0.9", "qom_0_split0.9_live_ball")] + NEW:
    t = T.get(arm, {})
    if len(t) == 3 and len(tw) == 3:
        say(f"  {tag:11s} {arm:40s} {t['code']:6.1f} / {t['ifeval']:6.1f} / {t['multi']:6.1f}   x{t['code'] / tw['code']:.3f} / x{t['ifeval'] / tw['ifeval']:.3f} / x{t['multi'] / tw['multi']:.3f}")

say("")
say("4. Manifests (masks/MANIFEST.md5, masks/MANIFEST.sha256) and one md5 per arm")
m5, s2 = [], []
for tag, arm in NEW:
    d = os.path.join(H, "masks", arm); per = []
    for n in sorted(f for f in os.listdir(d) if f.endswith(".npz")):
        b = open(os.path.join(d, n), "rb").read(); h5 = hashlib.md5(b).hexdigest(); per.append(f"{h5}  {arm}/{n}")
        s2.append(f"{hashlib.sha256(b).hexdigest()}  {arm}/{n}")
    m5 += per
    say(f"  {tag:11s} {arm:40s} files {len(per)}  arm md5 {hashlib.md5(chr(10).join(per).encode()).hexdigest()}")
open(os.path.join(H, "masks", "MANIFEST.md5"), "w").write("\n".join(m5) + "\n"); open(os.path.join(H, "masks", "MANIFEST.sha256"), "w").write("\n".join(s2) + "\n")
say("")
say("ALL OK" if not bad else f"FAILED: {len(bad)}")
open(os.path.join(H, "check_build.out"), "w").write("\n".join(lines) + "\n")
sys.exit(1 if bad else 0)
