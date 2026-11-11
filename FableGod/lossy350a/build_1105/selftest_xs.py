"""CPU self-test of the T1 patch (Fable A, Thu 5 Nov 2026). No GPU: MLX runs on its CPU device; no model is loaded.
Run:  ~/BigMoE/.venv/bin/python -I selftest_xs.py        (about a minute; writes SELFTEST.out; exit 0 = all PASS)
 1  the mask: dm_xs.mask on MLX's CPU device == an independent numpy definition, on random values, exact ties, zero
    groups and REAL map inputs (decision_map/out/xcap, all 48 layers), for bfloat16 / float16 / float32, at c 48 / 40 / 32;
    c = 64 returns the very same array object; sparsity and kept energy on the real inputs.
 2  the wiring: the real harness code (base = today's files, applied = patched) driven with a small fake model through
    both paths (>= 32 rows via sorted_experts and the fused combine; narrower via the switch call):
    - DM_XS unset: the patched harness == today's, bit for bit, and has no xs arm;
    - DM_XS=1: baseline, topk and xs64 == today's bit for bit; under xs48 / xs32 the switch reads exactly mask(x) while
      the router, the shared expert and its gate read x; outputs == a reference whose fake switch masks its own input;
      each row is masked and counted once, also when wrapped classes nest; clear() switches the arm off.
 3  measure(): the per-window record of masked switch rows (48 layers x rows) and the WARNING when a path escapes.
 4  the patch: applies to today's files (md5 checked against the live decision_map), gives applied/, reverses cleanly;
    apply.sh / revert.sh on a sandbox copy.
 5  the reader: read_xs.py reproduces fablec/wall_class_flips.out on arms already on disk.
"""
import glob, hashlib, json, os, shutil, subprocess, sys
import numpy as np
import mlx.core as mx
mx.set_default_device(mx.cpu)
HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE)
sys.path.insert(0, os.path.join(HERE, "applied"))
import dm_xs
OUT = []; FAILS = []
def say(s=""):
    print(s, flush=True); OUT.append(s)
def check(ok, what):
    say(("PASS  " if ok else "FAIL  ") + what)
    if not ok: FAILS.append(what)
def md5(p): return hashlib.md5(open(p, "rb").read()).hexdigest()
DM = os.path.expanduser("~/FableGod/decision_map")
PY = sys.executable

# ---------------------------------------------------------------- 1 the mask
say("== 1 the mask (MLX on " + str(mx.default_device()) + ") ==")
def mask_np(x, c):
    """keep a member iff fewer than c members of its 64-group are strictly larger in |x| (ties at the cut are kept)"""
    if c >= 64: return x
    a = np.abs(x.astype(np.float32)).reshape(*x.shape[:-1], -1, 64)
    bigger = (a[..., None, :] > a[..., :, None]).sum(-1)
    return np.where((bigger < c).reshape(x.shape), x, np.zeros_like(x))
rng = np.random.default_rng(1105)
def as_bf16(a): return np.array(mx.array(a).astype(mx.bfloat16).astype(mx.float32))
rnd = rng.standard_normal((37, 2560)).astype(np.float32) * 3
ties = rnd.copy(); ties[0, :64] = 0.0; ties[1, 64:128] = 1.5; ties[2, :64] = np.tile([2.0, -2.0, 0.25, 0.0], 16); ties[3, 128:192] = np.sign(ties[3, 128:192]) * 0.5
real = []
for f in sorted(glob.glob(os.path.join(DM, "out", "xcap", "*.npz")))[::12][:3]:
    x = np.load(f)["x"]; real.append((os.path.basename(f), x[: min(16, len(x))].astype(np.float32)))
sets = [("random", rnd), ("ties and zero groups", ties)] + [(f"real map x, {nm} (16 rows x 48 layers)", x.reshape(-1, 2560)) for nm, x in real]
for name, x32 in sets:
    for dt_name, dt in (("bfloat16", mx.bfloat16), ("float16", mx.float16), ("float32", mx.float32)):
        xm = mx.array(x32).astype(dt); xv = np.array(xm.astype(mx.float32))      # the values the model would hold
        ok = True; kept = {}
        for c in (48, 40, 32):
            for shape in (xv.shape, (1,) + xv.shape):
                y = dm_xs.mask(xm.reshape(shape), c)
                ok &= (y.dtype == dt) and (tuple(y.shape) == tuple(shape))
                ok &= np.array_equal(np.array(y.astype(mx.float32)).reshape(xv.shape), mask_np(xv, c))
            kept[c] = (mask_np(xv, c) != 0).reshape(len(xv), 40, 64).sum(-1)
        ok &= dm_xs.mask(xm, 64) is xm and dm_xs.mask(xm, None) is xm
        check(bool(ok), f"mask == numpy reference, dtype and shape kept, c 64 is the same object: {name}, {dt_name}")
    if name.startswith("real") or name == "random":
        xv = x32; e = (xv.astype(np.float64) ** 2).sum(-1)
        say("      " + name + ": " + "; ".join(
            f"c {c}: kept per group {(mask_np(xv, c) != 0).reshape(len(xv), 40, 64).sum(-1).mean():.2f} (max {(mask_np(xv, c) != 0).reshape(len(xv), 40, 64).sum(-1).max()}),"
            f" energy kept {((mask_np(xv, c).astype(np.float64) ** 2).sum(-1) / e).mean():.4f}" for c in (48, 40, 32)))
for nm, x in real:                                           # per layer, the kept energy at c 32 (the mask is per row and layer)
    xl = x.reshape(len(x), 48, 2560).astype(np.float64); m = mask_np(x.reshape(-1, 2560), 32).reshape(xl.shape).astype(np.float64)
    ke = ((m ** 2).sum(-1) / (xl ** 2).sum(-1)).mean(0)
    say(f"      {nm}: energy kept at c 32 by layer: min {ke.min():.4f} (L{int(ke.argmin())}), median {np.median(ke):.4f}, max {ke.max():.4f} (L{int(ke.argmax())})")
check(len(real) == 3, "three real map windows read from decision_map/out/xcap")

# ---------------------------------------------------------------- 2 the wiring
say("\n== 2 the wiring: the harness's own code on a fake model, both MoE paths ==")
T = os.path.join(HERE, "selftest", "tmp"); shutil.rmtree(T, ignore_errors=True); os.makedirs(T)
def case(tag, src, variant, xs):
    env = dict(os.environ); env.pop("DM_XS", None)
    if xs: env["DM_XS"] = "1"
    out = os.path.join(T, tag)
    r = subprocess.run([PY, "-I", os.path.join(HERE, "selftest", "harness_case.py"), os.path.join(HERE, src), out, variant],
                       env=env, capture_output=True, text=True)
    if r.returncode != 0:
        say(r.stdout[-800:] + r.stderr[-1500:]); return None, None
    return json.load(open(out + ".json")), (dict(np.load(out + ".npz")) if os.path.exists(out + ".npz") else None)
def same(a, b, keys):
    return all(a[k].tobytes() == b[k].tobytes() and a[k].shape == b[k].shape for k in keys)
jb, ab = case("base_unset", "base", "streamed", False)
ju, au = case("applied_unset", "applied", "streamed", False)
check(jb is not None and ju is not None, "scenarios ran: today's harness and the patched one, DM_XS unset")
if jb and ju:
    keys = [k for k in ab if k in au]
    check(len(keys) == 10 and same(ab, au, keys), "DM_XS unset: patched harness == today's harness, bit for bit (baseline, topk, both paths, 10 outputs)")
    check(ju["_meta"]["xs_imported"] is False and ju["_meta"]["xs_arms"] == [], "DM_XS unset: dm_xs is not imported and no xs arm is registered")
    check(jb["baseline"]["switch_is_mask64"] and jb["ref_mask32"]["switch_is_mask32"] and not jb["baseline"]["switch_is_mask32"],
          "the fake model's own reference masking works (the comparison is not vacuous)")
    check(not same(ab, {k: ab[k.replace("ref_mask32", "baseline")] for k in ab}, ["ref_mask32_wide", "ref_mask32_narrow"]), "masking the switch input changes the outputs (wide and narrow)")
for variant in ("streamed", "fused", "mixed"):
    jr, ar = case("ref_" + variant, "base", variant, False)
    jx, ax = case("xs_" + variant, "applied", variant, True)
    if not (jr and jx): check(False, f"{variant}: scenarios ran"); continue
    v = variant
    check(same(ar, ax, ["baseline_wide", "baseline_narrow", "topk8_wide", "topk8_narrow"]), f"{v}: DM_XS=1, no xs arm: baseline and topk == today's, bit for bit")
    check(ax["xs64_wide"].tobytes() == ar["baseline_wide"].tobytes() and ax["xs64_narrow"].tobytes() == ar["baseline_narrow"].tobytes()
          and jx["xs64"]["switch_is_mask64"] and jx["xs64"]["xs_rows"] == 0, f"{v}: xs64 == today's baseline, bit for bit; nothing masked")
    for c in (32, 48):
        f = jx[f"xs{c}"]
        check(f[f"switch_is_mask{c}"] and f["shared_is_x"] and not f["switch_is_mask64"], f"{v}: xs{c}: the switch reads exactly mask(x, {c}); router, shared expert and its gate read x")
        check(ax[f"xs{c}_wide"].tobytes() == ar[f"ref_mask{c}_wide"].tobytes() and ax[f"xs{c}_narrow"].tobytes() == ar[f"ref_mask{c}_narrow"].tobytes(),
              f"{v}: xs{c} outputs == the reference whose switch masks its own input, bit for bit (wide and narrow)")
        check(f["xs_rows"] == (40 + 8) * 3 and f["xs_calls"] == 6, f"{v}: xs{c}: every row masked and counted once ({f['xs_rows']} = 48 rows x 3 layers, {f['xs_calls']} calls)")
    f = jx["topk+xs32"]
    check(f["switch_is_mask32"] and f["shared_is_x"] and ax["topk+xs32_wide"].tobytes() == ar["ref_topk_mask32_wide"].tobytes()
          and ax["topk+xs32_narrow"].tobytes() == ar["ref_topk_mask32_narrow"].tobytes(), f"{v}: the stacked arm topk+xs32 == topk on the masked switch input, bit for bit (the harness's own narrow path and its combine)")
    check(all(jx[k]["xs_c_after_clear"] is None for k in jx if k != "_meta") and ax["baseline_again_wide"].tobytes() == ar["baseline_wide"].tobytes()
          and ax["baseline_again_narrow"].tobytes() == ar["baseline_narrow"].tobytes(), f"{v}: clear() switches the arm off; the baseline after the xs arms == today's, bit for bit")
    check(jx["_meta"]["xs_arms"] == ["xs32", "xs40", "xs48", "xs64"], f"{v}: arms registered: {jx['_meta']['xs_arms']}; wrapped classes {jx['xs32']['wrapped']}")

# ---------------------------------------------------------------- 3 measure()
say("\n== 3 measure(): the masked-row record and its WARNING ==")
jm, _ = case("measure", "applied", "measure", True); jk, _ = case("measure_broken", "applied", "measure_broken", True)
if jm and jk:
    ok = all(jm[a][w]["xs"] == {"c": c, "rows": 48 * n} and jm[a][w]["warn"] is None for a, c in (("xs48", 48), ("topk3+xs32", 32)) for w, n in (("w_a.npz", 46), ("w_b.npz", 59)))
    check(ok, "xs48 and topk3+xs32: each window records 48 x (ids - 1) masked switch rows, no warning")
    check(all(jm["xs64"][w]["xs"] is None and jm["xs64"][w]["warn"] is None for w in jm["xs64"]), "xs64: no xs record, no warning (nothing is masked)")
    check(all(jk["xs48"][w]["warn"] and "WARNING xs masked" in jk["xs48"][w]["warn"] for w in jk["xs48"]), "a layer whose experts escape the mask raises the WARNING and marks the window")
    check(jm["xs48_file"] and jm["topk3+xs32_file"], "the arm files are written under their own names (xs48.json, topk3+xs32.json)")
else: check(False, "measure scenarios ran")

# ---------------------------------------------------------------- 4 the patch
say("\n== 4 the patch ==")
want = {"dm_common.py": "78f789c9dd45f742581f9403205a4132", "dm_ablations.py": "e5c608ed0b572ad6e2f419e03da5b48f"}
check(all(md5(os.path.join("base", f)) == m for f, m in want.items()), "base/ holds the files the patch was cut against")
live = {f: md5(os.path.join(DM, f)) for f in want}
check(live == want, "the live decision_map files are still those files (the patch applies to today's HEAD)" + ("" if live == want else f": NOW {live}"))
check(not os.path.exists(os.path.join(DM, "dm_xs.py")) or md5(os.path.join(DM, "dm_xs.py")) == md5("dm_xs.py"), "decision_map has no other dm_xs.py")
S = os.path.join(T, "sandbox"); os.makedirs(S)
for f in want: shutil.copy(os.path.join("base", f), S)
r = subprocess.run(["patch", "-p1", "--dry-run", "-i", os.path.join(HERE, "t1_xs.patch")], cwd=S, capture_output=True, text=True)
check(r.returncode == 0 and "fuzz" not in r.stdout and "offset" not in r.stdout, "patch --dry-run on today's files: clean, no fuzz, no offset")
r = subprocess.run(["sh", os.path.join(HERE, "apply.sh"), S], capture_output=True, text=True)
check(r.returncode == 0 and all(md5(os.path.join(S, f)) == md5(os.path.join("applied", f)) for f in list(want) + ["dm_xs.py"]), "apply.sh on a sandbox copy gives exactly applied/ (three files)")
r2 = subprocess.run(["sh", os.path.join(HERE, "apply.sh"), S], capture_output=True, text=True)
check(r2.returncode != 0, "apply.sh refuses to apply twice (the md5 guard)")
r = subprocess.run(["sh", os.path.join(HERE, "revert.sh"), S], capture_output=True, text=True)
check(r.returncode == 0 and all(md5(os.path.join(S, f)) == m for f, m in want.items()) and not os.path.exists(os.path.join(S, "dm_xs.py")), "revert.sh restores today's files and removes dm_xs.py")
r = subprocess.run([PY, "-I", "-m", "py_compile", "applied/dm_common.py", "applied/dm_ablations.py", "applied/dm_xs.py"], capture_output=True, text=True)
check(r.returncode == 0, "the three patched files compile")

# ---------------------------------------------------------------- 5 the reader
say("\n== 5 the reader ==")
r = subprocess.run([PY, "-I", "read_xs.py", "qom_0_live_ball+code", "top_k8", "topk10", "t3_all", "t4_all", "topk5"], capture_output=True, text=True)
rows = {l.split("|")[1].strip(): [c.strip() for c in l.split("|")[3:8]] for l in r.stdout.splitlines() if l.startswith("| ") and "arm" not in l}
ref = {}
for f in ("wall_class_flips.out", "wall_class_flips_out.out"):
    p = os.path.expanduser("~/FableGod/fablec/" + f)
    if os.path.exists(p):
        for l in open(p):
            if l.startswith("| "): c = [x.strip() for x in l.split("|")]; ref.setdefault(c[1], c[2:5])
got = [(a, rows.get(a, [])[:3], ref.get(a)) for a in ("qom_0_live_ball+code", "top_k8", "topk10")]
check(all(g[1] == g[2] and g[2] is not None for g in got), "read_xs.py == fablec/wall_class_flips on arms on disk (all-row, non-tie, tie): " + "; ".join(f"{a} {x[0]} | {x[1]}" for a, x, _ in got if x))
for a in ("t4_all", "t3_all", "top_k8", "topk5"):
    if a in rows: say(f"      neighbour {a}: all-row {rows[a][0]}; non-tie {rows[a][1]}")
F = os.path.join(T, "fakeout"); os.makedirs(F)
src = os.path.join(DM, "out")
for dst, frm in (("baseline", "baseline"), ("xs64", "topk10"), ("xs40", "top_k8"), ("xs32", "topk5"), ("qom_0_live_ball+code", "qom_0_live_ball+code"), ("qom_0_live_ball+xs48+code", "top_k8")):
    shutil.copy(os.path.join(src, frm + ".json"), os.path.join(F, dst + ".json"))
r = subprocess.run([PY, "-I", "read_xs.py", "--out", F, "xs64", "xs40", "xs32", "qom_0_live_ball+xs48+code"], capture_output=True, text=True)
o = r.stdout
check(r.returncode == 0 and "xs64: identity arm, 0 flips: OK" in o and "xs40 code: non-tie 0.21 % (predicted 0.17-0.24)" in o and "xs40 ifeval: non-tie 1.90 %" in o and "OVER" in o
      and "xs32: lever check, non-tie code 1.03 %" in o and "ADDED by xs48" in o and "qom_0_live_ball+xs48+code code: all-row" in o,
      "read_xs.py's verdict paths on stand-in files (identity, standalone fit and kill check, stacked arm vs its parent)")
shutil.rmtree(T, ignore_errors=True)
for d in glob.glob(os.path.join(HERE, "**", "__pycache__"), recursive=True): shutil.rmtree(d, ignore_errors=True)
say(f"\n{'ALL PASS' if not FAILS else 'FAILED: ' + str(len(FAILS))} ({sum(1 for l in OUT if l.startswith('PASS'))} checks passed, {len(FAILS)} failed)")
open("SELFTEST.out", "w").write("\n".join(OUT) + "\n")
sys.exit(1 if FAILS else 0)
