"""Pass/fail table for a variant against both anchors (floor = same weights, accepted = ar4). CPU only.

Teacher-forced metrics are paired by position against the cached reference; CIs come from a cluster bootstrap
(clusters = generated sequences, and 1,024-token blocks of the long documents). Multiple-choice deltas are paired per
item (normal-approximation CI on the paired differences, exact McNemar for the discordant pairs)."""
from __future__ import annotations
import json, math, os, sys
from statistics import NormalDist
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
REFC = os.path.join(ROOT, "cache", "ref", "tf_ref.npz")
MARGINS = json.load(open(os.path.join(ROOT, "evalsuite", "margins.json")))
Z = NormalDist().inv_cdf(0.5 + MARGINS["ci_level"] / 2)
ANCHOR, FLOOR = "ar4", "floor"
GEN_SETS = ("tf_gsm", "tf_ifeval", "tf_code", "tf_multi")


# ---------------------------------------------------------------------------------------------- statistics
def mcnemar_p(b, c):
    """exact two-sided McNemar (binomial on the discordant pairs)"""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def paired_ci(d):
    d = np.asarray(d, float)
    m, se = d.mean(), d.std(ddof=1) / math.sqrt(len(d)) if len(d) > 1 else 0.0
    return m, m - Z * se, m + Z * se


def cluster_boot(num, den, cl, B, seed, stat="ratio"):
    """ratio of sums (num / den) with a cluster bootstrap CI; num, den per position, cl cluster id per position"""
    u, inv = np.unique(cl, return_inverse=True)
    sn, sd = np.bincount(inv, num, len(u)), np.bincount(inv, den, len(u))
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, len(u), (B, len(u)))
    if stat == "ratio":
        est = sn.sum() / sd.sum()
        bs = sn[pick].sum(1) / np.maximum(sd[pick].sum(1), 1e-30)
    else:                                         # difference of means (den = 1 per position as weights)
        cnt = np.bincount(inv, minlength=len(u)).astype(float)
        est = (sn.sum() - sd.sum()) / cnt.sum()
        bs = (sn[pick].sum(1) - sd[pick].sum(1)) / cnt[pick].sum(1)
    a = (1 - MARGINS["ci_level"]) / 2
    return est, float(np.quantile(bs, a)), float(np.quantile(bs, 1 - a))


def quantile_ratio(x, y, cl, q, B, seed):
    u, inv = np.unique(cl, return_inverse=True)
    groups = np.split(np.argsort(inv, kind="stable"), np.cumsum(np.bincount(inv))[:-1])
    rng = np.random.default_rng(seed)
    est = np.quantile(x, q) / max(np.quantile(y, q), 1e-30)
    bs = []
    for _ in range(B):
        idx = np.concatenate([groups[g] for g in rng.integers(0, len(u), len(u))])
        bs.append(np.quantile(x[idx], q) / max(np.quantile(y[idx], q), 1e-30))
    a = (1 - MARGINS["ci_level"]) / 2
    return est, float(np.quantile(bs, a)), float(np.quantile(bs, 1 - a))


# ---------------------------------------------------------------------------------------------- loading
def tf(name):
    p = os.path.join(RES, name, "tf.npz")
    return dict(np.load(p)) if os.path.exists(p) else None


def mc(name):
    p = os.path.join(RES, name, "mc.npz")
    return dict(np.load(p)) if os.path.exists(p) else None


def clusters(z):
    sets = z["sets"][z["seq"]]
    doc = sets == "tf_doc"
    return np.where(doc, 100000 + z["seq"] * 1000 + z["rel"] // MARGINS["bootstrap"]["doc_block_tokens"], z["seq"]), doc


def overlap(ra, rb, chunk=20000):
    """mean over layers of |top-10(a) & top-10(b)| / 10, per position"""
    out = np.empty(len(ra), np.float32)
    for s in range(0, len(ra), chunk):
        a, b = ra[s:s + chunk].astype(np.int32), rb[s:s + chunk].astype(np.int32)
        out[s:s + chunk] = (a[..., :, None] == b[..., None, :]).any(-1).sum(-1).mean(-1) / a.shape[-1]
    return out


# ---------------------------------------------------------------------------------------------- tables
def tf_rows(variant):
    ref = dict(np.load(REFC)) if os.path.exists(REFC) else None
    zv, za, zf = tf(variant), tf(ANCHOR), tf(FLOOR)
    if ref is None or zv is None:
        return [], {}
    bt = MARGINS["bootstrap"]
    nm = MARGINS["noninferiority"]
    cl, doc = clusters(zv)
    rows, info = [], {"positions": int(len(cl)), "mean_ref_tail_mass": float(zv["p_rest"].mean())}
    for part, m in (("own outputs", ~doc), ("documents", doc)):
        if not m.any():
            continue
        kv = zv["kld"][m]
        r = {"test": f"TF KLD mean ({part})", "variant": f"{kv.mean():.5f}"}
        if zf is not None:
            r["floor"] = f"{zf['kld'][m].mean():.5f}"
        if za is not None:
            ka = za["kld"][m]
            e, lo, hi = cluster_boot(kv, ka, cl[m], bt["resamples"], bt["seed"])
            r.update(accepted=f"{ka.mean():.5f}", vs_accepted=f"x{e:.2f} [{lo:.2f}, {hi:.2f}]",
                     ok=hi <= nm["kld_mean_ratio_max"])
        rows.append(r)
        r = {"test": f"TF KLD P99.9 ({part})", "variant": f"{np.quantile(kv, 0.999):.4f}"}
        if zf is not None:
            r["floor"] = f"{np.quantile(zf['kld'][m], 0.999):.4f}"
        if za is not None:
            ka = za["kld"][m]
            e, lo, hi = quantile_ratio(kv, ka, cl[m], 0.999, bt["quantile_resamples"], bt["seed"])
            r.update(accepted=f"{np.quantile(ka, 0.999):.4f}", vs_accepted=f"x{e:.2f} [{lo:.2f}, {hi:.2f}]",
                     ok=hi <= nm["kld_p999_ratio_max"])
        rows.append(r)
        av = zv["agree"][m].astype(float)
        r = {"test": f"TF top-1 agreement ({part})", "variant": f"{100 * av.mean():.2f}%"}
        if zf is not None:
            r["floor"] = f"{100 * zf['agree'][m].mean():.2f}%"
        if za is not None:
            aa = za["agree"][m].astype(float)
            e, lo, hi = cluster_boot(av, aa, cl[m], bt["resamples"], bt["seed"], stat="diff")
            r.update(accepted=f"{100 * aa.mean():.2f}%", vs_accepted=f"{100 * e:+.2f} [{100 * lo:+.2f}, {100 * hi:+.2f}] pt",
                     ok=100 * lo >= -nm["top1_agree_drop_max_points"])
        rows.append(r)
        if "routes" in zv and "routes" in ref:
            ov = overlap(zv["routes"][m], ref["routes"][m])
            r = {"test": f"Router top-10 overlap ({part})", "variant": f"{100 * ov.mean():.2f}%"}
            if zf is not None and "routes" in zf and zf["routes"].any():
                r["floor"] = f"{100 * overlap(zf['routes'][m], ref['routes'][m]).mean():.2f}%"
            if za is not None and "routes" in za:
                oa = overlap(za["routes"][m], ref["routes"][m])
                e, lo, hi = cluster_boot(ov, oa, cl[m], bt["resamples"], bt["seed"], stat="diff")
                r.update(accepted=f"{100 * oa.mean():.2f}%", vs_accepted=f"{100 * e:+.2f} [{100 * lo:+.2f}, {100 * hi:+.2f}] pt",
                         ok=100 * lo >= -nm["router_overlap_drop_max_points"])
            rows.append(r)
    # KLD by position on the documents: buckets, and the long-range rule
    if doc.any():
        rel = zv["rel"]
        edges = [0, 1024, 4096, 16384, 1 << 30]
        parts = []
        for a, b in zip(edges, edges[1:]):
            m = doc & (rel >= a) & (rel < b)
            if not m.any():
                continue
            s = f"{a // 1024}K+: {zv['kld'][m].mean():.4f}"
            if za is not None:
                s += f" (acc {za['kld'][m].mean():.4f})"
            parts.append(s)
        info["kld_by_position_docs"] = parts
        m = doc & (rel >= 16384)
        if m.any() and za is not None:
            e, lo, hi = cluster_boot(zv["kld"][m], za["kld"][m], cl[m], bt["resamples"], bt["seed"])
            first = doc & (rel < 4096)
            flat_v = zv["kld"][m].mean() / max(zv["kld"][first].mean(), 1e-30)
            flat_a = za["kld"][m].mean() / max(za["kld"][first].mean(), 1e-30)
            rows.append({"test": "TF KLD at 16K+ (documents)", "variant": f"{zv['kld'][m].mean():.5f}",
                         "floor": f"{zf['kld'][m].mean():.5f}" if zf is not None else "",
                         "accepted": f"{za['kld'][m].mean():.5f}", "vs_accepted": f"x{e:.2f} [{lo:.2f}, {hi:.2f}]",
                         "ok": hi <= nm["long_position_ratio_max"],
                         "note": f"16K+/0-4K: variant {flat_v:.2f}, accepted {flat_a:.2f}"})
    return rows, info


def tfstep(name):
    p = os.path.join(RES, name, "tfstep.npz")
    return dict(np.load(p)) if os.path.exists(p) else None


def at_positions(z, zv):
    """z's per-position arrays at zv's (seq, pos) positions (same item order), or None if any is missing"""
    if z is None:
        return None
    idx = {sp: j for j, sp in enumerate(zip(z["seq"].tolist(), z["pos"].tolist()))}
    sel = [idx.get(sp) for sp in zip(zv["seq"].tolist(), zv["pos"].tolist())]
    if any(j is None for j in sel):
        return None
    return {k: z[k][np.asarray(sel)] for k in ("kld", "agree")}


def tfstep_rows(variant):
    """the one-token-per-call teacher-forced pass (stages tfstep) on the own-output sequences. floor = the reference
    itself through the same one-token calls (the step path's own noise); accepted = ar4's layer-major pass at the same
    positions (without that noise, so the comparison leans against the variant)."""
    zv = tfstep(variant)
    if zv is None or variant == "ref":
        return [], {}
    zf, za = at_positions(tfstep("ref"), zv), at_positions(tf(ANCHOR), zv)
    bt, nm = MARGINS["bootstrap"], MARGINS["noninferiority"]
    cl, kv, av = zv["seq"], zv["kld"], zv["agree"].astype(float)
    sets = zv["sets"][zv["seq"]]
    info = {"tfstep_positions": int(len(kv)), "tfstep_sequences": int(len(np.unique(cl))),
            "tfstep_kld_by_set": {str(s): round(float(kv[sets == s].mean()), 5) for s in np.unique(sets)}}
    r = {"test": "TF-step KLD mean (own outputs)", "variant": f"{kv.mean():.5f}"}
    if zf is not None:
        r["floor"] = f"{zf['kld'].mean():.5f}"
    if za is not None:
        e, lo, hi = cluster_boot(kv, za["kld"], cl, bt["resamples"], bt["seed"])
        r.update(accepted=f"{za['kld'].mean():.5f}", vs_accepted=f"x{e:.2f} [{lo:.2f}, {hi:.2f}]",
                 ok=hi <= nm["kld_mean_ratio_max"])
    rows = [r]
    r = {"test": "TF-step KLD P99.9 (own outputs)", "variant": f"{np.quantile(kv, 0.999):.4f}"}
    if zf is not None:
        r["floor"] = f"{np.quantile(zf['kld'], 0.999):.4f}"
    if za is not None:
        e, lo, hi = quantile_ratio(kv, za["kld"], cl, 0.999, bt["quantile_resamples"], bt["seed"])
        r.update(accepted=f"{np.quantile(za['kld'], 0.999):.4f}", vs_accepted=f"x{e:.2f} [{lo:.2f}, {hi:.2f}]",
                 ok=hi <= nm["kld_p999_ratio_max"])
    rows.append(r)
    r = {"test": "TF-step top-1 agreement (own outputs)", "variant": f"{100 * av.mean():.2f}%"}
    if zf is not None:
        r["floor"] = f"{100 * zf['agree'].mean():.2f}%"
    if za is not None:
        aa = za["agree"].astype(float)
        e, lo, hi = cluster_boot(av, aa, cl, bt["resamples"], bt["seed"], stat="diff")
        r.update(accepted=f"{100 * aa.mean():.2f}%", vs_accepted=f"{100 * e:+.2f} [{100 * lo:+.2f}, {100 * hi:+.2f}] pt",
                 ok=100 * lo >= -nm["top1_agree_drop_max_points"])
    rows.append(r)
    if zf is None:
        rows[0]["note"] = "no reference tfstep run yet (floor column = the step path's own noise)"
    return rows, info


def mc_correct(z):
    lv = np.where(np.arange(10)[None] < z["n_opts"][:, None], z["letters"], -np.inf)
    pred = lv.argmax(1)
    p = np.exp(lv - lv.max(1, keepdims=True))
    p /= p.sum(1, keepdims=True)
    return pred, (pred == z["gold"]).astype(float), p[np.arange(len(p)), z["gold"]], np.exp(z["letters"]).sum(1)


def mc_rows(variant):
    zr, zv, za, zf = mc("ref"), mc(variant), mc(ANCHOR), mc(FLOOR)
    if zr is None or zv is None:
        return [], {}
    nm = MARGINS["noninferiority"]
    pr, cr, gr, mass = mc_correct(zr)
    pv, cv, gv, _ = mc_correct(zv)
    rows, info = [], {"items": int(len(cr)), "ref_letter_mass_median": float(np.median(mass)),
                      "ref_accuracy": {s: float(cr[zr["sets"] == s].mean()) for s in np.unique(zr["sets"])}}
    for label, m in (("pooled", np.ones(len(cr), bool)),) + tuple((s, zr["sets"] == s) for s in np.unique(zr["sets"])):
        d = 100 * (cv[m] - cr[m])
        e, lo, hi = paired_ci(d)
        b, c = int(((cr[m] == 1) & (cv[m] == 0)).sum()), int(((cr[m] == 0) & (cv[m] == 1)).sum())
        r = {"test": f"MC accuracy delta ({label}, n={int(m.sum())})", "variant": f"{100 * cv[m].mean():.1f}% vs ref {100 * cr[m].mean():.1f}%",
             "vs_ref": f"{e:+.2f} [{lo:+.2f}, {hi:+.2f}] pt, McNemar p={mcnemar_p(b, c):.2f} ({b}/{c})"}
        if zf is not None:
            r["floor"] = f"{100 * (mc_correct(zf)[1][m] - cr[m]).mean():+.2f} pt"
        if za is not None:
            r["accepted"] = f"{100 * (mc_correct(za)[1][m] - cr[m]).mean():+.2f} pt"
        if label == "pooled":
            r["ok"] = lo > -MARGINS["equivalence_points"] and hi < MARGINS["equivalence_points"]
        rows.append(r)
    fv = (pv != pr).astype(float)
    r = {"test": "MC flips vs reference (pooled)", "variant": f"{100 * fv.mean():.2f}% ({int(fv.sum())})"}
    if zf is not None:
        ff = (mc_correct(zf)[0] != pr).astype(float)
        r["floor"] = f"{100 * ff.mean():.2f}% ({int(ff.sum())})"
    if za is not None:
        fa = (mc_correct(za)[0] != pr).astype(float)
        e, lo, hi = paired_ci(100 * (fv - fa))
        r.update(accepted=f"{100 * fa.mean():.2f}% ({int(fa.sum())})", vs_accepted=f"{e:+.2f} [{lo:+.2f}, {hi:+.2f}] pt",
                 ok=hi <= nm["mc_flip_rise_max_points"])
    rows.append(r)
    e, lo, hi = paired_ci(100 * (gv - gr))
    rows.append({"test": "MC p(correct letter) delta (pooled)", "variant": f"{100 * gv.mean():.2f}% vs ref {100 * gr.mean():.2f}%",
                 "vs_ref": f"{e:+.2f} [{lo:+.2f}, {hi:+.2f}] pt",
                 "accepted": f"{100 * (mc_correct(za)[2] - gr).mean():+.2f} pt" if za is not None else ""})
    return rows, info


def table(rows):
    cols = [("test", "test"), ("variant", "variant"), ("floor", "floor (same weights)"), ("accepted", "accepted (ar4)"),
            ("vs_accepted", "variant vs accepted [90% CI]"), ("vs_ref", "variant vs reference [90% CI]"), ("ok", "pass")]
    cols = [c for c in cols if any(c[0] in r for r in rows)]
    cell = lambda r, k: ("PASS" if r[k] else "FAIL") if k == "ok" and k in r else ("-" if k == "ok" else str(r.get(k, "")))
    w = [max(len(h), *(len(cell(r, k)) for r in rows)) for k, h in cols]
    line = lambda vals: "  ".join(v.ljust(n) for v, n in zip(vals, w))
    out = [line([h for _, h in cols]), line(["-" * n for n in w])]
    out += [line([cell(r, k) for k, _ in cols]) for r in rows]
    notes = [f"  {r['test']}: {r['note']}" for r in rows if r.get("note")]
    return "\n".join(out + notes)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    variant = argv[0]
    as_json = "--json" in argv
    tr, ti = tf_rows(variant)
    sr, si = tfstep_rows(variant)
    mr, mi = mc_rows(variant)
    rows, ti = tr + sr + mr, {**ti, **si}
    refrun = argv[argv.index("--ref") + 1] if "--ref" in argv else "ref"
    if "--decode" in argv:
        from . import decode_report
        dr, di = decode_report.rows(variant, ref=refrun)
        rows += dr
        mi["decode"] = di
    judged = [r for r in rows if "ok" in r]
    verdict = "PASS" if judged and all(r["ok"] for r in judged) else ("INCOMPLETE" if not judged else "FAIL")
    missing = [n for n in ("ref", FLOOR, ANCHOR, variant)
               if not any(os.path.exists(os.path.join(RES, n, f)) for f in ("run.json", "tfstep.npz", "decode.json"))]
    if as_json:
        print(json.dumps(dict(variant=variant, verdict=verdict, rows=rows, tf=ti, mc=mi, missing=missing), default=str, indent=1))
        return
    print(f"evalsuite: {variant} vs " + ("the 4-bit reference" if refrun == "ref" else f"{refrun} (as the reference)")
          + "  (anchors: floor = same weights, accepted = AutoRound int4 experts)")
    print(table(rows) if rows else "(no results yet)")
    for k, v in {**ti, **mi}.items():
        print(f"  {k}: {v}")
    if missing:
        print(f"  missing runs: {missing}")
    print(f"VERDICT: {verdict}  ({sum(1 for r in judged if r['ok'])}/{len(judged)} rules pass; margins: evalsuite/margins.json)")


if __name__ == "__main__":
    main()
