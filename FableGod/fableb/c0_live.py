"""c0_live.py (Fable B, mission B2): the live (unsynced, SB) warm 16-row check against the sum of its pieces' GPU evals,
per C0 probe (~/Leap/data/c0/c0_c160_probes.json). The difference bounds what the live check spends between kernels
outside the GPU evals (exposed CPU build, gate handshakes, command-buffer boundaries). Also the per-piece build / eval
split at 16 and 1 rows. CPU only; reads Leap's file."""
import json, statistics as st
P = json.load(open("/Users/kamen/Leap/data/c0/c0_c160_probes.json"))
SYNC = P["sync_ms"] if isinstance(P.get("sync_ms"), (int, float)) else 0.222
probes = P["probes"]
print("probes %d, sync %.3f ms" % (len(probes), SYNC))
PIECES = ("hc_a", "gdn", "att", "hc_m", "moe_pre", "experts", "hc_w", "ple", "hc_f")
rows = []
for pr in probes:
    x = pr.get("synced_16")
    wm = pr.get("warm") or []
    if not x or not wm:
        continue
    b = {}; e = {}; cnt = {}
    for kind, L, bb, ee in x["seg"]:
        b[kind] = b.get(kind, 0) + bb; e[kind] = e.get(kind, 0) + ee; cnt[kind] = cnt.get(kind, 0) + 1
    # the moe record nests moe_pre and experts (their build + eval sit inside moe's build span); what moe's own eval
    # span evaluates is the MoE tail, so moe_post eval = moe's eval - its syncs, and its build = moe's build - the
    # children's build + eval (Leap's kinds(): moe_post = inc(moe) - inc(moe_pre) - inc(experts) - syncs)
    moe_post_e = e.get("moe", 0) - cnt.get("moe", 0) * SYNC
    moe_post_b = b.get("moe", 0) - sum(b.get(k, 0) + e.get(k, 0) for k in ("moe_pre", "experts"))
    ev = sum(e.get(k, 0) - cnt.get(k, 0) * SYNC for k in PIECES) + moe_post_e
    bu = sum(b.get(k, 0) for k in PIECES) + max(moe_post_b, 0)
    top = x["wall"] - sum(bb + ee for kind, L, bb, ee in x["seg"] if kind in ("layer", "hc_f")) - SYNC   # embed / norm / head (build + eval)
    warm = [w["wall"] for w in wm]; wb = [w["build"] for w in wm]
    rows.append((pr["rows"], st.median(warm), st.median(wb), ev, bu, top, ev + top))
    print("  %-28s rows %2d: warm live wall %6.2f (forward's Python span %5.2f) | pieces' GPU eval %6.2f (moe_post %4.2f) + top-level %4.2f = %6.2f; pieces' build %5.2f | live - eval sum = %+5.2f ms"
          % (str(pr["key"])[:28], pr["rows"], st.median(warm), st.median(wb), ev, moe_post_e, top, ev + top, bu, st.median(warm) - (ev + top)))
r16 = [r for r in rows if r[0] >= 15]
print("16-row probes (%d): live warm wall median %.1f ms; GPU eval sum + top-level median %.1f; live - eval median %+.1f [%+.1f .. %+.1f]; pieces' CPU build median %.1f"
      % (len(r16), st.median(r[1] for r in r16), st.median(r[6] for r in r16), st.median(r[1] - r[6] for r in r16),
         min(r[1] - r[6] for r in r16), max(r[1] - r[6] for r in r16), st.median(r[4] for r in r16)))
