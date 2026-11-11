"""C3-R7 (Cloud Fable 3 via the orchestrator, Thu 2026-11-05 00:4x; answer rows only, its choice): the decision map's router top-10
SCORES beside the ids for the 36 windows, from the map's 11-03 xcapture (~/FableGod/decision_map/out/xcap, read-only).
What the scores are (dm_common.py moe_inner, CAPR "x" branch): per MoE layer, the block's router logits g -> softmax(g, fp32,
precise) -> the top-10 by descending probability (rank-sorted, the same order as out/routing's ids) -> renormalised over those 10
(norm_topk_prob) -> float16. Rows: the ANSWER rows only (positions a .. T-2 of the window; out/routing/<window>.npz ids[a:]
are the same ids, checked here per window). Baseline model (no perturbation), teacher-forced.
Out (per window): <window>.npz with ids [m, 48, 10] int16, scores [m, 48, 10] float16, a (the first answer row in
out/routing's row index), plus README.md. Usage (BigMoE venv): python export_scores.py <dest dir>"""
import json, os, sys
import numpy as np
DM = os.path.expanduser("~/FableGod/decision_map/out")
X, R = os.path.join(DM, "xcap"), os.path.join(DM, "routing")
dst = os.path.expanduser(sys.argv[1]); os.makedirs(dst, exist_ok=True)
base = json.load(open(os.path.join(DM, "baseline.json")))
rows = 0; bad = []
for nm in sorted(os.listdir(X)):
    z = np.load(os.path.join(X, nm)); ids, w, a = z["ids"], z["w"], int(z["a"])
    r = np.load(os.path.join(R, nm))["ids"]
    ok = (ids.shape == w.shape and ids.shape[1:] == (48, 10) and np.array_equal(ids, r[a:]) and ids.shape[0] == len(base[nm]["tk"])
          and np.allclose(w.astype(np.float32).sum(-1), 1.0, atol=1e-2))
    if not ok: bad.append(nm); continue
    np.savez_compressed(os.path.join(dst, nm), ids=ids.astype(np.int16), scores=w.astype(np.float16), a=np.int64(a))
    rows += ids.shape[0]
open(os.path.join(dst, "README.md"), "w").write(
    "# decision_map_scores (C3-R7; Breakthrough, Thu 5 Nov 2026)\n"
    "The decision map's router top-10 scores beside the ids, ANSWER rows only, for the 36 teacher-forced windows (same windows\n"
    "and names as data/decision_map_routing/).\n"
    "- Per window `<window>.npz`: `ids` [m, 48, 10] int16 (rank-sorted expert ids), `scores` [m, 48, 10] float16, `a` (int).\n"
    "- `scores`: per MoE layer, softmax of the router logits (fp32, precise), the top-10 by descending probability, renormalised\n"
    "  over those 10 (the model's norm_topk_prob), stored as float16; each row sums to 1 (within float16 rounding).\n"
    "- Rows: the answer rows only. Row i here is row a + i of decision_map_routing/<window>.npz's `ids` (identical ids,\n"
    "  checked per window); prompt rows have no scores in this export.\n"
    "- Source: the map's 11-03 xcapture (Breakthrough's harness, FableGod/decision_map/dm_common.py moe_inner, CAPR 'x'),\n"
    "  baseline model, no perturbation. Writer: Breakthrough scripts/dm_s7/export_scores.py.\n")
print(f"[export] {len(os.listdir(X)) - len(bad)} windows, {rows} answer rows -> {dst}; failed checks: {bad or 'none'}", flush=True)
sys.exit(1 if bad else 0)
