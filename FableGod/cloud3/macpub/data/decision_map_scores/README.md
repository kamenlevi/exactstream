# decision_map_scores (C3-R7; Breakthrough, Thu 5 Nov 2026)
The decision map's router top-10 scores beside the ids, ANSWER rows only, for the 36 teacher-forced windows (same windows
and names as data/decision_map_routing/).
- Per window `<window>.npz`: `ids` [m, 48, 10] int16 (rank-sorted expert ids), `scores` [m, 48, 10] float16, `a` (int).
- `scores`: per MoE layer, softmax of the router logits (fp32, precise), the top-10 by descending probability, renormalised
  over those 10 (the model's norm_topk_prob), stored as float16; each row sums to 1 (within float16 rounding).
- Rows: the answer rows only. Row i here is row a + i of decision_map_routing/<window>.npz's `ids` (identical ids,
  checked per window); prompt rows have no scores in this export.
- Source: the map's 11-03 xcapture (Breakthrough's harness, FableGod/decision_map/dm_common.py moe_inner, CAPR 'x'),
  baseline model, no perturbation. Writer: Breakthrough scripts/dm_s7/export_scores.py.
