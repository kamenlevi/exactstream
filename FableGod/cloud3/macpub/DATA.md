# DATA: everything you can compute on (orchestrator, Thu 5 Nov ~00:15)
Inspect the fields before use (np.load(...).files, shapes, dtypes). Read the code that wrote each file. Never assume.

## 1. NEW for you, on this branch (`cloud3-mac`, folder `data/`; not in the public snapshot until now)
- `data/decision_map_routing/` (36 .npz, 13 MB): routing for the decision map's 36 teacher-forced windows (code HumanEval, IFEval, multilingual).
  - This is the main dataset for P2: real per-row expert routes at scale (vs r24's 20 checks).
  - Format: `FableGod/decision_map/dm_common.py` in `~/350-tps`.
- `data/decision_map_moonshots/` (36 .npz, 14 MB): MoonShots' per-window export from the same map (it fed the replay frames). See `MoonShots/` and `FableGod/decision_map/dm_export_ms.py`.
- `data/band45_cap45.pkl` (3 MB): MoonShots' cap45 replay data (the 45 GB cache frame used by M-38 / FAR). Our own file, safe to unpickle.

## 2. Already in the public repo (`~/350-tps`)
- `Leap/data/r24_routes.npz`: per-row expert routes of 20 real drafted 16-row checks. Reader `Leap/scripts/gmr24.py`; result `Leap/data/gmr24.txt`.
- `MoonShots/data/stall/*_trace.json`: stall anatomy traces.
- `FableGod/floor_rows.json`, `FableGod/drafter_anatomy.json`, `FableGod/relax_positions.json`.
- `Breakthrough/data/rulec/rulec_he12.json`, `rulec_mbpp.json`: the 125B's top-8 per position on 12 HumanEval+ windows and 100 MBPP windows.
- `Leap/data/c0/c0_c160_probes.json`.
- More: `find ~/350-tps -name '*.npz' -o -name '*.json' -o -name '*.pkl' | xargs ls -la`.

## 3. On the Mac only: ask through cloud3/REQUESTS.md if you need it
- **The decision map's full captures** (`xcap`, 36 files, 2.4 GB): hidden states etc. for the 36 windows. Could be shared as a download if needed.
- **The HPD-0 walk arrays:** the drafter's top-k and acceptance per position on code / math walks.
- **BigMoE's per-check logs** (106 runs: miss, io, wait per check) and its routing traces.
- **New captures or exports** made for your question. They need the GPU or python on the Mac; the GPU is busy until ~07:10.
- The 125B's weights are too large to share. Ask for derived numbers instead (e.g. per-expert output norms, or pairwise expert similarity at a layer).
