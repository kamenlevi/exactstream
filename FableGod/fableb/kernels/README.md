# kernels/ (Fable B, DENSE_REBUILD v2, Fri 6 Nov 2026). Drafts: nothing here ran on a GPU.
See ../DENSE_REBUILD.md section 7 for the file table.
Checks (CPU only): `~/BigMoE/.venv/bin/python -I metal_check.py` (offline Metal compile, 10 instances),
`... -I gdn_chunk.py` (algebra), `... -I eprep.py` (logic), `... -I pack_check.py` (W's stream, ~2 min), `python3 -I ksplan.py`.
