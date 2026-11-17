# CERTIFY98 candidate pin: t30b AS IS (MTP Head, Tue 17 Nov; NewIdeas 09:29 / the orchestrator's SLIM capture)
- Checkpoint: ~/datasets/deepdraft/ckpt/dd_t30b.safetensors, sha256
  790c42db393ad158f0f61e64962c868967995268babb816b27cbf8f4e215d502.
  - State dd_t30b.json: status done at step 4206.
  - Recipe (BigMoE TTT30B): dd_ttt.py train --run t30b --manifest data/deepdraft/ttt30b/split.json --pass-seed 1113;
    depth 30, rows 192, lr 1e-5 cosine to 10 %, wdecay 0.97, start b5.
- Code (every file clean in its repo; full sha256 in data/t30b_pin_sha256.txt):
  - TrainLab mtp125.py (the head model; commit 0c855a2, 2026-10-22) 8ca9b5dcde912071...
  - BigMoE HEAD 31f92cd1: scripts/dd_ttt.py 7864d28d..., dd_d22_ttt.py a6c0e892..., dd_tree.py 71b7faf5...,
    dd_topk.py 35556b4e...
  - EvalSuite tools/certify98.py (the scorer) c0528ad7...
- Load path: mtp125.load(frac 0.03) -> Trainable(head).load_weights(dd_t30b.safetensors), as dd_tree.py / dd_topk.py
  do, with the weights as trained (fp32 Linears, the full vocab).
- Settings: CERTIFY98 amendment 8's (EvalSuite).
- FLAG for EvalSuite: the ENGINE form of this head (P2's C0: 8-bit Linears + a 32K screen + session tokens) differs from
  the dumped form.
  - Sandbox (LAB): post-hoc 8-bit is free (S17, x0.995-1.001 kept); a 32K screen is free (S19).
  - The 125B's own Q8 / H32 / H16 cuts are MoonShots' Wed 05:24 / 06:24 run.
  - Which form CERTIFY98 certifies is EvalSuite's call.
