# DRAFT, not filed: CERTIFY98's proof-run amendment, FULL (as registered: wd stored, the single look offline Thursday by day)

Prepared Tue 2026-11-17 10:0x, beside the (b) draft (results/certify98/proofrun_amendment_b_DRAFT_2026-11-17.md).
- **When FULL applies:** Kamen approves both deletes by Wed ~15:00 (the orchestrator 08:3x / 08:4x). Otherwise (b)
  applies, on its three conditions, or the capture slips.
- **When it is filed:** into docs/PREREG_certify98_2026-11-15.md after the capture (it names the capture's sha256s, as
  the timeline registers: "capture; freeze the manifest; file the proof-run amendment"), and before any candidate pass.
- Placeholders are in <angle brackets>.

## Amendment 9 (<Wed 2026-11-18 or Thu 2026-11-19> <HH:Mx>, after the capture, before any candidate pass): the proof run, FULL

### 1. The pins (section 2's list)
- **The candidate:** t30b AS IS (MTP Head 09:30, ~/TrainLab/data/t30b_pin.md at TrainLab 82171b2; NewIdeas 09:3x).
  - ~/datasets/deepdraft/ckpt/dd_t30b.safetensors, sha256
    790c42db393ad158f0f61e64962c868967995268babb816b27cbf8f4e215d502 (state done at step 4206). The file's 11
    tensors are all F32.
  - **Recipe:** BigMoE's TTT30B, dd_ttt.py train --run t30b, depth 30, rows 192.
  - **Code** (recomputed Tue 09:3x, every file clean in its repo):
    - TrainLab mtp125.py (0c855a2): 8ca9b5dcde9120715e3161d17d0e747177d82a922f6ea9d66f4e4f9a64534677;
    - BigMoE 31f92cd1:
      - dd_ttt.py 7864d28da79680c31cb19d56a94c3b1de6c1708bdf90855e64a9a05cc7c9d560;
      - dd_d22_ttt.py a6c0e892a1658b01fe704c37099024fabd0b5dccb22f760da7b74d66777f5b97;
      - dd_tree.py 71b7faf5ae8cbc98ce5c9ff3e13a2b23803cdc62043c5554fe8613aad092c3a2;
      - dd_topk.py 35556b4e46cb2dcfbf23c7a729a30cb6097f7524405af89c598dd6b9abf68f57.
    - If any of them changes before the pass, the pins are re-read and re-filed.
  - **Settings:** amendment 8's. The modules map (path: sha256) carries mtp125.py (amendment 7a, item 4).
- **The certified form:** the head as dd_topk.py / dd_tree.py load it, mtp125.load(frac 0.03) -> Trainable(head)
  .load_weights(dd_t30b.safetensors), with fp32 Linears and the full vocab.
- **The engine check** (the orchestrator 09:4x) runs the same stored form:
  - t30b as stored, on BigMoE's r57 stored-format path;
  - chain-32 (31 drafts + the current token, kept/32's cap) and exact acceptance;
  - >= 20 fresh HumanEval+ windows. The engine's tokens per check must be within +-3 % of kept_walk on the same
    windows. Failing it voids only the kept/32 claim.
  - **BigMoE confirmed the path (Tue 10:56):**
    - The flags: R57_DRAFT=t30b R57_T30B_FMT=stored R57_TREE_K=1 R57_TREE_NODES=31 (k = 1 best-first = the greedy
      31-draft chain), with exact acceptance (no R57_RELAX_DELTA).
    - The weights: dd_t30b.safetensors as stored over the shipped MTP layer. That means fp32 Linears, the full
      248,320-row lm_head, no screen and no session tokens.
      - r57 loads them through its own model load, not mtp125.load, so identity is checked, not assumed (the
        rehearsal below).
    - The numerics (BigMoE 07:50 Wed, via NewIdeas): the stored form mirrors dd_topk's own block-level cast, so the
      count check's drafts match the dumps' TOPK numerics. TrainLab mtp125.py computes the shared expert and its
      se_gatew gate in fp32 (lines 135-139), then casts the whole MoE block's output to the stream dtype (line 172);
      checked here 07:5x. The shipping forms (Q8H32, c2) keep MoonShots' CastLinear on the se_gatew output instead
      (ra_sf's live-head form). They are not certified here.
    - The precision: MLX_ENABLE_TF32 unset, i.e. TF32 on (MLX 0.32.2's default), as the dumps ran. runs.json prints
      it with the MLX version.
    - Per window: tokens, checks, kept per check, and an identity flag of the answer tokens against the capture's
      ids. A differing window is dropped and counted.
    - **A dev rehearsal** (Wed by day): heplus24 vs kept_walk from the dev TOPK dump.
      - Each check's drafts up to its first miss are compared with dd_topk's TOPK top-1 at (fi, anchor row, depth).
        At least 99 % must be equal, and each mismatch is listed with dd_topk's top-1 minus top-2 margin.
        - Why not 100 % (NewIdeas, received 08:1x Wed): MoonShots' r57 drafter piece (week98/draft/r57_draft.py,
          261b266) matched 453 / 454 positions in FMT=stored chain-32, and the miss was a near-tie. A load error would
          show as a far lower rate.
        - Near-tie flips can move the engine's tokens per check either way. The rehearsal's own tokens per check vs
          kept_walk on the dev windows shows their net effect before the fresh check, whose line stays +-3 %.
      - It also shows the dropped-window rate. A 32-row verify is outside REF2-T's certified 1-16 rows.
  - **The windows (agreed with BigMoE, Tue 13:51):** the first 24 fresh HumanEval+ windows in the manifest's order. If fewer than 20 of
    them have identical answer tokens, the next windows in order are added until 20 do. Every drop is reported.
  - The shipping forms (Q8 or bf16 + the 32K screen) are not certified here. The lossy gates cover them in lossy mode.
- **The scorers:** tools/certify98.py c0528ad765fd5e59885a1e1066292a3c4dbf6908064b83956b0280773b077eee,
  tools/certify98_tree.py 4a0ba21625497be1..., and tools/certify98_dumpcheck.py
  884e5cb415633eab53f89cb203b64a95d59b62e7106d8232daf09bda1c7d773f (its --gated mode; the (b) mode is not used).
- **The fresh set:** results/certify98/fresh_manifest_2026-11-15.json (manifest sha256 acdf3e62...). The capture's
  files_sha256.json is sha256 <...>, committed to results/certify98/ before any pass.
- **The everyday pool:** amendment 3's, unchanged.

### 2. The single look (Thursday by day, BigMoE; ~0.8 h head-only by the bank's price)
- TOPK as registered (layout 9f49e03), plus T8 at k 8 and 32 nodes (amendment 7a, item 2), offline from the stored
  fresh files.
- **DROPPED:** amendment 7a's conditional HE+ T8 at 127 nodes. NewIdeas 09:3x: early death is KILLED (stage 2, as
  pre-registered), and the ADAPTIVE K build is stopped. So the gated lines read T8-32 only, and the dev K-curve stays the
  record.
- BigMoE writes dumps_sha256.json with amendment 7's _run fields. EvalSuite commits a copy before scoring.

### 3. The integrity checks (before any scoring)
- tools/certify98_dumpcheck.py must PASS with `--gated <files_sha256.json> --pins <this amendment's pins> --fresh
  <the fresh-set manifest>`.
- The scorer runs only on dumps whose sha256 match the manifest, as registered. The RESULT quotes both manifests.

### 4. Unchanged
- Every line, bar and power of amendments 4 / 6 / 6a / 8 / 8a; the metric and its validity checks.
- The 2B rows (MoonShots) and C2-4B.
- LAB comes from TOPK's depth-0 cells.
- wd stays stored. A later candidate may be scored on these files only under section 2's one-candidate-per-look rule
  (a second candidate on the same sets uses a one-sided 97.5 % bound).
