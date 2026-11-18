# DRAFT, not filed: CERTIFY98's proof-run amendment, variant (b): slim capture + the candidate's dumps written in the capture sitting

Prepared Tue 2026-11-17 08:3x at the orchestrator's request (08:3x, cc NewIdeas) as the FALLBACK for Wednesday's capture.
It is filed into docs/PREREG_certify98_2026-11-15.md only if all three hold (the orchestrator 08:3x / 08:4x):
- Kamen has not approved BOTH deletes by Wed ~15:00. With both approved, FULL runs as registered: wd stored, the single
  look offline Thursday by day.
- Kamen HAS approved at least the 3.8 GB rawtaps delete by Wed ~15:00. Without it, (b) ends Wednesday at ~96-98 GiB:
  106 now, minus tonight's 0.1-1.5, minus (b)'s ~7.5-8. That is below Kamen's 100 GiB floor.
- The candidate is pinned by Wed ~18:00 (NewIdeas + MTP Head; NewIdeas expects t30b as-is to be pinnable on Tuesday).
If Kamen approves neither delete, the capture slips.
Placeholders are in <angle brackets>.

## Amendment 9 (Wed 2026-11-18 <HH:Mx>, BEFORE the capture and before any candidate pass): the proof run, variant (b)

### Why
- The disk: FULL stores wd, the trunk's top pre-mixer widened stream (10240 fp16, 20.5 KB per row), on every fresh
  file. That is ~11-13.5 GB, and without both deletes the Mac would end below Kamen's 100 GiB floor (the orchestrator
  08:3x).
- The candidate's single pass reads wd. scripts/dd_topk.py feeds the head z["wd"] as its state input, cast to bf16.
  - wd cannot be rebuilt afterwards. The raw taps are means of the HC streams after layers 5 / 15 / 25 / 35 / 45, not
    the top's four streams.
  - A re-prefill (dd_reprefill.py) rebuilds it on the prefill path only, and its answer rows differ from the runtime
    (verify) path. Section 6's validity requires runtime-path features.
- So the candidate's passes run IN the capture sitting, from each file's wd, and wd is not stored.
- This is amendment 5's single look, moved earlier. It therefore needs this amendment, with every pin, BEFORE the
  capture. Section 2's rule is unchanged: no drafter runs on the fresh files before this amendment.

### 1. The pins (section 2's list: they would otherwise be filed after the capture)
- **The candidate:** t30b AS IS (MTP Head 09:30, ~/TrainLab/data/t30b_pin.md at TrainLab 82171b2; NewIdeas 09:3x).
  - ~/datasets/deepdraft/ckpt/dd_t30b.safetensors, sha256
    790c42db393ad158f0f61e64962c868967995268babb816b27cbf8f4e215d502 (recomputed here, Tue 09:3x). State done at step
    4206.
  - Recipe: BigMoE's TTT30B, dd_ttt.py train --run t30b, depth 30, rows 192.
  - Code, recomputed here Tue 09:3x, every file clean in its repo:
    - TrainLab mtp125.py (commit 0c855a2): 8ca9b5dcde9120715e3161d17d0e747177d82a922f6ea9d66f4e4f9a64534677;
    - BigMoE 31f92cd1: dd_ttt.py 7864d28da79680c31cb19d56a94c3b1de6c1708bdf90855e64a9a05cc7c9d560; dd_d22_ttt.py
      a6c0e892a1658b01fe704c37099024fabd0b5dccb22f760da7b74d66777f5b97; dd_tree.py
      71b7faf5ae8cbc98ce5c9ff3e13a2b23803cdc62043c5554fe8613aad092c3a2; dd_topk.py
      35556b4e46cb2dcfbf23c7a729a30cb6097f7524405af89c598dd6b9abf68f57.
    - If any changes before the capture, the pins are re-read and re-filed.
  - Settings: amendment 8's. The modules map (path: sha256) carries mtp125.py (amendment 7a, item 4).
  - **The certified form** (MTP Head asked; it is the registered reading):
    - The cells measure the head as dd_topk.py / dd_tree.py load it: mtp125.load(frac 0.03) -> Trainable(head)
      .load_weights(dd_t30b.safetensors), with fp32 Linears (the file's 11 tensors are all F32) and the full vocab.
      That is the form CERTIFY98 certifies.
    - **The engine check runs the same stored form** (the orchestrator 09:4x, via NewIdeas):
      - t30b as stored, on BigMoE's r57 stored-format path;
      - chain-32 (31 drafts + the current token, kept/32's cap) and exact acceptance;
      - >= 20 fresh HumanEval+ windows. The engine's tokens per check must be within +-3 % of kept_walk on the same
        windows. Failing it voids only the kept/32 claim.
      - **BigMoE confirmed the path (Tue 10:56):**
        - The flags: R57_DRAFT=t30b R57_T30B_FMT=stored R57_TREE_K=1 R57_TREE_NODES=31 (k = 1 best-first = the
          greedy 31-draft chain), with exact acceptance (no R57_RELAX_DELTA).
        - The weights: dd_t30b.safetensors as stored over the shipped MTP layer. That means fp32 Linears, the full
          248,320-row lm_head, no screen and no session tokens.
          - r57 loads them through its own model load, not mtp125.load, so identity is checked, not assumed (the
            rehearsal below).
        - The numerics (BigMoE 07:50 Wed, via NewIdeas): the stored form mirrors dd_topk's own block-level cast, so
          the count check's drafts match the dumps' TOPK numerics. TrainLab mtp125.py computes the shared expert and
          its se_gatew gate in fp32 (lines 135-139), then casts the whole MoE block's output to the stream dtype
          (line 172); checked here 07:5x. The shipping forms (Q8H32, c2) keep MoonShots' CastLinear on the se_gatew
          output instead (ra_sf's live-head form). They are not certified here.
        - The precision: MLX_ENABLE_TF32 unset, i.e. TF32 on (MLX 0.32.2's default), as the dumps ran. runs.json
          prints it with the MLX version.
        - Per window: tokens, checks, kept per check, and an identity flag of the answer tokens against the capture's
          ids. A differing window is dropped and counted.
        - **A dev rehearsal** (Wed by day): heplus24 vs kept_walk from the dev TOPK dump.
          - Each check's drafts up to its first miss are compared with dd_topk's TOPK top-1 at (fi, anchor row,
            depth). At least 99 % must be equal, and each mismatch is listed with dd_topk's top-1 minus top-2 margin.
            - Why not 100 % (NewIdeas, received 08:1x Wed): MoonShots' r57 drafter piece (week98/draft/r57_draft.py,
              261b266) matched 453 / 454 positions in FMT=stored chain-32, and the miss was a near-tie. A load error
              would show as a far lower rate.
            - Near-tie flips can move the engine's tokens per check either way. The rehearsal's own tokens per check
              vs kept_walk on the dev windows shows their net effect before the fresh check, whose line stays +-3 %.
          - It also shows the dropped-window rate. A 32-row verify is outside REF2-T's certified 1-16 rows.
      - **The windows (agreed with BigMoE, Tue 13:51):** the first 24 fresh HumanEval+ windows in the manifest's order. If fewer than 20 of
        them have identical answer tokens, the next windows in order are added until 20 do. Every drop is reported.
    - The shipping forms (Q8 or bf16 + the 32K screen, chosen Wed ~07:30) are not certified here.
      - In lossy mode, the gates (tier 1 + the card gate) run on the chosen form, and the headline timings use it.
      - The RESULT names both forms. MTP Head's sandbox read (8-bit x0.995-1.001 kept; the 32K screen free) is
        information.
- **The capture script:** BigMoE's new scripts/dd_cert.py at commit <...>, sha256 <...> (BigMoE sends both before
  filing; this is the pinned capture_script_sha256).
  - It is built like dd_mid.py (31f92cd1): dd_bank's runtime path verbatim, with BIGMOE_EXT_BUILD=build_qom.
  - The passes are the UNCHANGED scripts/dd_topk.py and scripts/dd_tree.py (the shas above). They run in a second
    process that follows the capture file by file, with these arguments:
    - TOPK as registered (layout 9f49e03);
    - T8: k 8, 32 nodes (amendment 7a, item 2).
  - DROPPED: amendment 7a's conditional HE+ T8 at 127 nodes. NewIdeas 09:3x: early death is KILLED (stage 2, as
    pre-registered), and the ADAPTIVE K build is stopped. So neither needs a fresh 127-node tree, and the gated lines
    read T8-32 only. The dev K-curve stays the record.
- **The scorers:** tools/certify98.py c0528ad765fd5e59885a1e1066292a3c4dbf6908064b83956b0280773b077eee (MTP Head's pin
  file names the same), tools/certify98_tree.py 4a0ba21625497be1..., and
  tools/certify98_dumpcheck.py 884e5cb415633eab53f89cb203b64a95d59b62e7106d8232daf09bda1c7d773f (with its (b) mode,
  --incapture; section 3).
- **The fresh-set manifest:** results/certify98/fresh_manifest_2026-11-15.json (manifest sha256 acdf3e62...). The
  capture takes exactly these ids.
- **The everyday pool:** amendment 3's, unchanged.

### 2. The sitting (Wednesday night, BigMoE's capture)
- **Per fresh file, in this order:**
  1. The runtime-path capture as registered: each answer decoded to EOS on the runtime path, capped at 512 tokens,
     then teacher-forced.
  2. The candidate's TOPK and T8-32 passes on that file's wd.
     - wd is first written exactly as FULL would store it: fp16, (N - 1, 10240). It is then read back through the
       offline code path (dd_topk / dd_tree's loader: fp16 -> bf16), so each dump equals what FULL + an offline pass
       would give. The fp16 file is hashed BEFORE the passes, and that sha256 goes in the dump manifest
     (`_run.wd_sha256`, keyed by the fresh file's name).
  3. The slim file is written: the capture's arrays minus wd (<the exact list: ids, lp, tk, gen, raw taps ...>). Its
     sha256 goes in files_sha256.json, and the temporary wd file is deleted.
- **How it runs (BigMoE, Tue 10:56):**
  - The capture writes each file in the bank format, with wd (fp16, as FULL would), to a temp path. It records the
    file's sha256 BEFORE any pass.
  - The follower process reads wd from that temp file and writes its dump parts. It then writes the slim file (the same
    arrays minus wd) and deletes that wd.
- **Memory:** the capture takes ~22 GB (the mid capture peaked at 21.8) and the head ~12.5 GB.
  - One share lock covers both: MEM 36 GB.
  - The capture is gated on NO-SWAP >= 42, on two reads 30 s apart. The follower is gated on its own NO-SWAP read
    before each file.
- **Disk:** a wd file exists until the follower has read it.
  - **The queue cap** (the orchestrator 11:1x): the wd queue on disk is capped at ~1 GB. While it is full, the capture
    waits rather than queueing more. BigMoE built it into dd_cert.py and Wednesday's dev check (Tue 13:51).
    - Without the cap, a lagging follower could queue the set's wd (~4.4 GB, BigMoE).
    - With it, the whole sitting stays above Kamen's 100 GiB floor. (b) ends at ~100-102 GiB with the 3.8 GB rawtaps
      delete, which is still pending Kamen's answer.

### 3. The integrity checks
- **Before the capture (Wednesday by day, on the dev bank, not the fresh files):**
  - BigMoE runs the sitting's code on heplus24 (and mbpp100).
  - Its TOPK and T8-32 must equal dd_topk.py / dd_tree.py's offline output on the same files BITWISE, on every field
    and in every window. EvalSuite reads the record.
  - If they are not bitwise equal, (b) is not used.
- **After the capture, before any scoring:**
  - EvalSuite copies files_sha256.json and dumps_sha256.json to results/certify98/ and commits both.
  - tools/certify98_dumpcheck.py must pass in its (b) mode:
    `--gated <files_sha256.json> --pins <this amendment's pins> --fresh <the fresh-set manifest> --incapture <the slim dir>`.
    The pins file also carries capture_script_sha256. Beyond the registered checks (amendments 7 / 7a), the mode checks:
    - `_run.in_capture` is true, and `_run.capture_script_sha256` is set (and pinned);
    - `_run.wd_sha256` holds exactly one fp16-wd sha256 (64 hex) per file of files_sha256.json;
    - every file of files_sha256.json is a plain name, and is in the slim dir with that sha256. files_sha256.json's
      manifest_sha256 is acdf3e62... (--fresh);
    - the dumps were computed on the stored token sequences: every TOPK cell's y == ids[row + 1] of its slim file, and
      every T8 node's y == ids[an + depth + 2] (-1 past the file's end);
    - every T8 dump's anchors (a_fi, a_an) are exactly its TOPK dump's depth-0 cells (fi, row - 1);
    - every fresh file is in exactly one named TOPK dump and in a named T8 dump.
  - EvalSuite built and tested the mode on Tue 09:4x: 24 new checks, and the old 26 unchanged, 50 / 50 as wanted
    (results/certify98/dumpcheck_b_checks_2026-11-17.txt). Its identities hold exactly on BigMoE's dev dumps (heplus24:
    186,465 TOPK cells, 204,000 T8-32 nodes, 6,375 anchors).
- **The scorer** runs only on dumps whose sha256 match the manifest, as registered. The RESULT quotes both manifests.

### 4. Unchanged
- Every line, bar and power of amendments 4 / 6 / 6a / 8 / 8a; the metric and its validity checks.
- The 2B rows (MoonShots) and C2-4B: they read token ids, not wd.
- The engine check (Thu-Fri): live on the engine, not from the capture.
- LAB comes from TOPK's depth-0 cells.

### 5. A consequence, stated before the capture
- Without wd, no other head can be scored on these fresh files later. That is the design's one-candidate look, made
  physical.
- A later candidate needs a new fresh capture.
