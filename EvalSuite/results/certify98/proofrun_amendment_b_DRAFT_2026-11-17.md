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
- **The candidate:** <t30b-class head> at <checkpoint path>.
  - checkpoint sha256 <...>; code commit <TrainLab ...>; settings <as MTP Head names them>;
  - the modules map (path: sha256), ~/TrainLab/mtp125.py included (amendment 7a, item 4).
- **The capture + dump script:** BigMoE's <scripts/dd_capture_b.py or the capture with --dump> at commit <...>,
  sha256 <...>.
  - The passes are scripts/dd_topk.py (sha256 <...>) and scripts/dd_tree.py (sha256 <...>), called per file in the
    sitting, with these arguments:
    - TOPK as registered (layout 9f49e03);
    - T8: k 8, 32 nodes (amendment 7a, item 2).
  - NOT in the sitting: amendment 7a's conditional HE+ T8 at 127 nodes. It needs Wednesday's probe and a by-day window,
    and without wd it cannot run later. Under (b) it is dropped, and the dev K-curve stays the record. If NewIdeas
    wants it, it must be named here and run in the same sitting, before this amendment is filed.
- **The scorers:** tools/certify98.py <sha256>, tools/certify98_tree.py 4a0ba21625497be1..., and
  tools/certify98_dumpcheck.py <sha256 of its (b) mode, below>.
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
       would give. The fp16 file is hashed BEFORE the passes, and that sha256 goes in the dump manifest.
  3. The slim file is written: the capture's arrays minus wd (<the exact list: ids, lp, tk, gen, raw taps ...>). Its
     sha256 goes in files_sha256.json, and the temporary wd file is deleted.
- **Disk:** at most one file's wd exists at a time.
- **Memory:** the head-only passes need ~12.5 GB beside the capture's process. BigMoE states how it runs them (the
  same process, or an alternation), and the NO-SWAP gate it uses.

### 3. The integrity checks
- **Before the capture (Wednesday by day, on the dev bank, not the fresh files):**
  - BigMoE runs the sitting's code on heplus24 (and mbpp100).
  - Its TOPK and T8-32 must equal dd_topk.py / dd_tree.py's offline output on the same files BITWISE, on every field
    and in every window. EvalSuite reads the record.
  - If they are not bitwise equal, (b) is not used.
- **After the capture, before any scoring:**
  - EvalSuite copies files_sha256.json and dumps_sha256.json to results/certify98/ and commits both.
  - tools/certify98_dumpcheck.py must pass in its (b) mode, `--gated --pins <this amendment's pins> --fresh --incapture`.
    It checks:
    - `_run.in_capture: true`, with the capture + dump script's sha256;
    - one dump entry per fresh file;
    - each dump's fp16-wd sha256 is present;
    - each dump's row count and ids match its slim file's;
    - the slim files' sha256s match files_sha256.json, whose manifest_sha256 is acdf3e62...
  - The (b) mode is EvalSuite's to build and test before this amendment is filed. Its sha256 is pinned above.
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
