# QWEN 4 DAY-0 READINESS KIT (MoonShots; started Sat 7 Nov 2026 23:4x EET, the orchestrator's ask for Kamen)

> (a quote from the project owner, kept private)
that generation of models and document absolutely everything real time ... set everything up for when qwen 4 releases".

## 0. Status and how to use it (one screen)
- **Watcher: RUNNING** (session-only background loop; `qwen4/watch.py`, log `qwen4/watch.log`). After every account
  move or restart: `cd ~/MoonShots/qwen4 && ./watch_loop.sh` in the background.
> (a quote from the project owner, kept private)
    - **ANNOUNCED** (blog, Qwen Chat, GitHub, HF org page; section 1): `qwen4/ANNOUNCED.json`, a macOS notification and
      a `peer say` to the orchestrator. It keeps watching, every 2 min instead of 5. Section 3's T-1d rows start.
    - **WEIGHTS** (a new Qwen HF repo named or configured Qwen 4): `qwen4/FOUND.json`, notification, `peer say`, stop.
  - No LaunchAgent for now (Kamen, Sun 8 Nov). The plist stays written and NOT loaded:
    `qwen4/com.moonshots.qwen4watch.plist`.
- **First command when it fires:** `python3 qwen4/port_check.py <hf repo id>`.
  - It reads config.json only and prints what changed vs today's model, by engine part.
  - It also prints the derived store sizes and slots per layer at 45 / 24 / 12 GB, and the download size and license (for
    Kamen's download OK).
- **Day-0 blockers to clear BEFORE the release (section 5):**
  - **Disk:** 88 GB free of 1.8 TB (Sat 7 Nov). Today's model folder is 225 GB; a Qwen 4 bf16 repo of today's size is
    360 GB. Kamen decides storage later; at the latest in the day between announcement and weights.
  - **Download OK:** standing, via the orchestrator (section 5).
  - **Owners' pre-work:** the converter, kernels and screen parametrised by config (section 2).
- **Today's model is already the Qwen 4 architecture preview** (Qwen3.8-Flash-Next, `qwen4_exp_text`, "a preview of the
  architecture that will underpin Qwen4", arXiv 2608.30320). If Qwen 4 keeps the architecture, the port is mostly sizes,
  and port_check shows which.

## 1. Watch
| source | what fires | tier |
|---|---|---|
| qwen.ai article list (`qwen.ai/api/v2/article/retrieval`, what qwen.ai/research loads; ~5 MB, every ~15 min) | a new post naming Qwen 4 (title or slug) | ANNOUNCE |
| Qwen Chat model list (`chat.qwen.ai/api/models`, public) | a new model naming Qwen 4 (e.g. qwen4-max) | ANNOUNCE |
| GitHub org QwenLM (newest 30 created + 30 pushed) | a new repo named Qwen 4, or a rename (Qwen3 -> Qwen4) | ANNOUNCE |
| QwenLM/Qwen3 README; the Hugging Face Qwen org card | a new line naming Qwen 4 | ANNOUNCE |
| Hugging Face Qwen collections (50) and Spaces (40) | a new one named Qwen 4 | ANNOUNCE |
| Hugging Face, author Qwen (newest 40) | a new repo named Qwen 4 (not Qwen4-Exp), or any new Qwen repo whose config.json says model_type / architecture qwen4 | WEIGHTS: stops the loop |
| huggingface/transformers, ml-explore/mlx-lm issues + PRs titled qwen4 (not Exp) | e.g. "Add Qwen4": can come days early | signal |
| HF search "qwen4" by known converters (mlx-community, youssofal / MTPLX, lmstudio-community, unsloth) | a 4-bit MLX upload = the fastest day-0 download | signal |
| github.com/youssofal/mtplx releases | any new release (our runtime base) | signal |

- Every tier notifies: macOS, plus a fixed-text `peer say` to the orchestrator. Third-party text stays in the local
  JSON files.
- A new source's first run only records a baseline. Checked Sun 8 Nov 08:53: nothing current matches the pattern.
- Dead or not watched:
  - the old blog RSS (qwenlm.github.io: last post Sep 2025; the blog moved to qwen.ai);
  - X and Weibo (no read-only access without an account; X's embed endpoint answers 429; no sign-ups);
  - ModelScope (no public JSON listing found).
  An X-only teaser is missed until a blog post, a Qwen Chat model or a GitHub repo appears.
- Read-only: public metadata and config.json text only; downloads go through the orchestrator (section 5).

## 2. Port map (every model-specific part of the engine; config-driven vs hard-coded)
Read-only scan, Sat 7 Nov 23:5x: BigMoE, Leap, TrainLab, EvalSuite, the decision map, Breakthrough, plus MoonShots'
planes_engine, the drafter's HF cache and mtplx 2.12.2. Rows 26-33 added Fri 13 Nov 17:4x (orchestrator): this week's P2,
P3 and DEEPDRAFT parts and what 1.2.0 / 1.2.1 add, from a read-only scan of BigMoE, Leap release/out/pkg_120b /
pkg_121 and FableGod deepdraft.
- **The core is config-driven.** The expert count and piece shapes come from the safetensors headers, the record sizes from
  records3/index.json, and the ext takes both as arguments.
- **Everything around the core is baked in:** top-k 10, the 48 x 512 tables, the 640 x 2560 expert shape in the
  converters / p11 / kernel asserts, the GDN dims in the exact kernels, the drafter (taps, hc 4), the EOS ids, and 4-bit
  affine g64 throughout.
- `qwen4/port_check.py` prints which of these hazards a new config trips.

| # | part | key files (path:line) | baked in | source | port action | owner |
|---|---|---|---|---|---|---|
| 1 | engine screen r57 (+ _rlx) | BigMoE scripts/r57_screen.py:75, 80-81, 107-108, 305, 412, 463, 597, 682, 755, 863, 954 | tier plan "4433333333", C_l 112, MAXK 160 / XLINES 4, gate 40 / 160, EOS, drafter taps, 48 / 512 literals, hc 4 in the tap reshape (layers :412 derived) | HARD | derive from config + drafter config; re-pin presets | BigMoE |
| 2 | pinned per-model data | r57:48-55 (md5 laysz45 pools), :453-454 (pop_cap45 (48,512)), :568-594 (corr, prior); bigmoe/stream.py:153-155, 583-584 | pool sizes, popularity, priors, corrections | DATA | regenerate all (stream.py silently goes uniform past the list) | BigMoE |
| 3 | chat-switch MTP head | r57:381-391 (md5 d22_b5); TrainLab mtp125.py:72-93 | trained weights for today's MTP layer | MIXED | retrain or keep R57_CHATSW off | MTP Head |
| 4 | loader / streaming | bigmoe/loader.py:36-98; stream.py:19, 469-511, 633 | tensor names `language_model.model.layers.{L}.mlp.switch_mlp`; MTP streams `mtp.layers.0` only | CONFIG | keep names; loop over MTP layers if > 1 | BigMoE |
| 5 | tier pools / plane math | stream.py:563-599 (assert len(PLAN)==10 :594); bigmoe/tier.py:15-27, 78-83, 216-228, 487 | plan length = top-k; 4-bit g64 MSB-3 / LSB-1 split; 92-id threshold | MIXED | plan of length top-k; re-tune C / C_l | BigMoE |
| 6 | sync-free gate (Python + Metal) | bigmoe/gated.py:22-31, 91 (TLINES 17 = 527 experts), 95 | 16 rows x top-k <= 185 ids; residency tables <= 527 experts | HARD | TLINES = ceil(E/31); keep 16k <= 185 | BigMoE |
| 7 | C++ / Metal ext | BigMoE ext/bigmoe_ext.cpp:351-410, 587-591, 1082-1084, 1221-1251, 1375, 2054 | 30 / 31-id lines, <= 4 extra lines; `li = e/31` unchecked; slot_of(1024); native / p11 unpack | MIXED | raise caps, add bounds checks, rebuild | BigMoE |
| 8 | records (4-bit) store | BigMoE scripts/repack_records.py | header-driven | CONFIG | re-run | BigMoE |
| 9 | records3 converter | MoonShots patches/planes_engine/make_records3.py:19-22 | ALIGN 16384, G 64, SHAPES ((640,2560),(640,2560),(2560,640)) | HARD | shapes from headers; MSB = pad16K(sum O*K*3/8 + 2*(O*K/64)*2), LSB = pad16K(sum O*K/8 + (O*K/64)*2) | MoonShots (mine) |
| 10 | native + p11 + pad conversion | Leap scripts/p11.py:5-25 (PADW 9216 = 36,864 B), 160-195; convert_records_p11.py:102-107; convert_records_native.py:16; n7_repack.py:28; BigMoE scripts/r3layout_ref.py:16-19; convert_records3.sh:78 (recsz != 2785280) | 25,600 groups per piece -> 8,804 words; MSB 2,064,384 / LSB 720,896; p11 needs no zeros and exponent span <= 7 (data-dependent) | HARD | derive SH and PADW; run `--check` first | Leap + BigMoE |
| 11 | records3 index | BigMoE models/bare/records3/index.json | per-layer sizes, offsets, layout (+ rowmajor entry) | DATA | regenerated by the converters (engine adapts) | BigMoE |
| 12 | REF2 core | Leap scripts/ref2.py:202-205, 357, 364, 400-401, 573, 707, 1190-1193, 1211-1213, 1777-1779, 1922-1941 | R 16; SDPA classes measured on today's model; 4-bit g64; split-K table by shape; GDN guards (conv 10240, key 2048, kernel 4, head 128, 48 x 128); self_test shapes | HARD guards | update guards / self_test; re-measure SDPA classes | Leap |
| 13 | GDN exact kernels | Leap scripts/gdnbody.py:56, 243-275; MoonShots patches/planes_engine/gdn_exact.py:9-17, gdn_exact_patch.py:16-27 (+ Leap third_party/ms_gdn copy) | compile-time Metal constants 10240 / 2048 / 128 / 6144 / 48 heads | HARD | template on heads / dims | Leap (+ MoonShots copy) |
| 14 | 3c tierglu | MoonShots patches/planes_engine/tierglu.py:160, 243-245, 402 | G 64; gate / up copy MLX qmv_fast (K % 512), down copies qmv | MIXED | re-prove bitwise == MLX for the new K, N | MoonShots / Leap |
| 15 | 3c-N / NAX6 / NAX6-N / OOO-1C | Leap scripts/tierglu_3cn.py:152-154, 189; tierglu_nax6.py:345, 359; tierglu_nax6n.py:230, 308; ooo1c.py:451 | piece_words(N*40), D*10 (40 = 2560/64, 10 = 640/64); NAX6 ks 4 / ks_b 2 | HARD | K//64, N//64; re-tune split-K | Leap |
| 16 | dense / router / glue / HC kernels | Leap qmm16.py:18, 107; qmm26.py:182; router_fused.py:10, 104-110; glue_fused.py:48-60; leapk.py:241-330; hc8r.py:781-796; d7b_nax.py:151 | qmm16 G 64, N % 32, K % 64; router E % 128, k <= 32 (MLX softmax at 512); d7b drafter 2560 x 4096 / 7680 | MIXED | mostly templated; re-prove bitwise | Leap |
| 17 | mtplx model (third party) | mtplx 2.12.2 models/qwen4_exp.py:77-108, 1000-1162, 1771, 1861, 2631 | defaults overridden by config; fast paths gated on 10240 / 2048 / heads / hc 4 / hidden 2560 / lowrank 320; env-gated fusions with literal k_in 2560 (off by default) | CONFIG + HARD fast paths | needs mtplx support for the new model_type (watched) | upstream (youssofal/mtplx) |
| 18 | drafter DFlash2 L1-20 | r57:108, 682, 693-712; Leap third_party/dflash_model_mlx.py:353-372, 432-470; ~/.cache/huggingface models--incoai--Qwen3.8-Flash-Next-DFlash2 | trained for hidden 2560, vocab 248320, 48 layers, taps 5 / 15 / 25 / 35 / 45, mask 248077; shares the target's embeddings and lm_head | DATA + HARD taps | new / retrained drafter; read taps from draft.config.target_layer_ids | MTP Head / Leap |
| 19 | drafter training + walks | Leap dfl2c_train.py:43-44, dfl2c_extract.py:26, 38, l120_chain.sh; TrainLab sticky/d02b_walk.py:24-47, d23c_mtptrunc.py:72-73, 209-219 | literal 248320, 2560, 10240, taps, hc 4, EOS | HARD | parametrise | MTP Head / Leap |
| 20 | EvalSuite env + items | EvalSuite evalsuite/env.py:9-10; data.py:15-17, 51-58; decode.py:229-252 | AutoTokenizer + apply_chat_template(enable_thinking); pre-tokenised data/items/*.jsonl | CONFIG + DATA | rebuild the items with the new tokenizer | EvalSuite |
| 21 | EvalSuite runners | decode.py:20; relax.py:32-34, 249-259; live16.py:25-27, 144-155; live16b.py:40-41, 158; live16diag.py:19-20; stages.py:29; engine.py:20-21, 53; model.py:254-259, 597-609; cli.py:374 | EOS, KTOP 10, taps, hc 4, drafter path, TOPK 10 / 512, C_l 112 pin, RAM estimate 2.77 MB x 48 | HARD | parametrise | EvalSuite |
| 22 | relax_core (2 copies) | EvalSuite evalsuite/relax_core.py:8, 38, 253-258; BigMoE scripts/relax_core.py:9, 39 | R 16, k_top 10 default | MIXED | pass top-k | EvalSuite / BigMoE |
| 23 | AR reference outputs | EvalSuite results/ref2/decode.jsonl (read at r57:1249) | today's model's outputs | DATA | regenerate | EvalSuite |
| 24 | decision map | FableGod decision_map/dm_common.py:49-53, 164-228, 381-431, 475-570; dm_t3.py:26, 139-171; dm_ablations.py:186-270; run_map.py | TOPK 8, 10 ranks, range(48), 512, 640 x 2560; TrainLab walk windows + D-02a top-1 | HARD + DATA | parametrise + re-extract | Breakthrough |
| 25 | keep-set rule | Breakthrough keepset/qom_keep.py:18, 124-125; qom_vectors.npz | K = 10, arrays [R, 10] | HARD | pass K = top-k; regenerate the test vectors | Breakthrough |
| 26 | P2 first token at 24 GB (1.2.0; LOSSY, gated) | BigMoE data/r57/snap/w24p2/r57_screen_w24p2.py:523-525 (R57_P2 / R57_P2_ROWS), 1060 (R57_P2_VERIFY), 1123 (prompts <= P2MAX take the decode's check path, not the prefill), 112-113 (EOS, TAPS); shipped as Leap release/out/pkg_120b/engine/loop/r57_screen_w24p2.py; configs chat-24.env / lossy-24.env:2 (R57_P2=256, R57_P2_ROWS=13, R57_P2_VERIFY=0) | 13 rows x top-10 = 130 ids per check, under the 141-slot 24 GB base pool and the gate's 185; P2MAX 256; the rest as row 1 (48 / 512 tables, EOS, taps, MAXK 160); the gate PASS is today's model's | MIXED | P2_ROWS = floor(min(185, base slots) / top-k); re-run EvalSuite's 24 GB P2 gate before shipping P2; row 1's port | BigMoE / EvalSuite / Leap |
| 27 | P2 at 45 GB (chat-45, package 1.2.1, not pushed) | BigMoE snapshot c45p2 (3bf3dbd): data/r57/snap/c45p2/r57_screen_c45p2.py:500-501, 1033, 1096 (the same P2 code), pflm.py, relax_* / inert_span.py; Leap release/out/pkg_121/engine/loop/r57_screen_c45p2.py; configs/chat-45.env:5 (R57_P2=256, R57_P2_ROWS=16, R57_P2_VERIFY=0) | 16 x top-10 = 160 ids (gate cap 185); the 45 GB P2 gate PASS (EvalSuite, 13 Nov) is today's model's | MIXED | as row 26; re-gate at 45 GB | BigMoE / EvalSuite / Leap |
| 28 | P3 exact fold (first token) | BigMoE data/r57/snap/p3/ (c477b60): r57_screen_p3.py (P2 code at :500-501, 1033, 1096 too); pflm.py:682 install_pfd, :732 stage_many, :298 unp11 (11-word blocks), :313 native layout (t*KG + g)*32 + r, :535 1,024-word buffers, :550 record offsets 3p / 9+2p / 10+2p (w3, s3, b3, l1, b), :158 / :175 hybrid asserts (shared_expert_gate, ssm mask), :41 MLX 0.32.2 split-K note | the records3 / p11 / native layouts (rows 9-11), E 512 scratch, the hybrid layer types, MLX 0.32.2 | MIXED | layouts from index.json (rows 9-11); re-prove exact (bitwise == the default prefill) on the new model; hazard 11 | BigMoE |
| 29 | DEEPDRAFT capture | BigMoE scripts/dd_capture.py:11-18 (npz: h, wd, tk, lp, ids), 75 (EOS {248044, 248046}), 200-220 (taps = the mean of the 4 HC streams; DFlash2 weights Leap data/l120_ft.safetensors), 389 (h (N-1, 2560)), 394 (assert wd (N-1, 10240)); dd_capture_slot.sh / dd_capture_night.sh (45 GB engine, 164 GB disk check); output ~/TrainLab/data/dlog/n1112 | hidden 2560, widened stream 10240 (hc 4 x 2560), EOS, the tap rule; the captures are today's model's hidden states | HARD + DATA | shapes / hc / EOS from config; recapture on the new model (one night, section 3 item 5) | BigMoE / MTP Head |
| 30 | DEEPDRAFT TTT (deep MTP head) | BigMoE scripts/dd_ttt.py:23 (--xb 10240 / 30720 branch), 176-189 (starts from TrainLab's d22_b5), 324; dd_d22_ttt.py (verbatim TrainLab sticky/d22_ttt.py: head.layers[0] :85, :97; d16w wd inputs :128; held-out HumanEval 24 / gsm 60 :14); dd_ttt_slot.sh / dd_ttt_alone.sh; checkpoints ~/datasets/deepdraft/ckpt/dd_t30*; pre-registrations FableGod deepdraft/PREREG_TTT30*.md, PLAN.md | one MTP layer of hidden 2560 on a 10240-wide input, the b5 start point, trained weights (t30b ...), the eval items | HARD + DATA | retrain from the new model's own MTP layer on a fresh capture; re-register the gate rules for the new model | BigMoE / MTP Head |
| 31 | agent server tool calls (1.2.0) | Leap release/src/agentio.py:3-12, 30 (the chat template's `<tool_call>` / `<function=NAME>` / `<parameter=ARG>` markup); serve.py | today's chat template's tool-call syntax | HARD | re-derive from the new chat template; re-run EvalSuite's agent tests (AGENTCC) | Leap / EvalSuite |
| 32 | conversation cache (1.2.0) | Leap release/out/pkg_121/engine/loop/convcache.py:7-21 (GDN / PLE state checkpoints at prefill-chunk edges; row 0 of layer 35's QSA keys), 46 and 240 (window 2048), 92-100 (_snapshot_gdn / _restore_gdn) | the hybrid's state layout (GDN, PLE), the layer-35 QSA detail, a 2,048-token window | MIXED | state lists from the cache objects, window from config; re-run the bitwise clone test (T1e) | BigMoE / Leap |
| 33 | setup, verify data, agent launchers (1.2.0) | Leap release/src/setup.sh:45-46 (REPO / REV: the MTPLX 4-bit pack of Qwen3.8-Flash-Next, the DFlash2 base), 12-17 (105.4 / 1.2 / 1.4 / 68.5 GB); setup/files.tsv (checksums); verify/*.json (reference tokens per config, incl. chat-24_w24p2, chat-45_p2); tools/claude_code.sh:15-18 (context 64K at 45 / 32 GB, 128K at 24, from today's memory curve) | pins, sizes, checksums, reference outputs, context defaults | DATA + CONFIG | new pins / sizes / checksums; regenerate verify/*.json; re-measure memory for the context defaults | Leap |

Not read by any live engine code (mtplx handles them from config.json): the indexer, the n-gram sidecar, per-layer
embeddings (REF2 finds PLE through the module's `.ple` attribute). Special token literals in live code: EOS {248044,
248046} only (im_start / think / image ids appear only in config and mtplx).

**Top hazards, silent ones first:**
1. **SILENT, > 527 experts:** wrong experts, no error. gated.py:91 TLINES 17 x 31; the ext's `li = e/31` is unchecked
   (bigmoe_ext.cpp:1082-1084, 1221-1249); slot_of(1024) overflows above 1,024 experts.
2. **SILENT, GDN dims changed:** ref2's guards (ref2.py:1190-1213, 1777-1779) fall back to MTPLX's non-exact path. The
   model still runs, but "spec decoding == plain AR, bit for bit" is lost.
3. **SILENT, vocab / EOS / tokenizer changed:**
   - EOS {248044, 248046} is a literal in r57, 6 EvalSuite runners and TrainLab;
   - EvalSuite's items and the AR reference are today's tokenizer's;
   - so generation may not stop, and comparisons go wrong.
4. **SILENT, pinned per-model data:** pools, popularity, priors and corrections are md5-pinned and applied without a
   shape check past layer 48.
5. **top-k > 11:** 16 x k > the gate's 185 ids (gated.py:25-30, ext:587-591; r57 MAXK 160).
6. **top-k != 10:** plan length 10, qom_keep K 10, relax_core k_top 10, EvalSuite KTOP / TOPK, the map's 10 ranks.
7. **Expert shape != 640 x 2560:**
   - make_records3 SHAPES; p11 PADW 9216, where larger pieces OVERLAP;
   - r3layout_ref 8804 words; convert_records3.sh recsz 2785280;
   - the kernel asserts N*40 / D*10.
8. **"Bitwise == MLX" rests on today's shapes:** tierglu copies MLX's qmv_fast / qmv by K. If moe_intermediate is a
   multiple of 512, MLX may change kernels under us. The router is modelled at 512 experts, and ref2 self_test covers only
   today's shapes.
9. **The drafter is coupled to this model:** taps 5 / 15 / 25 / 35 / 45 and hc 4 appear in 6+ files, plus the md5-pinned
   MTP head.
10. **4-bit affine g64 throughout:** a bf16 upstream must be quantised first (the mtplx recipe: base 4-bit g64, mlp.gate +
    indexer 8-bit, hc mixers + PLE bf16).
11. **MLX version (the orchestrator + Leap, Sun 8 Nov):** MLX's sorted gather_qmm is wrong past 32,767 rows when
    M % 64 != 0 (MLX #4632; fixed upstream in 0.32.3+ per the orchestrator, 0.32.4 per Leap's note). We pin 0.32.2 with
    the pad-to-64 fix, because 0.32.4 changes tokens vs our exact references. A Qwen 4 port on MLX must pad, or choose
    its MLX version deliberately, and take its exact references from that same version.
12. **Per-model gates and row counts (rows 26-28):** P2 is lossy and its gates (24 and 45 GB) passed on today's model
    only; its check sizes R57_P2_ROWS 13 / 16 assume top-10 (130 / 160 ids against the 141-slot 24 GB pool and the gate's
    185). P3 is exact only as proven on today's model. Re-gate P2 and re-prove P3 before either ships for Qwen 4.

## 3. Runbook: from the announcement (T-1d) to "weights public" (T+0) and on
| T | step | who | ready now? |
|---|---|---|---|
| T-1d +0:00 | ANNOUNCED fires. Read the post and Qwen Chat entry as data: sizes, architecture changes, license, release date if given. Run `qwen4/port_check.py` on any public config (a transformers / mlx-lm PR, a GitHub repo). Tell the orchestrator the expected download size, the disk it needs, and which owners' rows change. | MoonShots | yes |
| T-1d +0:30 | Storage: Kamen's decision (the orchestrator has the candidates), so T+0 does not wait. Free space for the expected download + our store + 20 GB swap. | orchestrator, Kamen | **needs Kamen** |
| T-1d +1:00 | Owners re-check their section 2 rows against the announcement (shapes, experts, top-k, vocab, MTP layers) and fix the MLX version (hazard 11). Smoke-run the parametrised converter / kernels on today's model. | owners | due Tue eve |
| T-1d | Drafts: the "what Qwen 4 is" note, the numbers table header, prior art refreshed for whatever the post claims (section 4). | MoonShots | templates ready |
| +0:00 | WEIGHTS fires. Run `qwen4/port_check.py <repo>` and post its diff, sizes, license and download size to the orchestrator. The orchestrator asks Kamen for the download OK. | MoonShots, orchestrator | yes |
| +0:05 | Pick the artifact: a 4-bit MLX upload (watch signal: MTPLX / mlx-community; today's model came from MTPLX's 4-bit at ~75 GB + n-gram tables) if it exists, else the bf16 repo plus our own conversion. Start the download. | orchestrator / BigMoE | needs disk + OK |
| +0:10 | Freeze the facts: model card, license, architecture deltas, tokenizer + chat template diff (special token ids), MTP layers shipped? First finding note: "what Qwen 4 is" (TEMPLATE_finding.md). | MoonShots | template ready |
| +0:30 | Baselines on the same Mac while our store builds: MTPLX / mlx-lm stock decode tok/s, plus prefill, on the 12-prompt sets. This is the honest comparison row for every later claim. | BigMoE | scripts exist for today's model |
| +0:45 | Build our store: records3 (MSB-3 + LSB) from the 4-bit tensors, then the native+p11+pad layout. The converter must take the new shapes from config (section 2). | BigMoE + Leap | **needs parametrising now** |
| +1:30 | Exact reference: AR greedy outputs on 12 HumanEval / 12 MBPP / 12 IFEval / 12 math (the REF2-T-style reference the engine must equal). | EvalSuite / BigMoE | **needs a model-path switch now** |
| +2:00 | **FIRST EXACT NUMBER:** 45 GB, whole decode tok/s, tokens == the reference on n / 12. Drafter = the model's own MTP layer if shipped (day-0, no training), else plain AR. Note + table row. | BigMoE, MoonShots | engine needs the port |
| +2:30 | First 24 GB and 12 GB exact numbers (store sizes from port_check; the 12 GB exact line is our likely "first", RESULTS_FIRST.md). | BigMoE / Fit12 | same |
| +3:00 | **FIRST LOSSY NUMBER:** the FAST stack (QOM gate, 3-bit sure rows, DRAFT17) needs the drafter's per-row confidence: MTP probabilities day-0, DFlash2 only after retraining. Quality gate = the map's flips vs the exact reference (decision map baseline run first). | BigMoE, Breakthrough | the map needs a model switch |
| +3:30 | Prefill tok/s at 1K / 4K / 8K and chat / math whole-decode rows. | BigMoE | scripts exist |
| +4:00 | Findings notes: one per number (TEMPLATE_finding.md), the live table (TEMPLATE_numbers_table.md), prior-art tags (section 4). Commit + push; public posts are Kamen's call. | MoonShots | templates ready |
| +4 to +24 h | Re-run our levers on the new model: per-layer pools, OOO, QOM, split, relaxed rules, FAR-style foresight, the RESULTS_FIRST claims. Then the drafter: retrain DFlash2 on Qwen 4 hidden states, or MTP-head training. | all | per-lever scripts exist for today's model |

Pre-write NOW (owners; MoonShots cannot edit their trees):
1. **BigMoE:**
   - the records3 builder and the native+p11+pad converter, driven by config (experts, moe_intermediate, hidden,
     layers); dry run: rebuild today's store and match its md5s;
   - the r57 screen with a model-path argument and config-derived sizes;
   - one runner script for every table cell.
2. **Leap:** kernels whose shapes are compile-time today (NAX6-N, 3c, dense, GDN, attention) made shape-generic, or a
   one-command rebuild for new H / I; the native repack driven by config.
3. **EvalSuite:** an AR-reference generator with a model-path argument; the same-scores harness with a model switch.
4. **Breakthrough:** the decision map's baseline capture for a new model (top-8 rows, xcapture) with a model switch.
5. **MTP Head / TrainLab:** a day-0 drafter decision tree: (a) the model's own MTP layer (zero training), (b) DFlash2
   retraining (hours: capture + train), (c) prompt-lookup copy drafting (model-agnostic, day-0), (d) the TTT30 recipe
   (DEEPDRAFT, rows 29-30): a deep MTP head retrained in about one capture night plus one training pass. Capture n1112
   ran 12 h 22 min (01:08-13:30, 12 Nov; 2,277 prompt files, 32.8 GB by its log, ~31 GiB on disk) [FableGod
   deepdraft/capture/capture_n1112.log lines 97, 2683]; t30b ran 4,206 steps, 02:29-10:08 on 13 Nov, with pauses (20,463 s
   of training; one restart at step 1,271), peak 15.6 GB [BigMoE data/deepdraft/ttt30b/night.log lines 6-16;
   ~/datasets/deepdraft/ckpt/dd_t30b.json].
6. **MoonShots:** watcher, port_check (with the port map's hazards), note templates, prior art: done. Still mine:
   make_records3.py (port map row 9, my own file: shapes from the headers) and the planes_engine copies of
   gdn_exact / tierglu (rows 13-14) with Leap.

## 4. Prior art to tag "first" claims fast and honestly
- **Results, not methods (RESULTS_FIRST.md, Wed 4 Nov; full tables in research/results_first/):**
  - Public SSD-streaming engines already ran the Qwen 4 preview on M5-class Macs at 56-61 tok/s (Kapitsa, Splash;
    2.4-bit experts, 16 GB store).
  - Stock Splash: 28.8 tok/s at 24 GB; a fork 48.8 with lossy routing. SlotStream exact: 1.48 tok/s at <= 12 GB.
  - Palm-Infra: 12.4 tok/s at 5.9 GiB, lossy.
  - Expect the same engines (and MTPLX, mlx-lm, llama.cpp) to support Qwen 4 within hours to days. "First" claims will be
    about EXACTNESS at small memory (our 12 GB exact line) and about measured speed with stated workloads (the two-speed
    caveat: HumanEval re-emits its prompt, PRIOR_ART_1105.md lead 5).
- **Methods (NOVELTY.md, PRIOR_ART_1105.md, FAR_NOVELTY_CHECK.md, results_first/EXTERNAL_PAPERS.md):**
  - expert offloading / caching / prefetch: MoE-Infinity, Fiddler, HOBBIT, ProMoE, AdapMoE, ExpertFlow, fMoE, SeqMoE,
    KTransformers (Expert Deferral, SOSP '25), Pre-gated MoE;
  - draft-fed expert forecasting: SP-MoE, MoE-SpeQ / Specter, SPICE, DraftExpert, MoE-SpAc;
  - verifier-side expert reduction: PSR section 7, AcceptMoE, OEA, MoE-Spec, cache-conditional routing;
  - speculative decoding: EAGLE-3, DFlash, Medusa, MTP (native), prompt lookup / CopySpec / REST;
  - lossy verification: Judge Decoding, AutoJudge, Fuzzy SD;
  - quantisation: GPTQ, AWQ, HQQ, QuIP#, QTIP / trellis, MLX affine, GGUF k-quants, mxfp4 / NVFP4;
  - kernel layouts: Marlin and MoE-Marlin (pre-shuffled fragments);
  - determinism / batch invariance: Thinking Machines (2025), Memory-Sovereign Inference (2608.23805: exact SSD
    streaming, 3090).
- **Our tags so far:** every lever in the live defaults is published or adapted; our own contribution is exactness (16-row
  checks bit-identical to one-token decode on Apple's GPU at 125B) plus measurements. Day-0 claims should say "first
  measured ..." with the workload, the memory line, and exact vs lossy, never "first method".

## 5. Approvals and asks (open)
Status (the orchestrator, Sun 8 Nov 00:0x; Kamen's answers relayed 09:0x):
> (a quote from the project owner, kept private)
  download goes through the orchestrator, which logs name / source / size and tells Kamen. MoonShots does not start
  downloads itself.
- **Disk / storage:** Kamen decides later; the orchestrator holds the candidates. It is needed by the announcement at
  the latest (section 3, T-1d +0:30).
> (a quote from the project owner, kept private)
> (a quote from the project owner, kept private)
- **Owners' pre-work:** NewIdeas puts it in the QUEUE behind the CHAT-150 pieces, due Tue evening.

> (a quote from the project owner, kept private)
   download (~75-130 GB) plus our store (~72 GB scaled), or ~450 GB for the bf16 path. Sat 7 Nov: 88 GB free. The
   announcement gives about a day to clear it.
2. **Kamen: answered** (standing download OK, >= 20 GB left for swap, each download via the orchestrator).
3. **Kamen: answered** (no LaunchAgent for now).
4. **Owners:** the section 3 pre-writes, by Tue 10 Nov evening; MoonShots will message each owner with its rows of the port
   map.

## 6. Publish (posting is Kamen's call)
- **License: GPL-3** for the published engine package (Kamen confirmed, Sun 8 Nov). Bundled code keeps its own license:
  MTPLX is Apache-2.0 (GPL-3 compatible), MLX is MIT. The weights keep the model's license (port_check prints it).
- What goes out per result: a row in the live table (`qwen4/notes/TEMPLATE_numbers_table.md`), a finding note
  (`qwen4/notes/TEMPLATE_finding.md`) with its prior-art tag (section 4), and the commit + exact command.
- MLX version and pad fix stated with every exact number (hazard 11).
