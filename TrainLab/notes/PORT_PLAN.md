# PORT: the DFlash2 drafter trainer in plain PyTorch for Kaggle (MTP Head, Wed 11-04 20:5x)

The orchestrator asked for this at 20:0x: the long pole of notes/DRAFTER_PLAN_1104.md, CPU-only work. The code is in
`port/` (TrainLab). Kamen creates the Kaggle account only when a probe passes; nothing here needs an account or the
network.

## 1. What exists, and what is verified

| Piece | File | Verified (CPU, numbers in REPORT / TRIED) |
|---|---|---|
| Model: layers, dynamic convs, grouped attention, selector, frozen 4-bit embedding / head | port/dflash2_torch.py | full-size L1-20 step-1 loss **5.3434 vs the logged 5.344** (0.01 %); head = bf16(float64 truth) on 99.98 % (argmax 4/4, input grad rel 1e-3); embedding bitwise = MLX's dequantize |
| SDPA attention path (CUDA memory-efficient) | same | vs explicit: 7e-7 |
| Anchor-chunked attention (less memory) | same | bitwise = whole |
| MLX AdamW (no bias correction) + clip | port/optim_port.py | vs MLX 0.32.2 over 6 steps: params 3e-8, norms 2e-6 |
| Batches, groups, RNG order | port/data_port.py | shards == npz for all 401 L1-20 train batches (bitwise, pmask on) |
| Memory-mapped shards for Kaggle datasets | port/pack_shards.py | the same test |
| Trainer: exact resume, deadline, DDP / ZeRO-1, fp16 AMP, row-chunked head + loss with recompute | port/train_port.py | 4 steps straight == 2 + resume (new process) + 2, losses and weights bitwise; gloo DDP and ZeRO-1 (2 processes) == single process, bitwise |
| Kaggle runner + upload folder | port/kaggle/ | not run (no account) |

**Finding on the way:** MLX 0.32.2's CPU `QuantizedMatmul` on bf16 inputs is far off: 3.97 % of logits equal the truth,
argmax 3/4, input gradient rel 0.44. Its GPU kernel accumulates in fp32. Never use MLX on CPU as a quantized reference.

**Not yet verified:**
- **Steps 2..N against L1-20's log** (4.6958, 4.7962, 4.6455, 3.7235). Needs a ~12 GB CPU run: weights + Adam + grads
  9.1 GB. Run it in a free slot, or as the first thing on Kaggle (`--head bf16`, exact).
- The fp16 AMP path on a real T4. Only CPU-testable here.
- ZeRO-1 resume across processes. Quick; next.
- An FSDP / ZeRO-2 path for ~1B drafters (see §3).

## 2. Kaggle procedure (for when a probe passes)

1. Kamen creates the account and phone-verifies it, for GPU and TPU quotas.
2. Weights dataset: `python port/kaggle/make_upload.py OUT`. It holds config.json + model.safetensors (incoai DFlash2) +
   target_embed_head.safetensors (the 125B's 4-bit embedding + head, 6 tensors, ~0.7 GB). Total ~3 GB. **Check the two
   licenses before uploading, even privately.**
3. Features dataset(s): `python port/pack_shards.py OUT NPZ_DIR...` writes 5,317 bytes per token (h bf16 = 5,120).
   That is ~188K tokens per GB, **~36.7M tokens per 195 GB dataset** (Kaggle's cap is 200 GB). `--max-gb` stops before
   the cap.
4. Notebook: 2 x T4, internet off. Attach both datasets and run
   `bash port/kaggle/run_kaggle.sh --epochs 1 --lr 2e-5 --S 4 --A 64 --bs 32 --pmask 0.5 ...`.
   - It runs as torchrun x 2, ZeRO-1, fp16 autocast, the fp16 head, SDPA, per-layer checkpointing.
   - It checkpoints every 30 min and at `--deadline-hours` (11.3; Kaggle GPU sessions end at 12 h), keeping ONE copy,
     because /kaggle/working holds 20 GB.
5. Run it as a committed version ("Save & Run All"), so /kaggle/working persists. Next session: attach the previous
   version's output; the runner resumes from its ckpt/last.pt exactly (weights, Adam, data order, RNG).
6. Output: drafter.safetensors (bf16, MLX names). The Mac's walk (sticky/d02b_walk.py, TL_DRAFT=...) scores it as it
   scores the MLX trainer's files.

## 3. Cost on 2 x T4 (estimate; the first Kaggle session should time it)

Per anchor (one 32-row block, L1-20's recipe), forward + backward:
- the 5 layers ~85 GFLOP (442M matmul params x 32 rows x 6);
- the head ~118 GFLOP (31 rows x 248,320 x 2,560 x 2, for forward, input grad and the recompute);
- attention ~16 GFLOP; context K/V ~5 GFLOP.

That is **~225 GFLOP per anchor**, ~57 TFLOP per step of 256 anchors. A T4 sustains maybe 15-20 TFLOPS in fp16 on these
shapes (peak 65), so 2 x T4 gives **~1.5-2 s per step, ~130-170 anchors/s**. The Mac does 31/s (L1-20: 8.3 s per step).

- **An epoch over 20M assistant tokens (= anchors) is ~35-45 h on 2 x T4**: more than Kaggle's 30 GPU-hours per week.
  My DRAFTER_PLAN line "one epoch over 20M tokens ~2-3 h" was wrong by ~15x. That plan's earliest dates (Nov 13-18)
  need re-deriving.
- Levers, each to be tested on L1-20's data before use:
  - anchor subsampling (1 in 4 assistant positions as anchors; DFlash's own recipe samples anchors; `--anchor-frac`,
    group span widened to 2A/frac): measured 3.08x fewer steps on L1-20's short windows, closer to 4x on long ones;
  - bs 16 instead of 32: ~2x, but depths 16-31 are then untrained;
  - a 1B drafter (10 layers) costs ~310 GFLOP per anchor (~1.4x).
- Memory per T4 (16 GB):
  - 5 layers (570M trained params, fp32 master): weights 2.3 + grads 2.3 + Adam/2 (ZeRO-1) 2.3 + head fp16 1.3 +
    activations ~1-2 = **~10 GB, fits**.
  - 1B: 4 + 4 + 4 + 1.3 + 1-2 = ~15 GB, too tight. It needs ZeRO-2 (shard grads too, -2 GB) or FSDP (~8 GB). Not
    built yet.

## 4. JAX / TPU for a 2-4B drafter (plan only, no code yet)

Kaggle TPU v3-8: 8 cores x 16 GB HBM, ~20 h/week, 9 h sessions.
- **Route 1 (try first): torch_xla on this port.** The model is plain torch ops; the custom head Function and
  checkpoint/topk lower to XLA.
  - Make shapes static: pad the context U to 2,176 and keep A and BS fixed. Otherwise every distinct U is a new compile.
  - Shard with SPMD FSDP (xs.mark_sharding over 8 cores). A 2-4B drafter's fp32 master weights + Adam (32-64 GB) split
    to 4-8 GB per core.
  - bf16 is native on the MXU, so the head matches MLX's bf16 semantics.
  - Checkpoint via torch_xla's distributed checkpoint, or gather to host.
- **Route 2 (if route 1 lowers badly): a JAX/Flax rewrite** (~1-2 days).
  - pjit over a (data, model) mesh.
  - Adam written MLX-style (optax's adam bias-corrects).
  - The same shards read with numpy memmap.
  - The same parity tests: step-1 loss vs L1-20's log, resume bitwise, multi-core == single-core.
- Steps:
  1. 1-core smoke on the synthetic model.
  2. Static-shape compile time.
  3. 8-core FSDP s/step.
  4. A 9 h session with resume.
  5. Scale to 2-4B by block expansion of DFlash2, as P-CAP does.
- Budget sense: 4B is ~4.5x the 5-layer's layer FLOPs. On a v3-8 (~420 TFLOPS bf16 peak, maybe 150 sustained)
  that is ~400 GFLOP per anchor, ~375 anchors/s, ~15 h per 20M anchors. That needs ~2 weeks of the 20 h/week quota
  per epoch, unless anchor subsampling holds.

## 5. Next steps (in order)

1. ZeRO-1 resume test; fp16 AMP smoke on CPU (synthetic).
2. Steps 2-5 against L1-20's log, in a ~12 GB CPU slot.
3. Re-derive DRAFTER_PLAN's dates with §3's cost.
4. Test anchor subsampling on L1-20's data: kept vs anchors trained.
