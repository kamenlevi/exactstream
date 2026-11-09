# ExactStream: a 125B mixture-of-experts model on a 64 GB Mac

Kamen Levi

The engine that runs Qwen3.8-Flash-Next (125B parameters, 512 experts per layer) on one Apple-silicon laptop by
streaming the experts from the internal SSD into fixed memory pools, with speculative decoding checked on the matrix
units. Exact configs produce tokens bitwise equal to the exact reference; lossy configs passed (or are labelled as not
having passed) gates on the model's known benchmarks. The papers that describe it (an overview: `docs/index.html`):
- [MAIN](docs/pdf/main.pdf): exact and gated lossy decoding of the 125B model on one 64 GB laptop
- [KERNELS](docs/pdf/kernels.pdf): batch-invariant kernels on Apple's M5 GPU and its matrix units
- [DRAFTER](docs/pdf/drafter.pdf): drafting for a model that does not fit in memory
- [EVALUATION](docs/pdf/evaluation.pdf): same-scores gates on the model's known benchmarks
- [RELEASE](docs/pdf/release.pdf): this engine package

A paper on the 12 GB and 24 GB lines is coming soon. We will publish it by Wednesday, 11 November 2026.

Status: research code, released as measured. The decode loop is the research harness's own loop, unchanged; `run.py`
only turns a text prompt into a prompt item and prints the answer and the loop's timing.

## Requirements
- An Apple-silicon Mac with 64 GB of unified memory and the model on its internal SSD. Tested: MacBook Pro, M5 Pro,
  64 GB; macOS 26.5 (25F71).
- Python 3.12, CMake >= 3.27 and the Xcode command-line tools (for the extension).
- Disk: the converted expert store (48 layers x 512 experts x 2.79 MB = 68.5 GB) plus the model's own files; keep
  about 20 GB free for swap.

## Weights
1. The model, Qwen3.8-Flash-Next by Qwen (Copyright (c) 2026 Qwen; Qwen Community License 1.0), as MTPLX's flat 4-bit
   MLX pack on Hugging Face, `Youssofal/Qwen3.8-Flash-Next-MTPLX-Bare-Speed` (about 105 GB without its vision tower). The
   weights and anything derived from them, the converted store too, stay under the Qwen licence, not the GPL-3; its
   conditions bind whoever uses them (the licence text is in the model's files). Its directory must also hold the
   converted expert store `records3/` (see Conversion).
2. The DFlash2 drafter's base, `incoai/Qwen3.8-Flash-Next-DFlash2` by Inco AI (Apache-2.0), at revision
   `a22a1744e85c17364dcc54a6a420af7135d4c7f6`, in your Hugging Face cache (`HF_HOME`; 1.2 GB):

       .venv/bin/hf download incoai/Qwen3.8-Flash-Next-DFlash2 config.json model.safetensors --revision a22a1744e85c17364dcc54a6a420af7135d4c7f6

   `run.py` loads that revision even when your cache's default branch is newer (every measured run used it), and stops
   with this command if it is missing.
3. Our two drafter files, published as assets of this repository's release v1.0.0
   (https://github.com/kamenlevi/exactstream/releases/tag/v1.0.0), each with its notice and licence text:
   `l120_ft.safetensors` (1.14 GB; our fine-tune of DFlash2, under Apache-2.0: `third_party/NOTICE-l120_ft`,
   `third_party/LICENSE-dflash2`) and `d22_b5.safetensors` (277 MB; our training of the model's own MTP head, under the
   Qwen Community License 1.0: `third_party/NOTICE-d22_b5`, `third_party/LICENSE-qwen`). Put both in one directory.

## Install
    git clone <this repository> && cd <it>
    bash install.sh --store /path/to/model-dir --weights /path/to/drafter-dir

`install.sh` writes the clone's path into the code (where it says `@ROOT@`), creates `.venv` with the pinned packages
(`requirements.txt`; MLX stays at 0.32.2, see Limits), builds the extension into `engine/ext/build`, and links the store
and the drafter files. `--offline` installs from a local uv cache instead of the network.

## Run
    .venv/bin/python run.py --config exact --mem 45 --prompt "Explain how a refrigerator works."
    .venv/bin/python run.py --config chat --mem 32 --prompt-file question.txt --max-tokens 512
    .venv/bin/python run.py --config exact --mem 45 --keys ifeval/1220,ifeval/3280   # benchmark items

It prints the answer, then whole decode (tokens / decode wall), whole request (tokens / (first token + decode wall)),
first token (the prefill wall) and the process's lifetime peak memory (GB = 10^9 bytes). The first run of a process
loads the model and runs one discarded warm-up prompt (`--no-warmup` skips it; the first prompt then runs cold).

## Configs
| config | memory line | what it is | label |
|---|---|---|---|
| `exact` | 45 (C 256) / 32 (C 160) | config E: exact, tokens bitwise equal to the exact reference (NAX6-N) | exact |
| `lossy` | 45 / 32 | config L + COH | 45: lossy, same scores as `exact` on IFBench, GPQA Diamond and LiveCodeBench v6 (a card gate, both orders); 32: lossy, not gated (no gate ran at C 160; on IFEval it gives the 45 GB line's tokens on 12 / 12 prompts) |
| `chat` | 45 / 32 | config zs = M1-B + COH + SB1 | 45: lossy, same scores as `exact` on IFEval, IFBench, GPQA Diamond and LiveCodeBench v6 (gates on M1-B + COH, both orders; the shipped config gives its tokens on all 12 IFEval prompts); 32: lossy, not gated (its card gate did not run) |

Each config is one environment line in `configs/<config>-<mem>.env`, as the research runs used it. `--kind
chat|code|math` picks which benchmark family's switches apply to a prompt (default chat). The memory line (45 or 32 GB)
is the engine's lifetime peak footprint (phys_footprint, GB = 10^9 bytes). Besides it, every config holds a 7.45 GB
volatile expert tier as purgeable memory, which macOS reclaims before it compresses or swaps and which the footprint does
not count.

## Results
Measured on the test Mac, pooled over 12 prompts per set, 256 new tokens, greedy: whole decode / whole request (tok/s) /
first token (median, s) / lifetime peak (GB). From the RELEASE paper, section 2, which gives the sources.

| config | line | HumanEval | MBPP | IFEval (chat) | math |
|---|---|---|---|---|---|
| `exact` | 45 | 101.4 / 32.6 / 4.41 / 42.24 | 70.4 / 21.7 / 3.80 / 41.92 | 37.1 / 20.8 / 2.89 / 41.70 | 77.9 / 38.9 / 2.73 / 42.50 |
| `exact` | 32 | not measured | not measured | 33.8 / 19.8 / 2.83 / 30.62 | not measured |
| `lossy` | 45 | 164.0 / 36.2 / 4.60 / 42.94 | 112.7 / 25.3 / 3.72 / 42.60 | 45.2 / 22.4 / 3.01 / 42.42 (two passes) | 78.9 / 37.9 / 3.20 / 43.17 |
| `lossy` | 32 | not measured (1) | not measured (1) | 39.0 / 21.1 / 2.90 / 31.30 (one pass) | not measured |
| `chat` | 45 | not run | not run | 50.7 / 28.8 / 2.26 / 43.69 (one pass) | not run |
| `chat` | 32 | not run | not run | 51.3 / 22.5 / 2.99 / 31.22 | not run |

(1) Only before coherent reads (config L at C 160): HumanEval 150.7 / 36.0 / 4.43 / 31.78, MBPP 111.0 / 26.8 / 3.69 / 31.44.

## Check your install
    bash tools/clean_clone_test.sh <this repository> /tmp/cc prep
    bash tools/clean_clone_test.sh <this repository> /tmp/cc run

The `run` phase runs every config on two benchmark prompts (up to 256 tokens each) with `HOME` set to an empty directory
and compares the tokens with the references in `verify/`: exact configs must match token for token. The lossy and chat
configs' tokens depend on what the process decoded before, so a prompt of theirs is checked only when the earlier prompts
ran exactly as in the reference run (same prompts, same order, same lengths); otherwise it is reported as information.
On the test Mac (Mon 9 Nov 2026) the `run` phase passed in about 4 minutes: all six configs matched their references,
and every module and every file opened came from inside the allowed folders.

## Conversion
The expert store is built in three steps from the model's safetensors: `convert/repack_records.py` (the expert pieces,
one record per expert), `convert/make_records3.py` (each 4-bit code split into a 3-bit part and a 1-bit part, with a
verifier that checks every expert from disk), and `convert/convert_records_p11.py --pad` (the native piece order and
11-bit packed scales the kernels read).

    bash convert/convert_store.sh /path/to/model-dir /path/to/model-dir/records3

runs the three steps one layer at a time: about 3-4.4 GB of extra disk at a time beside the growing 68.5 GB store, and a
stopped run resumes at the next unfinished layer. `convert/test_layers.sh` runs them on a few layers and compares the
bytes with an existing store. On the test Mac a full conversion of all 48 layers took 9.5 minutes, and the store it built
equals, byte for byte, the one all our measurements used.

## Limits
- MLX is pinned to 0.32.2. Its sorted gather_qmm leaves rows unwritten past 32,767 sorted rows when the row count is
  not a multiple of 64 (MLX issue #4632); the engine pads to 64. 0.32.4 fixes the bug but changes tokens against our
  exact references.
- First token on short prompts at 45 GB takes 2.7-4.6 s (2.26 s on `chat`). Strata (MIT; github.com/Niko1221/Strata),
  which we ran on the same Mac and prompts, takes 0.5-1.0 s, so up to 256 generated tokens it wins or ties every
  whole-request number; our decode is ahead on code and math, about level on chat, and behind on exact chat.
- The chat prefill path (PF-D) is on for `chat` at 45 GB only and was measured on chat-length prompts (37-101 tokens).
- Lifetime peaks go above 45.0 GB on long prompts (45.1-46.4 GB on 7.9K tokens).
- At 32 GB, `exact` was measured on chat (IFEval) only, and `lossy` has no code or math rows of its own.
- The lossy configs are gated on the named benchmarks only. On a multilingual chat set, M1-B (the `chat` configs' lossy
  part) fails our own blunder check, which is information only.
- Tested on one Mac, with two IFEval prompts per config in the clean-clone test; other Macs and macOS versions are
  untested.

## Licence and credits
Powered by MTPLX
https://github.com/youssofal/mtplx

The model code for this architecture and the MTP decoding come from MTPLX by Youssof Altoukhi (Apache-2.0; its LICENSE
and NOTICE are in third_party/). The block drafter is DFlash2 by Inco AI (Apache-2.0), run with z-lab's DFlash MLX code
(MIT). Our MTP head file is a modified version of the model's own MTP head (Qwen Community License 1.0). Third-party
parts and credits, including Strata (MIT), are listed in THIRD_PARTY.md.

Copyright (C) 2026 Kamen Levi. Our code is free software: you can redistribute it and/or modify it under the terms of
the GNU General Public License, version 3, as published by the Free Software Foundation (LICENSE). It is distributed
without any warranty; see the licence for details. To cite it, see CITATION.cff.
