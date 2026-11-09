# Third-party code and credits

Our code in this package is GPL-3.0 (LICENSE). These parts are not ours:

| part | where | licence | notes |
|---|---|---|---|
| DFlash draft model (MLX) | `third_party/dflash_model_mlx.py` | MIT | vendored unmodified (below its header) from github.com/z-lab/dflash, `dflash/model_mlx.py`; licence text: `third_party/LICENSE-dflash` |
| nanobind 2.15.0 | `third_party/nanobind/` | BSD-3-Clause | the binding library the extension is built with (MLX 0.32.2 uses nanobind's ABI v21); licence text: `third_party/LICENSE-nanobind` (the text nanobind ships) |
| MLX 0.32.2, mlx-lm 0.31.3 | installed from PyPI (`requirements.txt`) | MIT | not included; pinned |
| MTPLX 2.12.2, by Youssof Altoukhi | installed from github.com/youssofal/MTPLX at commit 9882703f (`requirements.txt`) | Apache-2.0 | not vendored (pip-installed); the model code for this architecture (`mtplx.models.qwen4_exp`) and the MTP decode loop. Its NOTICE asks every product built on it to show "Powered by MTPLX" and https://github.com/youssofal/mtplx in-product: `run.py` prints both at startup and the README carries them; LICENSE and NOTICE copies: `third_party/LICENSE-mtplx`, `third_party/NOTICE-mtplx` |
| DFlash2 drafter checkpoint, by Inco AI | not included: read from the Hugging Face cache, `incoai/Qwen3.8-Flash-Next-DFlash2` at revision `a22a1744e85c17364dcc54a6a420af7135d4c7f6` (`run.py` pins it; WEIGHTS.md) | Apache-2.0 (its model card) | the block drafter's base in every config; its model card asks users to cite DFlash 2 and the original DFlash paper. Our fine-tune of it, `l120_ft.safetensors`, is not in this repository either; wherever it is distributed it carries the licence text `third_party/LICENSE-dflash2` and the notice of our changes `third_party/NOTICE-l120_ft` |
| The model's MTP head, by Qwen (Qwen3.8-Flash-Next) | not included: the model's own files; our trained tensors for it, `d22_b5.safetensors`, are not in this repository either (WEIGHTS.md) | Qwen Community License 1.0 | the drafter for chat prompts (the engine's chat switch: chat -> MTP head, code and math -> DFlash2): the engine loads the model's MTP head, then 11 of its tensors from `d22_b5.safetensors` over it; wherever that file is distributed it carries the licence text `third_party/LICENSE-qwen` (the text the model ships) and the notice of our changes `third_party/NOTICE-d22_b5` |
| Benchmark prompts | `harness/data/items/` | IFEval: Apache-2.0; GSM8K: MIT; MATH: MIT; HumanEval: MIT; MBPP: CC-BY-4.0 | small prompt subsets as token ids, for the warm-up and the token checks |

Credit: the chat prefill path PF-D follows the idea of Strata's prompt path (read only the experts that memory lacks);
Strata is MIT, github.com/Niko1221/Strata. No Strata code is included. If any is ported later, it keeps
"Copyright (c) 2026 Niko1221 and the Strata contributors" and the MIT notice.

The model weights are not part of this package: see WEIGHTS.md for their sources and licences.

The model, Qwen3.8-Flash-Next, is Copyright (c) 2026 Qwen, under the Qwen Community License 1.0, which also covers the
weights derived from it (the converted expert store and our MTP head file). The block drafter is DFlash2 by Inco AI
(Apache-2.0), run with z-lab's DFlash code (MIT).

To cite this work, use CITATION.cff (GitHub shows it as "Cite this repository").
