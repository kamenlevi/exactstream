# Weights

The weights are not in this repository. You need three things.

1. The model, Qwen3.8-Flash-Next by Qwen (Copyright (c) 2026 Qwen; Qwen Community License 1.0), as MTPLX's flat 4-bit
   MLX pack on Hugging Face, `Youssofal/Qwen3.8-Flash-Next-MTPLX-Bare-Speed` (about 105 GB without its vision tower). The
   weights and anything derived from them, the converted store too, stay under the Qwen licence, not the GPL-3; its
   conditions bind whoever uses them (the licence text is in the model's files). Its directory must also hold the
   converted expert store `records3/` (see [CONVERSION.md](CONVERSION.md)).
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
