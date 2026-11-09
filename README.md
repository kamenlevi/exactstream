# ExactStream

ExactStream runs Qwen3.8-Flash-Next, a 125-billion-parameter model, on a 64 GB Mac. The model doesn't fit in memory, so its experts are streamed from the SSD as they are needed, and a small drafter guesses tokens ahead so the big model can check several at once.

On a MacBook Pro with an M5 Pro and 64 GB, staying under 45 GB of memory (plus a 7.45 GB cache that macOS can reclaim), it generates about 164 tokens per second on HumanEval, 113 on MBPP, 79 on math and 51 on chat. Those numbers use the lossy and chat settings, which scored the same as the exact model on its known benchmarks. The exact settings give 101, 70, 78 and 37. The first token on a chat prompt takes about 2.3 seconds.

This is research code, released as measured, and it has only been tested on that one MacBook Pro.

## Install

You need an Apple-silicon Mac with 64 GB of memory, macOS 26, Python 3.12, CMake, the Xcode command-line tools and about 195 GB of free disk. The model weights are not in this repository; [WEIGHTS.md](WEIGHTS.md) says where to get them.

    git clone https://github.com/kamenlevi/exactstream && cd exactstream
    bash install.sh --store /path/to/model-dir --weights /path/to/drafter-dir

[INSTALL.md](INSTALL.md) has the details and a way to check the install.

## Run

    .venv/bin/python run.py --config chat --mem 45 --prompt "Explain how a refrigerator works."

`--config` picks the settings: `exact` gives the same tokens as normal decoding, `lossy` is the fastest on code, and `chat` is tuned for conversation. `--mem` is the memory limit in GB, 45 or 32. [CONFIGS.md](CONFIGS.md) lists every setting, [RESULTS.md](RESULTS.md) has the full numbers and [LIMITS.md](LIMITS.md) has the known limits and what we did not test.

## Papers

The whole project is written up in [What It Takes to Run a 125B LLM Fast on a 64 GB Mac, and How to Go Faster](docs/pdf/complete.pdf). The parts are also available separately: [main paper](docs/pdf/main.pdf), [kernels](docs/pdf/kernels.pdf), [evaluation](docs/pdf/evaluation.pdf), [drafter](docs/pdf/drafter.pdf) and [release](docs/pdf/release.pdf).

A 12 GB and 24 GB paper is coming soon; we will publish it by Wednesday, 11 November 2026.

## Credits

Powered by MTPLX: https://github.com/youssofal/mtplx

The model code and MTP decoding come from MTPLX by Youssof Altoukhi. The drafter is based on DFlash2 by Inco AI, and the chat prompt path follows Strata. Qwen3.8-Flash-Next is by Qwen, under the Qwen Community License 1.0. Licence texts are in third_party/, listed in [THIRD_PARTY.md](THIRD_PARTY.md).

## Licence

GPL-3.0. Copyright (C) 2026 Kamen Levi. To cite this work, see [CITATION.cff](CITATION.cff).
