# ExactStream

ExactStream runs Qwen3.8-Flash-Next, a 125-billion-parameter model, on a 64 GB Mac. The model doesn't fit in memory, so its experts are streamed from the SSD as they are needed, and a small drafter guesses tokens ahead so the big model can check several at once.

The model is the 4-bit (Q4) version, and its routed experts take 68.5 GB on the SSD. At run time each token's two strongest experts use all 4 bits and the other eight use 3, about 3.2 bits per expert weight on average.

On a MacBook Pro with an M5 Pro (18-core CPU, 20-core GPU), 64 GB of memory and a 2 TB SSD, staying under 45 GB of memory (plus a 7.45 GB cache that macOS can reclaim), it generates about 164 tokens per second on HumanEval, 113 on MBPP, 79 on math and 51 on chat. Those numbers use the lossy and chat settings, which scored the same as the exact model on its known benchmarks. The exact settings give 101, 70, 78 and 37. These are averages over 12 prompts per set, from a single run for the lossy and chat settings, and speed depends on the prompt: on the lossy setting the 12 HumanEval prompts ranged from 130 to 198 tokens per second, and the fastest ran above 190. The first token on a chat prompt takes about 2.3 seconds.

This is research code, released as measured, and it has only been tested on that one MacBook Pro.

## Install

You need an Apple-silicon Mac with 64 GB of memory (36 GB or more for the 24 GB setting, which we have not tested on such a Mac; see [INSTALL.md](INSTALL.md)), macOS 26, Python 3.12, CMake, the Xcode command-line tools and about 195 GB of free disk. The model weights are not in this repository; [WEIGHTS.md](WEIGHTS.md) says where to get them.

    git clone --single-branch https://github.com/kamenlevi/exactstream && cd exactstream
    bash install.sh --store /path/to/model-dir --weights /path/to/drafter-dir

[INSTALL.md](INSTALL.md) has the details and a way to check the install.

## Run

    .venv/bin/python run.py --config chat --mem 45 --prompt "Explain how a refrigerator works."

`--config` picks the settings: `exact` gives the same tokens as decoding one token at a time with these same weights, `lossy` is the fastest on code, and `chat` is tuned for conversation. `--mem` is the memory limit in GB, 45, 32 or 24. `--mem 24` runs one lossy setting (with `--config lossy` or `chat`) that scored the same as the exact model on IFBench, GPQA Diamond and LiveCodeBench and generates about 130 tokens per second on HumanEval, 91 on MBPP, 41 on math and 31 on chat. It stays under 24 GB on prompts up to about 500 tokens and uses more on longer ones, up to 28.2 GB at 30K tokens. [CONFIGS.md](CONFIGS.md) lists every setting, [RESULTS.md](RESULTS.md) has the full numbers and [LIMITS.md](LIMITS.md) has the known limits and what we did not test.

`chat.py` is a chat in the terminal that shows the speed as it writes, and `serve.py` serves chat apps on your Mac in the OpenAI and Anthropic API formats, text only for now ([INSTALL.md](INSTALL.md)).

## Papers

The whole project is written up in [What It Takes to Run a 125B LLM Fast on a 64 GB Mac, and How to Go Faster](docs/pdf/complete.pdf). The parts are also available separately: [main paper](docs/pdf/main.pdf), [kernels](docs/pdf/kernels.pdf), [evaluation](docs/pdf/evaluation.pdf), [drafter](docs/pdf/drafter.pdf) and [release](docs/pdf/release.pdf).

The 24 GB and 12 GB lines are written up in [A 125B mixture-of-experts model in 32, 24 and 12 GB of memory on one Mac](docs/pdf/small-ram.pdf).

Research data (every measurement, script and record the papers cite): the [`research` branch](https://github.com/kamenlevi/exactstream/tree/research).

## Credits

Powered by MTPLX: https://github.com/youssofal/mtplx

The model code and MTP decoding come from MTPLX by Youssof Altoukhi. The drafter is based on DFlash2 by Inco AI, and the chat prompt path follows Strata. Qwen3.8-Flash-Next is by Qwen, under the Qwen Community License 1.0. Licence texts are in third_party/, listed in [THIRD_PARTY.md](THIRD_PARTY.md).

## Licence

GPL-3.0. Copyright (C) 2026 Kamen Levi. To cite this work, see [CITATION.cff](CITATION.cff).
