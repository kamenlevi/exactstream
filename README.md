# ExactStream

Runs Qwen3.8-Flash-Next (125B) on a 64 GB Mac by streaming its experts from the SSD.

    git clone https://github.com/kamenlevi/exactstream && cd exactstream
    bash install.sh --store /path/to/model-dir --weights /path/to/drafter-dir
    .venv/bin/python run.py --config exact --mem 45 --prompt "Hello"

The paper is [here](docs/pdf/complete.pdf). A 12 GB and 24 GB paper is coming soon; we will publish it by Wednesday, 11 November 2026.

Measured on one M5 Pro Mac. [INSTALL.md](INSTALL.md) has the requirements and setup, [WEIGHTS.md](WEIGHTS.md) where to get the model, [RESULTS.md](RESULTS.md) the speeds, [CONFIGS.md](CONFIGS.md) the settings and [LIMITS.md](LIMITS.md) the known limits.

Powered by MTPLX: https://github.com/youssofal/mtplx. Other credits are in [THIRD_PARTY.md](THIRD_PARTY.md). GPL-3.0. Copyright (C) 2026 Kamen Levi.
