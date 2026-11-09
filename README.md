# ExactStream

ExactStream runs Qwen3.8-Flash-Next, a 125B mixture-of-experts model, on a 64 GB Mac. It streams the experts from the
internal SSD into fixed memory pools, with speculative decoding checked on the matrix units. Its exact configs give
tokens bitwise equal to the exact reference; its lossy configs are labelled by gates on the model's known benchmarks.
This is research code, released as measured.

The papers are in docs/: the [complete report](docs/pdf/complete.pdf) (all five papers as Parts I-V), or separately
the [main paper](docs/pdf/main.pdf) and the papers on the [kernels](docs/pdf/kernels.pdf), the
[evaluation](docs/pdf/evaluation.pdf), the [drafter](docs/pdf/drafter.pdf) and this [release](docs/pdf/release.pdf).

A paper on the 12 GB and 24 GB lines is coming soon. We will publish it by Wednesday, 11 November 2026.

## Requirements

An Apple-silicon Mac with 64 GB of unified memory, Python 3.12, CMake and the Xcode command-line tools, and about 174 GB
of disk for the model and its converted expert store, plus about 20 GB free for swap. The weights are not in this
repository; [WEIGHTS.md](WEIGHTS.md) says where to get them.

## Install and run

    git clone https://github.com/kamenlevi/exactstream && cd exactstream
    bash install.sh --store /path/to/model-dir --weights /path/to/drafter-dir
    .venv/bin/python run.py --config exact --mem 45 --prompt "Explain how a refrigerator works."

[INSTALL.md](INSTALL.md) has the details and an install check, [CONFIGS.md](CONFIGS.md) the configs and their labels,
[RESULTS.md](RESULTS.md) the measured speeds and [CONVERSION.md](CONVERSION.md) how to build the expert store. Everything
was measured on one M5 Pro Mac; the known limits, and what we did not test, are in [LIMITS.md](LIMITS.md).

## Credits and licence

Powered by MTPLX
https://github.com/youssofal/mtplx

The model code for this architecture and the MTP decoding come from MTPLX by Youssof Altoukhi (Apache-2.0). The block
drafter is DFlash2 by Inco AI (Apache-2.0), run with z-lab's DFlash code (MIT). The chat prefill path follows the idea of
Strata's prompt path (MIT; github.com/Niko1221/Strata). Qwen3.8-Flash-Next is Copyright (c) 2026 Qwen, under the Qwen
Community License 1.0, which also covers the weights derived from it. Licence texts and notices are in third_party/ and
[THIRD_PARTY.md](THIRD_PARTY.md).

Copyright (C) 2026 Kamen Levi. Our code is under the GNU General Public License, version 3 (LICENSE). To cite it, see
[CITATION.cff](CITATION.cff).
