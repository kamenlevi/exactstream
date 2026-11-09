# Install and run

## Requirements
- An Apple-silicon Mac with 64 GB of unified memory and the model on its internal SSD. Tested: MacBook Pro, M5 Pro
  (18-core CPU, 20-core GPU), 64 GB, 2 TB SSD; macOS 26.5 (25F71).
- Python 3.12, CMake >= 3.27 and the Xcode command-line tools (for the extension).
- Disk: about 174 GB in all, for the model's own files (about 105 GB without its vision tower; [WEIGHTS.md](WEIGHTS.md))
  and the converted expert store (48 layers x 512 experts x 2.79 MB = 68.5 GB; [CONVERSION.md](CONVERSION.md)); keep
  about 20 GB free for swap.

## Install
    git clone https://github.com/kamenlevi/exactstream && cd exactstream
    bash install.sh --store /path/to/model-dir --weights /path/to/drafter-dir

`install.sh` writes the clone's path into the code (where it says `@ROOT@`), creates `.venv` with the pinned packages
(`requirements.txt`; MLX stays at 0.32.2, see [LIMITS.md](LIMITS.md)), builds the extension into `engine/ext/build`, and
links the store and the drafter files ([WEIGHTS.md](WEIGHTS.md)). `--offline` installs from a local uv cache instead of
the network.

## Run
    .venv/bin/python run.py --config exact --mem 45 --prompt "Explain how a refrigerator works."
    .venv/bin/python run.py --config chat --mem 32 --prompt-file question.txt --max-tokens 512
    .venv/bin/python run.py --config exact --mem 45 --keys ifeval/1220,ifeval/3280   # benchmark items

It prints the answer, then whole decode (tokens / decode wall), whole request (tokens / (first token + decode wall)),
first token (the prefill wall) and the process's lifetime peak memory (GB = 10^9 bytes). The first run of a process
loads the model and runs one discarded warm-up prompt (`--no-warmup` skips it; the first prompt then runs cold).
The decode loop is the research harness's own loop, unchanged; `run.py` only turns a text prompt into a prompt item and
prints the answer and the loop's timing. The configs are in [CONFIGS.md](CONFIGS.md).

## Check your install
    bash tools/clean_clone_test.sh <this repository> /tmp/cc prep
    bash tools/clean_clone_test.sh <this repository> /tmp/cc run

The `run` phase runs every config on two benchmark prompts (up to 256 tokens each) with `HOME` set to an empty directory
and compares the tokens with the references in `verify/`: exact configs must match token for token. The lossy and chat
configs' tokens depend on what the process decoded before, so a prompt of theirs is checked only when the earlier prompts
ran exactly as in the reference run (same prompts, same order, same lengths); otherwise it is reported as information.
On the test MacBook Pro (Mon 9 Nov 2026) the `run` phase passed in about 4 minutes: all six configs matched their references,
and every module and every file opened came from inside the allowed folders.
