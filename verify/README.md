# Reference tokens for the install check

`tools/clean_clone_test.sh run` compares each config's tokens with the file here whose name starts with the config.
Exact configs must match their reference token for token (as a prefix: the shorter length is compared). The files are
the decode loop's tokens.json format, {key: {arm: [[tokens]]}}; their sources are listed in MANIFEST.json.

Since 1.2.0 the 24 GB configs (with P2) are checked against P2's gated arm on the 12 IFEval prompts at cap 256 (BigMoE's
W24T run w4p1_100; its second P2 run gave the same tokens on 12 of 12 prompts). The other references are 1.1.0's.
