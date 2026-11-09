# Reference tokens for the install check

`tools/clean_clone_test.sh run` compares each config's tokens with the file here whose name starts with the config.
Exact configs must match their reference token for token (as a prefix: the shorter length is compared). The files are
the decode loop's tokens.json format, {key: {arm: [[tokens]]}}; their sources are listed in MANIFEST.json.
