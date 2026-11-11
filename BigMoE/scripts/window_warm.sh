#!/bin/sh
# lmp4pd + pool warm-up (PF_WARM): rebuild ext, then ar2 64 with lmp4pd: sha must stay c09a8163f9, decode back to ~27 tok/s.
cd ~/BigMoE
(cd ext/build && make 2>&1 | grep -E "error|\[100" | tail -1)
V1="BIGMOE_DUPROW=1 BIGMOE_COMPILE_HC=1 BIGMOE_HC_VREAD=1 BIGMOE_DUPROW_CHUNK=5 BIGMOE_MAXK=80 BIGMOE_ASYNC_EVERY=1 BIGMOE_PF_JIT=1 BIGMOE_PF_QD=4 BIGMOE_PREFETCH=12 BIGMOE_PF_PREFILL=1 BIGMOE_PF_FULLSCR=1 BIGMOE_PREFILL=lmp4pd"
for w in 1 0; do
  until [ "$(memory_pressure | tail -1 | grep -o "[0-9]*")" -ge 85 ]; do sleep 5; done; ./scripts/gpu_quiet.sh > /dev/null
  sh -c "env $V1 BIGMOE_PF_WARM=$w .venv/bin/python scripts/bench.py 0.50 ar2 64 2>&1 | grep -E '^ar|Error|Traceback|assert|bigmoe:' | sed -E 's/.*decode +([0-9.]+) tok.*prefill\+first +([0-9.]+)s.*misses +([0-9.]+).*sha ([0-9a-f]+).*/decode \1 tok\/s prefill+first \2 s misses\/tok \3 sha \4/' | sed 's|^|[lmp4pd warm=$w] |'"
done
