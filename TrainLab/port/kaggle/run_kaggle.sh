#!/bin/bash
# PORT Kaggle runner (MTP Head, Wed 11-04 20:4x): port/train_port.py on a 2 x T4 notebook session. Attach as inputs:
#   the weights dataset (port/kaggle/make_upload.py: config.json, model.safetensors, target_embed_head.safetensors),
#   the features dataset (port/pack_shards.py: index.json + shards; <= 200 GB per dataset),
#   and, from the 2nd session on, the previous session's output (its ckpt/last.pt is the resume point).
# The run checkpoints every 30 min and before the session's end (--deadline-hours), keeping one copy (20 GB output cap).
# Usage (in the notebook): !bash port/kaggle/run_kaggle.sh --epochs 1 --lr 2e-5 --S 4 --A 64 --bs 32 --pmask 0.5
set -e
WTS=${WTS:-$(dirname "$(ls /kaggle/input/*/target_embed_head.safetensors | head -1)")}
FEAT=${FEAT:-$(dirname "$(ls /kaggle/input/*/index.json | head -1)")}
PREV=$(ls -t /kaggle/input/*/ckpt/last.pt 2>/dev/null | head -1 || true)
export PORT_DRAFT_DIR=$WTS PORT_TARGET=$WTS/target_embed_head.safetensors
echo "weights $WTS, features $FEAT, resume ${PREV:-none}"
torchrun --nproc_per_node 2 "$(dirname "$0")/../train_port.py" --data "$FEAT" --out /kaggle/working/drafter.safetensors \
  --ckpt-dir /kaggle/working/ckpt --ckpt-keep 1 ${PREV:+--resume-from "$PREV"} --deadline-hours "${HOURS:-11.3}" \
  --ckpt-every-min 30 --device cuda --amp fp16 --head native --attn sdpa --ckpt-layers --dist zero1 "$@"
