#!/bin/sh
# D-02a part 3 (MTP Head, Thu 10-29; NewIdeas' D-11 pre-check): incoai's ORIGINAL DFlash2 weights (no L1-20 tune) walked
# on all five sets (IFEval / multi / agent from data/d02a, code / math from Leap's dfl2d). Did L1-20's code + math tune
# cost the other domains? One quiet lock (~4 min). Output data/d02b_walk_d02a_orig.txt (+ npz per set).
cd $HOME/TrainLab
trap 'pkill -P $$; exit 143' TERM
echo "== D-02a orig walk $(date +%H:%M:%S)"
TL_DRAFT=/Users/kamen/.cache/huggingface/hub/models--incoai--Qwen3.8-Flash-Next-DFlash2/snapshots/a22a1744e85c17364dcc54a6a420af7135d4c7f6/model.safetensors TL_WALK_TAG=_d02a_orig \
  TL_WALK_SETS="ifeval=$HOME/TrainLab/data/d02a/tf_ifeval_*.npz,multi=$HOME/TrainLab/data/d02a/tf_multi_*.npz,agent=$HOME/TrainLab/data/d02a/agent_*.npz,code=$HOME/Leap/data/dfl2d_code/*.npz,math=$HOME/Leap/data/dfl2d_math/*.npz" \
  HF_HUB_OFFLINE=1 $HOME/BigMoE/.venv/bin/python sticky/d02b_walk.py 2>&1 | grep --line-buffered -v "Token indices"
echo "== done $(date +%H:%M:%S)"
