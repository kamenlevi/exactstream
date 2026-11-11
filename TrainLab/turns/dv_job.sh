#!/bin/sh
# DIVERSE2 job (MTP Head; REPORT 'DIVERSE2'), inside a held SHARE lock, the big-test window only; each step resumable:
# 1. sticky/dv_starts.py (CPU); 2. hpd_train.py from incoai (no --init), L1-20's recipe: dv_m0 (no weights), dv_mw (x4 where
# L1-20 was wrong), dv_mws (+ x4 at L1-20's live block starts); measured peak ~14.4 GB (HPD-1) -> gate 23 GB; 3. per arm the
# teacher-forced CF walk at HPD-0's anchors (code + math) and the joint-chain ensemble with L1-20; 4. sticky/dv_read.py.
cd $HOME/TrainLab
PY=$HOME/BigMoE/.venv/bin/python
trap 'pkill -f "hpd_train.py --data"; pkill -f d02b_walk; pkill -P $$; exit 143' TERM
swap() { sysctl -n vm.swapusage | awk '{v = $6; sub("M", "", v); print int(v)}'; }
noswap() { vm_stat | awk '/page size of/ {ps=$8} /^Pages (free|speculative|purgeable):/ || /^File-backed pages:/ {v=$NF; sub("[.]","",v); s+=v} END {printf "%d", s*ps/1e9}'; }
guard() { SW0=$(swap); ( while sleep 10; do if [ $(( $(swap) - SW0 )) -gt 2048 ]; then echo "== swap grew > 2 GB ($(swap) MB): stopping $1"; pkill -f "$1"; exit; fi; done ) & G=$!; }
need() { i=0; while [ "$(noswap)" -lt $1 ]; do i=$((i + 1)); [ $i -gt 20 ] && { echo "== DIVERSE2: NO-SWAP < $1 GB for 10 min: stopping"; exit 1; }; sleep 30; done; }
# 0. the conf-out pass (L1-20 on its own data: cf + right / wrong), micro-batched since 11-02 21:0x (its first run took a 51 GB
#    footprint with 32 groups per call and was stopped); gate 23 GB until its printed MLX peak is known
if [ "$(ls data/hpd/conf_l120_right/*.npy 2>/dev/null | wc -l | tr -d ' ')" != 109 ]; then
  need 23; echo "== DIVERSE2 conf-out $(date +%H:%M:%S) ($(noswap) GB NO-SWAP)"; guard "hpd_train.py --data"
  $PY sticky/hpd_train.py --data $HOME/Leap/data/l120 --init $HOME/Leap/data/l120_ft.safetensors --noval --bs 32 --micro 1 --S 32 --A 8 \
    --cache-gb 0.5 --epochs 2 --minutes 120 --pmask 0.5 --div fkl --conf-out $HOME/TrainLab/data/hpd/conf_l120 2>&1 | grep -E "conf-out|rror|Trace"
  kill $G 2>/dev/null
  [ "$(ls data/hpd/conf_l120_right/*.npy 2>/dev/null | wc -l | tr -d ' ')" = 109 ] || { echo "== DIVERSE2: conf-out incomplete"; exit 1; }
fi
[ "$(ls data/hpd/starts_l120/*.npy 2>/dev/null | wc -l | tr -d ' ')" = 109 ] || $PY -I sticky/dv_starts.py
ARGS="--data $HOME/Leap/data/l120 --noval --bs 32 --micro 1 --S 32 --A 8 --cache-gb 0.5 --epochs 2 --minutes 120 --pmask 0.5 --div fkl"
for v in m0 mw mws; do
  W=$HOME/TrainLab/data/dv_$v.safetensors; ST=data/dv_$v.state.json
  grep -q '"status": "done"' $ST 2>/dev/null && continue
  case $v in m0) X="";; mw) X="--dv-right $HOME/TrainLab/data/hpd/conf_l120_right --dv-w 4";;
    mws) X="--dv-right $HOME/TrainLab/data/hpd/conf_l120_right --dv-starts $HOME/TrainLab/data/hpd/starts_l120 --dv-w 4";; esac
  need 23; R=""; grep -q '"status"' $ST 2>/dev/null && R="--resume"
  echo "== DIVERSE2 train $v $(date +%H:%M:%S) ($(noswap) GB NO-SWAP)"
  guard "hpd_train.py --data"
  $PY sticky/hpd_train.py $ARGS $X --out $W $R 2>&1 | grep --line-buffered -E "DONE|windows;|rror|peak"
  kill $G 2>/dev/null
  grep -q '"status": "done"' $ST || { echo "== DIVERSE2 $v training did not finish (resumable)"; exit 1; }
done
for v in m0 mw mws; do
  need 14; guard d02b_walk
  [ -f data/d02b_walk_dv_${v}_code.npz ] || TL_DRAFT=$HOME/TrainLab/data/dv_$v.safetensors TL_WALK_CF=1 TL_WALK_ANCHORS=_hpd0 TL_WALK_TAG=_dv_$v \
    TL_WALK_SETS="code=$HOME/Leap/data/dfl2d_code/*.npz,math=$HOME/Leap/data/dfl2d_math/*.npz" HF_HUB_OFFLINE=1 $PY sticky/d02b_walk.py 2>&1 | grep -E "samples|rror"
  [ -f data/d02b_walk_dve_$v.txt ] || TL_WALK_ENS=$HOME/TrainLab/data/dv_$v.safetensors TL_WALK_TAG=_dve_$v HF_HUB_OFFLINE=1 \
    $PY sticky/d02b_walk_ens.py 2>&1 | grep -E "samples|DUALDRAFT|rror"
  kill $G 2>/dev/null
done
$PY -I sticky/dv_read.py
echo "== DIVERSE2 done $(date +%H:%M:%S)"
