"""PORT trainer (MTP Head, Wed 11-04 20:3x; the orchestrator 20:0x): the MLX DFlash2 trainer (sticky/d09_train.py,
L1-20's recipe) in plain PyTorch, for Kaggle's 2 x T4 (drafters <= ~1B). Same data, groups, batches and RNG order, same
loss, clip, MLX AdamW (no bias correction) and lr schedule, so a CPU run reproduces the MLX losses. Added for Kaggle:
- exact checkpoint / resume across sessions: fp32 weights + Adam moments + GradScaler + data order + both RNG states
  (the MLX trainer saves bf16 weights only and restarts Adam with a ramp); atomic files, the last two kept;
- a wall-clock deadline (--deadline-hours, for 9-12 h sessions), periodic checkpoints, SIGTERM -> checkpoint, exit 3;
- 2 GPUs: torchrun --nproc_per_node 2 (manual gradient all-reduce: exact sums; --dist zero1 shards Adam's moments with
  torch.distributed.optim.ZeroRedundancyOptimizer). Every rank builds every micro-batch (the pmask RNG stays aligned)
  and trains on its share;
- T4 has no bf16: --amp fp16 (autocast + GradScaler) and --head native (fp16 dequantized head, 1.27 GB);
- memory: row-chunked head + loss with recompute, optional per-layer checkpointing (--ckpt-layers);
- data from the extractor's npz files or memory-mapped shards (port/pack_shards.py, sized for a 200 GB dataset).
Output: --out = the trained weights in bf16 with MLX names (the MLX walk reads it as it reads the MLX trainer's file).
Usage: python train_port.py --data DIR --out W.safetensors --ckpt-dir DIR [L1-20's flags: --epochs 2 --lr 2e-5 --S 4
       --A 64 --micro 1 --bs 32 --pmask 0.5] [--device cuda --amp fp16 --head native --attn sdpa --ckpt-layers]"""
import argparse, json, math, os, signal, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
import torch.distributed as dist
import dflash2_torch as M
import data_port as DP
from optim_port import MLXAdamW, clip_grad_norm_mlx, lr_schedule

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="~/Leap/data/l120"); ap.add_argument("--out", default="")
ap.add_argument("--epochs", type=float, default=1.0); ap.add_argument("--minutes", type=float, default=1e9)
ap.add_argument("--lr", type=float, default=2e-5); ap.add_argument("--S", type=int, default=4)
ap.add_argument("--A", type=int, default=64); ap.add_argument("--micro", type=int, default=1)
ap.add_argument("--bs", type=int, default=32); ap.add_argument("--pmask", type=float, default=0.0)
ap.add_argument("--val-every", type=int, default=100); ap.add_argument("--no-val", action="store_true")
ap.add_argument("--init", default="hf"); ap.add_argument("--layers", type=int, default=None)
ap.add_argument("--intermediate", type=int, default=None); ap.add_argument("--expand", default=None)
ap.add_argument("--lr-new", type=float, default=None)
ap.add_argument("--device", default="auto"); ap.add_argument("--amp", default="none")
ap.add_argument("--head", default="auto"); ap.add_argument("--attn", default="explicit")
ap.add_argument("--ckpt-layers", action="store_true"); ap.add_argument("--rows", type=int, default=256)
ap.add_argument("--attn-chunk", type=int, default=0)
ap.add_argument("--bf16-store", action="store_true")
ap.add_argument("--anchor-frac", type=float, default=1.0)
ap.add_argument("--embed-rounding", default="once")   # "twice" = MLX-CPU dequantize arithmetic   # train on this share of anchors (seeded per window)   # --loss-only: weights in bf16 storage (exact for bf16 inits)
ap.add_argument("--dist", default="none"); ap.add_argument("--threads", type=int, default=0)
ap.add_argument("--ckpt-dir", default=""); ap.add_argument("--ckpt-every-min", type=float, default=20)
ap.add_argument("--ckpt-keep", type=int, default=2)   # 1 = overwrite in place (Kaggle: /kaggle/working holds 20 GB)
ap.add_argument("--resume-from", default=""); ap.add_argument("--deadline-hours", type=float, default=0)
ap.add_argument("--max-steps", type=int, default=0); ap.add_argument("--loss-only", action="store_true")
ap.add_argument("--log-json", default=""); ap.add_argument("--synthetic", action="store_true")
A = ap.parse_args()
T_START = time.time()

# ---------------------------------------------------------------- devices, processes
world, rank = int(os.environ.get("WORLD_SIZE", "1")), int(os.environ.get("RANK", "0"))
dev = A.device if A.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
if world > 1:
    local = int(os.environ.get("LOCAL_RANK", "0"))
    if dev == "cuda":
        torch.cuda.set_device(local)
        dev = f"cuda:{local}"
    dist.init_process_group("nccl" if dev.startswith("cuda") else "gloo")
if A.threads:
    torch.set_num_threads(A.threads)
cuda = dev.startswith("cuda")
if cuda:
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
head_mode = A.head if A.head != "auto" else ("native" if cuda else "bf16")


def log(msg):
    if rank == 0:
        print(f"[port] {msg}", flush=True)


# ---------------------------------------------------------------- model
if A.synthetic:
    m = M.build_synthetic(layers=A.layers or 2, head_mode=head_mode, attn_impl=A.attn, device=dev)
    m.ckpt_layers = A.ckpt_layers
    if A.expand == "interleave":
        M.expand_interleave(m)
else:
    m = M.build(n_layers=A.layers, intermediate=A.intermediate, init=A.init, head_mode=head_mode, attn_impl=A.attn,
                ckpt_layers=A.ckpt_layers, device=dev, expand=A.expand,
                param_dtype=torch.bfloat16 if A.bf16_store else torch.float32)
if head_mode == "native":
    m.head.q4.build_cache(torch.float16 if A.amp == "fp16" or cuda else torch.float32)
else:
    m.head.q4.unpack_u8()
m.head.q4.packed = torch.empty(0, dtype=torch.int32, device=dev)          # the head needs only q8 / its cache
m.attn_chunk = A.attn_chunk
m.embed_rounding = A.embed_rounding
if A.bf16_store:
    assert A.loss_only, "--bf16-store is for --loss-only runs"
    M.bf16_storage(m)
NEW = set(range(1, len(m.layers), 2)) if A.expand == "interleave" else set()
params = [p for p in m.parameters() if p.requires_grad]
n_par = sum(p.numel() for p in m.parameters())

# ---------------------------------------------------------------- data (identical RNG use to the MLX trainer)
wins = DP.synthetic_windows() if A.synthetic else DP.load_windows(A.data)
tr, va, rng = DP.split_groups(wins, A.A, anchor_frac=A.anchor_frac)
PM = np.random.default_rng(20)
n_anchor = sum(len(g) for _, g in tr)
steps = max(1, int(math.ceil(A.epochs * len(tr) / A.S)))
lr_at = lr_schedule(A.lr, steps)
log(f"{len(wins)} windows; train {len(tr)} groups / {n_anchor} anchors; val {len(va)} groups; {steps} steps; "
    f"{len(m.layers)} layers, {n_par / 1e6:.0f}M trained params; device {dev} x {world} ({A.dist}), amp {A.amp}, "
    f"head {head_mode}, attn {A.attn}")

# ---------------------------------------------------------------- optimizer
if A.expand:
    new_p = [p for i, L in enumerate(m.layers) if i in NEW for p in L.parameters()]
    ids = {id(p) for p in new_p}
    groups = [dict(params=new_p, scale=(A.lr_new or A.lr) / A.lr), dict(params=[p for p in params if id(p) not in ids], scale=1.0)]
else:
    groups = [dict(params=params, scale=1.0)]
if A.dist == "zero1" and world > 1:
    from torch.distributed.optim import ZeroRedundancyOptimizer
    opt = ZeroRedundancyOptimizer(groups[0]["params"], optimizer_class=MLXAdamW, lr=A.lr, weight_decay=0.0)
    for g_ in groups[1:]:
        opt.add_param_group(dict(params=g_["params"]))
    for pg, g_ in zip(opt.param_groups, groups):
        pg["scale"] = g_["scale"]
else:
    opt = MLXAdamW([dict(params=g_["params"], scale=g_["scale"]) for g_ in groups], lr=A.lr, weight_decay=0.0)
scaler = torch.amp.GradScaler("cuda" if cuda else "cpu", enabled=(A.amp == "fp16"))
amp_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(A.amp)


def autocast():
    return torch.autocast(device_type="cuda" if cuda else "cpu", dtype=amp_dtype, enabled=amp_dtype is not None)


def set_lr(x):
    for pg in opt.param_groups:
        pg["lr"] = x * pg.get("scale", 1.0)


# ---------------------------------------------------------------- one step
def run_parts(items, pmask):
    """every rank builds every micro-batch in the MLX order (the pmask RNG stays identical), then keeps its share"""
    parts = [DP.make_batch(items[k:k + A.micro], A.A, A.bs, m.MASK, pmask, PM) for k in range(0, len(items), A.micro)]
    norm = max(1.0, sum(float(p_["w"].sum()) for p_ in parts))
    return parts[rank::world], norm


def step(items):
    mine, norm = run_parts(items, A.pmask)
    opt.zero_grad(set_to_none=True)
    loss = torch.zeros((), dtype=torch.float64)
    for nb in mine:
        b = DP.to_torch(nb, dev)
        with torch.set_grad_enabled(not A.loss_only), autocast():
            hid = m.fwd(b["hc"], b["upos"], b["anc"], b["tok"])
            l = m.loss_sum(hid, b["pred"], b["tkl"], b["lpl"], b["y"], b["w"], rows=A.rows) / norm
        if not A.loss_only:
            scaler.scale(l).backward()
        loss += float(l.detach())
    if world > 1:
        lt = loss.to(dev)
        dist.all_reduce(lt)
        loss = lt.cpu()
    if A.loss_only:
        return float(loss), float("nan")
    if world > 1:                                                     # exact sum of every rank's micro-batch gradients
        for p in params:
            if p.grad is None:
                p.grad = torch.zeros_like(p)
            dist.all_reduce(p.grad)
    scaler.unscale_(opt)
    gn = clip_grad_norm_mlx(params, 1.0)
    scaler.step(opt)
    scaler.update()
    return float(loss), gn


@torch.no_grad()
def evaluate(groups):
    KD = A.bs - 1
    hit_l, hit_s, den = np.zeros(KD), np.zeros(KD), np.zeros(KD)
    for k in range(0, len(groups), A.micro):
        b = DP.to_torch(DP.make_batch(groups[k:k + A.micro], A.A, A.bs, m.MASK), dev)
        with autocast():
            hid = m.fwd(b["hc"], b["upos"], b["anc"], b["tok"])
            am, pk = m.picks(hid, b["pred"], rows=A.rows)
        wn = b["w"].cpu().numpy()
        hit_l += ((am == b["y"]).cpu().numpy() * wn).sum(0); hit_s += ((pk == b["y"]).cpu().numpy() * wn).sum(0)
        den += wn.sum(0)
    return hit_l / np.maximum(den, 1), hit_s / np.maximum(den, 1), den


def val(tag):
    if A.no_val or not va or rank != 0:
        return
    al, as_, den = evaluate(va)
    state["log"]["val"].append({"tag": tag, "argmax": al.tolist(), "selector": as_.tolist(), "n": den.tolist()})
    log(f"VAL {tag}: on-path top-1 by depth " + " ".join(f"{x:.3f}" for x in al) + " | selector "
        + " ".join(f"{x:.3f}" for x in as_) + f" | n {int(den[0])}..{int(den[-1])}")


def mem():
    if cuda:
        return f"cuda peak {torch.cuda.max_memory_allocated() / 1e9:.1f} GB"
    try:
        import resource
        return f"rss peak {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (2 ** 30 if sys.platform == 'darwin' else 2 ** 20):.1f} GB"
    except Exception:
        return ""


# ---------------------------------------------------------------- checkpoints
CK = os.path.expanduser(A.ckpt_dir) if A.ckpt_dir else ""


def save_ckpt(status):
    if A.loss_only or not CK:
        return
    osd = None
    if A.dist == "zero1" and world > 1:
        opt.consolidate_state_dict(to=0)
    if rank == 0:
        osd = opt.state_dict()
        os.makedirs(CK, exist_ok=True)
        st = dict(state)
        st.update(status=status, rng=rng.bit_generator.state, pm=PM.bit_generator.state, wall=time.time() - T_START)
        blob = dict(model={k: v.detach().cpu() for k, v in m.state_dict().items() if not k.endswith("inv")},
                    opt=osd, scaler=scaler.state_dict(), state=st, args=vars(A), steps=steps)
        last, prev = os.path.join(CK, "last.pt"), os.path.join(CK, "prev.pt")
        if A.ckpt_keep == 1:          # one copy on disk at any time (the session's input dataset keeps the previous one)
            if os.path.exists(last):
                os.remove(last)
            torch.save(blob, last + ".tmp")
        else:
            torch.save(blob, last + ".tmp")
            if os.path.exists(last):
                os.replace(last, prev)
        os.replace(last + ".tmp", last)
        json.dump(st, open(os.path.join(CK, "state.json.tmp"), "w")); os.replace(os.path.join(CK, "state.json.tmp"),
                                                                                 os.path.join(CK, "state.json"))
    if world > 1:
        dist.barrier()


def find_resume():
    for p in [os.path.join(CK, "last.pt") if CK else "", A.resume_from]:   # this session's own (newer) one first
        if p and os.path.exists(os.path.expanduser(p)):
            return os.path.expanduser(p)
    return ""


STOP = {"now": False}
signal.signal(signal.SIGTERM, lambda *_: STOP.update(now=True))
signal.signal(signal.SIGINT, lambda *_: STOP.update(now=True))
rp = find_resume()
if rp:
    blob = torch.load(rp, map_location="cpu", weights_only=True)    # tensors, numbers, strings, lists, dicts only
    assert blob["steps"] == steps, f"the checkpoint is for {blob['steps']} steps, this run has {steps}"
    m.load_state_dict({k: v.to(dev) for k, v in blob["model"].items()}, strict=False)
    opt.load_state_dict(blob["opt"])
    scaler.load_state_dict(blob["scaler"])
    state = blob["state"]
    rng.bit_generator.state = state.pop("rng"); PM.bit_generator.state = state.pop("pm")
    state["resumes"] = state.get("resumes", 0) + 1
    log(f"RESUME from {rp} at step {state['s']}/{steps} ({state['train_time'] / 60:.1f} min trained)")
else:
    state = {"s": 0, "order": [], "train_time": 0.0, "ema": None, "resumes": 0,
             "log": {"args": vars(A), "steps": steps, "train_groups": len(tr), "train_anchors": n_anchor, "val": [],
                     "loss": [], "gnorm": []}}
    val("before")

# ---------------------------------------------------------------- loop (the MLX trainer's)
s0, last_ck = state["s"], time.time()
deadline = T_START + A.deadline_hours * 3600 if A.deadline_hours else None
while state["s"] < steps:
    s = state["s"]
    if not state["order"]:
        state["order"] = [int(x) for x in rng.permutation(len(tr))]
    items = [tr[state["order"].pop()] for _ in range(min(A.S, len(state["order"])))]
    set_lr(lr_at(s))
    t1 = time.time(); l, gn = step(items); state["train_time"] += min(time.time() - t1, 60.0)
    state["s"] = s + 1; state["ema"] = l if state["ema"] is None else 0.98 * state["ema"] + 0.02 * l
    state["log"]["loss"].append(l); state["log"]["gnorm"].append(gn)
    if (s + 1) % 20 == 0 or s == s0 or A.max_steps:
        log(f"step {s + 1}/{steps} loss {l:.4f} (ema {state['ema']:.3f}) grad-norm {gn:.4f} lr {lr_at(s):.2e} "
            f"{state['train_time'] / 60:.1f} min trained, {time.time() - t1:.1f}s, {mem()}")
    if STOP["now"] or (deadline and time.time() > deadline):
        save_ckpt("stopped")
        log(f"STOPPED ({'signal' if STOP['now'] else 'deadline'}) at step {s + 1}: checkpoint saved; rerun to resume")
        sys.exit(3)
    if (s + 1) % A.val_every == 0:
        val(f"step {s + 1}")
    if time.time() - last_ck > A.ckpt_every_min * 60:
        save_ckpt("running"); last_ck = time.time()
    if state["train_time"] > A.minutes * 60 or (A.max_steps and state["s"] - s0 >= A.max_steps):
        log(f"stopping at step {s + 1}/{steps} (budget)")
        break
if A.log_json and rank == 0:
    json.dump(state["log"], open(os.path.expanduser(A.log_json), "w"))
if A.loss_only:
    sys.exit(0)
done = state["s"] >= steps
if done:
    val("after")
save_ckpt("done" if done else "running")
if A.out and rank == 0:
    from safetensors.torch import save_file
    sd = {k: v.detach().to(torch.bfloat16).cpu().contiguous() for k, v in m.state_dict().items()
          if not k.endswith("inv") and any(k == n for n, p in m.named_parameters() if p.requires_grad)}
    save_file(sd, os.path.expanduser(A.out))
    log(f"DONE: saved {A.out} after {state['train_time'] / 60:.1f} min of training, {state['s']} steps")
if world > 1:
    dist.destroy_process_group()
