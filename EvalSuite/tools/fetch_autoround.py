"""Accepted-build anchor: Intel's AutoRound int4 routed experts for Qwen3.8-Flash-Next, in the reference's MLX layout.

Intel/Qwen3.8-Flash-Next-W4A16-AutoRound quantizes only the routed experts (everything else stays BF16 there): int4,
symmetric, group 128, auto_gptq packing (qweight [in/8, out], qzeros stored as zero - 1, fp16 scales). This tool
fetches just those expert tensors by HTTP range requests (~63 GB of the repo's 181 GB) and converts them exactly to
MLX affine 4-bit g64, the format of the reference's switch_mlp tensors:
  weight[e] = qweight.T (both pack 8 consecutive inputs per uint32, input j at bits 4j)
  scales    = fp16 scale -> bf16 (round to nearest even), each g128 scale repeated for its two g64 groups
  biases    = -zero * scale (zero = 8: exact in bf16)
so W = scale * (q - zero) up to the fp16 -> bf16 rounding of the scale (relative 2^-9; the reference's own scales are
bf16 too). One safetensors file per layer (models/ar4/zz-ar-layer-LL.safetensors); `--link` then builds the composite
model dir models/ar4 that BigMoE's ExpertStreamer reads: every reference file symlinked (not records/), the index
re-pointed for the 48 x 9 routed-expert tensors. The dense trunk stays the reference's, so the anchor differs from the
reference exactly where expert-precision variants do.

usage: fetch_autoround.py [--layers A-B] [--par 4] [--rate 1900k] [--verify] [--link]
Resumable (finished layers are skipped). Stops if free disk would drop below 112 GB."""
from __future__ import annotations
import argparse, concurrent.futures as cf, json, os, shutil, struct, subprocess, sys, time
import numpy as np

REPO = "https://huggingface.co/Intel/Qwen3.8-Flash-Next-W4A16-AutoRound/resolve/main/"
REF = os.path.expanduser("~/BigMoE/models/bare")
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "models", "ar4")
META = os.path.join(HERE, "models", "autoround_meta")
PROJ = ("gate_proj", "up_proj", "down_proj")
E, FLOOR_GB = 512, 112


def free_gb():
    st = os.statvfs("/")
    return st.f_bavail * st.f_frsize / 1e9


def curl_range(url, a, b, rate, tries=6):
    """bytes [a, b) of url"""
    for t in range(tries):
        r = subprocess.run(["curl", "-sfL", "--limit-rate", rate, "-r", f"{a}-{b - 1}", url], capture_output=True)
        if r.returncode == 0 and len(r.stdout) == b - a:
            return r.stdout
        time.sleep(2 + 3 * t)
    raise RuntimeError(f"range {a}-{b} of {url} failed after {tries} tries")


def header(shard, rate):
    os.makedirs(META, exist_ok=True)
    p = os.path.join(META, shard + ".header.json")
    if os.path.exists(p):
        return json.load(open(p))
    url = REPO + shard
    n = struct.unpack("<Q", curl_range(url, 0, 8, rate))[0]
    h = json.loads(curl_range(url, 8, 8 + n, rate))
    h.pop("__metadata__", None)
    json.dump({"n": n, "h": h}, open(p, "w"))
    return {"n": n, "h": h}


def to_bf16(x):
    """float array -> bf16 bit patterns (uint16), round to nearest even"""
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)


def from_bf16(u16):
    return (u16.astype(np.uint32) << 16).view(np.float32)


def unpack_cols(words):
    """[R, C/8] packed int32 (value j at bits 4j) -> [R, C] uint8"""
    w = words.view(np.uint32)
    return np.stack([(w >> (4 * j)) & 15 for j in range(8)], axis=-1).reshape(w.shape[0], -1).astype(np.uint8)


def convert_expert(qweight, qzeros, scales):
    """one expert projection: auto_gptq int4 g128 -> MLX (weight [out, in/8] u32, scales/biases [out, in/64] bf16)"""
    w = np.ascontiguousarray(qweight.view(np.uint32).T)
    zero = unpack_cols(qzeros).astype(np.float32) + 1.0               # [in/128, out]; auto_gptq stores zero - 1
    s = from_bf16(to_bf16(scales.astype(np.float32)))                   # [in/128, out], bf16-exact values
    b = -zero * s
    s2 = np.repeat(s.T, 2, axis=1)                                      # [out, in/64]
    b2 = np.repeat(b.T, 2, axis=1)
    return w, to_bf16(s2), to_bf16(b2)


def write_safetensors(path, tensors):
    """tensors: name -> (dtype tag, np array of the raw bits)"""
    hdr, off = {}, 0
    for k, (dt, a) in tensors.items():
        hdr[k] = {"dtype": dt, "shape": list(a.shape), "data_offsets": [off, off + a.nbytes]}
        off += a.nbytes
    hb = json.dumps(hdr, separators=(",", ":")).encode()
    hb += b" " * (-len(hb) % 8)
    tmp = path + ".part"
    with open(tmp, "wb") as f:
        f.write(struct.pack("<Q", len(hb)))
        f.write(hb)
        for _, a in tensors.values():
            f.write(np.ascontiguousarray(a).tobytes())
    os.replace(tmp, path)


def layer_file(L):
    return os.path.join(OUT, f"zz-ar-layer-{L:02d}.safetensors")


def fetch_layer(L, where, rate, par):
    """where: shard -> {'n', 'h'}; returns name -> np array for all 512 x 3 x 3 expert tensors of layer L"""
    pre = f"model.language_model.layers.{L}.mlp.experts."
    raw = {}
    for shard, H in where.items():
        names = [k for k in H["h"] if k.startswith(pre)]
        if not names:
            continue
        a = min(H["h"][k]["data_offsets"][0] for k in names)
        b = max(H["h"][k]["data_offsets"][1] for k in names)
        base = 8 + H["n"]
        step = 64 << 20
        parts = [(x, min(x + step, b)) for x in range(a, b, step)]
        buf = bytearray(b - a)
        with cf.ThreadPoolExecutor(par) as ex:
            for (x, y), data in zip(parts, ex.map(lambda p: curl_range(REPO + shard, base + p[0], base + p[1], rate), parts)):
                buf[x - a:y - a] = data
        for k in names:                                  # views into buf (no copies; buf lives as long as they do)
            v = H["h"][k]
            x, y = v["data_offsets"]
            dt = {"I32": np.int32, "F16": np.float16}[v["dtype"]]
            raw[k] = np.frombuffer(buf, dtype=dt, count=(y - x) // np.dtype(dt).itemsize, offset=x - a).reshape(v["shape"])
    return raw


def convert_layer(L, raw):
    pre = f"model.language_model.layers.{L}.mlp.experts."
    out = {}
    for p in PROJ:
        ws, ss, bs = [], [], []
        for e in range(E):
            k = f"{pre}{e}.{p}."
            w, s, b = convert_expert(raw[k + "qweight"], raw[k + "qzeros"], raw[k + "scales"])
            ws.append(w); ss.append(s); bs.append(b)
        name = f"language_model.model.layers.{L}.mlp.switch_mlp.{p}."
        out[name + "weight"] = ("U32", np.stack(ws))
        out[name + "scales"] = ("BF16", np.stack(ss))
        out[name + "biases"] = ("BF16", np.stack(bs))
    return out


def verify(L=0, experts=(0, 1, 511)):
    """MLX's own dequantize of the converted tensors == the auto_gptq formula with bf16-rounded scales"""
    import mlx.core as mx
    from safetensors import safe_open
    mx.set_default_device(mx.cpu)
    where = {s: header(s, "10M") for s in shards_for(L)}
    worst = 0.0
    f = mx.load(layer_file(L))
    for p in PROJ:
        name = f"language_model.model.layers.{L}.mlp.switch_mlp.{p}."
        for e in experts:
            k = f"model.language_model.layers.{L}.mlp.experts.{e}.{p}."
            sh = [s for s, H in where.items() if k + "qweight" in H["h"]][0]
            H = where[sh]
            t = {}
            for f_ in ("qweight", "qzeros", "scales"):
                v = H["h"][k + f_]
                x, y = v["data_offsets"]
                dt = {"I32": np.int32, "F16": np.float16}[v["dtype"]]
                t[f_] = np.frombuffer(curl_range(REPO + sh, 8 + H["n"] + x, 8 + H["n"] + y, "10M"), dtype=dt).reshape(v["shape"])
            q = unpack_cols(t["qweight"].view(np.uint32).T.copy()).astype(np.float32)        # [out, in]
            zero = unpack_cols(t["qzeros"]).astype(np.float32) + 1.0                        # [in/128, out]
            s = from_bf16(to_bf16(t["scales"].astype(np.float32)))
            ref = (q - np.repeat(zero.T, 128, axis=1)) * np.repeat(s.T, 128, axis=1)
            got = mx.dequantize(f[name + "weight"][e], f[name + "scales"][e], f[name + "biases"][e], group_size=64, bits=4)
            d = float(np.abs(np.array(got.astype(mx.float32)) - ref).max())
            worst = max(worst, d)
            print(f"verify L{L} {p} e{e}: max |mlx - gptq| = {d:.3g} (max |w| {np.abs(ref).max():.3g})", flush=True)
    return worst


def shards_for(L):
    idx = json.load(open(os.path.join(META, "index.json")))["weight_map"]
    pre = f"model.language_model.layers.{L}.mlp.experts."
    return sorted({v for k, v in idx.items() if k.startswith(pre)})


def link():
    """composite model dir: reference files symlinked, the routed-expert tensors re-pointed to the converted files"""
    missing = [L for L in range(48) if not os.path.exists(layer_file(L))]
    assert not missing, f"layers not converted yet: {missing}"
    for f in os.listdir(REF):
        if f in ("records", "model.safetensors.index.json") or f.startswith("."):
            continue
        dst = os.path.join(OUT, f)
        if not os.path.lexists(dst):
            os.symlink(os.path.join(REF, f), dst)
    idx = json.load(open(os.path.join(REF, "model.safetensors.index.json")))
    for L in range(48):
        for p in PROJ:
            for f in ("weight", "scales", "biases"):
                k = f"language_model.model.layers.{L}.mlp.switch_mlp.{p}.{f}"
                assert k in idx["weight_map"], k
                idx["weight_map"][k] = os.path.basename(layer_file(L))
    idx.setdefault("metadata", {})["evalsuite_anchor"] = "routed experts from Intel/Qwen3.8-Flash-Next-W4A16-AutoRound"
    json.dump(idx, open(os.path.join(OUT, "model.safetensors.index.json"), "w"), indent=1)
    print("linked", OUT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="0-47")
    ap.add_argument("--par", type=int, default=4)
    ap.add_argument("--rate", default="1900k", help="curl --limit-rate per connection")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--link", action="store_true")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if not os.path.exists(os.path.join(META, "index.json")):
        os.makedirs(META, exist_ok=True)
        open(os.path.join(META, "index.json"), "wb").write(
            subprocess.run(["curl", "-sfL", REPO + "model.safetensors.index.json"], capture_output=True, check=True).stdout)
    if a.verify:
        print("worst", verify())
        return
    if a.link:
        link()
        return
    lo, hi = (int(x) for x in a.layers.split("-"))
    t0, done_bytes = time.time(), 0
    for L in range(lo, hi + 1):
        if os.path.exists(layer_file(L)):
            continue
        if free_gb() < FLOOR_GB + 2:
            sys.exit(f"stop: {free_gb():.0f} GB free, floor {FLOOR_GB} GB")
        where = {s: header(s, a.rate) for s in shards_for(L)}
        t = time.time()
        raw = fetch_layer(L, where, a.rate, a.par)
        nb = sum(x.nbytes for x in raw.values())
        tensors = convert_layer(L, raw)
        del raw
        write_safetensors(layer_file(L), tensors)
        done_bytes += nb
        el = time.time() - t0
        print(f"layer {L}: {nb / 1e9:.2f} GB in {time.time() - t:.0f}s; {done_bytes / el / 1e6:.1f} MB/s overall; "
              f"free {free_gb():.0f} GB", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
