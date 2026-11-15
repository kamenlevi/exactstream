"""c55_class_bytes.py (Breakthrough, CHECK55 step 0): weight bytes per profile class for the 12-layer slice (layers 0-11 + the
head), from the safetensors headers (no load), and each class's bandwidth floor at 280 GB/s against pass K's GPU ms.
Usage: python3 -I -B c55_class_bytes.py <profile tag>   -> data/m1/check55/<tag>_class_floors.txt"""
import glob, json, os, re, struct, sys
D = os.path.expanduser("~/BigMoE/models/bare")
T = {}
for f in sorted(glob.glob(D + "/*.safetensors")):
    with open(f, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        h = json.loads(fh.read(n))
    for k, v in h.items():
        if k != "__metadata__":
            T[k] = v["data_offsets"][1] - v["data_offsets"][0]
NL = 12
B = {}
def add(c, b):
    B[c] = B.get(c, 0) + b
for k, b in T.items():
    if not k.startswith("language_model."):
        continue
    m = re.match(r"language_model\.model\.layers\.(\d+)\.(.*)", k)
    if m:
        L, rest = int(m.group(1)), m.group(2)
        if L >= NL:
            continue
        if rest.startswith("linear_attn."):
            add("gdn_proj" if re.match(r"linear_attn\.(in_proj|out_proj)", rest) else "gdn_core", b)
        elif rest.startswith("self_attn."):
            add("att_proj" if re.search(r"_proj\.", rest) else "att_core", b)
        elif rest.startswith("mlp.switch_mlp."):
            add("moe_experts_all", b)
        elif rest.startswith("mlp.gate."):
            add("moe_router", b)
        elif rest.startswith("mlp.shared_expert"):
            add("moe_shared", b)
        elif "hyper_connection" in rest:
            add("hc", b)
        elif rest.startswith("ple."):
            add("ple_gathered" if "ngram" in rest or "embedding" in rest else "ple", b)
        else:
            add("other:" + rest.split(".")[0], b)
    elif k.startswith("language_model.lm_head."):
        add("head_lm", b)
    elif "hyper_connection_mixer" in k:
        add("head_mixer", b)
    elif "embed_tokens" in k:
        add("embed_gathered", b)
tag = sys.argv[1] if len(sys.argv) > 1 else None
out = ["weight bytes per class, layers 0-11 + head (MB) and the floor at 280 GB/s (ms):"]
for c in sorted(B):
    out.append(f"  {c:18s} {B[c] / 1e6:9.1f} MB  {B[c] / 280e6:7.3f} ms")
if tag:
    r = json.load(open(os.path.expanduser(f"~/Breakthrough/data/m1/check55/{tag}.json")))
    CI = ["disp", "gpu", "gpu_net", "build", "enc", "flushes", "cbs"]
    grp = {"dense projections": (["gdn_proj", "att_proj"], ["gdn_proj", "att_proj"]), "HC": (["hc_read", "hc_write"], ["hc"]),
           "MoE dense side": (["moe_router", "moe_shared", "moe_tail"], ["moe_router", "moe_shared"]),
           "head": (["head_mixer", "head_lm"], ["head_mixer", "head_lm"]), "PLE": (["ple"], ["ple"]),
           "GDN core": (["gdn_core"], ["gdn_core"]), "attention core": (["att_core"], ["att_core"])}
    for a in [x for x in ("A16", "C32", "A16k", "C32k") if x in r["K"]]:
        K = r["K"][a]
        sc = 1e3 * r["T"][a]["gpu"] / sum(K[c][1] for c in K)
        out.append(f"{a}: class | GPU ms (pass K, scaled) | weight floor ms | GPU / floor")
        for g, (cs, bs) in grp.items():
            gm = sc * sum(K[c][1] for c in cs)
            fm = sum(B.get(b, 0) for b in bs) / 280e6
            out.append(f"  {g:18s} {gm:7.2f} {fm:7.3f}  {gm / fm if fm else float('nan'):6.1f}x")
        eg = sc * K["moe_experts"][1]
        ef = r["floors"][a]["experts_gb"] * 1e3 / 280
        out.append(f"  {'MoE experts':18s} {eg:7.2f} {ef:7.3f}  {eg / ef:6.2f}x  (routed union {r['floors'][a]['union_mean']:.1f} / layer)")
print("\n".join(out))
if tag:
    open(os.path.expanduser(f"~/Breakthrough/data/m1/check55/{tag}_class_floors.txt"), "w").write("\n".join(out) + "\n")
