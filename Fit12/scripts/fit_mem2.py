"""Fit12 MEM2 (Mon 26 Oct 2026; PLAN.md 'Rapid screens, batch 1' MEM2): census of what is resident after loading the
12 GB stack exactly as fit_cmp.py loads it (its source up to the prompt loop is executed here, unchanged), then after one
p2k prefill + 64-token decode. Prints MLX active memory, the trunk pools, the model's parameters grouped by module (top 25),
and what neither accounts for. Usage: [env as fit_cmp] ~/BigMoE/.venv/bin/python -B fit_mem2.py TAG"""
import os, sys
from collections import defaultdict

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fit_cmp.py")
src = open(SRC).read()
cut = src.index('for pos, P in enumerate(os.environ.get("F12_PROMPTS"')
NS = {"__name__": "fit_cmp_prefix", "__file__": SRC}
sys.argv = [SRC] + sys.argv[1:]
exec(compile(src[:cut], SRC, "exec"), NS)
mx, model, st, B = NS["mx"], NS["model"], NS["st"], NS["B"]
from mlx.utils import tree_flatten


def census(tag):
    mx.eval(model.parameters())
    act = mx.get_active_memory() / 1e9
    pools = st.resident_gb()
    grp, seen, tot = defaultdict(int), set(), 0
    for k, v in tree_flatten(model.parameters()):
        if not isinstance(v, mx.array) or id(v) in seen:
            continue
        seen.add(id(v))
        parts = k.split(".")
        g = ".".join(p if not p.isdigit() else "N" for p in parts[:7])
        grp[g] += v.nbytes
        tot += v.nbytes
    print(f"[mem2] {tag}: MLX active {act:.3f} GB | trunk pools {pools:.3f} GB | parameters {tot / 1e9:.3f} GB | rest "
          f"{act - pools - tot / 1e9:.3f} GB | footprint {NS['fp']()[0]:.3f} GB", flush=True)
    for g, b in sorted(grp.items(), key=lambda x: -x[1])[:25]:
        print(f"[mem2]   {b / 1e9:7.3f} GB  {g}", flush=True)
    found = defaultdict(int)                           # arrays held by engine / harness modules, not parameters

    def walk(o, path, depth):
        if isinstance(o, mx.array):
            if id(o) not in seen:
                seen.add(id(o))
                found[path] += o.nbytes
            return
        if depth > 4 or id(o) in vis:
            return
        vis.add(id(o))
        if isinstance(o, dict):
            for k, v in list(o.items())[:4096]:
                walk(v, f"{path}.{k}" if depth < 2 else path, depth + 1)
        elif isinstance(o, (list, tuple)):
            for v in o[:4096]:
                walk(v, path, depth + 1)
        elif hasattr(o, "__dict__") and not isinstance(o, type) and type(o).__module__.split(".")[0] in MODS:
            walk(vars(o), path, depth + 1)
    vis = set()
    MODS = ("bigmoe_v3f", "bt_b12", "spec125", "mtplx", "fit_cmp_prefix")
    for name, m in list(sys.modules.items()):
        if name.split(".")[0] in MODS and m is not None:
            walk({k: v for k, v in vars(m).items() if not k.startswith("__")}, name, 1)
    walk({k: v for k, v in NS.items() if not k.startswith("__") and k not in ("model", "rt")}, "harness", 1)
    ft = sum(found.values())
    print(f"[mem2]   module-held arrays outside the parameters: {ft / 1e9:.3f} GB (rest after them "
          f"{act - tot / 1e9 - ft / 1e9:.3f} GB)", flush=True)
    for g, b in sorted(found.items(), key=lambda x: -x[1])[:15]:
        print(f"[mem2]   {b / 1e9:7.3f} GB  {g}", flush=True)


census("after load")
P = "p2k"
ids = NS["chat_ids"](NS["PROMPTS"][P]())
NS["reset_generation_state"](model)
state = B.b12_prefill_state(model, ids, 512, 4096)
out, s = NS["spec125"].generate(model, ids, 64, k_mtp=3, max_draft=3, lookup=0, prefill_chunk=512, prefilled=state, stats={})
del state
mx.clear_cache()
census("after a p2k prefill + 64-token decode")
print("[mem2] done", flush=True)
