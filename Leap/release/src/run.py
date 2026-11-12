#!/usr/bin/env python3
"""run.py: generate with the 125B mixture-of-experts engine on one Mac, with a published config.

  .venv/bin/python run.py --config exact --mem 45 --prompt "Explain how a refrigerator works."
  .venv/bin/python run.py --config chat --mem 32 --prompt-file question.txt --max-tokens 512
  .venv/bin/python run.py --config exact --mem 45 --keys ifeval/1220,ifeval/1313   # benchmark items (verification)

What it runs: the engine's own decode loop (engine/loop/r57_screen_<snapshot>.py), unchanged, the way the research
runner ran it (c45_runner.sh: R57_PRESET=45, one arm, one repetition, the warm-up prompt discarded), with the config's
environment line from configs/<config>-<mem>.env. This file only turns a text prompt into a prompt item (the harness's
own chat template, non-thinking), feeds it to the loop's item loader, and prints the answer and the loop's own timing.

Configs (see README for labels and gates): exact = bitwise equal to the exact reference (NAX6-N); lossy = config L + COH
(gated on known benchmarks); chat = M1-B + COH + SB1. Memory lines: 45 (C 256 expert slots per layer) or 32 (C 160).

--kind picks which benchmark family's switches apply to the prompt (the configs switch some levers per family, as their
gates measured them): chat (default; the IFEval / multilingual sets), code (HumanEval / MBPP), math (MATH / GSM8K).
Speed printed per STYLE: whole decode = tokens / decode wall; whole request = tokens / (first token + decode wall);
first token = the prefill wall; lifetime peak = the process's lifetime physical footprint (10^9 bytes)."""
import argparse, hashlib, json, os, runpy, statistics, subprocess, sys, time

ROOT = os.path.dirname(os.path.abspath(__file__))
KIND_SET = {"chat": "multi", "code": "humaneval", "math": "math"}     # the loop's SETOF names for a user prompt's key
DEFAULT_C = {"45": 256, "32": 160, "24": 100}
# The block drafter's base checkpoint (Inco AI, Apache-2.0) at the revision every measured run loaded. The drafter's loader
# asks the Hugging Face cache for the default branch, which may move on; the engine process pins this revision instead.
DRAFTER = ("incoai/Qwen3.8-Flash-Next-DFlash2", "a22a1744e85c17364dcc54a6a420af7135d4c7f6")
DRAFTER_FILES = ["*.safetensors", "*.json"]


def read_config(name, mem):
    p = os.path.join(ROOT, "configs", f"{name}-{mem}.env")
    if not os.path.exists(p):
        sys.exit(f"run.py: no config {name}-{mem} (configs/: {', '.join(sorted(os.listdir(os.path.join(ROOT, 'configs'))))})")
    C, lines = DEFAULT_C[mem], []
    for ln in open(p):
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        if ln.startswith("C="):
            C = int(ln[2:])
        elif not ln.startswith("SNAPSHOT="):      # the decode loop's name (snapshot() below)
            lines.append(ln)
    env = {}
    for line in lines:                            # comma-separated lines, applied in order (the 24 GB configs keep R57_PFCHUNK on
        last = None                               # a line of its own); a token without "=" continues the last value
        for tok in line.split(","):               # (R57_QOM_SETS=humaneval,mbpp,...)
            k, eq, v = tok.partition("=")
            if eq and k.strip().replace("_", "").isalnum() and k.strip().upper() == k.strip():
                last = k.strip(); env[last] = v
            elif last is not None:
                env[last] += "," + tok
    return C, env, p


def snapshot(cpath):
    """The decode loop a config runs on: its SNAPSHOT= line (the 24 GB configs: w24p2), else engine/loop/SNAPSHOT (rel1)."""
    for ln in open(cpath):
        if ln.strip().startswith("SNAPSHOT="):
            return ln.strip()[len("SNAPSHOT="):]
    return open(os.path.join(ROOT, "engine", "loop", "SNAPSHOT")).read().strip()


def tokenizer(store):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(store)


def drafter_cached():
    """DRAFTER's snapshot directory in the Hugging Face cache (local files only, no download), or None."""
    try:
        from huggingface_hub import snapshot_download
        p = snapshot_download(DRAFTER[0], revision=DRAFTER[1], allow_patterns=DRAFTER_FILES, local_files_only=True)
    except Exception:
        return None
    return p if all(os.path.exists(os.path.join(p, f)) for f in ("config.json", "model.safetensors")) else None


def engine_child(spec_path):
    """The engine process: feed the prompt items to the harness's item loader, then run the decode loop as __main__."""
    spec = json.load(open(spec_path))
    extra = spec["items"]                          # {"dec_<set>": [item, ...]}
    import importlib.abc, importlib.util

    def _items(mod):                               # the harness's item loader also returns the CLI's prompt items
        base = mod.load

        def load(nm):
            have = os.path.exists(os.path.join(mod.ITEMS, nm + ".jsonl"))
            return (base(nm) if have else []) + extra.get(nm, [])
        mod.load = load

    def _drafter(mod):                             # the drafter's loader gets DRAFTER's revision (it asks for the default)
        sd = mod.snapshot_download

        def snapshot_download(repo_id, *args, **kw):
            if repo_id == DRAFTER[0] and not kw.get("revision"):
                kw["revision"] = DRAFTER[1]            # a commit, resolved in the cache only (the engine runs offline): the
                kw.setdefault("local_files_only", True)   # cache's snapshots/<commit>, checked against its file listing if any
            return sd(repo_id, *args, **kw)
        mod.snapshot_download = snapshot_download
    patch = {"evalsuite.data": _items, "dflash_model_mlx": _drafter}

    class _Hook(importlib.abc.MetaPathFinder):     # patches each module right after the loop imports it (no early import:
        def find_spec(self, name, path, target=None):   # the loop sets its environment before the harness's env module)
            if name not in patch:
                return None
            fn = patch.pop(name)
            sys.meta_path.remove(self)
            try:
                sp = importlib.util.find_spec(name)
            finally:
                if patch:
                    sys.meta_path.insert(0, self)
            if sp is None:
                return None
            orig = sp.loader.exec_module

            def exec_module(mod):
                orig(mod)
                fn(mod)
            sp.loader.exec_module = exec_module
            return sp
    sys.meta_path.insert(0, _Hook())
    opened = set()
    if spec.get("trace"):                          # every file Python opens (audit hook), for the closure check
        def _audit(ev, args):
            if ev == "open" and args and isinstance(args[0], str):
                opened.add(os.path.abspath(args[0]))
        sys.addaudithook(_audit)
    os.chdir(os.path.join(ROOT, "engine"))
    sys.argv = [spec["screen"], str(spec["C"]), str(spec["cap"])]
    sys.path.insert(0, os.path.dirname(spec["screen"]))
    rc = 0
    try:
        runpy.run_path(spec["screen"], run_name="__main__")
    except SystemExit as e:
        rc = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
    finally:
        mods = []
        for m, mod in list(sys.modules.items()):
            f = getattr(mod, "__file__", None)
            if f and os.path.isfile(f):
                mods.append([m, os.path.abspath(f), hashlib.sha256(open(f, "rb").read()).hexdigest()[:16]])
        json.dump(dict(modules=sorted(mods), opened=sorted(opened)), open(spec["manifest"], "w"), indent=0)
    sys.exit(rc)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, choices=("exact", "lossy", "chat"))
    ap.add_argument("--mem", default="45", choices=("45", "32", "24"))
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--prompt")
    g.add_argument("--prompt-file")
    g.add_argument("--keys", help="comma-separated benchmark item keys from harness/data/items (e.g. ifeval/1220)")
    ap.add_argument("--kind", default="chat", choices=tuple(KIND_SET))
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--no-warmup", action="store_true", help="skip the discarded warm-up prompt (first prompt runs cold)")
    ap.add_argument("--out", default=os.path.join(ROOT, "out"))
    ap.add_argument("--trace", action="store_true", help="log every file the engine opens (closure check)")
    ap.add_argument("--env", action="append", default=[], help="extra KEY=VALUE for the engine (research switches, e.g. the "
                    "prefill paths: R57_PFD=1 PF-D, exact; R57_PFLM=1 PF-A, exact except when the prompt's last 1,024-token "
                    "chunk is under about 200 tokens; R57_PFLM=3 PF-B, which can differ when the prompt's last chunk is short)")
    ap.add_argument("--dry-run", action="store_true", help="print the engine command and environment, run nothing")
    a = ap.parse_args()

    print("Powered by MTPLX\nhttps://github.com/youssofal/mtplx", flush=True)   # MTPLX's NOTICE (Apache-2.0 section 4(d))
    C, cenv, cpath = read_config(a.config, a.mem)
    store = os.path.join(ROOT, "engine", "models", "bare")
    if not os.path.isdir(os.path.join(store, "records3")):
        sys.exit("run.py: engine/models/bare is not linked to a converted store (install.sh --store <dir>)")
    if drafter_cached() is None:
        sys.exit(f"run.py: the block drafter's base checkpoint {DRAFTER[0]} at revision {DRAFTER[1][:7]} is not in the Hugging "
                 f"Face cache (HF_HOME). Fetch it once (1.2 GB):\n  .venv/bin/hf download {DRAFTER[0]} config.json "
                 f"model.safetensors --revision {DRAFTER[1]}")
    snap = snapshot(cpath)
    screen = os.path.join(ROOT, "engine", "loop", f"r57_screen_{snap}.py")
    os.makedirs(a.out, exist_ok=True)
    tag = time.strftime("cli_%m%d_%H%M%S")
    items = {}
    if a.keys:
        keys = [k.strip() for k in a.keys.split(",") if k.strip()]
    else:
        text = a.prompt if a.prompt is not None else open(a.prompt_file).read()
        sys.path.insert(0, os.path.join(ROOT, "harness"))
        from evalsuite import data as ED                    # the harness's chat template (non-thinking), as every item
        tok = tokenizer(store)
        ids = ED.chat(tok, text)
        s = KIND_SET[a.kind]
        key = f"{s}/{tag}" if s != "math" else f"math500/{tag}"
        items[f"dec_{s}"] = [dict(key=key, set=s, prompt=text, user=text, prompt_ids=ids)]
        keys = [key]
    env = dict(os.environ)
    env.update({                                            # c45_runner.sh's base environment for one arm
        "R57_PRESET": "45", "BIGMOE_EXT_BUILD": "build", "R57_PROMPTS": str(len(keys)), "R57_REPS": "1",
        "R57_WARM": "0" if a.no_warmup else "1", "R57_TAG": tag, "R57_REFTOK": "", "R57_KEYS": ",".join(keys),
        "R57_QOM_SETS": "humaneval,mbpp", "HF_HUB_OFFLINE": "1", "PYTHONDONTWRITEBYTECODE": "1"})
    env.update(cenv)                                        # the config's own line wins (as an arm's env did)
    env["BIGMOE_EXT_BUILD"] = "build"                       # ONE extension build (engine/ext/build) for every config; the
                                                            # research arms used "build" (E) and "build_qom" (the others)
    for kv in a.env:
        k, _, v = kv.partition("=")
        env[k] = v
    spec = dict(screen=screen, C=C, cap=a.max_tokens, items=items, trace=a.trace,
                manifest=os.path.join(a.out, f"{tag}_modules.json"))
    sp = os.path.join(a.out, f"{tag}_spec.json")
    json.dump(spec, open(sp, "w"))
    shown = {k: env[k] for k in sorted(set(cenv) | {"R57_PRESET", "BIGMOE_EXT_BUILD", "R57_PROMPTS", "R57_REPS", "R57_WARM",
                                                       "R57_TAG", "R57_KEYS", "R57_QOM_SETS"})}
    print(f"config {a.config}-{a.mem} ({os.path.relpath(cpath, ROOT)}), C {C}, cap {a.max_tokens}, snapshot {snap}, "
          f"{len(keys)} prompt(s): {', '.join(keys)}; drafter base {DRAFTER[0]} @ {DRAFTER[1][:7]}", flush=True)
    if a.dry_run:
        print(json.dumps(shown, indent=1)); return 0
    log = os.path.join(a.out, f"{tag}.log")
    t0 = time.time()
    with open(log, "w") as lf:
        rc = subprocess.run([sys.executable, os.path.abspath(__file__), "--_engine", sp], env=env, stdout=lf,
                            stderr=subprocess.STDOUT).returncode
    wall = time.time() - t0
    od = os.path.join(ROOT, "engine", "data", "r57")
    try:
        toks = json.load(open(os.path.join(od, f"{tag}_tokens.json")))
        runs = json.load(open(os.path.join(od, f"{tag}_runs.json")))
    except FileNotFoundError:
        print(f"run.py: the engine wrote no tokens (rc {rc}); last lines of {log}:")
        print("".join(open(log).readlines()[-25:])); return rc or 1
    for f in (f"{tag}_tokens.json", f"{tag}_runs.json"):
        os.replace(os.path.join(od, f), os.path.join(a.out, f))
    tk = tokenizer(store)
    for r in runs:
        out = toks[r["key"]][r["arm"]][-1]
        print(f"\n=== {r['key']} [{r['arm']}, {r['type']}, drafter {r['drafter']}] {r['n']} tokens" + (f" ({r['bad']})" if r["bad"] else ""))
        print(tk.decode(out, skip_special_tokens=True))
    n = sum(r["n"] for r in runs)
    dec = sum(r["n"] / r["tps"] for r in runs if r["tps"])
    pfs = [r["pfs"] for r in runs if r.get("pfs") is not None]
    peak = max((r["mem"][6] for r in runs if r.get("mem") and len(r["mem"]) > 6), default=float("nan"))
    print(f"\nwhole decode {n / dec:.1f} tok/s | whole request {n / (sum(pfs) + dec):.1f} tok/s | first token "
          f"{statistics.median(pfs):.2f} s ({min(pfs):.2f}-{max(pfs):.2f}) | lifetime peak {peak:.2f} GB | {len(runs)} prompt(s), "
          f"{n} tokens, process wall {wall:.0f} s (load + warm-up included)")
    print(f"tokens: {os.path.join(a.out, tag + '_tokens.json')}; log: {log}; module manifest: {spec['manifest']}")
    return rc


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--_engine":
        engine_child(sys.argv[2])
    sys.exit(main())
