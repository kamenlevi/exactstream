"""GPU stages (one process per model load; the CLI runs each under a queued quiet lock and a memory gate).

  gen-tf          the reference's own greedy outputs for the teacher-forced prompts (once) -> cache/ref/tf_gen.jsonl
  prefill V...    for each variant: MC letter probabilities (mmlu_pro, gpqa) and the teacher-forced pass
                  (ref: caches its top-256 log-probs and routing; others: KLD / top-1 vs that cache)
                  -> results/<V>/{mc,tf}.npz
  tfstep V...     the teacher-forced pass over the own-output sequences, one token per call (decode semantics: the
                  prompt prefilled as generate_ar does it, then forward_ar per token), scored against the same cache;
                  for variants that act only on one-token calls (frugal). Resumable per sequence
                  (results/<V>/tfstep_parts/), assembled into results/<V>/tfstep.npz. ref = the step path's own noise.
  decode V...     decode sets (decode.py)
  live16 ref --arm A  R-56: the 16-row live drafting loop (arm ref2 or stock) vs its reference's AR rows (live16.py)
  live16diag ref  R-56's diagnostic: where the REF2 live loop first leaves REF2's AR in bits (live16diag.py)
  relax ref --arm A    RELAX (docs/PREREG_relax_2026-11-02.md): the gate / R1 arm A on the live loop (relax.py);
  relax-replay ref --arm A   REF2-T replays arm A's logged checks (replay.py)
  tfwin ref --run RUN  a run's committed tokens scored by the exact reference along its own prefix (tfwin.py)
  live16b ref --arm A  R-56b-e (arms in live16b.ARMS: REF2b, REF2-T, REF2-T / REF2c-T + OOO): every token + every kept row vs
                       the arm's own AR (live16b.py)
Variants sharing weights (ref, floor, plan*) run in one process; ar4 and frugal load their own runtimes."""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np
from . import env
import mlx.core as mx
from . import model as M, engine as E, data as D

CACHE = os.path.join(env.ROOT, "cache", "ref")
RESULTS = os.path.join(env.ROOT, "results")
EOS = (248044, 248046)
REF_TOPK = 256


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def save_npz(path, **arrays):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".part.npz"
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------------------------- generation
def greedy(rt, ids, max_tokens):
    from mtplx.sampling import SamplerConfig
    from mtplx import generation as G
    res = G.generate_ar(rt, list(ids), max_tokens=max_tokens, sampler=SamplerConfig(temperature=0.0, top_p=1.0, top_k=0))
    toks = getattr(res, "tokens", None) or getattr(res, "token_ids", None) or (res.get("tokens") if isinstance(res, dict) else None)
    if toks is None and isinstance(res, tuple):
        toks = res[0]
    toks = [int(t) for t in (toks or [])]
    for j, t in enumerate(toks):
        if t in EOS:
            return toks[:j + 1]
    return toks


def gen_tf(rt):
    out = os.path.join(CACHE, "tf_gen.jsonl")
    done = {}
    if os.path.exists(out):
        done = {r["key"]: r for r in map(json.loads, open(out))}
    rows = D.load("tf_prompts")
    os.makedirs(CACHE, exist_ok=True)
    t0, n = time.time(), 0
    with open(out, "a") as f:
        for r in rows:
            if r["key"] in done:
                continue
            toks = greedy(rt, r["prompt_ids"], r["max_tokens"])
            n += len(toks)
            f.write(json.dumps(dict(key=r["key"], set=r["set"], prompt_ids=r["prompt_ids"], completion_ids=toks)) + "\n")
            f.flush()
            log(f"gen {r['key']}: {len(toks)} tokens ({n / (time.time() - t0):.1f} tok/s overall)")
    return out


# ---------------------------------------------------------------------------------------------- prefill tests
def mc_items():
    return D.load("mmlu_pro") + D.load("gpqa")


def tf_items():
    gen = os.path.join(CACHE, "tf_gen.jsonl")
    rows = [json.loads(l) for l in open(gen)]
    want = {r["key"] for r in D.load("tf_prompts")}
    assert want == {r["key"] for r in rows}, "cache/ref/tf_gen.jsonl is incomplete: run gen-tf first"
    items = []
    for r in sorted(rows, key=lambda r: r["key"]):
        ids = r["prompt_ids"] + r["completion_ids"]
        items.append(dict(key=r["key"], set=r["set"], ids=ids, pos=list(range(len(r["prompt_ids"]) - 1, len(ids) - 1))))
    return items + D.load("tf_docs")


def run_batches(rt, items, score, capture=False, label=""):
    out, t0, ntok = {}, time.time(), 0
    bs = E.batches(items)
    for b, idx in enumerate(bs):
        sub = [items[i] for i in idx]
        t = time.time()
        res = E.forward(rt, sub, lambda j, pos, lg: score(idx[j], pos, lg), capture_routes=capture)
        for j, r in res.items():
            out[idx[j]] = r
        ntok += sum(len(s["ids"]) for s in sub)
        log(f"{label} batch {b + 1}/{len(bs)}: {len(sub)} seqs, {sum(len(s['ids']) for s in sub)} tokens, "
            f"{time.time() - t:.1f}s ({ntok / (time.time() - t0):.0f} tok/s overall)")
    return out


def run_mc(rt, v):
    items = mc_items()
    meta = json.load(open(os.path.join(D.ITEMS, "meta.json")))
    res = run_batches(rt, items, E.letter_head(meta["letter_ids"]), label=f"{v['name']} mc")
    letters = np.stack([res[i]["letters"][0] for i in range(len(items))]).astype(np.float32)
    save_npz(os.path.join(RESULTS, v["name"], "mc.npz"), letters=letters,
             top1=np.array([res[i]["top1"][0] for i in range(len(items))]),
             gold=np.array([it["gold"] for it in items]), n_opts=np.array([it["n_opts"] for it in items]),
             keys=np.array([it["key"] for it in items]), sets=np.array([it["set"] for it in items]),
             subjects=np.array([it.get("subject", "") for it in items]))


def _targets(items):
    return lambda i, pos: np.asarray(items[i]["ids"])[np.asarray(pos) + 1]


def run_tf(rt, v):
    items = tf_items()
    tg = _targets(items)
    if v["name"] == "ref":
        res = run_batches(rt, items, E.ref_head(REF_TOPK)(tg), capture=True, label="ref tf")
        path = os.path.join(CACHE, "tf_ref.npz")
    else:
        ref = load_ref_tf(items)
        res = run_batches(rt, items, E.var_head(ref, tg), capture=True, label=f"{v['name']} tf")
        path = os.path.join(RESULTS, v["name"], "tf.npz")
    names = sorted(res[0].keys())
    cat = {n: np.concatenate([res[i][n] for i in range(len(items))]) for n in names}
    seq = np.concatenate([np.full(len(it["pos"]), i, np.int32) for i, it in enumerate(items)])
    pos = np.concatenate([np.asarray(it["pos"], np.int32) for it in items])
    # position within the scored stretch (completion token index, or document position)
    rel = np.concatenate([np.asarray(it["pos"], np.int32) - it["pos"][0] for it in items])
    save_npz(path, seq=seq, pos=pos, rel=rel, keys=np.array([it["key"] for it in items]),
             sets=np.array([it["set"] for it in items]), **cat)
    return path


def load_ref_tf(items):
    z = np.load(os.path.join(CACHE, "tf_ref.npz"))
    assert list(z["keys"]) == [it["key"] for it in items], "reference cache does not match the items: rerun prefill ref"
    seq, pos, ids, lp = z["seq"], z["pos"], z["ids"], z["lp"]
    out = {}
    for i in range(len(items)):
        m = seq == i
        out[i] = dict(pos=pos[m], ids=ids[m], lp=lp[m])
    return out


# ---------------------------------------------------------------------------------------------- one token per call
def step_forward(rt, ids, pos, score):
    """one sequence teacher-forced through decode-style calls: MTPLX's AR prompt prefill of ids[:pos[0] + 1] as
    generate_ar runs it (the body multi-row, the last token as a 1-row call), then each later token of ids as a 1-row
    forward_ar call. score(pos_array, logits[1, V]) -> dict of lazy arrays; returns them per position (numpy)."""
    from mtplx import generation as G
    from mtplx.attention_context import attention_phase
    p0 = int(pos[0])
    assert list(pos) == list(range(p0, p0 + len(pos))), "tfstep scores a contiguous stretch"
    ps = G.restore_or_prefill_prompt_state(rt, [int(x) for x in ids[:p0 + 1]], base_hidden_variant=None,
                                           mtp_history_policy="cycle", capture_hidden=False)
    cache, logits, out = ps.trunk_cache, ps.logits, []
    for t in pos:
        if t > p0:
            with attention_phase("ar_decode"):
                logits = rt.forward_ar(mx.array([[int(ids[t])]]), cache=cache)[:, -1, :]
        r = score(np.array([t]), logits)
        mx.eval(r)
        out.append({k: np.array(x) for k, x in r.items()})
    return {k: np.concatenate([o[k] for o in out]) for k in out[0]}


def run_tfstep(rt, st, v, every=1, limit=0):
    items = tf_items()
    head = E.var_head(load_ref_tf(items), _targets(items))
    seqs = [i for i, it in enumerate(items) if it["set"] != "tf_doc"][::every]   # own outputs (no 64K documents)
    parts = os.path.join(RESULTS, v["name"], "tfstep_parts")
    os.makedirs(parts, exist_ok=True)
    t0, n, new = time.time(), 0, 0
    for k, i in enumerate(seqs):
        p = os.path.join(parts, f"{i:04d}.npz")
        if os.path.exists(p):
            continue
        if limit and new >= limit:
            return None
        it, t = items[i], time.time()
        r = step_forward(rt, it["ids"], it["pos"], lambda pos, lg: head(i, pos, lg))
        st.gate.check()                           # a timed-out spin gate invalidates the pass: never save it
        save_npz(p, **r)
        n, new = n + len(it["pos"]), new + 1
        log(f"{v['name']} tfstep {k + 1}/{len(seqs)} {it['key']}: {len(it['pos'])} positions in {time.time() - t:.1f}s, "
            f"KLD {r['kld'].mean():.4f} ({n / (time.time() - t0):.1f} tok/s overall)")
    res = [dict(np.load(os.path.join(parts, f"{i:04d}.npz"))) for i in seqs]
    names = sorted(res[0])
    path = os.path.join(RESULTS, v["name"], "tfstep.npz")
    save_npz(path, seq=np.concatenate([np.full(len(items[i]["pos"]), i, np.int32) for i in seqs]),
             pos=np.concatenate([np.asarray(items[i]["pos"], np.int32) for i in seqs]),
             rel=np.concatenate([np.arange(len(items[i]["pos"]), dtype=np.int32) for i in seqs]),
             keys=np.array([it["key"] for it in items]), sets=np.array([it["set"] for it in items]),
             every=np.int32(every), **{nm: np.concatenate([r[nm] for r in res]) for nm in names})
    log(f"{v['name']} tfstep: {len(seqs)} sequences -> {path}")
    return path


# ---------------------------------------------------------------------------------------------- main
def mem_free_gb():
    import subprocess
    r = subprocess.run(["memory_pressure"], capture_output=True, text=True).stdout
    pct = int(r.strip().splitlines()[-1].split(":")[-1].strip().rstrip("%"))
    return pct * 64 / 100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["gen-tf", "prefill", "tfstep", "decode", "live16", "live16diag", "live16b", "relax",
                                      "relax-replay", "tfwin"])
    ap.add_argument("variants", nargs="*", default=["ref"])
    ap.add_argument("--C", type=int, default=0, help="expert slots per layer (default 128 prefill, 192 decode)")
    ap.add_argument("--tests", default="mc,tf")
    ap.add_argument("--thinking", action="store_true", help="decode: thinking mode with paired sampling")
    ap.add_argument("--sets", default="math,ifeval,humaneval,ruler", help="decode: which sets")
    ap.add_argument("--every", type=int, default=1, help="tfstep: every n-th own-output sequence")
    ap.add_argument("--limit", type=int, default=0, help="tfstep: stop after this many new sequences; decode: the first n "
                                                         "items of each set (a check)")
    ap.add_argument("--max-rows", type=int, default=0, help="decode: at most this many new rows, then exit 0 (a chunk)")
    ap.add_argument("--chunk-minutes", type=float, default=0, help="decode: start no new row after this many minutes (a chunk)")
    ap.add_argument("--keys", default="", help="decode: only the items whose keys this file lists (whitespace-separated)")
    ap.add_argument("--smoke", action="store_true", help="tiny run: 8 MC items, 2 TF items, no files written")
    ap.add_argument("--arm", default="", help="live16 (R-56): ref2 or stock; relax: control, A, D, C or B")
    ap.add_argument("--run", default="", help="tfwin: the run whose decode rows the reference scores")
    ap.add_argument("--self", default="ref2tnax", help="tfwin: the exact run of the self-check (its own rows)")
    a = ap.parse_args()
    a.C = a.C or (128 if a.stage == "prefill" else 192)
    vs = [M.variant(n) for n in a.variants]
    models = {v["model"] for v in vs}
    assert len(models) == 1, "variants in one process must share weights (ar4 and frugal run alone)"
    assert not (models == {"frugal"} and a.stage in ("gen-tf", "prefill")), \
        "frugal acts on one-token calls only: run its tfstep and decode stages"
    assert models != {"fast24"} or (a.stage == "decode" and len(vs) == 1), \
        "fast24 variants: decode only (MTP), one per process (load_fast24's env is per budget)"
    assert models != {"b12"} or (a.stage == "decode" and len(vs) == 1), \
        "b12 / b8 / b8fast: decode only (MTP), one config per process (its switches are in the env)"
    log("provenance", json.dumps(env.provenance()), "selftest:", M.selftest())
    t0 = time.time()
    rt, st = M.load(vs[0], C=a.C)
    gb = st.resident_gb() if hasattr(st, "resident_gb") else float("nan")
    log(f"loaded {vs[0]['model']} at C={a.C} in {time.time() - t0:.0f}s; resident {gb:.1f} GB")
    if a.stage == "gen-tf":
        M.configure(M.variant("ref"), prefill=False)
        gen_tf(rt)
    elif a.stage == "tfstep":
        for v in vs:
            M.configure(v, prefill=False)         # deployed semantics: the variant acts on the one-token calls
            t = time.time()
            run_tfstep(rt, st, v, a.every, a.limit)
            log(f"{v['name']}: tfstep in {time.time() - t:.0f}s; stats {M.STATE.stats}")
    elif a.stage == "decode":
        from . import decode
        if not os.path.exists(os.path.join(D.ITEMS, "dec_ruler.jsonl")):     # decode items (incl. RULER) not built yet
            sys.path.append(os.path.join(env.ROOT, ".venv", "lib", "python3.12", "site-packages"))   # evalplus, after ours
            D.decode_items()
            log("built decode items")
        extra = [s for s in ("mgsm", "multi", "agent") if s in a.sets.split(",")]
        if extra and not all(os.path.exists(os.path.join(D.ITEMS, f"dec_{s}.jsonl")) for s in extra):
            D.decode_items_extra(rt.tokenizer)    # MGSM (scored) and the behaviour rows (multilingual, agentic)
            log("built the MGSM, multilingual and agentic decode items")
        for v in vs:
            t = time.time()
            r = decode.run(rt, v, thinking=a.thinking, sets=tuple(a.sets.split(",")), log=log, gate=st.gate,
                           limit=a.limit, max_rows=a.max_rows, chunk_s=a.chunk_minutes * 60,
                           keys=set(open(a.keys).read().split()) if a.keys else None)
            if r == "yield":
                st.gate.check()
                sys.exit(75)                      # yielded to a research ticket (cli.gpu re-queues it as [low])
            log(f"{v['name']}: decode {'chunk' if r == 'chunk' else 'sets'} done in {time.time() - t:.0f}s; stats {M.STATE.stats}")
    elif a.stage == "live16":                     # R-56: the 16-row live loop vs its reference's AR rows (live16.py)
        from . import live16
        live16.main(rt, st, a.arm, a.keys, a.chunk_minutes * 60, a.C, log)
    elif a.stage == "live16diag":                 # R-56's diagnostic: first bit difference, live loop vs REF2's AR
        from . import live16diag
        live16diag.main(rt, st, a.keys, a.C, log)
    elif a.stage in ("relax", "relax-replay"):    # RELAX: quality-defined modes (relax.py) and their exact replay (replay.py)
        from . import relax, replay
        if (relax if a.stage == "relax" else replay).main(rt, st, a, log) == "yield":
            st.gate.check()
            sys.exit(75)                          # yielded to a quiet ticket (cli.gpu --share asks again later)
    elif a.stage == "tfwin":                      # a run's tokens scored by the exact reference along its prefix (tfwin.py)
        from . import tfwin
        tfwin.main(rt, st, a, log)
    elif a.stage == "live16b":                    # R-56 on REF2b: every token + every kept row vs REF2b's AR (live16b.py)
        from . import live16b
        if live16b.main(rt, st, a, log) == "yield":
            st.gate.check()
            sys.exit(75)                          # yielded to a quiet ticket (cli.gpu --share asks again later)
    else:
        tests = a.tests.split(",")
        if a.smoke:
            smoke(rt, vs)
            return
        for v in vs:
            M.configure(v)
            t = time.time()
            if "mc" in tests:
                run_mc(rt, v)
            if "tf" in tests:
                run_tf(rt, v)
            log(f"{v['name']}: done in {time.time() - t:.0f}s; stats {M.STATE.stats}")
            os.makedirs(os.path.join(RESULTS, v["name"]), exist_ok=True)
            json.dump(dict(variant=v, provenance=env.provenance(), C=a.C, finished=time.time(), stats=M.STATE.stats),
                      open(os.path.join(RESULTS, v["name"], "run.json"), "w"), indent=1)
    st.gate.check()


def smoke(rt, vs):
    """sanity: (1) letter mass and argmax at the answer position on 8 MC items per variant; (2) full-vocab KLD and
    top-1 agreement vs ref over every position of two prompts, ref run twice (must be bit-identical)."""
    meta = json.load(open(os.path.join(D.ITEMS, "meta.json")))
    items = mc_items()[::200][:8]
    for v in vs:
        M.configure(v)
        t = time.time()
        res = E.forward(rt, items, E.letter_head(meta["letter_ids"]))
        lm = [float(np.exp(res[i]["letters"][0]).sum()) for i in range(len(items))]
        arg = [int(np.argmax(res[i]["letters"][0][:items[i]["n_opts"]])) for i in range(len(items))]
        log(f"smoke {v['name']}: letter mass {np.round(lm, 3).tolist()} argmax {arg} gold {[it['gold'] for it in items]} "
            f"({time.time() - t:.1f}s) stats {M.STATE.stats}")
    two = [dict(it, pos=list(range(len(it["ids"]) - 1))) for it in (items[0], items[3])]
    full = lambda i, pos, lg: {"lp": E.logsoftmax(lg)}
    runs = {}
    for v in [M.variant("ref")] + vs:
        M.configure(v)
        key = v["name"] if v["name"] not in runs else v["name"] + "#2"
        runs[key] = E.forward(rt, two, full, capture_routes=True)
    ref = runs["ref"]
    for name, r in runs.items():
        if name == "ref":
            continue
        kl, ag, ov = [], [], []
        for i in range(len(two)):
            p, q = ref[i]["lp"], r[i]["lp"]
            kl.append((np.exp(p) * (p - q)).sum(-1))
            ag.append(p.argmax(-1) == q.argmax(-1))
            a, b = ref[i]["routes"].astype(int), r[i]["routes"].astype(int)
            ov.append((a[..., :, None] == b[..., None, :]).any(-1).sum(-1).mean(-1) / 10)
        kl, ag, ov = np.concatenate(kl), np.concatenate(ag), np.concatenate(ov)
        log(f"smoke {name} vs ref: KLD mean {kl.mean():.3g} max {kl.max():.3g}, top-1 agree {100 * ag.mean():.2f}%, "
            f"router overlap {100 * ov.mean():.2f}% over {len(kl)} positions")


if __name__ == "__main__":
    main()
