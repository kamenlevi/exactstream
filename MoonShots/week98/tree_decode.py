"""WEEK98 TREE-E2E (MoonShots, Sun 15 Nov 2026), as pre-registered in LEVERS_A_MOONSHOTS.md 'TREE-E2E': the exact tree
check in real decoding on Qwen3.5-2B (Sandbox98's target, mlx-community 4-bit) with s30 drafting.
Arms (rows per check = the bonus row + the drafts):
  c16  s30's greedy chain of 15 drafts (depth 1, then 14 deeper steps)
  t16  best-first EAGLE-2 with 15 nodes: MTP Head's sim eagle2() (TrainLab sb98/tree_price.py) on the head's REAL
       distributions: a heap of nodes by path score (product of the head's softmax probabilities), top-3 children per
       node, pop until 15 nodes; a popped node's children come from the head run on that node's own path (deeper with
       the node's token, attending to the true pairs <= the anchor and its own ancestors' chain entries). No proxy.
  t32  the same with 31 nodes (optional arm)
Verify (Leap's route (a)): each root-to-leaf path is its own chain in a batch of rows. The 2B's cache at the check start
is forked to B rows (attention KV + GDN conv / recurrent states); the 2B steps the rows one token per forward ([B, 1]):
offset 0 = the bonus, offset j = each path's j-th token. The longest path prefix matching the 2B's argmax is accepted,
plus the 2B's argmax after it (the new bonus); the accepted row's cache becomes the cache. Stepping stops at the offset
where the last matching row ends, so nothing is rolled back.
--batch fixed: B = 16 rows (32 for t32), paths padded by repeating path 0; --batch paths: B = the path count.
--rows N (T32 run, LEVERS_A 'TREE-E2E T32'): one fixed B = N for every arm (c16 and t32 both at 32 rows).
--stop-at HH:MM: no new key pair starts after that time (the pre-registered stop rule).
--share "<until> --cpu N --gpu N": before each key pair, if another session holds a quiet lock or waits in the quiet
  queue (and `peer wait` confirms it), the runner releases this session's share lock, waits until the Mac is clear,
  then takes the share lock again with these arguments (a held share lock would keep a quiet ticket from its grant).
Per key: answer ids / text, per check: answer position of the bonus, accepted drafts, kept (committed), the accepted
path's slot sequence, paths, nodes, depth, the tree (tokens, parents), target steps; head calls; seconds.
--selfcheck (smoke): (b) deeper-with-a-token == sb.core.deeper bitwise, and the tree's all-#1 path == the chain drafted
at the same anchor. (c) is offline (tree_score.py): no path matches the committed text further than the accepted one.
--probe: the batch-shape probe (row 0 of a [B, 1] step vs a [1, 1] step from the same cache, bitwise) + ms per step.
Read-only: Sandbox98 sb/, EvalSuite items and panel, jcc/arm0.jsonl. Writes ~/MoonShots/week98/tree/ (resumable).
usage: ~/BigMoE/.venv/bin/python -I -B tree_decode.py --arm t16 [--keys 0:264] [--batch fixed] [--mem 4] [--tag smoke]
       ~/BigMoE/.venv/bin/python -I -B tree_decode.py --probe"""
import argparse, glob, heapq, json, os, shlex, subprocess, sys, time
import numpy as np

H = os.path.expanduser
sys.path.insert(0, H("~/Sandbox98")); sys.path.insert(0, H("~/MoonShots/week98"))
import jcc_decode as JD                                   # noqa: E402  (panel_items; the J(CC) loop's conventions)

OUT = H("~/MoonShots/week98/tree")
ARM0 = H("~/MoonShots/week98/jcc/arm0.jsonl")
NODES = {"c16": 15, "t16": 15, "t32": 31}
DEP = 31


# ---------------------------------------------------------------- the 2B's cache: fork per path, keep one row
def fork(cache, R):
    """v2 (T32 run, memory; v1 = the T16 run): an R-row KV buffer is built at length offset + KVCache.step, the buffer
    the first update_and_fetch builds from an exact-length one (same values, same strides), so no check pays a second
    R-row KV copy at its first step. Smoke-checked bitwise against v1's outputs."""
    import mlx.core as mx
    from mlx_lm.models.cache import KVCache, ArraysCache
    out = []
    for c in cache:
        if isinstance(c, KVCache):
            n = KVCache()
            k, v = c.state
            L = k.shape[2]
            if R > 1:
                kb = mx.zeros((R, k.shape[1], L + KVCache.step, k.shape[3]), k.dtype)
                vb = mx.zeros((R, v.shape[1], L + KVCache.step, v.shape[3]), v.dtype)
                kb[:, :, :L, :] = k
                vb[:, :, :L, :] = v
                n.keys, n.values = kb, vb
            else:
                n.keys, n.values = k, v
            n.offset = L
        else:
            assert isinstance(c, ArraysCache) and c.lengths is None and c.left_padding is None
            n = ArraysCache(size=len(c.cache))
            n.cache = [None if x is None else (mx.repeat(x, R, axis=0) if R > 1 else x) for x in c.cache]
        out.append(n)
    return out


def select(fc, p):
    """v2: the kept row is gathered into its own buffers and evaluated at once. v1's slices were views that pinned the
    whole R-row KV and GDN buffers through the next check."""
    import mlx.core as mx
    from mlx_lm.models.cache import KVCache, ArraysCache
    ix = mx.array([p], dtype=mx.int32)
    out, ev = [], []
    for c in fc:
        if isinstance(c, KVCache):
            n = KVCache()
            n.keys = mx.take(c.keys[:, :, :c.offset], ix, axis=0)
            n.values = mx.take(c.values[:, :, :c.offset], ix, axis=0)
            n.offset = c.offset
            ev += [n.keys, n.values]
        else:
            n = ArraysCache(size=len(c.cache))
            n.cache = [None if x is None else mx.take(x, ix, axis=0) for x in c.cache]
            ev += [x for x in n.cache if x is not None]
        out.append(n)
    mx.eval(ev)
    return out


def step(t, fc, feed):
    """one [R, 1] forward of the 2B -> post-norm hidden [R, d], fp32 logits [R, V]"""
    import mlx.core as mx
    post = t(mx.array(np.asarray(feed, np.int32))[:, None], cache=fc)
    return post[:, 0], t.embed_tokens.as_linear(post)[:, 0].astype(mx.float32)


# ---------------------------------------------------------------- the head: chain and best-first tree
def deeper_tok(C, D, h, tok, anc, j, K1, V1, ck, cv):
    """sb.core.deeper with the token given (tok [n], mx int32) instead of ida[anc + j]; returns (h, this depth's chain
    entries k, v [L, 1, KV, n, hd]) and leaves ck / cv alone"""
    import mlx.core as mx
    x = C.fuse(D, h, C.embed(tok)[None])
    ks, vs = [], []
    for li, L in enumerate(C.layers_of(D)):
        q, gate, k, v = C.qkv(L, L.input_layernorm(x), pos=anc + j - 1)
        ks.append(k); vs.append(v)
        o = C.ttt_attn(L.scale, q, K1[li], V1[li], [c[li] for c in ck] + [k], [c[li] for c in cv] + [v], anc)
        blk = L.o_proj(o.transpose(0, 2, 1, 3).reshape(1, h.shape[1], -1) * mx.sigmoid(gate))
        x = C.block(L, x, blk)
    return x, mx.stack(ks), mx.stack(vs)


def top3(C, D, h):
    """the head's #1..#3 tokens and softmax probabilities at h [1, 1, d]; slot 0 = the argmax (ties), as the sim"""
    import mlx.core as mx
    lg = C.lm_logits(D.norm(h))[0, 0]
    am = mx.argmax(lg)
    i3 = mx.argpartition(-lg, 2)[:3]
    v3 = lg[i3]
    lse = mx.logsumexp(lg)
    va = lg[am]
    mx.eval(am, i3, v3, lse, va)
    am, i3, v3, lse, va = int(am.item()), np.array(i3).tolist(), np.array(v3).tolist(), float(lse.item()), float(va.item())
    o = sorted(range(3), key=lambda r: (-v3[r], r))
    i3, v3 = [i3[r] for r in o], [v3[r] for r in o]
    if i3[0] != am:
        rest = [(a, b) for a, b in zip(i3, v3) if a != am][:2]
        i3, v3 = [am] + [a for a, _ in rest], [va] + [b for _, b in rest]
    return i3, [float(np.exp(v - lse)) for v in v3]


def draft_chain(C, D, h1, anc, K1, V1, n):
    """s30's greedy chain of n drafts (the J(CC) loop's drafts, made up front)"""
    import mlx.core as mx
    toks, ck, cv, h = [], [], [], h1
    for j in range(1, n + 1):
        if j >= 2:
            h, k, v = deeper_tok(C, D, h, mx.array([toks[-1]], dtype=mx.int32), anc, j, K1, V1, ck, cv)
            ck.append(k); cv.append(v)
        toks.append(int(mx.argmax(C.lm_logits(D.norm(h))[0, 0]).item()))
    return toks, n - 1


def draft_tree(C, D, h1, anc, K1, V1, n):
    """best-first EAGLE-2 (the sim's eagle2(), real distributions): nodes [(tok, parent, slots)], head calls"""
    import mlx.core as mx
    nodes, hs, kk, vv = [], {}, {}, {}
    t3, p3 = top3(C, D, h1)
    heap = [(-p3[r], (r,), -1, t3[r]) for r in range(3) if p3[r] > 0]
    heapq.heapify(heap)
    calls = 0
    while heap and len(nodes) < n:
        ns, rt, par, tk = heapq.heappop(heap)
        v = len(nodes)
        nodes.append((tk, par, rt))
        if len(nodes) < n and len(rt) < DEP:
            chain, u = [], par
            while u >= 0:
                chain.append(u); u = nodes[u][1]
            chain.reverse()                                    # ancestors at depths 1 .. d-1
            h, k, vk = deeper_tok(C, D, h1 if par < 0 else hs[par], mx.array([tk], dtype=mx.int32), anc, len(rt) + 1,
                                  K1, V1, [kk[u] for u in chain], [vv[u] for u in chain])
            hs[v], kk[v], vv[v] = h, k, vk
            calls += 1
            t3, p3 = top3(C, D, h)
            for r in range(3):
                if p3[r] > 0:
                    heapq.heappush(heap, (ns * p3[r], rt + (r,), v, t3[r]))
    return nodes, calls


def leaf_paths(nodes):
    """root-to-leaf token paths and their node ids, in node order of the leaves"""
    has_child = set(p for _, p, _ in nodes if p >= 0)
    out = []
    for v in range(len(nodes)):
        if v in has_child:
            continue
        seq, u = [], v
        while u >= 0:
            seq.append(u); u = nodes[u][1]
        seq.reverse()
        out.append(([nodes[u][0] for u in seq], seq))
    return out


# ---------------------------------------------------------------- yield to quiet tickets (peer)
PEERS = H("~/.claude-peers")


def _alive(pid):
    try:
        os.kill(int(pid), 0)
        return True
    except (OSError, ValueError, TypeError):
        return False


def quiet_wanted(me):
    """cheap pre-filter: another session holds a live quiet lock or waits in the quiet queue"""
    now = time.time()
    for f in glob.glob(os.path.join(PEERS, "queue", "*.json")):
        try:
            q = json.load(open(f))
        except (OSError, ValueError):
            continue
        if q.get("pid") != me and _alive(q.get("waiter", 0)):
            return True
    for f in glob.glob(os.path.join(PEERS, "locks", "*.json")):
        try:
            pid, lk = int(os.path.basename(f)[:-5]), json.load(open(f))
        except (OSError, ValueError):
            continue
        if pid != me and lk.get("mode") == "quiet" and lk.get("until", 0) > now and _alive(pid):
            return True
    return False


def yield_quiet(share, stop_at):
    """if peer confirms a quiet window is wanted: drop this session's share lock (it would keep the ticket from its
    grant), wait until the Mac is clear, take the share lock again"""
    me = int(os.environ.get("CLAUDE_PID", "0") or 0)
    if not me or not quiet_wanted(me):
        return
    if subprocess.run(["peer", "wait", "--timeout", "1m"], capture_output=True, text=True).returncode == 0:
        return                                            # an idle or stale lock: peer sees nothing to yield to
    t0 = time.time()
    print("[tree] quiet window wanted: share lock released, waiting", flush=True)
    subprocess.run(["peer", "unlock"], capture_output=True, text=True)
    while not (stop_at and time.time() >= stop_at):
        if subprocess.run(["peer", "wait", "--timeout", "30m"], capture_output=True, text=True).returncode != 0:
            continue
        r = subprocess.run(["peer", "lock", "share"] + shlex.split(share) +
                           ["MoonShots TREE-E2E T32 runner (resumed after a quiet window)"], capture_output=True, text=True)
        if r.returncode == 0:
            break
        time.sleep(10)
    print(f"[tree] resumed after {time.time() - t0:.0f} s", flush=True)


# ---------------------------------------------------------------- one answer
def decode_one(it, arm, D, model, tok, C, batch, selfcheck=False, rows=0):
    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache
    from sb.gen import chat_ids
    t = C.tm()
    prompt = chat_ids(tok, it["user"])
    cap, eos = int(it["max_tokens"]), set(tok.eos_token_ids)
    cache = make_prompt_cache(model)
    ids = list(prompt)
    post = t(mx.array(np.asarray(prompt, np.int32))[None], cache=cache)      # prefill: W rows 0..P-1
    lg = t.embed_tokens.as_linear(post[:, -1:])[0, 0].astype(mx.float32)
    ids.append(int(mx.argmax(lg).item()))                 # the first answer token: the prefill's argmax (no draft)
    Wl = [post[0]]
    nmax = NODES[arm]
    B = (rows or (32 if arm == "t32" else 16)) if batch == "fixed" else None
    checks, sc = [], dict(deeper_eq=0, deeper_n=0, path1_eq=0, path1_n=0, copy_eq=0, copy_n=0)
    calls = steps = 0
    t0 = time.perf_counter()
    while ids[-1] not in eos and len(ids) - len(prompt) < cap:
        S = len(ids)                                      # ids[S-1] = the bonus, not yet through the 2B
        W = mx.concatenate(Wl + [mx.zeros_like(Wl[0][:1])], 0)[None]
        ida = mx.array(np.asarray(ids, np.int32))
        h1, K1, V1 = C.depth1(D, W, ida, S - 2)
        anc = mx.array([S - 2])
        calls += 1
        if arm == "c16":
            toks, c = draft_chain(C, D, h1, anc, K1, V1, nmax)
            nodes = [(tk, i - 1, (0,) * (i + 1)) for i, tk in enumerate(toks)]
        else:
            nodes, c = draft_tree(C, D, h1, anc, K1, V1, nmax)
            if selfcheck:                                 # (b): the tree's all-#1 path vs the chain at this anchor
                ch, _ = draft_chain(C, D, h1, anc, K1, V1, nmax)
                p1, u = [], -1
                while True:
                    nx = [v for v, (_, p, rt) in enumerate(nodes) if p == u and rt[-1] == 0]
                    if not nx:
                        break
                    u = nx[0]; p1.append(nodes[u][0])
                sc["path1_n"] += 1; sc["path1_eq"] += int(p1 == ch[:len(p1)])
                # deeper_tok vs sb.core.deeper on the chain (bitwise, every depth)
                ida2 = mx.array(np.asarray(ids + ch, np.int32))
                hA = hB = h1; ckA, cvA, ckB, cvB = [], [], [], []
                for j in range(2, nmax + 1):
                    hA, k, v = deeper_tok(C, D, hA, mx.array([ch[j - 2]], dtype=mx.int32), anc, j, K1, V1, ckA, cvA)
                    ckA.append(k); cvA.append(v)
                    hB = C.deeper(D, hB, ida2, anc, j, K1, V1, ckB, cvB)
                    sc["deeper_n"] += 1; sc["deeper_eq"] += int(mx.array_equal(hA, hB).item())
        calls += c
        paths = leaf_paths(nodes)
        rows = [p for p, _ in paths]
        R = len(rows) if B is None else B
        assert len(rows) <= R, (len(rows), R)
        rows = rows + [rows[0]] * (R - len(rows))
        fc = fork(cache, R)
        b = ids[-1]
        alive, end, bon = list(range(R)), [None] * R, [None] * R
        feed, posts, j = [b] * R, [], 0
        while True:
            po, lgs = step(t, fc, feed)
            am = mx.argmax(lgs, -1)
            mx.eval(am, po)
            if selfcheck and R > len(paths):              # padded rows = copies of row 0: bitwise equal to it
                sc["copy_n"] += 1
                sc["copy_eq"] += int(mx.array_equal(lgs[len(paths):], mx.broadcast_to(lgs[:1], lgs[len(paths):].shape)).item())
            am = np.array(am).tolist()
            posts.append(po)
            steps += 1
            nxt = []
            for r in alive:
                pr = rows[r]
                if j < len(pr) and pr[j] == am[r]:
                    if pr[j] in eos:                      # an end token kept as a draft: the answer ends there
                        end[r], bon[r] = j + 1, None
                    else:
                        nxt.append(r)
                else:
                    end[r], bon[r] = j, am[r]
            if not nxt:
                break
            alive = nxt
            feed = [rows[r][j] if j < len(rows[r]) else rows[r][-1] for r in range(R)]
            j += 1
        p = max(range(R), key=lambda r: (end[r], -r))
        m = end[p]
        com = rows[p][:m] + ([bon[p]] if bon[p] is not None else [])
        room = cap - (len(ids) - len(prompt))
        cut = len(com) > room or any(x in eos for x in com)
        com = com[:room]
        if any(x in eos for x in com):
            com = com[:com.index(next(x for x in com if x in eos)) + 1]
        pi = paths[p][1] if p < len(paths) else paths[0][1]
        checks.append(dict(s=S - 1 - len(prompt), m=m, kept=len(com), slots=list(nodes[pi[m - 1]][2]) if m else [],
                           paths=len(paths), nodes=len(nodes), depth=max(len(rt) for _, _, rt in nodes),
                           steps=len(posts), cut=bool(cut), tree=[[tk for tk, _, _ in nodes], [pp for _, pp, _ in nodes]]))
        ids.extend(com)
        cache = select(fc, p)
        Wl.append(mx.stack([po[p] for po in posts[:m + 1]]))
        del fc, posts
    dt = time.perf_counter() - t0
    ans = ids[len(prompt):]
    stop = "eos" if ans and ans[-1] in eos else "cap"
    row = dict(key=it["key"], set=it["set"], arm=arm, batch=batch, rows=B, n=len(ans), stop=stop, ids=ans,
               text=tok.decode([x for x in ans if x not in eos]), checks=checks, head_calls=calls, target_steps=steps,
               secs=round(dt, 2), prompt_len=len(prompt))
    if selfcheck:
        row["selfcheck"] = sc
    return row


# ---------------------------------------------------------------- the batch-shape probe
def probe(C, model, tok, mem):
    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache
    from sb.gen import chat_ids
    t = C.tm()
    its = {it["key"]: it for it in JD.panel_items()}
    a0 = [json.loads(l) for l in open(ARM0)]
    r = max(a0, key=lambda r: r["prompt_len"] + r["n"])
    prompt = chat_ids(tok, its[r["key"]]["user"])
    ids = prompt + r["ids"]
    P = len(prompt)
    marks = sorted(set(min(max(c, P + 2), len(ids) - 40) for c in (300, 800, 1250)))
    print(f"[probe] {r['key']}: prompt {P}, answer {r['n']}, probe contexts {marks}", flush=True)
    cache = make_prompt_cache(model)
    post = t(mx.array(np.asarray(prompt, np.int32))[None], cache=cache)
    lg = t.embed_tokens.as_linear(post[:, -1:])[0, 0].astype(mx.float32)
    same_greedy = int(mx.argmax(lg).item()) == ids[P]
    res = {"key": r["key"], "contexts": marks, "rows": [], "timing_ms": {}}
    BS = (1, 2, 4, 8, 15, 16, 31, 32)
    for pos in range(P, max(marks) + 1):                  # teacher-forced one-token steps through arm0's answer
        x = ids[pos]
        if pos in marks:
            f1 = fork(cache, 1)
            po1, lg1 = step(t, f1, [x]); mx.eval(po1, lg1)
            del f1
            for Bn in BS:
                row0 = {}
                for mode in ("same", "distinct"):
                    if Bn == 1 and mode == "distinct":
                        continue
                    fb = fork(cache, Bn)
                    feed = [x] * Bn if mode == "same" else [x] + [ids[pos + 1 + i] for i in range(Bn - 1)]
                    pob, lgb = step(t, fb, feed); mx.eval(pob, lgb)
                    eq_l = bool(mx.array_equal(lgb[0], lg1[0]).item())
                    eq_h = bool(mx.array_equal(pob[0], po1[0]).item())
                    dl = float(mx.abs(lgb[0] - lg1[0]).max().item())
                    am_eq = int(mx.argmax(lgb[0]).item()) == int(mx.argmax(lg1[0]).item())
                    row0[mode] = (lgb[0], pob[0])
                    q = dict(ctx=pos, B=Bn, mode=mode, logits_equal=eq_l, post_equal=eq_h, max_abs_logit_diff=dl,
                             argmax_equal=am_eq)
                    if mode == "same":                    # row-index invariance: every row equals row 0 bitwise
                        q["all_rows_equal_row0"] = bool(mx.array_equal(lgb, mx.broadcast_to(lgb[:1], lgb.shape)).item()
                                                        and mx.array_equal(pob, mx.broadcast_to(pob[:1], pob.shape)).item())
                    else:                                 # row-content invariance: row 0 does not see the other rows
                        q["row0_equals_same_mode"] = bool(mx.array_equal(lgb[0], row0["same"][0]).item()
                                                          and mx.array_equal(pob[0], row0["same"][1]).item())
                    res["rows"].append(q)
                    del fb
                    mx.clear_cache()
            po0, lg0 = step(t, cache, [x])                # the original cache's own step (KVCache's own buffer)
            mx.eval(po0, lg0)
            res["rows"].append(dict(ctx=pos, B=0, mode="original_vs_fork1",
                                    logits_equal=bool(mx.array_equal(lg0[0], lg1[0]).item()),
                                    post_equal=bool(mx.array_equal(po0[0], po1[0]).item()),
                                    max_abs_logit_diff=float(mx.abs(lg0[0] - lg1[0]).max().item()), argmax_equal=True))
            print(f"[probe] ctx {pos}: " + ", ".join(f"B{q['B']}{q['mode'][0]}={'Y' if q['logits_equal'] and q['post_equal'] else 'N'}"
                                                   f"({q['max_abs_logit_diff']:.2g})" for q in res["rows"] if q["ctx"] == pos), flush=True)
        else:
            po0, lg0 = step(t, cache, [x])
            mx.eval(po0, lg0)
        if pos + 1 < len(ids) and int(mx.argmax(lg0[0]).item()) != ids[pos + 1]:
            same_greedy = False
    res["teacher_forced_argmax_equals_arm0"] = same_greedy
    # ms per [B, 1] step at the last context (fork, 6 steps, median of the last 5)
    for Bn in BS:
        fb = fork(cache, Bn)
        ms = []
        for i in range(6):
            t1 = time.perf_counter()
            pob, lgb = step(t, fb, [ids[-1]] * Bn)
            mx.eval(mx.argmax(lgb, -1), pob)
            ms.append(1000 * (time.perf_counter() - t1))
        res["timing_ms"][Bn] = round(float(np.median(ms[1:])), 2)
        del fb
        mx.clear_cache()
    res["peak_gb"] = round(mx.get_peak_memory() / 2 ** 30, 2)
    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(os.path.join(OUT, "probe.json"), "w"), indent=1)
    allq = [q for q in res["rows"] if q["B"] > 0]
    print(f"[probe] within a shape: all rows == row 0 (same input) in "
          f"{sum(q.get('all_rows_equal_row0', False) for q in allq)} / {sum('all_rows_equal_row0' in q for q in allq)}; "
          f"row 0 unchanged by the other rows' tokens in {sum(q.get('row0_equals_same_mode', False) for q in allq)} / "
          f"{sum('row0_equals_same_mode' in q for q in allq)}", flush=True)
    print(f"[probe] teacher-forced argmax == arm0 ids: {same_greedy}; rows bitwise equal (logits and post) in "
          f"{sum(q['logits_equal'] and q['post_equal'] for q in allq)} / {len(allq)}; fork(1) == original: "
          f"{all(q['logits_equal'] and q['post_equal'] for q in res['rows'] if q['B'] == 0)}; ms/step {res['timing_ms']}; "
          f"peak {res['peak_gb']} GB", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", help="c16 / t16 / t32, or a comma list: one process runs the arms key by key (paired "
                                  "coverage if a run stops early)")
    ap.add_argument("--keys", default="0:264")
    ap.add_argument("--batch", choices=["fixed", "paths"], default="fixed")
    ap.add_argument("--mem", type=float, default=4.0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--rows", type=int, default=0, help="one fixed B for every arm (0: 16, or 32 for t32)")
    ap.add_argument("--stop-at", default="", help="HH:MM: start no key pair after it")
    ap.add_argument("--share", default="", help='"<until> --cpu N --gpu N": yield to quiet tickets, then re-take')
    a = ap.parse_args()
    from sb import core as C
    import mlx.core as mx
    C.limit_memory(a.mem)
    model, tok = C.load_target()
    if a.probe:
        probe(C, model, tok, a.mem)
        return
    arms = a.arm.split(",")
    assert arms and all(x in NODES for x in arms), arms
    D = C.load_drafter("s30")
    its = JD.panel_items()
    lo, hi = (int(x) for x in a.keys.split(":"))
    os.makedirs(OUT, exist_ok=True)
    paths = {A: os.path.join(OUT, f"{A}_{a.batch}" + (f"_{a.tag}" if a.tag else "") + ".jsonl") for A in arms}
    done = {A: ({json.loads(l)["key"] for l in open(p)} if os.path.exists(p) else set()) for A, p in paths.items()}
    t_all, n_all = time.perf_counter(), 0
    stop_at = None
    if a.stop_at:
        hh, mm = (int(x) for x in a.stop_at.split(":"))
        lt = time.localtime()
        stop_at = time.mktime((lt.tm_year, lt.tm_mon, lt.tm_mday, hh, mm, 0, 0, 0, -1))
    for it in its[lo:hi]:
        if all(it["key"] in done[A] for A in arms):
            continue
        if stop_at and time.time() >= stop_at:
            print(f"[tree] stop rule: {a.stop_at} reached; no new key pair (next: {it['key']})", flush=True)
            break
        if a.share:
            yield_quiet(a.share, stop_at)
            if stop_at and time.time() >= stop_at:
                print(f"[tree] stop rule: {a.stop_at} reached while yielding; no new key pair", flush=True)
                break
        for A in arms:
            if it["key"] in done[A]:
                continue
            C.wait_pause()
            row = decode_one(it, A, D, model, tok, C, a.batch, selfcheck=a.selfcheck, rows=a.rows)
            with open(paths[A], "a") as f:
                f.write(json.dumps(row) + "\n")
            n_all += row["n"]
            ck = row["checks"]
            kp = sum(c["kept"] for c in ck) / max(len(ck), 1)
            print(f"[tree {A}] {row['key']}: {row['n']} tok ({row['stop']}), {len(ck)} checks, kept {kp:.2f}, "
                  f"steps {row['target_steps']}, head {row['head_calls']}, {row['n'] / max(row['secs'], 1e-9):.1f} tok/s"
                  + (f", selfcheck {row['selfcheck']}" if a.selfcheck else "")
                  + f", peak {mx.get_peak_memory() / 2 ** 30:.2f} GB", flush=True)
            mx.clear_cache()
    print(f"[tree {a.arm}] done: {n_all} tokens in {time.perf_counter() - t_all:.0f} s; "
          f"peak {mx.get_peak_memory() / 2 ** 30:.2f} GB", flush=True)


if __name__ == "__main__":
    main()
