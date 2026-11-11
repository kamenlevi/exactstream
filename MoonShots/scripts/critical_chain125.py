"""Round 32k: the dependent-kernel chain of a 125B verify pass, offline from the exported MLX graphs (data/pass_graph125.dot,
data/_layer.dot; round 26-29 exports of an unevaluated 4-row forward). DAG over primitives (prim -> prim when an output
array of one is an input of the other); weight 1 per kernel, 0 per free view op (Reshape, Broadcast, ExpandDims, Squeeze,
Flatten, Unflatten, Slice, Split, Transpose, AsStrided, StopGradient: round 29's classification). Prints the number of
kernel nodes, the longest weighted path (= dependent kernels on the critical chain), per layer (/48), and the label
histogram along the critical chain (what a chain-shortening fusion would have to merge).
usage: critical_chain125.py [dot ...]   (CPU, seconds)"""
import collections, re, sys

FREE = {"Reshape", "Broadcast", "ExpandDims", "Squeeze", "Flatten", "Unflatten", "Slice", "Split", "Transpose",
        "AsStrided", "StopGradient", "Depends"}
files = sys.argv[1:] or ["/Users/kamen/MoonShots/data/pass_graph125.dot", "/Users/kamen/MoonShots/data/_layer.dot"]
for f in files:
    # a primitive object can be shared by several ops (compiled / custom kernels reused across layers): each definition
    # line is its own node, and the edges that follow it (until the id's next definition) belong to that occurrence
    label, a2p_out, a2p_in = {}, {}, collections.defaultdict(list)   # node labels; array -> producing node; array -> consumers
    occ, cur = collections.Counter(), {}
    for line in open(f):
        m = re.match(r'\{ (\d+) \[label ="([^"]+)"', line.strip())
        if m:
            occ[m.group(1)] += 1
            cur[m.group(1)] = f"{m.group(1)}#{occ[m.group(1)]}"
            label[cur[m.group(1)]] = m.group(2)
            continue
        m = re.match(r'"?([^"\s]+)"? -> "?([^"\s]+)"?$', line.strip())
        if m:
            a, b = m.group(1), m.group(2)
            if a.isdigit():                                  # prim -> array (output)
                a2p_out[b] = cur[a]
            else:                                            # array -> prim (input)
                a2p_in[a].append(cur[b])
    succ = collections.defaultdict(set)
    indeg = collections.Counter()
    for arr, prod in a2p_out.items():
        for cons in a2p_in.get(arr, []):
            if cons != prod and cons not in succ[prod]:
                succ[prod].add(cons)
                indeg[cons] += 1
    nodes = list(label)
    w = {n: (0 if label[n] in FREE else 1) for n in nodes}
    order, q = [], [n for n in nodes if indeg[n] == 0]
    indeg2 = dict(indeg)
    while q:
        n = q.pop()
        order.append(n)
        for s in succ[n]:
            indeg2[s] -= 1
            if indeg2[s] == 0:
                q.append(s)
    dist, prev = {}, {}
    for n in order:
        d0 = dist.get(n, 0) + w[n]
        dist[n] = d0
        for s in succ[n]:
            if d0 > dist.get(s, -1) - 0:                     # longest path: keep the max predecessor
                if d0 >= dist.get(s, 0):
                    dist[s] = d0
                    prev[s] = n
    # dist[s] currently holds max over preds of (pred's dist incl. pred weight); add own weight when visited above
    end = max(order, key=lambda n: dist[n])
    path, n = [], end
    while n is not None:
        path.append(n)
        n = prev.get(n)
    path.reverse()
    chain = [label[n] for n in path if w[n]]
    kern = sum(w.values())
    print(f"{f.split('/')[-1]}: {len(nodes)} primitives, {kern} kernels; critical chain {len(chain)} kernels "
          f"({len(chain) / 48:.1f} per layer if this is a full pass); unordered nodes {len(nodes) - len(order)}", flush=True)
    print("   chain kernels by type:", dict(collections.Counter(chain).most_common(14)), flush=True)
    seg = chain[: min(60, len(chain))]
    print("   chain start:", " > ".join(seg[:40]), flush=True)

# per-layer chain segments: split the critical chain at the router's ArgPartition (one per MoE layer)
if len(sys.argv) <= 1:
    segs, cur_seg = [], []
    for k in chain:
        cur_seg.append(k)
        if k == "ArgPartition":
            segs.append(cur_seg); cur_seg = []
    lens = [len(s) for s in segs]
    print(f"\nlayer segments on the chain (ending at each router ArgPartition): {len(segs)}, kernels each: {lens}", flush=True)
    for i in (1, 2, 3, 4, 5):
        if i < len(segs):
            print(f"segment {i} ({len(segs[i])}): " + " > ".join(segs[i]), flush=True)
