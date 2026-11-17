# IDEAS_1117c idea 2 bytes screen: expert widths by the row's survival (path_p x gate); sure rows keep C6.
# Routing: Leap's r32_routes.npz (628 16-row checks x 12 layers x 10 ranks, softmax gates) -> K-row windows as treek_union does.
# path_p per row: BigMoE's 127-node t30b trees (heplus24), nodes 0..K-1 (best-first order), one real tree per window.
import numpy as np, sys
R = np.load('/Users/kamen/Leap/data/r32_routes.npz'); routes = R['routes'].astype(np.int32); gates = R['gates'].astype(np.float32)
T = np.load('/Users/kamen/BigMoE/data/deepdraft/ceiling/tree_t30b_heplus24_n127k8.npz')
node, pp, fi, an = T['node'], T['path_p'], T['fi'], T['an']
tree_id = fi * 100000 + an; uniq, inv = np.unique(tree_id, return_inverse=True)
P = np.zeros((len(uniq), 127), np.float32); P[inv, node] = pp          # path_p per tree per node
NL = 48; L_SLICE = routes.shape[1]; LAM = 1.8e-2
MB4, MB3, MB2 = 2.784, 2.064, 1.45   # 4-bit, MSB-3, MSB-2 (2 code bits + the same packed scales) per expert
def bytes_for(rows_routes, rows_gates, path_p, tau, mode):
    # rows_routes [K, L, 10]; width per (layer, expert) = max over uses; C6: rank 0-1 -> 4-bit, 2-9 -> MSB-3, gate < lam dropped
    K = rows_routes.shape[0]; tot_c6 = 0.0; tot_new = 0.0; n_demoted_exp = 0; n_exp = 0; n_drop = 0
    sure = path_p >= tau
    for l in range(L_SLICE):
        w_c6 = {}; w_new = {}; users = {}
        for r in range(K):
            for k in range(10):
                e = int(rows_routes[r, l, k]); g = rows_gates[r, l, k]
                if g < LAM: continue
                c6 = 4 if k < 2 else 3
                w_c6[e] = max(w_c6.get(e, 0), c6)
                if sure[r]: new = c6
                else: new = 3 if k < 2 else 2          # demoted row: one width down
                w_new[e] = max(w_new.get(e, 0), new)
                users.setdefault(e, []).append(r)
        for e, w in w_c6.items():
            tot_c6 += {4: MB4, 3: MB3}[w]; n_exp += 1
            wn = w_new[e]
            if mode == 'drop' and wn <= 2 and all(not sure[r] for r in users[e]) and len(set(users[e])) == 1:
                n_drop += 1; continue                   # single-use expert of a demoted row at rank 2-9: not loaded
            tot_new += {4: MB4, 3: MB3, 2: MB2}[wn]
            if wn < w: n_demoted_exp += 1
    scale = NL / L_SLICE / 1e3
    return tot_c6 * scale, tot_new * scale, n_demoted_exp / max(n_exp, 1), n_drop / max(n_exp, 1), (~sure).mean()
rng = np.random.default_rng(0)
for K in (16, 32):
    nwin = routes.shape[0] * 16 // K
    flat_r = routes.reshape(-1, L_SLICE, 10)[: nwin * K].reshape(nwin, K, L_SLICE, 10)
    flat_g = gates.reshape(-1, L_SLICE, 10)[: nwin * K].reshape(nwin, K, L_SLICE, 10)
    trees = rng.choice(len(uniq), size=nwin, replace=True)
    print(f"== K = {K} rows ({nwin} windows; path_p from {len(uniq)} real HE+ trees, nodes 0..{K-1}) ==")
    for mode in ('demote', 'drop'):
        for tau in (0.1, 0.2, 0.3, 0.5):
            acc = np.zeros(5)
            for w in range(nwin):
                acc += bytes_for(flat_r[w], flat_g[w], P[trees[w], :K], tau, mode)
            c6, new, dem, drp, rows = acc / nwin
            print(f"  {mode:6s} tau {tau:.1f}: C6 {c6:5.2f} GB -> {new:5.2f} GB per check (-{c6-new:4.2f} GB, -{(1-new/c6)*100:4.1f} %); "
                  f"rows demoted {rows*100:4.1f} %, experts demoted {dem*100:4.1f} %, dropped {drp*100:4.1f} %")
print("sizes: 4-bit %.3f, MSB-3 %.3f, MSB-2 %.2f MB per expert; lambda %.1e; 12-layer slice scaled x4" % (MB4, MB3, MB2, LAM))
