"""tree_gdn.py (Leap, Sun 15 Nov 2026 12:1x; TREE16 L1, data/week98/tree16_spec.md): the GDN body of one 125B layer for a
16-node tree in ONE dispatch, each node bitwise == gdnbody.gdn_body's row for that node when its root-to-node path is
run as a chain. scripts/gdnbody.py's 16-row kernel (non-G3, gates in-kernel) with three changes:
  A  each node's conv window comes from the tree's ancestor table (meta['gdn_win']: the 3 rows before it on its path,
     oldest first; -1 / -2 / -3 = conv-state slot 2 / 1 / 0), then the node itself: the chain's taps in the chain's order;
  B  the recurrence runs the nodes in DFS order (= node order, spec section 1): before a node whose parent is not the
     previous node it restores the parent's state from a device scratch slot; after a node with more than one child it
     saves its state there (meta['fork_ops']); the state is otherwise carried in registers as in the 16-row body;
  C  unchanged (per node).
The post-check conv / recurrent state is not written: tree_commit replays the accepted path from the snapshot (spec 4).
gdn_body_tree(qkv [1, T, 10240], z, a, b, conv_state, conv_w, A_log, dt_bias, state, norm_w, eps, scale_bf16, win [T, 3],
              restore [T], save [T], n_slots) -> y [1, T, 48, 128], q, k [1, T, 16, 128], v [1, T, 48, 128]
NOT RUN YET (written during the 11:3x pause); its unit test is tree_gdn_test.py."""
from functools import lru_cache
import os
import sys
import mlx.core as mx

sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_g1v3/Leap_scripts"))

_SRC = r"""
    constexpr int C = 10240, QK = 2048, HD = 128, VW = 6144, NHV = 48;
    const uint tid = thread_position_in_threadgroup.x;
    const uint sg = tid / 32, lane = tid % 32;
    const uint hv = threadgroup_position_in_grid.z;
    const uint hk = hv / 3;
    const bool own_qk = (hv % 3) == 0;
    threadgroup T qs[S_C * HD];
    threadgroup T ks[S_C * HD];
    threadgroup T vs[S_C * HD];
    threadgroup T os[S_C * HD];
    threadgroup float gt[S_C];
    threadgroup T bt[S_C];
    if (tid < (uint)S_C) {                                    // compute_g + sigmoid, once per (node, head)
        const uint s = tid;
        const T sx = a[s * NHV + hv] + dtb[hv];
        const T sp = GbLogAddExp()(sx, static_cast<T>(0));
        const float e = GbExp()(static_cast<float>(alog[hv]));
        const float m = (-e) * static_cast<float>(sp);
        gt[s] = GbExp()(m);
        bt[s] = GbSigmoid()(b[s * NHV + hv]);
    }

    // ---- A: prework of every node; taps 0-2 from the ancestor table, tap 3 = the node (the chain's order) ----
    for (uint task = sg; task < (uint)(S_C * 3); task += NSG) {
        const uint s = task / 3, part = task % 3;
        const uint gch = part == 0 ? hk * HD : (part == 1 ? QK + hk * HD : 2 * QK + hv * HD);
        const uint c0 = gch + lane * 4;
        T sv[4];
        for (int i = 0; i < 4; ++i) {
            const uint c = c0 + i;
            float acc = 0.0f;
            for (int t = 0; t < 4; ++t) {
                const int src = t < 3 ? win[s * 3 + t] : (int)s;
                const float xv = src < 0 ? (float)state[(3 + src) * C + c] : (float)qkv[src * C + c];
                acc = acc + (float)cw[c * 4 + t] * xv;
            }
            const T co = static_cast<T>(acc);
            sv[i] = TgMultiply()(co, TgSigmoid()(co));
        }
        if (part == 2) {
            for (int i = 0; i < 4; ++i) {
                vs[s * HD + lane * 4 + i] = sv[i];
                v_out[s * VW + hv * HD + lane * 4 + i] = sv[i];
            }
        } else {
            float xf[4], p = 0.0f;
            for (int i = 0; i < 4; ++i) xf[i] = (float)sv[i];
            for (int i = 0; i < 4; ++i) { const float sq = xf[i] * xf[i]; p += sq; }
            const float tot = simd_sum(p);
            const float r = metal::precise::rsqrt(tot + 1e-6f);
            for (int i = 0; i < 4; ++i) {
                const T nb = static_cast<T>(xf[i] * r);
                if (part == 0) {
                    const T qv = TgMultiply()(scale[0], nb);
                    qs[s * HD + lane * 4 + i] = qv;
                    if (own_qk) q_out[s * QK + hk * HD + lane * 4 + i] = qv;
                } else {
                    ks[s * HD + lane * 4 + i] = nb;
                    if (own_qk) k_out[s * QK + hk * HD + lane * 4 + i] = nb;
                }
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---- B: the recurrence over the nodes in DFS order, the state restored / saved at forks ----
    constexpr int RPS = HD / NSG;
    float sr[RPS][4];
    for (int j = 0; j < RPS; ++j) {
        const uint base = (hv * HD + sg + NSG * j) * HD + lane * 4;
        for (int i = 0; i < 4; ++i) sr[j][i] = static_cast<float>(dstate[base + i]);
    }
    for (int s = 0; s < S_C; ++s) {
        const int rs = rsl[s];
        if (rs >= 0) {                                        // a later sibling: its parent's state from the scratch slot
            for (int j = 0; j < RPS; ++j) {
                const size_t base = ((size_t)(rs * NHV + hv) * HD + sg + NSG * j) * HD + lane * 4;
                for (int i = 0; i < 4; ++i) sr[j][i] = slots[base + i];
            }
        }
        const threadgroup T* k_ = ks + s * HD;
        const threadgroup T* q_ = qs + s * HD;
        const float gv = gt[s];
        const T bv = bt[s];
        for (int j = 0; j < RPS; ++j) {
            const uint dv_idx = sg + NSG * j;
            float kv_mem = 0.0f;
            for (int i = 0; i < 4; ++i) {
                auto s_idx = 4 * lane + i;
                sr[j][i] = sr[j][i] * gv;
                kv_mem += sr[j][i] * k_[s_idx];
            }
            kv_mem = simd_sum(kv_mem);

            auto delta = (vs[s * HD + dv_idx] - kv_mem) * bv;

            float out = 0.0f;
            for (int i = 0; i < 4; ++i) {
                auto s_idx = 4 * lane + i;
                sr[j][i] = sr[j][i] + k_[s_idx] * delta;
                out += sr[j][i] * q_[s_idx];
            }
            out = simd_sum(out);
            if (lane == 0) {
                os[s * HD + dv_idx] = static_cast<T>(out);
            }
        }
        const int ss = ssl[s];
        if (ss >= 0) {                                        // a node with more children: keep its state for them
            for (int j = 0; j < RPS; ++j) {
                const size_t base = ((size_t)(ss * NHV + hv) * HD + sg + NSG * j) * HD + lane * 4;
                for (int i = 0; i < 4; ++i) slots[base + i] = sr[j][i];
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---- C: the gated norm of every node (unchanged) ----
    for (uint s = sg; s < (uint)S_C; s += NSG) {
        float acc = 0.0f;
        float hv4[4];
        for (int i = 0; i < 4; ++i) { hv4[i] = (float)os[s * HD + lane * 4 + i]; acc += hv4[i] * hv4[i]; }
        acc = simd_sum(acc);
        const float inv = metal::precise::rsqrt(acc / 128.0f + epsv[0]);
        for (int i = 0; i < 4; ++i) {
            const T xn = static_cast<T>(hv4[i] * inv);
            const T xw = TgMultiply()(w[lane * 4 + i], xn);
            const float gt_ = TgSigmoid()((float)z[s * VW + hv * HD + lane * 4 + i]);
            y[s * VW + hv * HD + lane * 4 + i] = static_cast<T>(gt_ * (float)xw);
        }
    }
"""


@lru_cache(None)
def _k():
    import gdnbody as GB
    return mx.fast.metal_kernel(
        name="leap_tree_gdn_body", input_names=["qkv", "state", "cw", "scale", "a", "b", "dtb", "alog", "dstate", "z", "w",
                                                "epsv", "win", "rsl", "ssl"],
        output_names=["y", "q_out", "k_out", "v_out", "slots"], source=_SRC, header=GB._HDR_G)


def fork_tables(parents):
    """(restore [T], save [T], n_slots) for nodes in DFS preorder (parents[n] < n): a node whose parent is not the
    previous node restores the parent's slot; a node with > 1 child saves to a slot freed after its last child starts.
    (tree_meta's fork_ops carries the same; this is the kernel's own, for tests and as a cross-check.)"""
    T = len(parents)
    kids = [[] for _ in range(T)]
    for n in range(1, T):
        kids[parents[n]].append(n)
    restore, save = [-1] * T, [-1] * T
    slot_of, free, used = {}, [], 0
    for n in range(T):
        p = parents[n]
        if n > 0 and p != n - 1:                          # not the first child of the previous node: restore
            restore[n] = slot_of[p]
            if kids[p][-1] == n:                          # the parent's last child: its slot is free afterwards
                free.append(slot_of.pop(p))
        if len(kids[n]) > 1:
            if free:
                save[n] = free.pop()
            else:
                save[n] = used
                used += 1
            slot_of[n] = save[n]
    return restore, save, max(used, 1)


def gdn_body_tree(qkv, z, a, b, conv_state, conv_w, A_log, dt_bias, state, norm_w, eps, scale_bf16, win, restore, save,
                  n_slots, nsg=8):
    S = qkv.shape[-2]
    assert qkv.shape[-1] == 10240 and 1 <= S <= 16 and state.dtype == mx.float32 and qkv.dtype == mx.bfloat16
    assert a.dtype == b.dtype == A_log.dtype == dt_bias.dtype == mx.bfloat16 and nsg in (8, 16, 32)
    win = mx.array(win, mx.int32).reshape(-1)
    rsl, ssl = mx.array(restore, mx.int32), mx.array(save, mx.int32)
    y, q, k, v, _ = _k()(
        inputs=[qkv.reshape(-1), conv_state.reshape(-1), conv_w.reshape(-1), scale_bf16.reshape(-1), a.reshape(-1),
                b.reshape(-1), dt_bias, A_log, state.reshape(-1), z.reshape(-1), norm_w.astype(qkv.dtype),
                mx.array([eps], mx.float32), win, rsl, ssl],
        template=[("T", qkv.dtype), ("S_C", S), ("NSG", nsg)], grid=(32 * nsg, 1, 48), threadgroup=(32 * nsg, 1, 1),
        output_shapes=[(S * 6144,), (S * 2048,), (S * 2048,), (S * 6144,), (n_slots * 48 * 128 * 128,)],
        output_dtypes=[qkv.dtype, qkv.dtype, qkv.dtype, qkv.dtype, mx.float32])
    return y.reshape(1, S, 48, 128), q.reshape(1, S, 16, 128), k.reshape(1, S, 16, 128), v.reshape(1, S, 48, 128)
