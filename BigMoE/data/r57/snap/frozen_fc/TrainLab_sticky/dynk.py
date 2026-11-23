"""R-26 DYNK-Q (NewIdeas Tue ~20:15; MoonShots' R-25 DYNK, the bytes side): per verify row, drop the rank-8 / rank-9
expert of the top-10 only when its top-10-renormalized gate weight is < tau, and renormalize the kept ones (fixed top-8
= S9 always drops both). Applied inside the suite's block on v3's own sorted scores: block_dynk is evalsuite/model.py's
block() (the STATE.sorted path, the same ops in the same order; tests/test_dynk.py compares it with the suite's source)
plus dynk_scores between the sort and the expert sum. So the ids, their rank order and v3's tiers by rank stay v3's; a
row that keeps all 10 keeps v3's scores bit for bit; a dropping row gets the dropped scores zeroed and the rest divided
by their sum. The dropped experts are still computed (x 0): the bytes side is MoonShots' replay. The scores sort
descending, so rank 8 below tau implies rank 9.
dynk_scores(sorted scores [..., k], tau) -> (scores, [rows, rows dropping rank 9, rows dropping 8 and 9, dropped weight]).
R-31 (UNIQ-Q, MoonShots' R-30): block_dynk also takes a rule(ids, scores) -> (scores, stats) instead of tau, e.g.
uniq_scores (an expert used only at ranks >= 4 in the whole check, with a summed weight < theta, is dropped from it)."""
import mlx.core as mx


def dynk_scores(scores, tau):
    k = scores.shape[-1]
    lo = scores[..., k - 2:] < tau
    drop = mx.concatenate([mx.zeros(lo.shape[:-1] + (k - 2,), mx.bool_), lo], axis=-1)
    kept = mx.where(drop, mx.zeros_like(scores), scores)
    out = mx.where(drop.any(axis=-1, keepdims=True), kept / kept.sum(axis=-1, keepdims=True), scores)
    f, d2 = scores.astype(mx.float32).reshape(-1, k), drop.reshape(-1, k)
    st = mx.stack([mx.array(float(f.shape[0])), d2[:, k - 1].sum().astype(mx.float32), d2[:, k - 2].sum().astype(mx.float32),
                   (f * d2).sum()])
    return out, st


def uniq_scores(inds, scores, theta, r0=4, renorm=True, E=512):
    """R-31 UNIQ-Q (MoonShots' R-30, check-level: uniq24.py UNIQ_MODE=mass): over the rows of one call (one verify check),
    expert e's mass = the sum of its top-10-renormalized scores over the rows that use it at ranks >= r0, plus 9 for every
    use at a rank < r0 (never dropped); its rank >= r0 uses are dropped from every row when that mass < theta, the kept
    scores renormalized per row (renorm; a row without a drop keeps v3's scores bit for bit) or left as they are.
    -> (scores, [rows, rows with a drop, dropped uses, dropped weight])"""
    k = scores.shape[-1]
    i2, s2 = inds.reshape(-1, k), scores.reshape(-1, k)
    sf = s2.astype(mx.float32)
    m = mx.zeros((E,), mx.float32).at[i2[:, r0:].reshape(-1)].add(sf[:, r0:].reshape(-1)).at[i2[:, :r0].reshape(-1)].add(9.0)
    drop = mx.concatenate([mx.zeros((i2.shape[0], r0), mx.bool_), mx.take(m, i2[:, r0:]) < theta], axis=-1)
    kept = mx.where(drop, mx.zeros_like(s2), s2)
    out = mx.where(drop.any(axis=-1, keepdims=True), kept / kept.sum(axis=-1, keepdims=True), s2) if renorm else kept
    st = mx.stack([mx.array(float(i2.shape[0])), drop.any(axis=-1).sum().astype(mx.float32), drop.sum().astype(mx.float32),
                   (sf * drop).sum()])
    return out.reshape(scores.shape), st


# block_dynk: chunk > 0 (R-31) runs the expert computation on at most `chunk` rows per call (routing still sees the
# whole check); rows are independent there, and > 8-row calls take the engine's full-E path (16 rows: 4.4x per position)
def block_dynk(self, x, tau, STATE, acc, chunk=0):
    gates = mx.softmax(self.gate(x), axis=-1, precise=True)
    k = self.top_k
    inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
    scores = mx.take_along_axis(gates, inds, axis=-1)
    if self.norm_topk_prob:
        scores = scores / scores.sum(axis=-1, keepdims=True)
    order = mx.argsort(-scores, axis=-1)
    inds, scores = mx.take_along_axis(inds, order, axis=-1), mx.take_along_axis(scores, order, axis=-1)
    if STATE.capture is not None:
        STATE.capture.setdefault(self.switch_mlp._bm["L"], []).append(inds)
    scores, st = tau(inds, scores) if callable(tau) else dynk_scores(scores, tau)   # R-26 (R-31: a rule on ids + scores)
    acc.append(st)                                                               # R-26
    if chunk and x.shape[-2] > chunk:                                            # R-31 chunk
        y = mx.concatenate([(self.switch_mlp(x[..., a:a + chunk, :], inds[..., a:a + chunk, :])  # R-31 chunk
                             * scores[..., a:a + chunk, :, None]).sum(axis=-2)  # R-31 chunk
                            for a in range(0, x.shape[-2], chunk)], axis=-2)    # R-31 chunk
    else:                                                                        # R-31 chunk
        y = (self.switch_mlp(x, inds) * scores[..., None]).sum(axis=-2)
    return y + mx.sigmoid(self.shared_expert_gate(x)) * self.shared_expert(x)
