"""Fit12 PR1 (2026-10-24): make B12LL's committed states independent of the draft policy (WV3: a keep = 1 commit replays the
PLE-carrying layer, layer 1, as a 1-row forward whose decode-width arithmetic differs from the multi-row verify's; keep >= 2
replays and full accepts are bitwise those of any multi-row chunking). With F12_PR1=1, MTPLX's
Qwen4ExpTextModel.commit_verified_window (patched in this process only; the package is read, never edited) replays that layer
on TWO rows (the kept row + the first rejected one, both from the verify's own captured inputs) inside
boundary_capture_scope([1]), then sets its GDN conv tail + recurrent state (slots 0, 1) and its PLE conv window + n-gram tail
(slots 2, 3) from the offset-1 capture: row 0's state with multi-row arithmetic. Every other commit path is unchanged. This
CHANGES the U1 token stream wherever U1 committed a keep = 1 pass at a near-tie (a new reference, U2, for the Eval suite);
afterwards any draft policy with verify width >= 2 must give U2's tokens. CALLS[0] counts the patched replays."""
import inspect, os, textwrap

ON = os.environ.get("F12_PR1") == "1"
CALLS = [0]
_OLD = """                ple.ple_embedding.stage(kept_ids, entry, ple.NGRAM_IDX)
                layer(
                    h_in[:, :keep_tokens],
                    input_ids=kept_ids,
                    ssm_mask=None,
                    cache=entry,
                )
"""
_NEW = """                if keep_tokens == 1 and h_in.shape[1] >= 2:    # [Fit12 PR1] 2-row replay, row 0's state
                    _F12_PR1[0] += 1
                    ids2 = ids[:, :2]
                    ple.ple_embedding.stage(ids2, entry, ple.NGRAM_IDX)
                    with boundary_capture_scope([1]):
                        layer(h_in[:, :2], input_ids=ids2, ssm_mask=None, cache=entry)
                    _cap = getattr(entry, _BOUNDARY_CAPTURE_ATTR)[1]
                    entry[0], entry[1] = _cap["gdn"]
                    entry[ple.CONV_IDX], entry[ple.NGRAM_IDX] = _cap["ple"]
                    setattr(entry, _BOUNDARY_CAPTURE_ATTR, None)
                else:
                    ple.ple_embedding.stage(kept_ids, entry, ple.NGRAM_IDX)
                    layer(
                        h_in[:, :keep_tokens],
                        input_ids=kept_ids,
                        ssm_mask=None,
                        cache=entry,
                    )
"""


def install(Q):
    """apply regardless of F12_PR1 (the Eval suite's hook); idempotent"""
    global ON
    ON = True
    return apply(Q)


def apply(Q):
    if not ON:
        return False
    cls = Q.Qwen4ExpTextModel
    if getattr(cls, "_f12_pr1", False):
        return True
    src = inspect.getsource(cls.commit_verified_window)
    assert src.count(_OLD) == 1, "fit_pr1: MTPLX's commit_verified_window changed"
    src = textwrap.dedent(src.replace(_OLD, _NEW))
    Q.__dict__["_F12_PR1"] = CALLS
    loc = {}
    exec(compile(src, Q.__file__ + " [fit_pr1]", "exec"), Q.__dict__, loc)
    cls.commit_verified_window = loc["commit_verified_window"]
    cls._f12_pr1 = True
    return True
