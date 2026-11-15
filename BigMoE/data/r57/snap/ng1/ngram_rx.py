"""N-GRAM receipt (a) and async warm (b') for the r57 decode loop (Sun 15 Nov 2026 11:3x; the orchestrator 11:2x-11:4x,
after Breakthrough's CHECK55 profile: cold PLE n-gram staging costs +5-8 ms per 16-row check on the 12-layer slice).
Both opt-in, default off; with both off nothing is installed and the loop is unchanged.

(a) R57_NGRAM_RX=1 -> runs.json "ngx" per run:
    - MTPLX's own PLE staging timer: ple_stage_s / ple_stage_calls (qwen4_exp.ple_stage_seconds / ple_stage_calls deltas;
      the staging includes the sidecar gather and its in-line cold reads);
    - the sidecar's hot_hits / hot_misses / pread_gathers / vectorized_gathers / prefetch_batches deltas;
    - per phase (prefill / verify / commit / other), from a wrapper on this sidecar instance's _rows_matrices (its single
      hot-LRU entry point): calls, unique rows, LRU misses, ms (sum, p50, p90, max);
    - verify-phase gathers split by whether (b') warmed that check (alt mode: a paired A/B inside one decode).
    R57_NGRAM_RX=2 also keeps every LRU-path call's unique row ids in order (int32, with phase and run index) and writes
    data/r57/<TAG>_ngram_rows.npz at exit (+ hot_cap_rows, hot_row_bytes), so the LRU hit rate at other sizes (1 / 2 / 4 /
    8 GB) can be simulated offline from one run (the orchestrator 11:4x).
(b') R57_NGRAM_PREFETCH=1 (every check) or =alt (odd checks only): at the drafts' readback, the verify window's n-gram rows
    that are not in the hot LRU go to the sidecar's async page-cache warm pool (submit_warm: preads on its threads; no LRU
    writes, no bytes produced). The rows come from MTPLX's own arithmetic (_rows_np) on [tok] + drafts with the PLE layer's
    committed history: the same inputs the verify's staging uses. A second call just before the verify warms rows the
    copy proposals changed. Scheduling only: the tokens must be bitwise identical (the on / off bit test).
The exact-45 loop drafts on DFlash2 (all drafts in one readback, right before the verify), so the warm can only overlap
the copy proposals, layer 0's graph build and layer 0's GPU time (REF2_PLE_LATE stages the PLE rows at layer 1)."""
import atexit, os, time
import numpy as np

RX = int(os.environ.get("R57_NGRAM_RX", "0") or 0)
_PFE = os.environ.get("R57_NGRAM_PREFETCH", "0")
assert _PFE in ("0", "1", "alt"), f"R57_NGRAM_PREFETCH={_PFE!r}: 0 / 1 / alt"
PF = _PFE != "0"
ON = bool(RX or PF)
PHC = {"prefill": 0, "verify": 1, "commit": 2, "other": 3}
_S = dict(phase="other", calls=[], rows=[], meta=[], run=0, chk=0, cur_w=False, pf_checks=0, pf_warmed=0, pf_rows=0,
          sub=set())


def install(TXT, ref2, tag, outdir="data/r57"):
    """after the model load: find the PLE layer's n-gram embedding and its sidecar; (a) wraps the sidecar's LRU path"""
    if not ON:
        return False
    from mtplx.models import qwen4_exp as Q
    idx = TXT._ple_stage_idx
    assert idx is not None, "R57_NGRAM: the model has no PLE layer"
    ple = TXT.layers[idx].ple
    emb = ple.ple_embedding
    sc = emb.ngram_embedding._sidecar
    assert sc is not None, "R57_NGRAM: no n-gram sidecar (the table is not SSD-streamed)"
    assert not PF or sc._pool is not None, "R57_NGRAM_PREFETCH: the sidecar has no warm pool (MTPLX_NGRAM_PREFETCH=0?)"
    _S.update(Q=Q, ref2=ref2, idx=idx, ngi=int(ple.NGRAM_IDX), emb=emb, sc=sc, tag=tag, outdir=outdir)
    if RX:
        orig = sc._rows_matrices                         # the bound method; the wrapper shadows it on this instance only

        def rec(flat, names):
            m0 = sc.hot_misses
            t0 = time.perf_counter()
            out = orig(flat, names)
            dt = time.perf_counter() - t0
            u = np.unique(np.asarray(flat, dtype=np.int64).reshape(-1))
            lru = bool(0 < len(u) <= sc._HOT_PATH_MAX_ROWS and sc._hot_cap_rows)
            ph = "commit" if ref2._IN_COMMIT[0] else _S["phase"]
            _S["calls"].append((PHC[ph], len(u), sc.hot_misses - m0, dt, lru, _S["cur_w"]))
            if RX >= 2 and lru:
                _S["rows"].append(u.astype(np.int32)); _S["meta"].append((PHC[ph], _S["run"]))
            return out
        sc._rows_matrices = rec
        if RX >= 2:
            atexit.register(_dump_rows)
    print(f"[ngx] installed: receipt {RX}, prefetch {_PFE}; PLE layer {idx}, hot cap {sc._hot_cap_rows} rows x "
          f"{sc._hot_row_bytes} B", flush=True)
    return True


def phase(p):
    if ON:
        _S["phase"] = p


def run_begin():
    if not ON:
        return
    sc, Q = _S["sc"], _S["Q"]
    _S["run"] += 1
    _S["b"] = dict(stage_s=Q.ple_stage_seconds(), stage_n=Q.ple_stage_calls(), hits=sc.hot_hits, misses=sc.hot_misses,
                   pread=getattr(sc, "pread_gathers", 0), vec=getattr(sc, "vectorized_gathers", 0),
                   pfb=getattr(sc, "prefetch_batches", 0), calls=len(_S["calls"]), pfc=_S["pf_checks"],
                   pfw=_S["pf_warmed"], pfr=_S["pf_rows"])


def warm(tok, dr, cache, new):
    """(b'): new=True at the drafts' readback (a new check), new=False just before the verify (rows changed since)"""
    if not PF:
        return 0
    emb, sc = _S["emb"], _S["sc"]
    if new:
        _S["chk"] += 1; _S["pf_checks"] += 1; _S["sub"] = set()
        _S["cur_w"] = _PFE == "1" or _S["chk"] % 2 == 1
        _S["pf_warmed"] += int(_S["cur_w"])
    if not _S["cur_w"]:
        return 0
    ent = cache[_S["idx"]]
    h = ent[_S["ngi"]] if ent is not None else None
    prev = np.asarray(h, dtype=np.int64) if h is not None else np.full((1, emb.context_len), emb.eos_id, np.int64)
    rows, _ = emb._rows_np(np.asarray([[int(tok)] + [int(x) for x in dr]], dtype=np.int64), prev)
    hot, sub = sc._hot, _S["sub"]
    miss = [r for r in np.unique(rows.reshape(-1)).tolist() if r not in hot and r not in sub]
    if miss:
        sc.submit_warm(np.asarray(miss, dtype=np.int64))
        sub.update(miss); _S["pf_rows"] += len(miss)
    return len(miss)


def _ms(x):
    x = np.asarray(x, dtype=np.float64) * 1e3
    return dict(n=int(len(x)), sum=round(float(x.sum()), 3), mean=round(float(x.mean()), 4),
                p50=round(float(np.median(x)), 4), p90=round(float(np.percentile(x, 90)), 4), max=round(float(x.max()), 4))


def run_end():
    """the receipt for runs.json (None when off)"""
    if not ON:
        return None
    sc, Q, b = _S["sc"], _S["Q"], _S["b"]
    r = dict(ple_stage_s=round(Q.ple_stage_seconds() - b["stage_s"], 5), ple_stage_calls=Q.ple_stage_calls() - b["stage_n"],
             hot_hits=sc.hot_hits - b["hits"], hot_misses=sc.hot_misses - b["misses"],
             pread_gathers=getattr(sc, "pread_gathers", 0) - b["pread"],
             vectorized_gathers=getattr(sc, "vectorized_gathers", 0) - b["vec"],
             prefetch_batches=getattr(sc, "prefetch_batches", 0) - b["pfb"],
             prefetch=_PFE, pf_checks=_S["pf_checks"] - b["pfc"], pf_warmed=_S["pf_warmed"] - b["pfw"],
             pf_rows=_S["pf_rows"] - b["pfr"])
    if RX:
        cs = _S["calls"][b["calls"]:]
        ph = {}
        for nm, code in PHC.items():
            sel = [c for c in cs if c[0] == code]
            if sel:
                ph[nm] = dict(calls=len(sel), uniq=int(sum(c[1] for c in sel)), miss=int(sum(c[2] for c in sel)),
                              lru=int(sum(c[4] for c in sel)), ms=_ms([c[3] for c in sel]))
        r["phase"] = ph
        v = [c for c in cs if c[0] == PHC["verify"]]
        if PF and v:                                     # the verify gathers of warmed vs unwarmed checks (alt: paired)
            r["verify_warmed"] = dict(_ms([c[3] for c in v if c[5]]), miss=int(sum(c[2] for c in v if c[5]))) \
                if any(c[5] for c in v) else None
            r["verify_unwarmed"] = dict(_ms([c[3] for c in v if not c[5]]), miss=int(sum(c[2] for c in v if not c[5]))) \
                if any(not c[5] for c in v) else None
    return r


def _dump_rows():
    if not _S["rows"]:
        return
    sc = _S["sc"]
    p = os.path.join(_S["outdir"], f"{_S['tag']}_ngram_rows.npz")
    np.savez_compressed(p, rows=np.concatenate(_S["rows"]), offs=np.cumsum([0] + [len(x) for x in _S["rows"]]),
                        phase=np.array([m[0] for m in _S["meta"]], np.int8), run=np.array([m[1] for m in _S["meta"]], np.int32),
                        phases=np.array(list(PHC)), hot_cap_rows=int(sc._hot_cap_rows), hot_row_bytes=int(sc._hot_row_bytes))
    print(f"[ngx] wrote {p}: {len(_S['rows'])} LRU-path calls", flush=True)


def selftest():
    """CPU only, no model: a fake sidecar + embedding through install / run_begin / warm / run_end"""
    from collections import OrderedDict
    import types
    global RX, PF, ON, _PFE
    RX, PF, ON, _PFE = 2, True, True, "alt"

    class SC:
        _HOT_PATH_MAX_ROWS = 4096

        def __init__(self):
            self._hot, self._hot_cap_rows, self._hot_row_bytes, self._pool = OrderedDict(), 100, 300, object()
            self.hot_hits = self.hot_misses = self.prefetch_batches = 0
            self.warmed = []

        def _rows_matrices(self, flat, names):
            u = np.unique(np.asarray(flat).reshape(-1)); miss = [int(r) for r in u if int(r) not in self._hot]
            for r in miss:
                self._hot[r] = (r,)
            self.hot_hits += len(u) - len(miss); self.hot_misses += len(miss)
            return {"weight": np.asarray(flat)}

        def submit_warm(self, rows):
            self.warmed.append(list(rows)); self.prefetch_batches += 1

    sc = SC()
    emb = types.SimpleNamespace(context_len=2, eos_id=0, ngram_embedding=types.SimpleNamespace(_sidecar=sc),
                                _rows_np=lambda ids, prev: (np.concatenate([prev, ids], 1)[:, 2:] * 10 + 1, None))
    TXT = types.SimpleNamespace(_ple_stage_idx=1, layers=[None, types.SimpleNamespace(ple=types.SimpleNamespace(
        ple_embedding=emb, NGRAM_IDX=3))])
    ref2 = types.SimpleNamespace(_IN_COMMIT=[False])
    import mtplx.models.qwen4_exp as Q
    s0, n0 = Q.ple_stage_seconds(), Q.ple_stage_calls()
    import tempfile
    td = tempfile.mkdtemp()
    install(TXT, ref2, "selftest", outdir=td)
    atexit.unregister(_dump_rows)
    run_begin()
    cache = [None, [None, None, None, np.array([[5, 6]])]]
    phase("verify")
    assert warm(7, [8, 9], cache, True) == 3 and sc.warmed[-1] == [71, 81, 91]      # check 1 (odd): warmed
    assert warm(7, [8, 9, 4], cache, False) == 1 and sc.warmed[-1] == [41]          # the changed row only
    sc._rows_matrices(np.array([71, 81, 91, 41]), ("weight",))
    assert warm(3, [2], cache, True) == 0 and len(sc.warmed) == 2                    # check 2 (even): not warmed in alt
    sc._rows_matrices(np.array([31, 21]), ("weight",))
    ref2._IN_COMMIT[0] = True; sc._rows_matrices(np.array([31]), ("weight",)); ref2._IN_COMMIT[0] = False
    phase("other")
    r = run_end()
    assert r["pf_checks"] == 2 and r["pf_warmed"] == 1 and r["pf_rows"] == 4 and r["hot_misses"] == 6, r
    assert r["phase"]["verify"]["calls"] == 2 and r["phase"]["commit"]["miss"] == 0, r
    assert r["verify_warmed"]["n"] == 1 and r["verify_unwarmed"]["n"] == 1, r
    assert Q.ple_stage_seconds() == s0 and Q.ple_stage_calls() == n0
    _dump_rows()
    f = os.path.join(td, "selftest_ngram_rows.npz")
    z = np.load(f)
    assert list(z["offs"]) == [0, 4, 6, 7] and list(z["phase"]) == [1, 1, 2], (z["offs"], z["phase"])
    os.remove(f); os.rmdir(td)
    print("SELFTEST OK: install / run_begin / warm (alt: odd checks; only new, cold rows) / phases / receipt / row dump")


if __name__ == "__main__":
    selftest()
