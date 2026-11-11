"""Python side of the sync-free exact gate (see ext/bigmoe_ext.cpp).

route_and_gate(L, seq, ids) -> slot ids (lazy mx.array), usable as gather indices.
The GPU writes ids into the shared routing row of layer L (a tiny kernel), commits, and
waits on layer L's event at value seq, which the native loader signals once those experts
are resident and their slot ids are written."""
import atexit, os, sys, time
_PAUSE = os.path.expanduser("~/.local/share/quiet/PAUSE")   # quiet windows: stop at a pass boundary (RULES.md, Sat 04:00)
import mlx.core as mx
import mlx.core as _mx
_BUILD = "build" if _mx.__version__ == "0.32.2" else "build" + _mx.__version__.replace(".", "")[1:]   # 0.32.4 -> build324
_BUILD = os.environ.get("BIGMOE_EXT_BUILD", _BUILD)   # e.g. build_next: a newer ext while another process holds ext/build
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ext", _BUILD))
import bigmoe_ext as ext

MAXK = int(os.environ.get("BIGMOE_MAXK", "40"))   # ids per sync-free gate row: 4 verify tokens x top-10; the row
                                                 # layout holds up to 61 (S=6 x top-10 = 60)
PRED_LINES = max(0, min(4, int(os.environ.get("BIGMOE_PRED_LINES", "1"))))   # prediction lines after line 3 (MoonShots)
assert MAXK <= 92, "BIGMOE_MAXK: the routing row carries at most 92 ids"
JOINT = os.environ.get("BIGMOE_JOINT", "0") not in ("", "0")   # [Fit12 B1] post a layer's rows before its first wait
XLINES = 1 if MAXK > 61 else 0   # one more id line (ids 61..91) at the end of the row when MAXK > 61
ROW = 96 + 32 * PRED_LINES + 32 * XLINES   # 4 x 128-byte lines: [seq, n, ids0..29] [seq, ids30..60] [seq, np, preds0..29] [seq, preds30..60] (+ more);
                   # each line carries its own seq, written last, so a line seen with the new seq is complete
_src = r"""
uint t = thread_position_in_grid.x;
if (t == 0) {
  device atomic_int* row = reinterpret_cast<device atomic_int*>(const_cast<device int*>(route) + meta[0] * ROWLEN);
  int n = meta[2];
  atomic_store_explicit(&row[1], n, memory_order_relaxed);
  for (int i = 0; i < n; i++) {
    int pos = i < 30 ? 2 + i : 33 + (i - 30);
    atomic_store_explicit(&row[pos], (int)ids[i], memory_order_relaxed);
  }
  atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
  atomic_store_explicit(&row[32], meta[1], memory_order_relaxed);
  atomic_store_explicit(&row[0], meta[1], memory_order_relaxed);
  out[0] = meta[1];
  if (DBGORDER) atomic_fetch_add_explicit((device atomic_int*)(const_cast<device uchar*>(bump)), 1, memory_order_relaxed);
}
"""
_DBG = int(os.environ.get("BIGMOE_DBGORDER", "0"))
_copy = mx.fast.metal_kernel(name="bigmoe_route_copy", input_names=["ids", "route", "meta", "bump"], output_names=["out"],
                             source=_src, header=f"#define ROWLEN {ROW}\n#define DBGORDER {_DBG}\n")


PAIRS = int(os.environ.get("BIGMOE_SPIN_PAIRS", "128"))   # spin mode: fresh line pairs per (layer, pass) region = max polls before timeout
_spin_src = r"""
uint t = thread_position_in_grid.x;
if (t == 0) {
  int seq = meta[1], n = meta[2];
  // fast path: post_ids already found every expert in the residency table published a pass ahead
  if (dep[0] == 1) {
    for (int i = 0; i < n; i++) out[i] = dep[1 + i];
    dep_out[0] = -2;
    atomic_fetch_add_explicit((device atomic_int*)const_cast<device int*>(status) + 16, 1, memory_order_relaxed);
    return;
  }
  if (DBGORDER) { int c = atomic_load_explicit((device atomic_int*)(const_cast<device uchar*>(bump)), memory_order_relaxed);
                  atomic_store_explicit((device atomic_int*)const_cast<device int*>(status) + 1 + (seq & 7), c, memory_order_relaxed); }
  const device int* reg = region;                                          // this (pass, layer)'s slice
  float x = 1.0f; float d = 32.0f; int got = -1;
  for (int p = 0; p < PAIRS_; p++) {
    const volatile device int* l0 = (const volatile device int*)(reg + p * 64);
    if (l0[0] == seq && l0[32] == seq) { got = p; break; }
    for (int j = 0; j < (int)d; j++) x = x * 1.0000001f + 1e-7f;    // back-off between polls
    d = min(d * 1.1f, 400000.0f);
  }
  if (got < 0) {
    atomic_fetch_add_explicit((device atomic_int*)const_cast<device int*>(status), 1, memory_order_relaxed);
    for (int i = 0; i < n; i++) out[i] = 0;
  } else {
    const volatile device int* l0 = (const volatile device int*)(reg + got * 64);
    for (int i = 0; i < n; i++) out[i] = i < 31 ? l0[1 + i] : l0[32 + 1 + (i - 31)];
  }
  if (x == 12345.0f) out[0] = 0;      // keep the back-off loop
  dep_out[0] = got;
}
"""
TLINES = 17        # residency table: 17 lines x 31 entries >= 512 experts
_spin = mx.fast.metal_kernel(name="bigmoe_spin_gate", input_names=["dep", "region", "meta", "status", "bump"],
                             output_names=["out", "dep_out"], source=_spin_src,
                             header=f"#define PAIRS_ {PAIRS}\n#define DBGORDER {_DBG}\n#define TLINES {TLINES}\n")
FASTPATH = os.environ.get("BIGMOE_FASTPATH", "1") == "1"
MODE = os.environ.get("BIGMOE_GATE", "spin")
assert PRED_LINES == 0 or MODE == "spin", "BIGMOE_PRED_LINES needs the spin gate (post_ids writes the extra lines)"


class Gate:
    def __init__(self, nl):
        self.nl = nl
        self.route = mx.zeros((max(4096, nl * ROW),), mx.int32)        # > 4 KB: bound as device memory
        self.ring = mx.zeros((4 << 20,), mx.int32)                     # 16 MB slot-id ring
        # 64 MB dummy input of the id-copy kernel: pushes the command buffer past MLX's per-buffer
        # MB limit, so MLX commits it right after the copy (the GPU must run it before the wait)
        self.bump = mx.zeros((64 << 20,), mx.uint8) if MODE == "event" else mx.zeros((4096,), mx.uint8)
        self.nreg = 128     # (layer, pass) regions in the ring (4 MB: small MB count per buffer; reused after
                            # ~3 tokens, long after any GPU cache could still hold a line)
        self.region = mx.zeros((self.nreg * PAIRS * (96 if XLINES else 64),), mx.int32)        # 32 MB
        self.status = mx.zeros((4096,), mx.int32)
        self.nopred = mx.zeros((1,), mx.int32)
        self.ntab = 1024
        self.tables = mx.zeros((self.ntab * TLINES * 32,), mx.int32)            # 2.2 MB
        mx.eval(self.route, self.bump, self.ring, self.region, self.status, self.tables)
        ext.start(self.route, self.ring, nl, MAXK)
        if MODE == "spin":
            ext.set_regions(self.region, PAIRS)
            if int(os.environ.get("BIGMOE_DEFER", "0")) > 0:   # [Breakthrough v3f] phase-B regions (late experts)
                self.region_b = mx.zeros_like(self.region); mx.eval(self.region_b); ext.set_regions_b(self.region_b)
            ext.set_status(self.status)
            if FASTPATH:
                ext.set_tables(self.tables, TLINES)
        atexit.register(ext.stop)
        self.seq = 0
        self._pend = []                                  # [Fit12 B1] rows posted since the last phase-0 wait

    def new_pass(self):
        while os.path.exists(_PAUSE):
            time.sleep(0.2)
        self.seq += 1
        return self.seq

    def route_and_gate(self, L, ids, seq=None, preds=None, counts=None):
        seq = self.seq if seq is None else seq
        flat = ids.reshape(-1).astype(mx.int32)
        n = flat.size
        assert n <= MAXK, n
        if MODE == "spin":
            pr = preds.reshape(-1).astype(mx.int32) if preds is not None else self.nopred
            c = tuple(counts) if counts else ((pr.size if preds is not None else 0),)
            n1, n2, n3, rk = (c + (0, 0, 0, 0))[:4]
            npr = n1 | (n2 << 8) | (n3 << 16) | (rk << 24)   # layers L+1, L+2, L+3; rk: ranked unique lists
            dep = ext.post_ids(flat, pr, L, seq, npr)  # GPU writes the routing row (+ next-layer predictions)
            if os.environ.get("BIGMOE_SKIPSPIN"):       # timing only: wrong slots, no wait
                return (flat + dep[0:1] * 0).reshape(ids.shape)
        else:
            meta = mx.array([L, seq, n], mx.int32)
            dep = _copy(inputs=[flat, self.route, meta, self.bump], grid=(32, 1, 1), threadgroup=(32, 1, 1),
                        output_shapes=[(1,)], output_dtypes=[mx.int32])[0]
        if MODE == "spin":
            out = ext.spin_wait(dep, L, seq, n)          # native: no per-layer Python kernel call
            return out.reshape(ids.shape)
        g = ext.gate(self.ring, dep, L, seq, n)
        return g.reshape(ids.shape)

    def post(self, L, ids, seq=None, preds=None, counts=None, wk=0):
        """OoO first half: write the routing row; returns (dep = [ok, slot or -1 per id], n). Lazy."""
        assert MODE == "spin"
        seq = self.seq if seq is None else seq
        flat = ids.reshape(-1).astype(mx.int32)
        n = flat.size
        assert n <= MAXK, n
        if JOINT and self._pend:                         # [Fit12 B1 v2] posted after the layer's earlier row (base first)
            flat = mx.depends([flat], [self._pend[-1]])[0]
        pr = preds.reshape(-1).astype(mx.int32) if preds is not None else self.nopred
        c = tuple(counts) if counts else ((pr.size if preds is not None else 0),)
        n1, n2, n3, rk = (c + (0, 0, 0, 0))[:4]
        npr = n1 | (n2 << 8) | (n3 << 16) | (rk << 24)
        dep = ext.post_ids(flat, pr, L, seq, npr, wk)   # [v3f] wk = nw | nk << 8 (nw 255: GPU flags on the ids)
        if JOINT:
            self._pend.append(dep)
        return dep, n

    def spin(self, dep, L, n, seq=None, phase=0):
        """OoO second half: the slot ids once the loader released (L, seq). Lazy. phase 1: the late (phase-B) pairs."""
        if JOINT and phase == 0 and self._pend:         # [Fit12 B1] the rows posted with this one go out before its wait
            others = [d for d in self._pend if d is not dep]
            self._pend = []
            if others:
                dep = mx.depends([dep], others)[0]
        return ext.spin_wait(dep, L, self.seq if seq is None else seq, n, phase)

    def fast_hits(self):
        import numpy as np
        return int(np.frombuffer(memoryview(self.status), np.int32)[16])

    def check(self):
        """Raise if any spin gate timed out (its pass is invalid). Cheap: one int."""
        import numpy as np
        bad = int(np.frombuffer(memoryview(self.status), np.int32)[0])
        if bad:
            raise RuntimeError(f"bigmoe: {bad} spin gate(s) timed out -- results of that pass are invalid")

    def stop(self):
        ext.stop()
