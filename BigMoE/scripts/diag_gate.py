"""Which data does a mismatched gather actually see? For each mismatch, compare the GPU's
slot ids (gate output) with the loader's table, and the GPU checksum with every expert's."""
import os, sys
sys.path.insert(0, os.path.expanduser("~/BigMoE"))
exec(open(os.path.expanduser("~/BigMoE/scripts/test_gate.py")).read().split("lines = []")[0])
allck = {L: {e: cpu_ck(L, e) for e in range(E)} for L in range(NL)}
inv = {L: {} for L in range(NL)}
for L in range(NL):
    for e, c in allck[L].items(): inv[L].setdefault(c, []).append(e)
def token_dbg(ids):
    h = x0; out = []
    gate.new_pass()
    for L in range(NL):
        slots = gate.route_and_gate(L, mx.array(ids[L], mx.int32))
        g = mx.take(pools[L], slots, axis=0); ck = (g[:, ::4099].astype(mx.uint64)).sum(axis=1)
        out.append((slots + 0, ck)); h = h + (ck.sum() % 7).astype(h.dtype) * 0.0
        y = mx.quantized_matmul(h, Wq, Ws, Wb, transpose=True, group_size=64, bits=4); h = h + y[:, :D] * 1e-3
    mx.eval(h, *[a for p in out for a in p]); return out
dbg = mx.zeros((4 << 20,), mx.int32); mx.eval(dbg); ext.set_debug(dbg); DBG = np.frombuffer(memoryview(dbg), np.int32)
kinds = {"loader saw wrong ids": 0, "ok": 0, "slot ids stale/wrong": 0, "pool bytes stale (other expert)": 0, "garbage": 0}
for rep in range(int(os.environ.get("REPS", 8))):
    ids = [rng.choice(E, K, replace=False).astype(np.int32) for _ in range(NL)]
    out = token_dbg(ids)
    tables = [ext.ensure(L, mx.array(ids[L])) for L in range(NL)]     # resident now -> pure lookup
    for L in range(NL):
        gs = np.array(out[L][0]).tolist(); gck = np.array(out[L][1]).tolist(); ts = np.array(tables[L]).tolist()
        for j, e in enumerate(ids[L].tolist()):
            off = ext.ring_off(L, gate.seq); seen = DBG[off:off + K].tolist()
            if gck[j] != allck[L][e] and seen != ids[L].tolist(): kinds["loader saw wrong ids"] += 1
            elif gck[j] == allck[L][e]: kinds["ok"] += 1
            elif gs[j] != ts[j]: kinds["slot ids stale/wrong"] += 1
            elif gck[j] in inv[L]: kinds["pool bytes stale (other expert)"] += 1
            else: kinds["garbage"] += 1
gate.check(); gate.stop()
print(kinds)
open(os.path.expanduser("~/BigMoE/data/diag_gate.txt"), "w").write(str(kinds) + "\n")
