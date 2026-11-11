# NEVER-WAIT in ref2's one-call: design (Leap, Fri 2026-11-06 09:3x)

Spec: ~/FableGod/far/NEVERWAIT_TEST.md section 6. Interface: ~/BigMoE/docs/NEVERWAIT_IFACE.md (BigMoE 6343ff1).
Switches: REF2_NEVERWAIT=1 (CFG["neverwait"]), REF2_NW_FORCE=<spec> (test only). Off = today's code, bitwise.

## Mechanism (no kernel change)
Per one-call (layer index k = NW["k"], counted per check; NW_RESET() zeroes it):
1. After the posts: cold = (pre_m == -1); late = cold & demand (not QOM's bit-21 optional use) | FORCE.
   - k == 47: rank 0 is never late (the kernel waits as today); other late pairs are DROPPED (weight 0, counted).
2. Layer k's own kernel runs with pre_m -> 0 for late pairs (any valid slot; weight 0), pre_l -> -2 for late ranks < R4
   (U3's no-LSB path), sc -> 0 for late pairs. No spin for them.
3. Saved for k + 1 (NW["pend"]): x (xin), ids_k, pre_m_late = where(late, pre_m, 0), pre_l_late = where(late, pre_l,
   -2), sc_late = where(late, sc, 0), t's pools (base, sideP), offs, the kernel form.
4. At layer k + 1's one-call: the late call = the same one-call kernel on layer k's pool with the saved arrays. Late pairs
   with pre -1 resolve through layer k's release region (still intact: regions cycle over 64). It waits only if a read
   has not landed after a whole layer (the first build caps lateness at one layer). Output delta_k [M, D].
5. ref2's _hc_write: the first write after a one-call is that layer's MLP-side write. It captures inj[k] (the inject the
   MoE output is written with) and, after the normal write, writes delta_{k-1} through inj[k-1]: y = _hc_write_c(y,
   delta, inj_prev). One extra write per layer.
6. Nothing survives the check: after k == 47, pend / due must be empty (asserted at NW_RESET and k == 47).
- The one-call must serve every MoE layer while NW is on: a decline raises.
- Warm (nothing cold): late is empty, the own call is unchanged, the late call is all weight 0 -> delta == 0 -> the extra
  write adds +0 (bitwise same).
- Cost when nothing is late: one extra one-call (slot 0 at weight 0 for every pair) + one write per layer.

## Bit checks
1. off == today (self_test + structure). 2. on + warm == off bitwise. 3. FORCE == an MLX reference built from ordinary
calls (layer k without the forced groups; their MLP on k's input; written through k's inject after k + 1's write).
4. BigMoE's 2-prompt pre-flight.
