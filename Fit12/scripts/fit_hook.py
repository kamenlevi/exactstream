"""Fit12 hook for the Eval suite (2026-10-23): B12LL + a per-pass expert budget, as one load call.
B12LL = bt_b12.load_b12(0.10, mtp=True) with B12_LATE3=2,10 B12_NORM=mass (the config that passed 5/5); under MTP 3:3:0 every
verify row uses its exact top-10 (all read and waited), i.e. v3 natural routing at 10 % residency. The budget is BigMoE's
FM1 router hook (copy: fit_budget.py): in 2..8-row calls each row takes its top-10 within the pass set of the U experts with
the largest summed w^2 (NON-EXACT; a row's routing depends on its pass-mates). One-row calls and the prompt prefill are
untouched. Decode exactly as the suite's _gen_mtp (b12_prefill_state 512/4096, spec125 3:3:0, lookup 0, greedy).

Env the suite sets for its b12ll rows, plus: BIGMOE_PF_GROUP=64 (MTP-layer prefill scratch; same greedy sha per Breakthrough,
-1.4 GB) and BIGMOE_ASYNC_EVERY=1 (same shas as without in Fit12 F1, +4 %). Both are speed / memory switches only.

API: load_b12ll_budget(budget="b16", ugb=0.0, pr1=False) -> (rt, st). pr1=True (2026-10-24, U2): MTPLX's
commit_verified_window replays the PLE layer on 2 rows for keep-1 commits (fit_pr1.py), so B12LL's tokens no longer depend
on the draft policy; exact math otherwise (U2 = load_b12ll_budget('', pr1=True)). topk=k / topp=p (L1-17, NON-EXACT): each verify row keeps
its top-k experts / the smallest prefix holding >= p of its top-10 weight, renormalized (fit_topk.py); topk=10 or
topp=1.0 = B12LL bitwise (identity check). budget in {"b16", "b18", "b20", ...}; "" = plain B12LL.
ugb > 0 = uncertainty-gated budget (NewIdeas' UGB): the budget applies to a verify pass only when the MTP head's lowest
draft-token probability in that pass is >= ugb; other passes use B12LL's natural routing. Installed by patching spec125's
_MTPHead in this process (want_probs on; draft_finish sets fit_budget.GATE), so the suite's _gen_mtp needs no change."""
import os, sys


def _install_gate(theta):
    import spec125
    import fit_budget
    init0, fin0 = spec125._MTPHead.__init__, spec125._MTPHead.draft_finish

    def init(self, m):
        init0(self, m)
        self.want_probs = True

    def fin(self, *a, **kw):
        out = fin0(self, *a, **kw)
        on = bool(self.last_probs) and min(self.last_probs) >= theta
        fit_budget.GATE.update(on=on, n=fit_budget.GATE["n"] + 1, on_n=fit_budget.GATE["on_n"] + int(on))
        return out
    spec125._MTPHead.__init__, spec125._MTPHead.draft_finish = init, fin


def load_b12ll_budget(budget="b16", ugb=0.0, pr1=False, topk=0, topp=0.0):
    for k, v in (("B12_LATE3", "2,10"), ("B12_NORM", "mass")):
        assert os.environ.get(k) == v, f"b12ll needs {k}={v}"
    sys.path.insert(0, os.path.expanduser("~/Breakthrough/scripts"))
    sys.path.insert(1, os.path.expanduser("~/Fit12/scripts"))
    import bt_b12
    if topk or topp:                             # L1-17 (Fit12 TK1, NON-EXACT): fewer experts per verify row, before load
        import fit_topk
        assert fit_topk.apply(bt_b12, topk, topp), "fit_topk not installed"
        print(f"fit12 hook: verify rows keep top-k {topk} / top-p {topp} of the top-10 weight (renormalized; prefill,"
              f" one-row calls and the MTP head keep top-10)", flush=True)
    rt, st = bt_b12.load_b12(0.10, mtp=True)
    if pr1:
        from mtplx.models import qwen4_exp as Q
        import fit_pr1
        assert fit_pr1.install(Q), "fit_pr1 not installed"
        print("fit12 hook: PR1 on (keep-1 commits replay the PLE layer on 2 rows: draft-policy-invariant B12LL = U2)", flush=True)
    if budget:
        import fit_budget
        n = fit_budget.install(rt.model, budget)
        assert n == 48, f"budget wrapped {n} routers, expected the 48 trunk layers"
        print(f"fit12 hook: B12LL + budget {budget} on {n} routers", flush=True)
        if ugb:
            sys.path.insert(2, os.path.expanduser("~/SpecDecode"))
            _install_gate(float(ugb))
            print(f"fit12 hook: uncertainty gate on: budget only when the min MTP draft prob >= {ugb}", flush=True)
    return rt, st
