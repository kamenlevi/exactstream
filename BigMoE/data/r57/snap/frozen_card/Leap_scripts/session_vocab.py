"""SESSION-VOCAB (Leap, Wed 2026-11-04 17:3x; the orchestrator's route (b): make tomorrow's SVH + BP-9 wiring in BigMoE's
r57_screen a 10-minute job). ONE object feeds the session's token ids to both screened heads:
  - SVH (ref2 CFG screened_head, FAST): the TRUNK's lm_head at <= 16 rows (REF2 on = the verify pass) over FR-Spec's top-98,304
    + the session's ids, scattered into a -inf full-vocabulary array (TABLE_350 SVH: -0.79 ms per check);
  - BP-9 (d7b_nax.DraftSubset, exact-safe: drafts are verified): the DRAFTER's head over the same kind of subset on qmm16c
    (TABLE_350 BP-9: -0.99 ms per propose, kept code x1.0000 / math x0.9995).
r57_screen hook points (four), shown in ~/Leap/data/r57_session_vocab.diff:
  1. after the model + drafter load:   SV = session_vocab.SessionVocab(model.language_model.lm_head, svh=..., dsub=...)
  2. run(), BEFORE prefill(ids):        SV.new_prompt(ids)      (a short last prefill chunk takes the SVH path too)
  3. run(), after `out += new`:         SV.committed(new)       (and SV.committed([tok]) for the prefill's first token)
  4. head2s(hid):                       if SV.dsub: return SV.head2s(draft, hid)
check=True (smokes): every screened draft-head call also runs the full head and counts the rows whose full top-1 is outside the
subset (the would-be deviations); SV.report() returns the counters (+ ref2's screened_head count)."""
import os, sys
import numpy as np
import mlx.core as mx
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_card/Leap_scripts"))
import d7b_nax


class SessionVocab:
    def __init__(self, lm_head, svh=False, dsub=False, check=False, base_n=98304, cap=4096):
        self.svh, self.check = bool(svh), bool(check)
        self.ref2 = None
        if self.svh:
            import ref2
            self.ref2 = ref2
            assert "screened_head" in ref2.CFG, "Leap's ref2 has no SVH (needs ~/Leap f5404cc)"
            ref2.screened_head_install(lm_head, base_n=base_n, cap=cap)
            ref2.CFG["screened_head"] = True
        self.sub = d7b_nax.DraftSubset(lm_head, base_n=base_n, cap=cap) if dsub else None
        self.dsub = self.sub is not None
        self.n = dict(prompts=0, committed=0, head2s=0, chk_rows=0, chk_out=0)

    def new_prompt(self, prompt_ids):
        """a new request: drop the last session's ids, then add this prompt's"""
        ids = np.asarray(prompt_ids).reshape(-1)
        if self.svh:
            self.ref2.screened_head_reset()
            self.ref2.screened_head_add(ids)
        if self.dsub:
            self.sub.reset()
            self.sub.add(ids)
        self.n["prompts"] += 1

    def committed(self, ids):
        """each check's committed ids (kept drafts + the target's token); cheap: only ids not seen yet are gathered"""
        ids = np.asarray(ids).reshape(-1)
        if ids.size == 0:
            return
        if self.svh:
            self.ref2.screened_head_add(ids)
        if self.dsub:
            self.sub.add(ids)
        self.n["committed"] += int(ids.size)

    def head2s(self, draft, hid, K=16):
        """r57_screen's head2s over the drafter's subset (two-stage top-K, candidates mapped back to token ids)"""
        self.n["head2s"] += 1
        cand, unary = d7b_nax.screened_head2s(draft, hid, self.sub, K)
        if self.check:                                   # smoke mode: the full head's top-1 outside the subset (host sync)
            full = mx.argmax(draft.compute_logits(hid).reshape(-1, draft.config.vocab_size), axis=-1)
            f = np.array(full).reshape(-1).tolist()
            self.n["chk_rows"] += len(f)
            self.n["chk_out"] += sum(1 for t in f if t not in self.sub.set)
        return cand, unary

    def report(self):
        r = dict(self.n)
        if self.svh:
            e = next(iter(self.ref2._SH.values()))
            r["svh_extra"] = len(e["extra"])
            r["svh_calls"] = int(self.ref2.COUNTS.get("screened_head", 0))
        if self.dsub:
            r["dsub_extra"] = len(self.sub.extra)
        return r
