"""R57_D17 (MTP Head, Wed 11-04; BigMoE's ask for QOM's last-row gate; DRAFT17 RESULT + D17-1 in REPORT): the drafter proposes
one more draft so the check's last row has a draft (and its cf) to gate, at ~0 ms with Leap's FORM_ROWS 16..64 (05fafe7).
The verify itself is unchanged: the extra draft is never verified as a row; it only informs the last row's gate."""


def split17(path, cf, R=16):
    """a 17-position pass (R drafts) -> today's 15-draft block (path, cf) and the extra draft (token, cf) for row R - 1"""
    return list(path[:R - 1]), list(cf[:R - 1]), int(path[R - 1]), float(cf[R - 1])


def last_row_draft(path, cf, c, x16=None, c16=None):
    """the draft (token, cf) for a check's last row after the cut kept c drafts: the own draft past the cut (c < 15, it
    exists today) or the extra draft (c == 15, only with R57_D17)"""
    if c < len(path):
        return int(path[c]), float(cf[c])
    return (x16, c16) if x16 is not None else (None, None)


PATCH17 = """r57_screen.py (BigMoE's file) behind R57_D17=1, DFlash2 path, R_ == 16 only (the copy's 32-row path unchanged):
0. Needs Leap's draft_fused with FORM_ROWS 16..64 (05fafe7): with an older one, 17 rows take the per-call weight-copy fallback
   (+8.6 ms, DRAFT17). D17-1: 17-position pass 9.67 vs 16-position 9.76 ms (no extra cost).
1. The draft: pth, cf = propose(mx.array([[tok] + [MASK] * R_]), ctx[None].astype(mx.bfloat16), dcache)  (17 positions,
   16 drafts) instead of [MASK] * (R_ - 1).
2. Right after: dr, cfl, x16, c16 = split17(np.array(pth[0]).tolist(), np.array(cf[0]).tolist())  (today's 15-draft block
   + the extra draft), so the copy override, R57_FAST's gates, the t cut and the verify see a 15-draft block as today.
3. QOM's last row: tok_last, cf_last = last_row_draft(full 15-draft path, its cf, len(dr), x16, c16): the draft past the cut
   for a cut segment (it exists today), the extra draft for a full block (the 32 % of code segments that lack one). Feed
   cf_last to the row gate for row len(dr) instead of keeping it exact (fk_[len(dr)] = True / sq_[len(dr)] = False today).
4. The verify input stays [tok] + dr; the drafter cache handling is unchanged (block rows are never cached).
5. The drafts change (unmasked 17th position, bidirectional block attention): D17-1 on 689 code checks: the same 15 tokens on
   83.3 % of checks (96.5 % per draft), cf never bit-equal, the t 0.3 cut at the same depth on 89.3 %. Exact tokens either way
   (the check decides); kept under the 17-position drafts is not measured. If drafts 1-15 must stay as today, mask the extra
   position from rows 0-15 in draft_fused's att (block keys: kk < ctx_len + 16 for query rows < 16): untested.
6. The extra draft = the 125B's token on 79.8 % of full blocks (all 15 checked drafts right; 163 of 689 checks)."""
