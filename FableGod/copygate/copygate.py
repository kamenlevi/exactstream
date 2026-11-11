"""copygate.py (FableGod copygate, GATE.md; Mon 2 Nov 2026): the gate for the session match override of
~/TrainLab/sticky/d02b_walk_match.py. Pure Python, CPU, no numpy; one object per session (window).

CopyGate(ids, g, mode): ids = the window's token ids (index 0 = the first stored token), g = the index where generation
starts (gen - off): a copy whose source token sits at an index <= g comes from the prompt. Modes:
  'all'       the MATCH override as walked (every match >= 8 replaces the draft): must reproduce match16 / match32
  'prompt'    G0: only copies whose source token lies in the prompt, not in the answer being written
  'pduel'     G1: G0 + the session duel: at checked positions where the copy and the drafter disagreed, the copy's win
              share with a Beta(3, 1) prior must be >= 0.6
  'copygate'  G2: G1 + a short line-boundary copy (ext < 24, and the copy's token or the source token before it holds a
              newline) needs its own duel share, Beta(1, 1) prior, >= 0.6
propose(dr, r) -> (out, used, info): dr = the drafter's picks for indices r+1.., r = the anchor index. The table holds
  every k-gram (k 8-16) whose next token index is <= r (Matcher.commit); at each position the longest k >= 8 suffix of
  (window + this block's chosen tokens so far) found in it proposes the token after its most recent occurrence;
  info[p] = (copy token, k, source index, ext = the copy's aligned length so far, boundary) or None.
learn(dr, info, true, k): after the check (k drafts accepted), with dr = the drafter's own picks: positions p <= k label
  the duel (both tokens are known on the host; the truth is committed)."""
import json
import os

_NL = None


def nl_ids(path=os.path.expanduser("~/BigMoE/models/bare/vocab.json")):
    """token ids whose text holds a newline (byte-level BPE: 'Ċ')"""
    global _NL
    if _NL is None:
        _NL = {i for t, i in json.load(open(path)).items() if "Ċ" in t}
    return _NL


class CopyGate:
    def __init__(self, ids, g, mode="copygate"):
        assert mode in ("all", "prompt", "pduel", "copygate"), mode
        self.ids, self.g, self.mode = [int(x) for x in ids], int(g), mode
        self.T = {k: {} for k in range(8, 17)}
        self.upto = 0
        self.duel = [0, 0]                                   # copy-only-right, drafter-only-right (checked positions)
        self.duelb = [0, 0]                                  # the same on short line-boundary copies
        self.NL = nl_ids()

    def commit(self, r):
        ids = self.ids
        for e in range(self.upto, r):
            for k in range(8, 17):
                if e - k + 1 < 0:
                    break
                self.T[k][tuple(ids[e - k + 1:e + 1])] = e + 1
        self.upto = max(self.upto, r)

    def allow(self, hit):
        o, k, j, ext, bnd = hit
        if self.mode == "all":
            return True
        if j > self.g:
            return False
        if self.mode == "prompt":
            return True
        if (self.duel[0] + 3) / (self.duel[0] + self.duel[1] + 4) < 0.6:
            return False
        if self.mode == "pduel":
            return True
        return ext >= 24 or not bnd or (self.duelb[0] + 1) / (self.duelb[0] + self.duelb[1] + 2) >= 0.6

    def propose(self, dr, r):
        self.commit(r)
        ids, blk = self.ids, []
        tok = lambda i: ids[i] if i <= r else blk[i - r - 1]
        out, used, info = [], [], []
        for p, d in enumerate(dr):
            x, hit = r + 1 + p, None
            for k in range(16, 7, -1):
                if x - k < 0:
                    continue
                j = self.T[k].get(tuple(tok(i) for i in range(x - k, x)))
                if j is not None:
                    m = 0
                    while m < 256 and j - 1 - m >= 0 and tok(x - 1 - m) == ids[j - 1 - m]:
                        m += 1
                    hit = (ids[j], k, j, m, ids[j] in self.NL or ids[j - 1] in self.NL)
                    break
            u = hit is not None and self.allow(hit)
            t = hit[0] if u else int(d)
            out.append(t)
            used.append(u)
            info.append(hit)
            blk.append(t)
        return out, used, info

    def learn(self, dr, info, true, k):
        for p in range(min(k + 1, len(info), len(true))):
            h = info[p]
            if h is None:
                continue
            ok_o, ok_d = h[0] == int(true[p]), int(dr[p]) == int(true[p])
            if ok_o != ok_d:
                self.duel[0 if ok_o else 1] += 1
                if h[4] and h[3] < 24:
                    self.duelb[0 if ok_o else 1] += 1
