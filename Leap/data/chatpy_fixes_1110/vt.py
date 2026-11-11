"""A small VT100/xterm subset to render chat.py's terminal output in tests: printable text with autowrap and a pending
wrap at the right margin, CR, LF / IND inside a scroll region (DECSTBM), BS, CUU, CUP, EL, DECSC / DECRC, SGR and private
modes ignored. DECRC comes in three flavours, as terminals differ on the pending wrap: 'keep' restores it, 'clear' drops
it, 'advance' moves to the next row without scrolling (the worst case). `bad` counts text written on the status row (the
last row) outside a status draw (a CUP to the last row up to the next DECRC)."""
import re, unicodedata


class VT:
    def __init__(self, rows, cols, restore="keep"):
        self.R, self.C, self.restore = rows, cols, restore
        self.g = [[" "] * cols for _ in range(rows)]
        self.y = self.x = 0
        self.pend, self.top, self.bot = False, 0, rows - 1
        self.saved, self.in_status, self.bad, self.regions, self.titles = (0, 0, False), False, 0, [], []

    def lf(self):
        if self.y == self.bot:
            del self.g[self.top]
            self.g.insert(self.bot, [" "] * self.C)
        elif self.y < self.R - 1:
            self.y += 1

    def put(self, ch):
        w = 0 if unicodedata.combining(ch) else 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1
        if w == 0:
            return
        if self.pend or self.x + w > self.C:
            self.x, self.pend = 0, False
            self.lf()
        if self.y == self.R - 1 and not self.in_status:
            self.bad += 1
        self.g[self.y][self.x] = ch
        if w == 2 and self.x + 1 < self.C:
            self.g[self.y][self.x + 1] = ""
        if self.x + w >= self.C:
            self.x, self.pend = self.C - 1, True
        else:
            self.x += w

    def feed(self, s):
        i, n = 0, len(s)
        while i < n:
            ch = s[i]
            if ch == "\x1b" and i + 1 < n:
                nx = s[i + 1]
                if nx == "7":
                    self.saved = (self.y, self.x, self.pend)
                    i += 2
                    continue
                if nx == "8":
                    y, x, p = self.saved
                    if self.restore == "keep":
                        self.y, self.x, self.pend = y, x, p
                    elif self.restore == "clear":
                        self.y, self.x, self.pend = y, x, False
                    else:                                  # 'advance': a pending wrap becomes a move to the next row
                        self.y, self.x, self.pend = (min(y + 1, self.R - 1), 0, False) if p else (y, x, False)
                    self.in_status = False
                    i += 2
                    continue
                if nx == "]":                              # OSC (a window title): up to BEL or ST, not shown
                    j = min([k for k in (s.find("", i), s.find("\\", i)) if k >= 0] or [n - 1])
                    self.titles.append(s[i + 2:j])
                    i = j + (2 if s.startswith("\\", j) else 1)
                    continue
                if nx == "[":
                    m = re.match(r"\x1b\[([?]?)([0-9;]*)([@-~])", s[i:])
                    if not m:
                        i += 2
                        continue
                    priv, par, f = m.groups()
                    a = [int(v) if v else 0 for v in par.split(";")] if par else []
                    if not priv:
                        if f == "A":
                            self.y, self.pend = max(0, self.y - (a[0] if a and a[0] else 1)), False
                        elif f == "H":
                            self.y = (a[0] if a and a[0] else 1) - 1
                            self.x = (a[1] if len(a) > 1 and a[1] else 1) - 1
                            self.pend = False
                            self.in_status = self.y == self.R - 1
                        elif f == "K":
                            mode = a[0] if a else 0
                            rng = range(self.x, self.C) if mode == 0 else range(self.C)
                            for c in rng:
                                self.g[self.y][c] = " "
                        elif f == "r":
                            self.top = (a[0] if a and a[0] else 1) - 1
                            self.bot = (a[1] if len(a) > 1 and a[1] else self.R) - 1
                            self.regions.append((self.top + 1, self.bot + 1))
                            self.y = self.x = 0
                            self.pend = False
                    i += m.end()
                    continue
                i += 2
                continue
            if ch == "\r":
                self.x, self.pend = 0, False
            elif ch == "\n":
                self.pend = False
                self.lf()
            elif ch == "\b":
                self.x, self.pend = max(0, self.x - 1), False
            elif ch >= " " and ch != "\x7f":
                self.put(ch)
            i += 1

    def text(self):
        return ["".join(r).rstrip() for r in self.g]
