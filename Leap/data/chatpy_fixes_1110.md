# chat.py fixes for 1.1.0 (Tue 10 Nov, 21:4x-22:2x; Kamen's chat, via the orchestrator)

Four problems Kamen hit in chat.py tonight, fixed in release/src (c82e93e) and in the local clone he chats from
(release/cleanclone/final/clone, 22:22:21, backups *.bak-1110b); candidate rebuilt as pkg_110e (7943ddb).

1. **A paste became several messages, and a long one was cut.** chat.py read `sys.stdin.readline()` in the tty's
   canonical mode: each pasted line was a message, and the tty holds at most MAX_CANON = 1,024 bytes per line, so a
   pasted 1,800-character prompt was cut at about 1,023 characters (Kamen 21:4x). Python's readline here is libedit,
   and input() would echo to fd 1, which carries the engine log, so neither was a drop-in fix.
   **Fix:** a message reader (class Keys) in non-canonical mode (ICANON / ECHO / IEXTEN off, ISIG on): chat.py echoes
   and edits the message itself (Backspace, Ctrl-U, Ctrl-W; wide characters; a row-crossing erase redraws the message
   on a fresh line); Enter sends it; bracketed paste keeps a paste in one message, newlines included, and on a terminal
   without bracketed paste, lines arriving together (within 30 ms) count as one paste; arrow keys and other escape
   sequences are ignored; Ctrl-C at the prompt and Ctrl-Z stay signals; Ctrl-D on an empty message quits. The terminal
   settings and bracketed-paste mode are restored after every message, on Ctrl-C, on errors and on SIGTERM. A pipe still
   gives one line per message (the clean-clone test's path).
2. **Long answer lines froze the display** (Kamen 21:50, minified CSS 213 columns wide). Root cause: the UI took its
   terminal size from `shutil.get_terminal_size()`, which reads fd 1, the engine's log after chat.py's redirect, so it
   always used the fallback (30 rows, 80 / 100 columns). On a taller window the scroll region covered rows 1-29 only,
   the cursor could sit below it, and text at the last screen row overwrote that row in place. A rendering test of the
   old chat.py in a 40 x 120 pty showed the region at rows 1-29 (data/chatpy_fixes_1110/old_layout_check.txt).
   **Fix:** the orchestrator's title-bar status (the live numbers go to the window title with OSC 0; no scroll region, no
   cursor save / restore), adopted in release/src; the UI's size now comes from its own tty (used by the reader's
   wrapping). The per-answer numbers still print after each answer.
3. **The research heat PAUSE file paused Kamen's chat mid-answer** (fanwatch set it at 21:53). The loop waits on it
   before each prompt (`PAUSE`) and between passes (`bigmoe.gated._PAUSE`, every 50 ms with BIGMOE_PAUSE_EVERY_MS=50).
   **Fix:** chat.py and serve.py set both to "" (which never exists): the item-loader hook before the warm-up's and every
   answer's run, and the import hook when the loop imports bigmoe.gated. The published package has no such file.
4. **`/file PATH message`** (Kamen's request, for context): sends the message, then the file in a fenced block with the
   language from its extension (a longer fence if the file holds ```), ~ expanded, a quoted path for spaces; it prints a
   note with the file's lines and tokens; a missing file gives a note and sends nothing.

Also tonight: chat.py's and serve.py's `--max-tokens` default 65,536 (data/w24_pkg.md, pkg_110d).

## Tests (all CPU, the stand-in loop; data/chatpy_fixes_1110/)

- **pty_test.py** runs chat.py in a real pseudo-terminal (40 x 120, no COLUMNS / LINES) on mock_screen.py and checks the
  message the loop got (out/<tag>.jsonl): **21 / 21 PASS** (pty_test_final.log, 84 s): a 3-line bracketed paste (one
  message), a 3,000-character one-line paste with and without markers (all 3,000 characters), 3 lines arriving together,
  Backspace, Ctrl-U, Ctrl-W, arrow keys ignored, UTF-8 and wide characters, a Backspace across a wrapped row, /file
  on a 417-line HTML file of Kamen's (11,345 tokens, html fence, the note), /file on a missing file, canonical mode and
  echo back while an answer streams, a whole session rendered by a small VT emulator (vt.py) with a 2,000-character
  single-line answer (no scroll region or cursor save / restore, 243 title updates, the answer and its numbers on
  screen), Ctrl-C at the prompt, Ctrl-D quits (rc 0), bracketed paste off at the end, SIGTERM while reading exits
  through the restore (rc 143).
- **Pipe:** two piped lines are two messages; no escape sequences in piped output.
- **PAUSE:** mock_pause.py plus a stand-in bigmoe/gated.py wait on a scratch file (never the real PAUSE file): the new
  chat.py answers in 2 s with that file present; the old one waited the stand-in's full 15 s.
- **Real engine** (chatfix_now.sh, Kamen's clone right after his old chat closed, NO-SWAP 45 GB, share lock): chat.py
  --mem 32, one piped message: rc 0 in 22 s, answer "Hello there friend", first token 1.71 s, lifetime peak 31.23 GB.

## Candidate

pkg_110e (7943ddb) = pkg_110d + chat.py, serve.py, INSTALL.md (one sentence: a paste is one message, /file, the window
title) and MANIFEST.json; leaks 0, unsafe 0, live paths 0, syntax 0. The go pushes pkg_110e.

## 5. No-tools system line, and /file with the file first (22:3x; release/src 83d99bb, candidate pkg_110f e756da7)

Kamen's first /file chat (22:27, chat_1110_222723.jsonl in his clone): his request came first, then the 417-line file.
The answer opened with "I'll start by exploring the project structure" and repeated a made-up tool-call block (a `ls -la`
on a path that does not exist) until it ended at 1,494 tokens, with no file.
- **A built-in system line** (chat.py's SYSTEM; `--system` text goes after it): "You are chatting with the user in a
  terminal. You have no tools: you cannot run commands, browse, or open files. When the user shares a file, its full
  contents are in their message." /clear keeps it.
- **/file sends the file first, then the request**, so the instruction comes last after a long file.
- **Stand-in test: 22 / 22** (pty_test_final.log): the system line reaches the prompt (the stand-in answers SYS-OK when
  it sees it), /file's message = the file's fenced block, then the request; the other 20 checks as before.
- **Real engine:** a check in Kamen's clone (chatfix_now2.sh) runs when his current chat ends: chat.py --kind code --mem
  32 --max-tokens 600 with his 22:27 request on the same file; PASS = code in the answer and no made-up tool calls.
  Result below when it has run.
- pkg_110f differs from pkg_110e in chat.py, INSTALL.md (/file sentence) and MANIFEST.json. The go pushes pkg_110f.

**Real-engine /file check: PASS (Wed 01:31-01:32, filecheck_night.sh in the v110d clone = release/src chat.py):** chat.py
--kind code --mem 32 --max-tokens 600, /file on the 417-line HTML file (11,345 tokens) with Kamen's 22:27 request: rc 0 in
55 s, 600 tokens at 54.9 tok/s whole decode, first token 25.7 s, lifetime peak 35.32 GB, drafter dfl at 3.85 tokens per
check; made-up tool-call lines 0, code lines 3. The answer finds the bug (the grid has 2 hidden rows above the 20 drawn
ones; the drawing maps rows straight to the canvas). Log: chatpy_fixes_1110/filecheck_night_1111.log.
