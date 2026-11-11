"""ENC2 SCREEN (Leap, Sun 2026-11-08 08:3x; BigMoE CHS5: the chat check is CPU-bound, bld ~31.5 of ~33 ms; XPT: per check the
> (a quote from the project owner, kept private)
full to 'prep' the math". Question 1: does MLX 0.32.2 release the GIL while async_eval encodes? Question 2: can a second
thread run SB's mid-loop async_eval (encode layer group k) while the main thread builds group k+1, bitwise equal, faster?
Toy check: 48 "layers" x ~60 small ops on 3 rows (dense-like glue: matmul 2560x2560 bf16 slices, norms, elementwise),
async_eval every 2 layers (SB 2), final eval; serial vs two-thread. Pre-registered line: PASS = GIL released (side thread
keeps >= 70 % of its solo rate during eval) AND two-thread wall <= 0.8x serial with bitwise equal outputs and no crash
in 200 checks. -> stdout"""
import queue
import sys
import threading
import time

import mlx.core as mx

D, L, ROWS, SBN = 2560, 48, 3, 2
mx.random.seed(3)
W = [(mx.random.normal((D, D)) * 0.02).astype(mx.bfloat16) for _ in range(4)]
g = (mx.random.normal((D,)) * 0.1 + 1).astype(mx.bfloat16)
mx.eval(W, g)


def layer(h, i):
    """~60 small kernels of dense-glue shape on ROWS rows"""
    for j in range(6):
        a = mx.fast.rms_norm(h, g, 1e-6)
        b = a @ W[(i + j) % 4]
        c = mx.sigmoid(b[:, :640]) * b[:, 640:1280]
        d = mx.concatenate([c, c, c, c], axis=-1)
        h = h + 0.1 * d.astype(h.dtype) - 0.05 * mx.tanh(b) * 0.1
        h = h * (1.0 / (1.0 + mx.abs(h).mean(axis=-1, keepdims=True)))
        e = mx.softmax(b[:, :256].astype(mx.float32), axis=-1).astype(h.dtype)
        h = h + mx.pad(e, ((0, 0), (0, D - 256))) * 0.01
    return h


def check_serial(x):
    h = x
    for i in range(L):
        h = layer(h, i)
        if (i + 1) % SBN == 0 and i + 1 < L:
            mx.async_eval(h)
    mx.eval(h)
    return h


class Enc:
    """the encoder thread owns a GPU stream (MLX 0.32 streams are thread-local): the main thread builds the check's graph
    on that stream; this thread async_evals each queued group in order and runs the final eval"""
    def __init__(self):
        self.q = queue.Queue()
        self.ready = threading.Event()
        self.t = threading.Thread(target=self.run, daemon=True)
        self.t.start()
        self.ready.wait()

    def run(self):
        self.s = mx.new_stream(mx.gpu)
        self.ready.set()
        while True:
            a = self.q.get()
            if a is None:
                return
            if isinstance(a, tuple):                          # ("final", array, event)
                with mx.stream(self.s):
                    mx.eval(a[1])
                a[2].set()
            else:
                with mx.stream(self.s):
                    mx.async_eval(a)


ENC = None


def check_two(x):
    h = x
    with mx.stream(ENC.s):                                  # ops bound to the encoder thread's stream
        for i in range(L):
            h = layer(h, i)
            if (i + 1) % SBN == 0 and i + 1 < L:
                ENC.q.put(h)
    ev = threading.Event()
    ENC.q.put(("final", h, ev))
    ev.wait()
    return h


