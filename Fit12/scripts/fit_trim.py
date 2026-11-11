"""Fit12 L2-02 / L2-03 (2026-10-24; handed over by Breakthrough 02:15): exact memory trims for B12LL, so C_m 59 fits < 12 GB.
Both act inside bigmoe_v3f.stream.ExpertStreamer.install, i.e. BEFORE the loader's mx.eval(model.parameters()):
  T2 embed (F12_EMBED=1): embed_tokens -> fit_embed.StreamedEmbedding (memmap row gather + the same mx.dequantize): the
     358 MB resident table is never read.
  T1 hc_pre (F12_HCPRE=1): hc8's 8-bit swap of the hyper-connection mixers + PLE K/V (Breakthrough's v3f/hc8.py, same call,
     same bits) on the still-LAZY bf16 weights. load_b12 runs hc8 only after the loader has evaluated every weight, so the
     bf16 mixers (~1.25 GB) are resident next to their 8-bit copies for a moment: MLX's load peak is ~1.1 GB above the
     steady state (M1: peak 10.80 vs active 9.68 GB). Quantizing lazily makes the bf16 tensors graph intermediates, freed
     as soon as their quantize has run. Same mx.quantize on the same bf16 input -> bitwise the same 8-bit weights;
     load_b12's own hc8.install then finds no bf16 mixer left (0 swapped).
Use from_env(B) after importing bt_b12 and before load_b12 (it installs after load_b12's _env())."""
import os
import fit_embed

STATS = {}


def install(embed=True, hc_pre=True):
    import bigmoe_v3f.stream as BS
    if getattr(BS.ExpertStreamer.install, "_fit12_trim", False):
        return
    inst0 = BS.ExpertStreamer.install
    hc_bits = int(os.environ.get("B12_HC", "8"))

    def inst(self, model, layers=None):
        r = inst0(self, model, layers)
        tm = model.language_model.model
        if embed:                                     # first: hc8.install evaluates TM.parameters() at its end
            qe = tm.embed_tokens
            tm.embed_tokens = fit_embed.StreamedEmbedding(self.model_dir, qe.group_size, qe.bits,
                                                          getattr(qe, "mode", "affine"))
            STATS["embed"] = f"{qe.bits}-bit g{qe.group_size}"
            print(f"fit12 trim: embed_tokens streamed ({STATS['embed']}); the resident table is never evaluated", flush=True)
        if hc_pre and hc_bits:
            import hc8                                # v3f/hc8.py, as load_b12 imports it
            n, before, after = hc8.install(model, bits=hc_bits)
            STATS["hc_pre"] = (n, before, after)
            print(f"fit12 trim: hc{hc_bits} applied before the loader's eval: {n} modules, {before:.0f} -> {after:.0f} MB",
                  flush=True)
        return r
    inst._fit12_trim = True
    BS.ExpertStreamer.install = inst


def from_env(B):
    """Harness hook: F12_EMBED=1 / F12_HCPRE=1 (B = the imported bt_b12 module). The patch goes in from inside load_b12,
    right after its _env(): the ext reads BIGMOE_DEFER / SPARE / PF_NOPUBLISH once, when bigmoe_v3f is first imported.
    Importing it before _env() left phase B off in the ext while the gate expected it, so the first 1-row pass timed out
    at every layer with a late part (47 spin timeouts; Fit12 B1 isolation, 04:40)."""
    import sys
    e, h = os.environ.get("F12_EMBED") == "1", os.environ.get("F12_HCPRE") == "1"
    if not (e or h):
        return
    _env0 = B._env

    def _env():
        _env0()
        if B.V3F not in sys.path:
            sys.path.insert(0, B.V3F)
        install(embed=e, hc_pre=h)
    B._env = _env
