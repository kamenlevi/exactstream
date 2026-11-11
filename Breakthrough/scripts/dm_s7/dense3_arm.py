# NewIdeas 16:5x: map arm for Leap's DENSE-3 (the orchestrator's FAST dense diet (b)); paste into
# ~/FableGod/decision_map/dm_ablations.py after the s7_ref3 line. Arm name: qom_0_live_ball_c147_t3+dense3
# (screen 6's FAST code stack + DENSE-3, via ablation()'s "+" stack); twin = screen 6 (qom_0_live_ball_c147_t3).
def dense3(which=None):
    """Leap's DENSE-3 sim (~/Leap/scripts/dense3.py): the trunk's dense 4-bit g64 projections (GDN in / out, attention
    q / k / v / o) at their MSB-3 values in the 4-bit format (codes & 0xEEEEEEEE, biases bf16(b + s / 2)); the router,
    shared expert, indexer, experts, embed and head untouched. Restored by clear() (TEARDOWN), as hc8w."""
    def setup():
        import sys
        assert os.environ.get("BIGMOE_PKD1") != "1" and os.environ.get("BIGMOE_SCOUT_BUNDLE") != "1", \
            "dense3: a fusion repacked the projections at load; the sim would not reach the forward"
        sys.path.insert(0, os.path.expanduser("~/Leap/scripts"))
        import dense3 as D3
        from dm_common import RT, TEARDOWN
        rt = RT["rt"]; model = rt.model if hasattr(rt, "model") else rt
        W = which or D3.WHICH
        saved = [(m, m["weight"], m["biases"]) for _, m in D3._targets(model, W)]
        n, mb = D3.sim(model, W)
        def restore():
            for m, w, b in saved: m.weight = w; m.biases = b
        TEARDOWN.append(restore)
        assert n == len(saved) and n >= 96, f"dense3 converted only {n} modules"
        print(f"[dm] dense3: {n} modules at MSB-3 ({mb:.0f} MB of 4-bit weights rewritten)", flush=True)
    return setup
ABLATIONS["dense3"] = dense3()
