# REFC: the reference contract in r57's EXACT build (Leap, Wed 18 Nov 2026 11:5x)

CONTRACT YES (Kamen, relayed 11:1x; NewIdeas CONTRACT_YES_PLAN 899926d section 1). The EXACT build only; the lossy stack
and CERTIFY98 stay on NAX6-N.

## a. Installer + env (BigMoE wires; default off == today's build, bitwise)
- **Env: `REF2_REFC=1`.** Unset or 0: refc is never installed and nothing is patched, so the build is today's, bit for bit.
- **One line in r57's exact paths (chain + tree), after `ref2.apply()` with the TARGET model loaded:**
  `import refc; refc.install_from_env(ref2, model)` (refc = `~/Leap/scripts/week98/refc.py`; it needs `~/Leap/scripts`
  and `~/Leap/scripts/week98` on sys.path, as the c25 / E14 harnesses have).
- **Placement:** after ref2.apply(), with the target model loaded. bofi (if installed) checks the contract at call time, so the
  order does not matter. v32 / v32one are needed only for 17-32-row checks; chain-16 / tree-16 run the <= 16-row forms.
- **Shape limits:**
  - Dense: rows 1-32, bf16 activations, 4-bit g64, the 7 (N, K) shapes. <= 16 rows: qmm16c(ks) padded to 16. 17-32 rows:
    ck.qmm16c2a_ks, and only with v32 + v32one installed.
  - HC read: 16 padded rows; 32 rows via one32 with v32one.
  - Masks: the contract's forms are row-local and never see a mask.
- **Log `refc.report()` at the start and the end of a run.** It shows ON, the target arrays found per contract shape, and
  the forms actually taken. A shape with 0 target arrays keeps REF2-T's own <= 16-row form: still row-invariant, but no
  gain on that shape.
- **Scope (UPDATED Wed 12:2x, ec06e43, refc.py md5 1a70f599):**
  - DEFAULT `REF2_REFC_SCOPE=ref2`: every contract form fires while ref2 is on, on any array. This is bofi's reviewed
    scope (r57 removes ref2 before every draft) and E14's behavior.
  - The 11:4x allow-list (`REF2_REFC_SCOPE=target`: the target's QuantizedLinear arrays) found only 3 / 7 shapes on BigMoE's
    s98m. The fused in-projections (ref2's in_proj_fused objects), QSA qkv_fused and the shared cat are not modules.
    That was safe (row-invariant) but lost most of the gain, so it is now opt-in only.
  - report() prints the dense forms taken per shape. ENGAGED = all 7 > 0 after the first check.
  - Drafter proof: the per-shape counts stay flat across a draft block.

## The named set for the snapshot (EvalSuite's pin) and its bits
| lever | in the contract build | why |
|---|---|---|
| **REF2_REFC=1** | ON | The contract: 7 dense shapes at fixed K-chunks (qmm16c ks 4/10/4/4/5/10/4, AR padded to 16) + the HC read's S4 partials at 2,048-key partitions. |
| REF2_BOF / BOF2-O | OFF (inert either way) | bofi falls through while the contract is on; its shapes are the contract's. |
| REF2_HC_B | **OFF this week (recommended)**, or ON only if M4C runs with it ON | Kernel B also routes through `leapk.s4_partials`, so it takes the contract's partials (eligible at the contract's NP 5). At stub level its `mixed` differs from the contract's S4 + gate1 path, BUT the same stub with the contract OFF differs the same way (control 11:5x). So the stub does not reproduce E14's engine-level bitwise D1-B, and the pair is inconclusive below the engine. In the contract build only self-consistency matters (its own AR), so HC_B ON is fine if M4C (row invariance end to end) runs with it ON. |
| GDN16 / gdn_fast (if used) | may stay ON | The GDN core is untouched by the contract; bitwise == mlx_lm's kernel. |
| tgl16e (if used in exact) | may stay ON | The experts are untouched by the contract; bitwise == ooo1cq. |
| REF2_GDN_LAZY | OFF | Chain-shaped (see 11-17 16:1x): not for tree-16 without a tree bit test. |

- **Bits.** This is a NEW reference: REF2-T's forms everywhere except the contract's 7 dense shapes and the HC read's S4
  partials. The self-test shows the contract's rows differ from MLX's own kernel on 6 of 7 shapes (b|a 96x2560: 1/16).
  Verdicts are against the contract build's own AR, never against NAX6-N or REF2-T.
- **Gain.** E14 (exact engine, Tue 00:5x): the contract is -5.76 ms per 16-row check and -7.04 per 32-row, vs plain REF2-T.
  With BOF inert (-2.36) and HC_B out (-0.81), the net vs today's REF2-T + BOF + HC_B build is ~-2.6 ms per 16-row check [P].
- **Files for the snapshot (md5, 11:5x):** refc.py 1a70f599 (ec06e43), qmm16.py 42766c32, ck.py 29e91d37, leapk.py d54a930a,
  one32.py c1bebad7, ref2.py 589b840c (all under ~/Leap/scripts).

## b. EvalSuite's VOID gate: row invariance
- **Kernel level (Leap, done 11:5x): `[BigMoE venv] python ~/Leap/scripts/week98/refc_rowinv.py`: 18 / 18 PASS**
  (data/week98/check25/refc_rowinv.txt). All bit for bit:
  - Every contract dense form, rows 1-16 vs the 16-row call, plus 8 tree-shaped row sets (any size, any order) and the 3-D
    form.
  - The HC read's S4 partials + gemm under row permutations and zeroed tails.
  - The guards: contract OFF, ref2 OFF and a weight outside the target set all fall through to today's kernel bitwise.
- **Engine level with tree masks (BigMoE's M4C, Thu):** tree-16 == chain-16 == AR on every token, mbpp/16 + mbpp/30
  included. Tree masks act only on attention and the GDN recurrence, which the contract leaves on REF2-T's forms. So the
  contract's share is row-invariant by the kernel test, and the masked share is what M4C checks.

## c. The drafter
- **Untouched by construction.**
  - The contract's dense forms take only the target's own weight arrays, and only while ref2 is on (r57 removes ref2 for
    drafts).
  - t30b Q8H32 is 8-bit, and c2 / ra_sf builds its own arrays, so neither is in the target set.
  - PK's pk_level runs its own kernels (8-bit dense asserted, experts via its own path), not mx.quantized_matmul.
  - So PK == SF is unaffected and the draft ms are unchanged.
- **Runtime proof for the key test:** `refc.report()`'s "forms taken" must not move across a draft block. Print it before
  and after one draft call.
