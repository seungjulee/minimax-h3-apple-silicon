# Fix NaN / black frames on MPS in the pruned-base AdaLN injection

**Fixes #14.**

## Summary

On Apple Silicon (MPS), `MiniMaxH3TurboLoRA` produces **black frames and NaN audio** on
every render against a pruned / curve-mode base. The NaN comes from the runtime AdaLN
injection: on a curve-mode base the bf16 LoRA operands are upcast to fp32, and the
resulting `[out, rank] @ [rank, M]` product (≈ `[96768, 16] @ [16, 1]`) returns NaN on MPS.

The patch computes that product in the LoRA's own dtype on MPS and casts only the result.

```diff
-            av = a.to(x.device, x.dtype)
-            bv = b.to(x.device, x.dtype)
-            sv = st.to(x.device, x.dtype)
-            x = x + (bv @ (av @ sv.T)).T                              # [M, out]
+            # On a curve-mode base `x` is fp32 (adaln_dtype), so the bf16 operands get
+            # upcast; on MPS that fp32 [out, rank] @ [rank, M] product returns NaN and
+            # poisons the whole sample (#14). Compute in the operands' own dtype there
+            # and cast only the result. Non-MPS backends keep their current behaviour.
+            cd = a.dtype if (x.device.type == "mps" and a.dtype.is_floating_point) else x.dtype
+            av = a.to(x.device, cd)
+            bv = b.to(x.device, cd)
+            sv = st.to(x.device, cd)
+            x = x + (bv @ (av @ sv.T)).T.to(x.dtype)                  # [M, out]
```

## Why the upcast happens only on pruned bases

`comfy/ldm/minimax/model.py`:

```python
"adaln_dtype": torch.float32 if self.use_adaln_curves else dtype
```

| Base | `x.dtype` in `_make_adaln_forward` | operands cast to | compute dtype today |
|---|---|---|---|
| full / non-pruned | `dtype` (bf16) | bf16 | **bf16** |
| pruned / curve-mode | **fp32** | **fp32** | **fp32** ← only here |

The fp32 compute is incidental — `a.to(x.device, x.dtype)` matches `x` so the add lines
up, not because fp32 was wanted. Non-pruned bases already compute this product in bf16 on
every backend today, and `a`, `b` and the shipped `h3_silu_temb_grid.safetensors` are all
bf16 on disk, so the upcast added no information.

## Reproduction

**Environment:** M5 Max / 128 GB / macOS 26.6.1 / PyTorch 2.13.0 / ComfyUI 0.32.0 /
node at `546b502`. Base `minimax_h3_fl2va_pruned_int8_convrot.safetensors` (Comfy-Org),
text encoder `qwen3vl_32b_minimax_h3_nvfp4_awq`, both H3 VAEs.

1. Stock H3 t2v graph.
2. Insert `MiniMaxH3TurboLoRA` (`minimax_h3_turbo_v4_step600_ema`, strength 1.0,
   `low_vram=False`) between `UNETLoader` and `BasicGuider`.
3. `MiniMaxH3TurboSampler` → `SamplerCustomAdvanced`, `BasicScheduler(simple, steps=6)`.
4. Render anything — **256×256 / 22 frames is enough**.

```
[H3TURBO fwd pruned/bypass] call#1 ... video_rms=1.0050 audio_rms=1.0070   <- input clean
[H3TURBO step 0] denoised_rms=nan x_rms=nan d_rms=nan                      <- output NaN
...
av.error.ArgumentError: Invalid argument: 'avcodec_send_frame()' returned 22
```

The `avcodec` error is downstream — `VAEDecodeAudio` emits NaN and the AAC encoder
rejects the frame. 100% deterministic: every render, every resolution, both checkpoints.

#14 reports the same failure independently on a **pruned BF16** base with PyTorch 2.12.1 /
ComfyUI 0.31.1 / 64 GB. Two different pruned variants (BF16 and INT8) and two different
PyTorch versions, same signature — consistent with the trigger being the curve-mode fp32
upcast rather than any particular quantisation.

## How it was localised

Each row is a one-line edit to `_make_adaln_forward`, one render each, same seed.

| Probe | Result | Conclusion |
|---|---|---|
| `pass` — skip the injection | ✅ works | NaN is inside this block |
| `x + torch.zeros_like(x)` | ✅ works (rms identical to `pass`) | the add is fine |
| `x + (bv @ (av @ sv.T)).T * 0.0` | ❌ **NaN** | **×0 cannot create NaN → the product is already NaN** |
| reduction after the **inner** matmul `av @ sv.T` | ❌ NaN | inner product is fine |
| reduction after the **outer** matmul `bv @ _i` | ✅ works | **the outer product is the faulty op** |
| whole chain in bf16 | ✅ works | the fp32 upcast is the trigger |

The `* 0.0` row is the decisive one: multiplying by zero cannot introduce a NaN, so the
product is already NaN before the add. That rules out the add, the `.view()`/`.chunk()`
that follow it, and any interaction with the consumer.

## Ruled out

| Hypothesis | Test | Result |
|---|---|---|
| Bad LoRA checkpoint | v4-600 **and** v1-850 | both fail |
| bypass vs merge | `low_vram=True` | fails |
| Third-party MPS accel node | `ComfyUI-AppleSilicon-FP8` fully removed | fails |
| …its Metal INT8 kernel | `ASFP8_INT8_EXT=off` | fails |
| In-place add in `_FrugalLoRA` | out-of-place `torch.add` | fails |
| Non-contiguous transposed operand | `.T.contiguous()` | fails |
| Associativity / output transpose | `(sv @ av.T) @ bv.T` | fails |
| Operand lifetime / buffer reuse | keep `av`,`bv`,`sv` alive past the forward | fails |
| Host readback | `_d.view(-1)[0].item()` | fails |
| Command-queue ordering | `torch.mps.synchronize()` | fails |
| Materialising copy | `.clone()`, `torch.add(out=)`, `* 1.0` | all fail |
| Memory pressure / size | 256×256, 416×736 @ 22 / 124 / 192 frames | fails at **every** size |

`torch.mps.synchronize()` and a host readback both fail while **any** reduction
(`.sum()`, `.abs().max()`) works — reductions are the only probes that force the product
to be evaluated as a standalone tensor rather than folded into the consuming graph.

## Verification

Pruned INT8 base on MPS, patch applied:

| Case | Result |
|---|---|
| v4-600, strength 1.0, 6 steps | ✅ valid video + 32 kHz audio |
| v1-850 checkpoint | ✅ |
| `low_vram=True` (merge path) | ✅ |
| strength 0.8 | ✅ |
| 8 steps | ✅ |
| 416×736 / 124 frames | ✅ `denoised_rms` 1.4552 → 1.1228, all finite |

Against the fp32 path forced to work with a `.sum()` barrier (same seed): both produce
valid, comparable output. Not bit-identical (PSNR ≈ 22 dB) because diffusion amplifies any
numerical change into a different trajectory — the same as swapping any low-level kernel.
Neither is degraded; the bf16 frame is marginally sharper.

## Impact on CUDA / ROCm

**None — the fast path is gated to MPS.**

I only have Apple Silicon, so I deliberately did not change other backends. Worth noting
that #21 (ROCm, pruned FP8) reaches a *row-count* `RuntimeError` at this same line, which
means the fp32 product **executes fine there** — so this is an MPS-specific kernel problem,
not a general one, and there is no reason to alter numerics for backends that work.

If a maintainer would rather have one dtype policy everywhere, dropping the
`x.device.type == "mps"` clause makes the pruned path match what non-pruned bases already
do on every backend, and also avoids upcasting the largest operand in the expression. Both
versions are verified working here; I've proposed the gated one because it has the smaller
blast radius.

## Relationship to the open PRs

#12, #16, #19 and #25 all fix a different bug in this area — the **row-count** mismatch in
`_unique_t` / `_inject_adaln_egrid` when audio or video references are present, which
surfaces as `RuntimeError: The size of tensor a (3) must match the size of tensor b (2)`.
This PR changes only the operand dtype and is orthogonal to all of them.

#16 is the only one that also edits `_make_adaln_forward`; it adds a
`if st is not None and st.shape[0] == x.shape[0]:` guard *above* the cast block and leaves
the casts themselves unchanged. So a textual conflict is possible but trivial, and the two
changes compose — happy to rebase on whichever lands first.

## Not reproducible standalone

The same expression at the exact shapes and dtypes (`[96768,16] @ [16,1]` fp32, `x` from
`Linear(8, 96768)`) run on MPS outside the model — including under `inference_mode`, with
51 concurrent modules, and with **deferred** NaN checks so the check adds no reduction
barrier — produces **zero** NaN over hundreds of iterations.

So I can't offer a minimal PyTorch-level repro and I'm not claiming this is definitively a
PyTorch bug. What is established: the faulty operation, that the fp32 upcast triggers it,
that the upcast is incidental to curve mode, and that removing it on MPS fixes the node
deterministically across every configuration tested.
