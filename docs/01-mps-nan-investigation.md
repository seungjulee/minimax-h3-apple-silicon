# MiniMaxH3TurboLoRA: NaN on MPS with pruned/curve-mode bases

**Repo:** [`Larryvrh/ComfyUI-MiniMax-H3-Turbo`](https://github.com/Larryvrh/ComfyUI-MiniMax-H3-Turbo)
**Environment:** M5 Max / 128 GB / macOS 26.6.1 / PyTorch 2.13.0 / ComfyUI 0.32.0
**Base:** `minimax_h3_fl2va_pruned_int8_convrot.safetensors` (Comfy-Org)

## Symptom

Applying `MiniMaxH3TurboLoRA` to a pruned base on MPS produces **pure black frames and
NaN audio** on every render, surfacing downstream as

```
av.error.ArgumentError: Invalid argument: 'avcodec_send_frame()' returned 22
```

during audio encode, because the sampler emitted NaN. The node's own debug shows a clean
input and a NaN on the very first forward:

```
[H3TURBO fwd pruned/bypass] call#1 ... video_rms=1.0050 audio_rms=1.0070   <- input fine
[H3TURBO step 0] denoised_rms=nan x_rms=nan d_rms=nan                      <- output NaN
```

## Root cause

In `_make_adaln_forward` (the pruned/curve-mode adaln injection):

```python
av = a.to(x.device, x.dtype)      # bf16 -> fp32, because x is fp32 on a curve base
bv = b.to(x.device, x.dtype)
sv = st.to(x.device, x.dtype)
x = x + (bv @ (av @ sv.T)).T
```

On a pruned base the model sets `adaln_dtype = torch.float32`, so `x.dtype` is fp32 and
all three bf16 operands are upcast. **The outer product `bv @ (av @ sv.T)` —
`[96768, 16] @ [16, 1]` in fp32 — silently returns NaN inside this graph on MPS.**

Evidence it is that specific matmul:

| Probe | Result |
|---|---|
| `_s = _d.sum()` / `_d.abs().max()` after the full chain | ✅ works |
| Reduction after the **inner** matmul only (`av @ sv.T`) | ❌ NaN |
| Reduction after the **outer** matmul (`bv @ _i`), before `.T` | ✅ works |
| Whole chain computed in **bf16**, result cast to fp32 | ✅ works |
| `x + (chain) * 0.0` | ❌ **NaN** — ×0 cannot create NaN, so the chain itself is NaN |
| `x + torch.zeros_like(x)` | ✅ works (identical rms to no-injection) |

The `* 0.0` result is the decisive one: it proves the product is already NaN before it
reaches the add, so this is not an add/fusion issue and not about the delta's values.

## The fix

Compute the low-rank update in the LoRA's **native dtype** and cast only the result:

```python
cd = a.dtype if a.dtype.is_floating_point else x.dtype
av = a.to(x.device, cd)
bv = b.to(x.device, cd)
sv = st.to(x.device, cd)
x = x + (bv @ (av @ sv.T)).T.to(x.dtype)
```

This avoids the bad kernel and is also **cheaper** — `out` is ~96k rows, so the fp32
upcast was the most expensive operand in the expression and added no information
(`a`, `b` and the shipped grid are all bf16 on disk).

Verified: `denoised_rms` = 1.4552, 1.2989, 1.1620, 1.1129, 1.1098, 1.1228 (finite),
valid video + 32 kHz audio, correct four-season content.

## Ruled out

| Hypothesis | Test | Result |
|---|---|---|
| Bad LoRA checkpoint | v4-600 **and** v1-850 | both fail |
| bypass-vs-merge path | `low_vram=True` | fails |
| `ComfyUI-AppleSilicon-FP8` | node fully removed | fails |
| Its Metal INT8 kernel | `ASFP8_INT8_EXT=off` | fails |
| In-place add in `_FrugalLoRA` | out-of-place `torch.add` | fails |
| Non-contiguous transposed operand | `.T.contiguous()` | fails |
| Matmul associativity / output transpose | `(sv @ av.T) @ bv.T` | fails |
| Operand lifetime / buffer recycling | keep `av`,`bv`,`sv` alive past the forward | fails |
| Host readback | `_d.view(-1)[0].item()` | fails |
| Command-queue ordering | `torch.mps.synchronize()` | fails |
| Materialising copy | `.clone()`, `torch.add(out=)`, `* 1.0` | fail |
| Memory pressure / size | 256×256, 416×736, 124f and 192f | fails at **all** sizes |

Note `torch.mps.synchronize()` and a host readback both fail while any **reduction**
works — reductions are the only ops here that force the product to be evaluated as a
standalone tensor rather than folded into the consuming graph.

## Not reproducible standalone

The same expression, at the exact shapes and dtypes (`[96768,16] @ [16,1]` fp32,
`x` from a `Linear(8, 96768)`), run on MPS outside the model — including under
`inference_mode`, with 51 concurrent modules, and with deferred NaN checks so the check
itself introduces no barrier — produces **zero** NaN over hundreds of iterations.

So the trigger requires something about the surrounding DiT graph that I could not
isolate. The in-model reproduction is 100% deterministic at every size tested; the
standalone one never fires. That gap is unexplained, and it is the reason this is
reported as a characterised workaround rather than a full PyTorch-level diagnosis.
