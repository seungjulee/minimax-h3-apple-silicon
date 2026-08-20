# MiniMax H3 / Music3 on Apple Silicon — ComfyUI notes, fixes, and tools

Running MiniMax H3 (video) and MiniMax Music3 (music) locally through ComfyUI on
Apple Silicon (M1–M5), and the bugs found and fixed along the way.

## What's here

- **`scripts/h3_lora_mps_test.sh`** — self-contained repro/fix-verification harness
  for the Turbo LoRA NaN bug. Bootstraps ComfyUI + models from scratch on any
  Apple Silicon Mac, then can reproduce the bug, apply candidate fixes, or run a
  no-LoRA control. See usage comment at the top of the file.
- **`scripts/test_pr107_mps_int8.sh`** — verifies
  [comfy-kitchen#107](https://github.com/Comfy-Org/comfy-kitchen/pull/107) (MPS
  INT8 fallback) against an existing install, with the third-party accel node
  disabled so only the PR's own code is under test.
- **`docs/01-mps-nan-investigation.md`** — the elimination trail for the Turbo
  LoRA NaN bug: every hypothesis tested, what worked, what didn't, and the parts
  that remain unexplained.
- **`docs/02-pr-body-fix-nan.md`** — the PR description filed upstream.
- **`comfyui_desktop/`** — a small native macOS (Swift/AppKit + WKWebView) shell
  that starts/stops the local ComfyUI server and shows it in a real window, for
  when the full Electron ComfyUI Desktop app doesn't make sense (it currently
  pins ComfyUI 0.22.3, which predates H3 support).

## Upstream contributions

- [Larryvrh/ComfyUI-MiniMax-H3-Turbo#26](https://github.com/Larryvrh/ComfyUI-MiniMax-H3-Turbo/pull/26) —
  fix for the Turbo LoRA producing black frames / NaN audio on MPS with a pruned
  base. Verified on M1 Max and M5 Max.
- [Comfy-Org/comfy-kitchen#107](https://github.com/Comfy-Org/comfy-kitchen/pull/107#issuecomment-5286104533) —
  independent verification (not authored here) that the MPS INT8 fallback works
  on both M1 Max (float fallback path) and M5 Max (fused Metal 4 path).

## Known-good local stack (M5 Max, 128GB)

| Component | Source |
|---|---|
| ComfyUI | `comfyanonymous/ComfyUI`, upstream |
| `ComfyUI-MiniMax-H3-Turbo` | fork/patch in this repo's PR #26 (LoRA NaN fix) |
| `ComfyUI-AppleSilicon-FP8` | `pawel-mazurkiewicz/ComfyUI-AppleSilicon-FP8` — required for INT8 on MPS until comfy-kitchen#107 merges |
| `ComfyUI-SolAttn-MPS` | `yshenaw/ComfyUI-SolAttn-MPS` — optional attention accelerator, opt-in per-workflow |

## Models

- H3: `Comfy-Org/MiniMax-H3` — `fl2va_pruned_int8_convrot` DiT + `nvfp4_awq` text
  encoder + fp16/fp32 VAEs. INT8 needs the AppleSilicon-FP8 node above (or
  comfy-kitchen#107 once merged) — otherwise it fails with
  `NotImplementedError: aten::_int_mm`.
- Music3: `Comfy-Org/MiniMax-Music-3` — fp16 DiT recommended over the int8
  variant on Apple Silicon for the same reason.

### LTX-2.5 (video + audio)

Confirmed working, M5 Max, 512x256/2s sanity render, 90s total, first attempt.
Script: `scripts/ltx25/ltx25.py`. Hand-built API graph (not exported from the
UI) because the bundled `video_ltx2_5_t2v` template's prompt-rewriter cluster
(`TextGenerateLTX2Prompt` + a small `gemma4_e2b_it_int8_convrot.safetensors`
CLIP loader) depends on a file that **does not exist at any resolvable
location** — not in `Lightricks/LTX-2.5` (confirmed via `HfApi.model_info`),
not in any `Comfy-Org/ltx-*` repo, not findable via HF search, and the UI's
own "Download" button silently no-ops without a Comfy.org login. The
prompt-rewriter is a convenience feature (auto-expands short prompts), not
load-bearing — the fix is to skip that whole node cluster and wire your own
prompt text directly into `CLIPTextEncode`, which is what the script does.

The real pipeline (traced from the template's subgraph JSON, then verified
node-by-node against `/object_info` to get exact input names before wiring
anything — a hand-built two-stage graph is easy to get subtly wrong) is a
**two-stage** sampler: base-resolution `LTXVDualCFGGuider` + `ManualSigmas`
(9-value schedule, ~8 steps) → `LTXVLatentUpsampler` (2x) → a second pass at
upscaled resolution (4-value schedule, ~3 steps) → `VAEDecodeTiled`. Video and
audio run as separate latents joined via `LTXVConcatAVLatent` /
`LTXVSeparateAVLatent` at each stage boundary.

## Known issues on Apple Silicon

- **Turbo LoRA NaN on MPS** — fixed, see PR #26 above.
- **INT8 `aten::_int_mm` unimplemented on MPS** — fixed by comfy-kitchen#107
  (unmerged) or the AppleSilicon-FP8 node.
- **Memory pressure**: H3 needs the DiT (~20GB) + text encoder (~15GB) + VAEs
  resident at once; Music3's DiT+encoder is a further ~23GB. Heavy concurrent
  load (other local LLM servers, browsers) can starve the render process enough
  that it appears to hang indefinitely rather than fail cleanly — watch `swap`
  usage, not just RAM, if a render seems stuck at model load or first-kernel-compile.

## Update log

### 2026-08-20 — Stack upgrade + Music3 verified working

- **ComfyUI**: `0.33.0` → `0.33.3`. Frontend `1.48.7` → `1.49.6`, workflow
  templates `0.11.41` → `0.11.44`. `comfy-kitchen` was already at latest (`0.2.31`).
- **`ComfyUI-MiniMax-H3-Turbo`**: pulled latest (picks up upstream PR #16, a
  separate fix for a row-count mismatch in the adaln injection — unrelated to
  and non-conflicting with the NaN fix in this repo's PR #26). Our patch was
  reapplied after the pull and reverified with a real render.
- **New: `ComfyUI-SolAttn-MPS`** (`yshenaw/ComfyUI-SolAttn-MPS`) — an opt-in
  Metal attention accelerator for H3 with real end-to-end benchmarks on M3 Ultra
  (1.3–1.8x at 480p/720p). Installed but not yet benchmarked on M5 Max here.
  It's a graph node (`Patch Sol-Attn`), not automatic — existing workflows are
  unaffected until it's explicitly wired in.
- **Music3 confirmed working end-to-end**: `minimax_music3_dit_fp16` +
  `minimax_music3_text_encoder_bf16` (deliberately not the `int8_convrot`
  variants — same reasoning as H3: quantized paths are exactly where the MPS
  bugs live, and there's no memory pressure to justify them on 128GB). Three
  30s tracks rendered successfully, 185–235s each. Scripts in `scripts/music3/`.
  Note the official template's `SaveAudioAdvanced` node needs `format.quality`
  as a *separate* nested key, not a flat `quality` field — the flat form fails
  ComfyUI's schema validation.

### Gotcha: renders can hang indefinitely under memory pressure, not fail cleanly

Hit this twice while restarting after the upgrade: the process would sit at
"compiling the INT8 Metal kernel" at 0% CPU indefinitely, then die with a
generic "Global interrupt" on the next request. Root cause was unrelated —
`swap` was at ~13GB/13GB (essentially full) because of an unrelated 33GB
`llama-server` process plus browsers competing for the same unified memory
pool as ComfyUI's ~35GB of resident weights. Once that process exited and swap
dropped to 0, the identical render completed in 45s. **If a render seems stuck
at model load or first-kernel-compile with no CPU activity, check `sysctl
vm.swapusage` before assuming the code regressed.**
