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

## Known issues on Apple Silicon

- **Turbo LoRA NaN on MPS** — fixed, see PR #26 above.
- **INT8 `aten::_int_mm` unimplemented on MPS** — fixed by comfy-kitchen#107
  (unmerged) or the AppleSilicon-FP8 node.
- **Memory pressure**: H3 needs the DiT (~20GB) + text encoder (~15GB) + VAEs
  resident at once; Music3's DiT+encoder is a further ~23GB. Heavy concurrent
  load (other local LLM servers, browsers) can starve the render process enough
  that it appears to hang indefinitely rather than fail cleanly — watch `swap`
  usage, not just RAM, if a render seems stuck at model load or first-kernel-compile.
