# H3 vs LTX-2.5 — matched performance comparison (M5 Max)

Date: 2026-08-20

## Setup

Matched as tightly as the two pipelines allow: same resolution, same target
duration, same seed, each model run in its own fast-mode configuration
(Turbo LoRA for H3, default two-stage distilled schedule for LTX-2.5).

| Parameter | Value |
|---|---|
| Resolution | 512x512 |
| Target duration | ~2.3s @ 24fps |
| Seed | 777 |
| Hardware | M5 Max, 128GB unified memory |

H3 config: `MiniMaxH3TurboLoRA` (`minimax_h3_turbo_v4_step600_ema.safetensors`,
strength 1.0) + `MiniMaxH3TurboSampler`, 6 steps.

LTX-2.5 config: default two-stage distilled schedule — 8 steps at base
resolution (`STAGE1_SIGMAS`), 2x latent upscale, 3 steps at full resolution
(`STAGE2_SIGMAS`), 11 steps total.

Scripts: `scripts/h3_lora_mps_test.sh`-adjacent `render.py` (H3) and
`scripts/ltx25/ltx25.py` (LTX-2.5), both in `~/code/h3_scratch/` locally
(not yet checked into this repo).

## Important caveat: prompts were not matched

`render.py`'s H3 builder has a **hardcoded prompt baked into the script**
(a static-camera Kyoto temple scene, left over from earlier work) — it does
not take a `prompt` argument the way `ltx25.py` does. LTX-2.5 was given a
distinct prompt (a fox walking through snow at dawn). Both happen to be
winter scenes, which makes the outputs superficially comparable, but this
is a **timing/compute comparison only** — not a controlled content A/B.
Fixing this (parameterizing `render.py`'s prompt) is a follow-up if a real
quality A/B is wanted.

## Results

| | MiniMax H3 (Turbo LoRA) | LTX-2.5 (2-stage distilled) |
|---|---|---|
| Wall time | **496s** (8m16s) | **317s** (5m17s) |
| Steps | 6 (Turbo) | 8 stage1 + 3 stage2 = 11 |
| Frames | 56 @ 24fps (2.333s) | 49 @ 24fps (2.042s) |
| Output | 512x512, h264+aac | 512x512, h264+aac |
| File size | 231KB | 292KB |

## Performance

LTX-2.5 is **~1.6x faster** than H3 at matched 512x512 resolution — 317s vs
496s — despite running nearly double the total denoising steps (11 vs 6).
LTX-2.5's two-stage architecture (cheap base-resolution pass + upscale) is
evidently cheaper per-step at this size than H3's Turbo path on MPS, or the
Turbo LoRA carries its own per-step overhead here.

This is consistent with the earlier finding that LTX-2.5's compute scales
sub-linearly with resolution (fixed per-render overhead — model load, text
encoding, VAE setup — dominates at small sizes; see the LTX-2.5 section of
the main README). Expect the timing gap between the two models to compress
at higher resolutions, since fixed overhead becomes a smaller fraction of
total time for both.

## Quality

Not a fair head-to-head given the mismatched prompts above — per-model
impressions only:

- **H3** (static-camera temple scene): sharp architectural detail, clean
  snow texture, coherent geometry — an easy case (locked-off camera, no
  complex subject motion).
- **LTX-2.5** (fox walking through snow): motion reads naturally and the
  gait is smooth, but the frame is noticeably softer/blurrier than H3's.
  Could be a genuine model-quality difference, or an artifact of skipping
  LTX-2.5's prompt-rewriter node cluster (see main README — the rewriter
  file has no resolvable source, so this pipeline feeds raw prompt text
  directly into `CLIPTextEncode` instead of the expanded form the model
  was likely tuned against).

## Bottom line

One matched data point at one resolution/duration, with mismatched prompt
content, is not a benchmark. Treat the 496s vs 317s timing as real and
reproducible (same seed, same hardware, same session) — it is the strongest
claim this data supports. Don't read the quality notes above as more than
a first look.
