#!/usr/bin/env bash
# =============================================================================
# Test comfy-kitchen PR #107 ("Add MPS fallback for INT8 linear") on your Mac.
#   PR:    https://github.com/Comfy-Org/comfy-kitchen/pull/107
#   Issue: https://github.com/Comfy-Org/comfy-kitchen/issues/92
#
# WHY THIS MATTERS
#   torch has no `aten::_int_mm` on MPS, so INT8 checkpoints crash instantly on
#   Apple Silicon with stock ComfyUI. PR #107 adds an MPS path. It ships fused
#   Metal 4 kernels *plus* a floating-point fallback -- and Metal 4 tensor ops
#   need M5-class hardware, so on M1/M2/M3/M4 this exercises the FALLBACK, which
#   nobody in the PR thread has tested yet. That is the useful data point.
#
# WHAT IT DOES
#   1. reuses ~/h3-lora-repro (from h3_lora_mps_test.sh) or sets it up
#   2. backs up comfy_kitchen, applies PR #107
#   3. moves ComfyUI-AppleSilicon-FP8 OUT of custom_nodes, so nothing else can
#      supply an INT8 path -- this is the whole point of the test
#   4. runs plain H3, NO LoRA, 256x256/22f/6steps
#   5. restores everything it touched
#
#   bash test_pr107_mps_int8.sh
#
# Verified on M5 Max: without PR #107 -> NotImplementedError aten::_int_mm
#                     with    PR #107 -> renders fine, no third-party node
# =============================================================================
set -euo pipefail

ROOT="${H3_REPRO_ROOT:-$HOME/h3-lora-repro}"
COMFY="$ROOT/ComfyUI"
PORT="${H3_PORT:-8188}"
step(){ printf "\n\033[1;36m==> %s\033[0m\n" "$1"; }
ok(){   printf "    \033[32m%s\033[0m\n" "$1"; }
warn(){ printf "    \033[33m%s\033[0m\n" "$1"; }
die(){  printf "\n\033[31mERROR: %s\033[0m\n" "$1" >&2; exit 1; }

printf "\033[1;34mtest_pr107_mps_int8.sh  rev 2\033[0m\n"
[[ "$(uname -m)" == "arm64" ]] || die "Apple Silicon only."
[[ -d "$COMFY" ]] || die "No ComfyUI at $COMFY. Run h3_lora_mps_test.sh first (it downloads the models)."

# shellcheck disable=SC1091
source "$COMFY/venv/bin/activate"
SP="$(python -c 'import site;print(site.getsitepackages()[0])')"
CK="$SP/comfy_kitchen"
[[ -d "$CK" ]] || die "comfy_kitchen not found in $SP"
ok "comfy_kitchen: $CK  (v$(pip show comfy-kitchen 2>/dev/null | awk '/^Version/{print $2}'))"

FFPROBE="$(brew --prefix)/bin/ffprobe"; [[ -x "$FFPROBE" ]] || FFPROBE=ffprobe
FFMPEG="$(brew --prefix)/bin/ffmpeg";   [[ -x "$FFMPEG"  ]] || FFMPEG=ffmpeg

DISABLED="$HOME/.h3_disabled_nodes"
restore() {
  step "Restoring your setup"
  [[ -d /tmp/ck_backup_$$ ]] && { rm -rf "$CK"; cp -R /tmp/ck_backup_$$ "$CK"; rm -rf /tmp/ck_backup_$$; ok "comfy_kitchen restored"; }
  if [[ -d "$DISABLED/ComfyUI-AppleSilicon-FP8" ]]; then
    mv "$DISABLED/ComfyUI-AppleSilicon-FP8" "$COMFY/custom_nodes/" && ok "AppleSilicon-FP8 restored"
  fi
  pkill -f "main.py --listen 127.0.0.1 --port $PORT" 2>/dev/null || true
}
trap restore EXIT

step "1/5  Backing up comfy_kitchen"
cp -R "$CK" /tmp/ck_backup_$$ && ok "backup at /tmp/ck_backup_$$"

step "2/5  Applying PR #107"
rm -rf /tmp/ck107_$$ && git clone -q --depth 1 --branch codex/mps-int8-bf16-fallback \
  https://github.com/ikeyan/comfy-kitchen.git /tmp/ck107_$$ || die "clone of PR branch failed"
cp /tmp/ck107_$$/comfy_kitchen/backends/eager/mps_int8.py "$CK/backends/eager/mps_int8.py"
ok "mps_int8.py installed"
curl -fsSL "https://patch-diff.githubusercontent.com/raw/Comfy-Org/comfy-kitchen/pull/107.diff" -o /tmp/pr107_$$.diff \
  || die "could not fetch PR diff"
awk '/^diff --git a\/comfy_kitchen\/backends\/eager\/quantization.py/,/^diff --git a\/tests/' /tmp/pr107_$$.diff \
  | sed '$d' > /tmp/q_$$.patch
( cd "$SP" && patch -p1 --forward --silent < /tmp/q_$$.patch ) \
  && ok "quantization.py patched" \
  || warn "patch reported conflicts - continuing, but results may be unreliable"
python - <<'PYCHK' || die "patched comfy_kitchen does not import"
from comfy_kitchen.backends.eager import quantization as q
assert hasattr(q, "_mps_int8_linear"), "PR #107 hook missing"
print("    import OK, _mps_int8_linear present")
PYCHK

step "3/5  Disabling ComfyUI-AppleSilicon-FP8 (so ONLY PR #107 can provide INT8)"
mkdir -p "$DISABLED"
if [[ -d "$COMFY/custom_nodes/ComfyUI-AppleSilicon-FP8" ]]; then
  mv "$COMFY/custom_nodes/ComfyUI-AppleSilicon-FP8" "$DISABLED/" && ok "moved out of custom_nodes"
else
  ok "not installed - nothing to disable"
fi
# a dot-prefixed dir is NOT skipped by ComfyUI 0.33, so it must leave the tree
git -C "$COMFY/custom_nodes/ComfyUI-MiniMax-H3-Turbo" checkout -- __init__.py 2>/dev/null || true

step "4/5  Starting ComfyUI and rendering plain H3 (no LoRA)"
pkill -f "main.py --listen 127.0.0.1 --port $PORT" 2>/dev/null || true; sleep 6
( cd "$COMFY" && nohup python main.py --listen 127.0.0.1 --port "$PORT" > "$ROOT/pr107.log" 2>&1 & )
printf "    waiting for server "
for _ in $(seq 1 150); do sleep 3; printf "."
  curl -s "http://127.0.0.1:$PORT/system_stats" >/dev/null 2>&1 && break; done; echo
curl -s "http://127.0.0.1:$PORT/system_stats" >/dev/null 2>&1 || die "server did not start; see $ROOT/pr107.log"
N=$(grep -ci "AppleSilicon-FP8" "$ROOT/pr107.log" || true)
[[ "$N" == "0" ]] && ok "confirmed: no third-party accel node loaded" \
                  || die "AppleSilicon-FP8 still loaded ($N lines) - test would be invalid"

export H3_COMFY_ROOT="$COMFY" H3_PORT="$PORT" H3_FFPROBE="$FFPROBE" H3_FFMPEG="$FFMPEG"
python - <<'PYRUN'
import json, os, subprocess, sys, time, urllib.request
API = "http://127.0.0.1:" + os.environ.get("H3_PORT", "8188")
COMFY = os.environ["H3_COMFY_ROOT"]
FFPROBE, FFMPEG = os.environ["H3_FFPROBE"], os.environ["H3_FFMPEG"]
def sh(c): return subprocess.run(c, shell=True, capture_output=True, text=True).stdout.strip()

print("\n" + "="*74)
print("ENVIRONMENT   <-- paste into the PR/issue")
print("="*74)
for k, v in {
  "chip":   sh("system_profiler SPHardwareDataType 2>/dev/null | awk -F': ' '/Chip/{print $2}'"),
  "memory": sh("system_profiler SPHardwareDataType 2>/dev/null | awk -F': ' '/Memory/{print $2}'"),
  "macOS":  sh("sw_vers -productVersion"),
}.items(): print(f"  {k:16s} {v}")
import torch; print(f"  {'torch':16s} {torch.__version__}")
print(f"  {'comfyui':16s} " + json.load(urllib.request.urlopen(API+'/system_stats'))['system']['comfyui_version'])
print(f"  {'comfy_kitchen':16s} " + sh("pip show comfy-kitchen | awk '/^Version/{print $2}'") + " + PR #107")

G = {
 "6":  {"class_type":"UNETLoader","inputs":{"unet_name":"minimax_h3_fl2va_pruned_int8_convrot.safetensors","weight_dtype":"default"}},
 "13": {"class_type":"CLIPLoader","inputs":{"clip_name":"qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors","type":"minimax","device":"default"}},
 "11": {"class_type":"VAELoader","inputs":{"vae_name":"minimax_h3_video_vae_fp16.safetensors"}},
 "24": {"class_type":"VAELoader","inputs":{"vae_name":"minimax_h3_audio_vae_fp32.safetensors"}},
 "104":{"class_type":"MiniMaxH3ImageToVideo","inputs":{"clip":["13",0],"vae":["11",0],
        "prompt":"A red fox walking through fresh snow at dawn.","width":256,"height":256,"length":22}},
 "16": {"class_type":"BasicGuider","inputs":{"model":["6",0],"conditioning":["104",0]}},
 "17": {"class_type":"MiniMaxH3TurboSampler","inputs":{}},
 "9":  {"class_type":"BasicScheduler","inputs":{"model":["6",0],"scheduler":"simple","steps":6,"denoise":1.0}},
 "15": {"class_type":"RandomNoise","inputs":{"noise_seed":42}},
 "14": {"class_type":"SamplerCustomAdvanced","inputs":{"noise":["15",0],"guider":["16",0],"sampler":["17",0],
        "sigmas":["9",0],"latent_image":["104",1]}},
 "10": {"class_type":"VAEDecode","inputs":{"samples":["14",0],"vae":["11",0]}},
 "23": {"class_type":"VAEDecodeAudio","inputs":{"samples":["14",0],"vae":["24",0]}},
 "91": {"class_type":"CreateVideo","inputs":{"images":["10",0],"fps":24.0,"audio":["23",0]}},
 "92": {"class_type":"SaveVideo","inputs":{"video":["91",0],"filename_prefix":"video/pr107_test","format":"auto","codec":"auto"}},
}
print("\n" + "="*74)
print("RUNNING  plain H3, NO LoRA, 256x256/22f/6steps, INT8 checkpoint")
print("="*74)
t0 = time.time()
try:
    pid = json.load(urllib.request.urlopen(urllib.request.Request(API+"/prompt",
          data=json.dumps({"prompt":G,"client_id":"pr107"}).encode(),
          headers={"Content-Type":"application/json"})))["prompt_id"]
except urllib.error.HTTPError as e:
    print("SETUP ERROR:", e.read().decode()[:600]); sys.exit(1)
while time.time()-t0 < 3600:
    time.sleep(5)
    q = json.load(urllib.request.urlopen(API+"/queue"))
    if not q.get("queue_running") and not q.get("queue_pending"): break
el = time.time()-t0

log = open(os.path.join(os.path.dirname(COMFY.rstrip("/")), "pr107.log"), errors="ignore").read()
hist = json.load(urllib.request.urlopen(f"{API}/history/{pid}")).get(pid, {})
files = [os.path.join(f.get("subfolder",""), f["filename"])
         for o in hist.get("outputs",{}).values() for f in o.get("images",[])]

print("\n" + "="*74); print("RESULT"); print("="*74)
if "_int_mm" in log and "NotImplementedError" in log:
    print(f"  VERDICT: *** PR #107 DID NOT FIX IT *** ({el:.0f}s)")
    print("  Still hitting aten::_int_mm:")
    for l in [x for x in log.splitlines() if "_int_mm" in x or "NotImplementedError" in x][-4:]:
        print("    " + l[:170])
    sys.exit(0)
if not files:
    print(f"  VERDICT: *** FAILED (no output) *** ({el:.0f}s)")
    for l in [x for x in log.splitlines() if x.strip()][-20:]: print("    " + l[:170])
    sys.exit(0)
p = os.path.join(COMFY, "output", files[0])
size = os.path.getsize(p)
aud  = sh(f'"{FFPROBE}" -v error -select_streams a -show_entries stream=codec_name -of csv=p=0 "{p}" | head -1')
vid  = sh(f'"{FFPROBE}" -v error -select_streams v:0 -show_entries stream=codec_name -of csv=p=0 "{p}" | head -1')
subprocess.run(f'"{FFMPEG}" -v error -i "{p}" -frames:v 1 /tmp/_pr107f.png -y', shell=True, capture_output=True)
fr = os.path.getsize("/tmp/_pr107f.png") if os.path.exists("/tmp/_pr107f.png") else 0
print(f"  file {size} B   video={vid or 'NONE'}  audio={aud or 'NONE'}  frame={fr} B")
# LOG is authoritative: the failure this PR fixes is an exception, and some
# machines have an ffprobe that cannot parse a perfectly good output file.
nan_hits = log.count("denoised_rms=nan")
executed = "Prompt executed" in log
if executed and nan_hits == 0:
    print(f"\n  VERDICT: *** PR #107 WORKS *** ({el:.0f}s)")
    print("  INT8 H3 renders on MPS with NO third-party accel node.")
    if not (vid and fr > 15000):
        print("  (ffprobe could not parse the file, but the render completed cleanly -")
        print(f"   open it and check visually: {p})")
else:
    print(f"\n  VERDICT: *** ran, but did not complete cleanly *** ({el:.0f}s)")
    print(f"  denoised_rms=nan x{nan_hits}, prompt_executed={executed}")
PYRUN

step "5/5  Done"
cat <<EOF

Please post the ENVIRONMENT block + VERDICT at:
  https://github.com/Comfy-Org/comfy-kitchen/pull/107
(especially useful on M1-M4: those lack Metal 4 tensor ops, so they exercise
 PR #107's floating-point FALLBACK, which has not been tested yet.)
EOF
