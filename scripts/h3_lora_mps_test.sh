#!/usr/bin/env bash
# =============================================================================
# MiniMax-H3 Turbo LoRA -- Apple Silicon NaN reproduction.
# Ref: https://github.com/Larryvrh/ComfyUI-MiniMax-H3-Turbo/issues/14
#
# SELF-CONTAINED. Run this on a fresh Mac that has nothing but Homebrew.
# It installs everything it needs, downloads the models, starts ComfyUI,
# runs one tiny render, and tells you whether the LoRA is broken on your chip.
#
#   curl -fsSL <raw-url> -o h3_lora_mps_test.sh && bash h3_lora_mps_test.sh
#
# Then, to check whether the proposed patch fixes it on your machine (fast --
# everything is already downloaded by the first run):
#
#   bash h3_lora_mps_test.sh --fix
#
# Or test every candidate fix in one go (each needs its own server restart,
# handled automatically) -- this is the useful one right now:
#
#   bash h3_lora_mps_test.sh --candidates
#
# CONTROL: does plain H3 work here at all, with no LoRA involved? If this fails
# too, the machine has a more basic problem and LoRA results mean nothing:
#
#   bash h3_lora_mps_test.sh --no-lora
#
# It installs (only if missing):  python@3.12, ffmpeg, git  (via Homebrew)
# It creates:                     ~/h3-lora-repro/   (ComfyUI + venv + models)
# It does NOT touch:              any existing ComfyUI, any system Python
# To undo everything:             rm -rf ~/h3-lora-repro
#
# Needs ~50 GB free disk. Models are ~43 GB, all public, no token/API key.
# First run: ~30-90 min, nearly all of it downloading. Re-runs resume/skip.
# =============================================================================
set -euo pipefail

MODE="baseline"
for a in "$@"; do
  case "$a" in
    --fix|--candidates|--no-lora|--fp8|--bf16) ;;
    *) printf "\033[31mUnknown argument: %s\033[0m\n" "$a"
       printf "Valid: --fix | --candidates | --no-lora [--fp8|--bf16]\n"; exit 2 ;;
  esac
done
case "${1:-}" in
  --fix)        MODE="fix" ;;
  --candidates) MODE="candidates" ;;
  --no-lora)    MODE="nolora" ;;
esac
case "${2:-${1:-}}" in
  --fp8)  VARIANT="fp8"  ;;
  --bf16) VARIANT="bf16" ;;
  *)      VARIANT="${VARIANT:-int8}" ;;
esac

ROOT="${H3_REPRO_ROOT:-$HOME/h3-lora-repro}"
COMFY="$ROOT/ComfyUI"
PORT="${H3_PORT:-8188}"

step() { printf "\n\033[1;36m==> %s\033[0m\n" "$1"; }
ok()   { printf "    \033[32m%s\033[0m\n" "$1"; }
warn() { printf "    \033[33m%s\033[0m\n" "$1"; }
die()  { printf "\n\033[31mERROR: %s\033[0m\n" "$1" >&2; exit 1; }

printf "\033[1;34mh3_lora_mps_test.sh  rev 11  (mode: %s)\033[0m\n" "$MODE"
[[ "$(uname -s)" == "Darwin" ]] || die "macOS only."
[[ "$(uname -m)" == "arm64"  ]] || die "Apple Silicon only (this bug is MPS-specific)."
command -v brew >/dev/null 2>&1 || die "Homebrew required. Install from https://brew.sh then re-run."

AVAIL_GB=$(df -g / | awk 'NR==2{print $4}')
[[ "${AVAIL_GB:-0}" -ge 50 ]] || warn "only ${AVAIL_GB}GB free; ~50GB recommended"

# ---------------------------------------------------------------- 1. packages
step "1/6  Installing prerequisites (skipped if present)"
for pkg in python@3.12 ffmpeg git; do
  if brew list --versions "$pkg" >/dev/null 2>&1; then
    ok "$pkg already installed"
  else
    echo "    brew install $pkg ..."
    brew install "$pkg" >/dev/null || die "failed to install $pkg"
    ok "$pkg installed"
  fi
done
PY312="$(brew --prefix python@3.12 2>/dev/null)/libexec/bin/python3"
[[ -x "$PY312" ]] || PY312="$(brew --prefix)/bin/python3.12"
[[ -x "$PY312" ]] || die "python3.12 not found after install"
ok "$($PY312 --version)"
# Use Homebrew's binaries explicitly: a conda/venv on PATH can shadow them with a
# stripped build that cannot decode H.264, which silently breaks verification.
FFPROBE="$(brew --prefix)/bin/ffprobe"; [[ -x "$FFPROBE" ]] || FFPROBE="$(command -v ffprobe || true)"
FFMPEG="$(brew --prefix)/bin/ffmpeg";   [[ -x "$FFMPEG"  ]] || FFMPEG="$(command -v ffmpeg  || true)"
[[ -x "$FFPROBE" ]] || die "no usable ffprobe"
ok "ffprobe: $FFPROBE"

# ---------------------------------------------------------------- 2. ComfyUI
step "2/6  Installing ComfyUI into $COMFY"
mkdir -p "$ROOT"
if [[ -d "$COMFY/.git" ]]; then
  ok "already present"
else
  git clone --depth 1 https://github.com/comfyanonymous/ComfyUI.git "$COMFY" >/dev/null 2>&1 \
    || die "git clone of ComfyUI failed"
  ok "cloned"
fi

# ------------------------------------------------------------- 3. python env
step "3/6  Creating venv and installing PyTorch + ComfyUI deps"
[[ -d "$COMFY/venv" ]] || "$PY312" -m venv "$COMFY/venv"
# shellcheck disable=SC1091
source "$COMFY/venv/bin/activate"
python -m pip install --quiet --upgrade pip
if python -c "import torch" 2>/dev/null; then
  ok "torch $(python -c 'import torch;print(torch.__version__)') already installed"
else
  echo "    installing torch (several minutes) ..."
  pip install --quiet torch torchvision torchaudio || die "torch install failed"
  ok "torch installed"
fi
python - <<'PYCHK' || die "PyTorch reports MPS unavailable - cannot reproduce an MPS bug here."
import sys, torch
sys.exit(0 if torch.backends.mps.is_available() else 1)
PYCHK
ok "MPS available"
echo "    installing ComfyUI requirements ..."
pip install --quiet -r "$COMFY/requirements.txt" || die "ComfyUI requirements failed"
pip install --quiet huggingface_hub
ok "dependencies ready"

# ------------------------------------------------------------ 4. custom node
step "4/6  Installing ComfyUI-MiniMax-H3-Turbo"
NODE_DIR="$COMFY/custom_nodes/ComfyUI-MiniMax-H3-Turbo"
if [[ -d "$NODE_DIR/.git" ]]; then
  ok "already present"
else
  git clone --depth 1 https://github.com/Larryvrh/ComfyUI-MiniMax-H3-Turbo "$NODE_DIR" >/dev/null 2>&1 \
    || die "git clone of the Turbo node failed"
  ok "cloned"
fi
NODE_COMMIT="$(git -C "$NODE_DIR" rev-parse --short HEAD 2>/dev/null || echo unknown)"
ok "node commit $NODE_COMMIT"

# INT8/FP8 checkpoints do not run on MPS without this compatibility layer.
ASFP8="$COMFY/custom_nodes/ComfyUI-AppleSilicon-FP8"
if [[ -d "$ASFP8/.git" ]]; then
  ok "ComfyUI-AppleSilicon-FP8 already present"
else
  git clone --depth 1 https://github.com/pawel-mazurkiewicz/ComfyUI-AppleSilicon-FP8 "$ASFP8" >/dev/null 2>&1 \
    && ok "ComfyUI-AppleSilicon-FP8 cloned (needed for INT8 on MPS)" \
    || warn "could not clone ComfyUI-AppleSilicon-FP8"
fi
[[ -f "$ASFP8/requirements.txt" ]] && pip install --quiet -r "$ASFP8/requirements.txt" 2>/dev/null || true

# ---------------------------------------------------------------- 5. weights
step "5/6  Downloading models (~43 GB, public, resumable)"
mkdir -p "$COMFY/models"/{diffusion_models,text_encoders,vae,loras}
get () { # repo  path-in-repo  local-dir  friendly-name
  if [[ -f "$3/$(basename "$2")" ]]; then ok "have $4"; return; fi
  echo "    downloading $4 ..."
  hf download "$1" "$2" --local-dir "$COMFY/models" >/dev/null || die "download failed: $4"
}
case "$VARIANT" in
  fp8)  DIT_FILE="minimax_h3_fl2va_pruned_fp8_scaled.safetensors";  DIT_DESC="DiT pruned fp8 (21 GB)" ;;
  bf16) DIT_FILE="minimax_h3_fl2va_pruned_bf16.safetensors";        DIT_DESC="DiT pruned bf16 (40 GB)" ;;
  *)    DIT_FILE="minimax_h3_fl2va_pruned_int8_convrot.safetensors"; DIT_DESC="DiT pruned int8 (21 GB)" ;;
esac
ok "checkpoint variant: $VARIANT -> $DIT_FILE"
get Comfy-Org/MiniMax-H3 "diffusion_models/$DIT_FILE" \
    "$COMFY/models/diffusion_models" "$DIT_DESC"
get Comfy-Org/MiniMax-H3 text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors \
    "$COMFY/models/text_encoders" "text encoder (16 GB)"
get Comfy-Org/MiniMax-H3 vae/minimax_h3_video_vae_fp16.safetensors \
    "$COMFY/models/vae" "video VAE (5 GB)"
get Comfy-Org/MiniMax-H3 vae/minimax_h3_audio_vae_fp32.safetensors \
    "$COMFY/models/vae" "audio VAE (0.6 GB)"
if [[ -f "$COMFY/models/loras/minimax_h3_turbo_v4_step600_ema.safetensors" ]]; then
  ok "have turbo LoRA"
else
  echo "    downloading turbo LoRA (0.8 GB) ..."
  hf download larryvrh/MiniMax-H3-Turbo-Lora minimax_h3_turbo_v4_step600_ema.safetensors \
      --local-dir "$COMFY/models/loras" >/dev/null || die "LoRA download failed"
fi
ok "models ready"

cat > "$ROOT/_run_test.py" <<'PYEOF'
import json, os, platform, subprocess, sys, time, urllib.request
API  = "http://127.0.0.1:" + os.environ.get("H3_PORT", "8188")
COMFY = os.environ["H3_COMFY_ROOT"]

def sh(c):
    return subprocess.run(c, shell=True, capture_output=True, text=True).stdout.strip()

FFPROBE = os.environ.get("H3_FFPROBE", "ffprobe")
FFMPEG  = os.environ.get("H3_FFMPEG", "ffmpeg")

print("=" * 74)
print("ENVIRONMENT   <-- paste this block into the issue")
print("=" * 74)
env = {
    "chip":   sh("system_profiler SPHardwareDataType 2>/dev/null | awk -F': ' '/Chip/{print $2}'")
              or sh("sysctl -n machdep.cpu.brand_string"),
    "memory": sh("system_profiler SPHardwareDataType 2>/dev/null | awk -F': ' '/Memory/{print $2}'"),
    "macOS":  sh("sw_vers -productVersion"),
    "python": platform.python_version(),
    "node_commit": os.environ.get("H3_NODE_COMMIT", "?"),
}
try:
    import torch
    env["torch"] = torch.__version__
    env["mps_available"] = torch.backends.mps.is_available()
except Exception as e:
    env["torch"] = f"<import failed: {e}>"
try:
    env["comfyui"] = json.load(urllib.request.urlopen(API + "/system_stats"))["system"]["comfyui_version"]
except Exception as e:
    env["comfyui"] = f"<unreachable: {e}>"
for k, v in env.items():
    print(f"  {k:16s} {v}")

GRAPH = {
  "6":  {"class_type": "UNETLoader", "inputs": {
            "unet_name": os.environ.get("H3_DIT", "minimax_h3_fl2va_pruned_int8_convrot.safetensors"),
            "weight_dtype": "default"}},
  "30": {"class_type": "MiniMaxH3TurboLoRA", "inputs": {
            "model": ["6", 0], "lora_name": "minimax_h3_turbo_v4_step600_ema.safetensors",
            "strength": 1.0, "low_vram": False}},
  "13": {"class_type": "CLIPLoader", "inputs": {
            "clip_name": "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
            "type": "minimax", "device": "default"}},
  "11": {"class_type": "VAELoader", "inputs": {"vae_name": "minimax_h3_video_vae_fp16.safetensors"}},
  "24": {"class_type": "VAELoader", "inputs": {"vae_name": "minimax_h3_audio_vae_fp32.safetensors"}},
  "104": {"class_type": "MiniMaxH3ImageToVideo", "inputs": {
            "clip": ["13", 0], "vae": ["11", 0],
            "prompt": "A red fox walking through fresh snow at dawn.",
            "width": 256, "height": 256, "length": 22}},
  "16": {"class_type": "BasicGuider", "inputs": {"model": ["30", 0], "conditioning": ["104", 0]}},
  "17": {"class_type": "MiniMaxH3TurboSampler", "inputs": {}},
  "9":  {"class_type": "BasicScheduler", "inputs": {
            "model": ["30", 0], "scheduler": "simple", "steps": 6, "denoise": 1.0}},
  "15": {"class_type": "RandomNoise", "inputs": {"noise_seed": 42}},
  "14": {"class_type": "SamplerCustomAdvanced", "inputs": {
            "noise": ["15", 0], "guider": ["16", 0], "sampler": ["17", 0],
            "sigmas": ["9", 0], "latent_image": ["104", 1]}},
  "10": {"class_type": "VAEDecode", "inputs": {"samples": ["14", 0], "vae": ["11", 0]}},
  "23": {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["14", 0], "vae": ["24", 0]}},
  "91": {"class_type": "CreateVideo", "inputs": {"images": ["10", 0], "fps": 24.0, "audio": ["23", 0]}},
  "92": {"class_type": "SaveVideo", "inputs": {
            "video": ["91", 0], "filename_prefix": "video/h3_lora_test",
            "format": "auto", "codec": "auto"}},
}

if os.environ.get("H3_NOLORA") == "1":
    del GRAPH["30"]
    GRAPH["16"]["inputs"]["model"] = ["6", 0]
    GRAPH["9"]["inputs"]["model"] = ["6", 0]

print("\n" + "=" * 74)
print("RUNNING  256x256, 22 frames, 6 steps" +
      ("  NO LoRA (control)" if os.environ.get("H3_NOLORA") == "1" else "  Turbo LoRA on a pruned base"))
print("=" * 74)
print("  (first run also loads ~37 GB of weights - allow several minutes)")

req = urllib.request.Request(API + "/prompt",
        data=json.dumps({"prompt": GRAPH, "client_id": "repro"}).encode(),
        headers={"Content-Type": "application/json"})
try:
    pid = json.load(urllib.request.urlopen(req))["prompt_id"]
except urllib.error.HTTPError as e:
    print("\nSETUP ERROR (not the bug) - ComfyUI rejected the graph:")
    print(e.read().decode()[:900]); sys.exit(1)

t0 = time.time()
while time.time() - t0 < 3600:
    time.sleep(5)
    q = json.load(urllib.request.urlopen(API + "/queue"))
    if not q.get("queue_running") and not q.get("queue_pending"):
        break
elapsed = time.time() - t0

hist  = json.load(urllib.request.urlopen(f"{API}/history/{pid}")).get(pid, {})
files = [os.path.join(f.get("subfolder", ""), f["filename"])
         for o in hist.get("outputs", {}).values() for f in o.get("images", [])]

MODE = os.environ.get("H3_MODE", "baseline")
print("\n" + "=" * 74); print(f"RESULT  (mode: {MODE})"); print("=" * 74)
LOG = os.environ.get("H3_LOG") or os.path.join(os.path.dirname(COMFY.rstrip("/")), "comfyui.log")
def log_signature():
    """Distinguish the LoRA NaN bug from any other failure."""
    try:
        txt = open(LOG, errors="ignore").read()[-20000:]
    except Exception:
        return None, ""
    tail = [l for l in txt.splitlines() if l.strip()][-6:]
    if "avcodec_send_frame()" in txt and "denoised_rms=nan" in txt:
        return "NAN", "\n".join(tail)
    if "denoised_rms=nan" in txt:
        return "NAN", "\n".join(tail)
    if "avcodec_send_frame()" in txt:
        return "AVCODEC_ONLY", "\n".join(tail)
    return "OTHER", "\n".join(tail)

if not files:
    sig, tail = log_signature()
    if os.environ.get("H3_MODE") == "fix" and sig == "NAN":
        print(f"  VERDICT: *** PATCH DID NOT FIX IT *** (NaN, no output, {elapsed:.0f}s)")
        sys.exit(0)
    if sig and sig != "NAN":
        print(f"  VERDICT: *** FAILED, BUT NOT THE NaN BUG *** (sig={sig}, {elapsed:.0f}s)")
        print("  This machine is failing for a different reason. Last log lines:")
        for l in tail.splitlines():
            print("    " + l[:160])
        sys.exit(0)
    print("  --- last 25 log lines ---")
    for l in [x for x in (tail or "").splitlines()][-25:]:
        print("    " + l[:170])
    if os.environ.get("H3_NOLORA") == "1":
        print(f"  VERDICT: *** H3 ITSELF IS BROKEN HERE *** (no output, {elapsed:.0f}s)")
        print("  Plain H3 with no LoRA fails on this machine, so LoRA results are")
        print("  not interpretable. This is a different and more basic problem.")
    elif MODE == "fix":
        print(f"  VERDICT: *** PATCH DID NOT FIX IT *** (no output, {elapsed:.0f}s)")
        print("  Important negative result - please report it.")
    else:
        print(f"  VERDICT: *** BUG REPRODUCED *** (no output, {elapsed:.0f}s)")
        print("  The sampler emitted NaN; the AAC encoder then rejected the frame")
        print("  (avcodec_send_frame() returned 22). This is issue #14.")
    sys.exit(0)

path = os.path.join(COMFY, "output", files[0])
size = os.path.getsize(path) if os.path.exists(path) else 0

def probe(entries, stream=None):
    sel = f"-select_streams {stream} " if stream else ""
    return sh(f'"{FFPROBE}" -v error {sel}-show_entries {entries} -of default=noprint_wrappers=1:nokey=1 "{path}"')

vcodec = probe("stream=codec_name", "v:0")
acodec = probe("stream=codec_name", "a:0")
w      = probe("stream=width",  "v:0")
h      = probe("stream=height", "v:0")
nbf    = probe("stream=nb_frames", "v:0")
dur    = probe("format=duration")

# extract 3 frames without relying on a select filter
frames = []
for idx in (0, 5, 11):
    fp = f"/tmp/_h3f{idx}.png"
    subprocess.run(f'"{FFMPEG}" -v error -i "{path}" -vf "select=gte(n\\,{idx})" -frames:v 1 {fp} -y',
                   shell=True, capture_output=True)
    frames.append(os.path.getsize(fp) if os.path.exists(fp) else 0)

# did the sampler produce NaN?
LOG = os.environ.get("H3_LOG") or os.path.join(os.path.dirname(COMFY.rstrip("/")), "comfyui.log")
try:
    logtxt = open(LOG, errors="ignore").read()[-40000:]
except Exception:
    logtxt = ""
nan_hits = logtxt.count("denoised_rms=nan")
avc      = "avcodec_send_frame()" in logtxt

print(f"  video   {vcodec or 'NONE':6s} {w}x{h}  frames={nbf or '?'}  dur={dur or '?'}")
print(f"  audio   {acodec or 'NONE'}")
print(f"  file    {size} B")
print(f"  frames  n=0:{frames[0]}B  n=5:{frames[1]}B  n=11:{frames[2]}B   (black is ~1900 B)")
print(f"  log     denoised_rms=nan x{nan_hits}   avcodec_error={avc}")

real_frames = [f for f in frames if f > 0]
video_ok = bool(vcodec) and len(real_frames) >= 2 and max(real_frames) > 15000
audio_ok = bool(acodec)
executed = "Prompt executed" in logtxt

# The bug's signature lives in the LOG, not the file: a NaN sampler is what the
# issue is about. File checks are corroboration only, because a shadowed ffmpeg
# can fail to read a perfectly good mp4.
log_readable = bool(logtxt.strip())
if nan_hits > 0:
    sampler = "NaN"
elif executed:
    sampler = "OK"
elif not log_readable and video_ok and audio_ok:
    sampler = "OK"          # no log, but the output is demonstrably good
elif not log_readable and not (video_ok or audio_ok):
    sampler = "NaN"         # no log, and nothing usable came out
else:
    sampler = "UNKNOWN"
if not log_readable:
    print(f"  (note: could not read {LOG} - judging from output file only)")

print()
print(f"  sampler   {sampler}   (denoised_rms=nan x{nan_hits}, prompt_executed={executed})")
print(f"  file      video_ok={video_ok} audio_ok={audio_ok}"
      + ("" if (video_ok and audio_ok) else "   <- if sampler is OK this is likely an ffmpeg problem, not H3"))

NOLORA = os.environ.get("H3_NOLORA") == "1"
print()
if NOLORA:
    if sampler == "OK":
        print(f"  VERDICT: plain H3 SAMPLES CORRECTLY here ({elapsed:.0f}s)")
        print("  -> LoRA results on this machine ARE interpretable.")
    elif sampler == "NaN":
        print(f"  VERDICT: plain H3 emits NaN even without any LoRA ({elapsed:.0f}s)")
        print("  -> a different, more basic problem.")
    else:
        print(f"  VERDICT: inconclusive - the render did not complete ({elapsed:.0f}s)")
elif MODE == "fix":
    if sampler == "OK":
        print(f"  VERDICT: *** PATCH FIXES IT *** ({elapsed:.0f}s)   no NaN in {nan_hits} steps")
    elif sampler == "NaN":
        print(f"  VERDICT: *** PATCH DID NOT FIX IT *** ({elapsed:.0f}s)   NaN x{nan_hits}")
    else:
        print(f"  VERDICT: inconclusive ({elapsed:.0f}s)")
else:
    if sampler == "NaN":
        print(f"  VERDICT: *** BUG REPRODUCED *** ({elapsed:.0f}s)   NaN x{nan_hits}")
    elif sampler == "OK":
        print(f"  VERDICT: NOT REPRODUCED - sampler is clean here ({elapsed:.0f}s)")
    else:
        print(f"  VERDICT: inconclusive ({elapsed:.0f}s)")
PYEOF

# ------------------------------------------------------------------- 6. test
if [[ "$MODE" == "candidates" ]]; then
  step "5b/6  Testing every candidate fix"
  # Any earlier run may have left the node patched. Re-clone it outright so the
  # starting state is unambiguous - it is a tiny repo.
  ok "re-cloning the node for a clean baseline"
  rm -rf "$NODE_DIR"
  git clone --depth 1 https://github.com/Larryvrh/ComfyUI-MiniMax-H3-Turbo "$NODE_DIR" >/dev/null 2>&1 \
    || die "re-clone failed"
  NODE_COMMIT="$(git -C "$NODE_DIR" rev-parse --short HEAD)"
  ok "node commit $NODE_COMMIT"
  if grep -q "av = a.to(x.device, x.dtype)" "$NODE_DIR/__init__.py"; then
    ok "anchor found - node is in the expected state"
  else
    die "anchor missing in a fresh clone - upstream changed this code; please report that at issue #14"
  fi
  ORIG="$NODE_DIR/__init__.py.candidates_orig"
  cp "$NODE_DIR/__init__.py" "$ORIG"
  SUMMARY=""
  for CAND in none reduce_delta reduce_result bf16_native; do
    printf "\n\033[1;35m--- candidate: %s ---\033[0m\n" "$CAND"
    cp "$ORIG" "$NODE_DIR/__init__.py"
    H3_CAND="$CAND" python3 - "$NODE_DIR/__init__.py" <<'PYCAND'
import os, sys
P = sys.argv[1]; src = open(P).read()
OLD = """            av = a.to(x.device, x.dtype)
            bv = b.to(x.device, x.dtype)
            sv = st.to(x.device, x.dtype)
            x = x + (bv @ (av @ sv.T)).T                              # [M, out]"""
HEAD = """            av = a.to(x.device, x.dtype)
            bv = b.to(x.device, x.dtype)
            sv = st.to(x.device, x.dtype)
"""
BODY = {
 "none":          OLD,
 "reduce_delta":  HEAD + "            _d = (bv @ (av @ sv.T)).T\n            _s = _d.sum()\n            x = x + _d",
 "reduce_result": HEAD + "            x = x + (bv @ (av @ sv.T)).T\n            _s = x.sum()",
 "bf16_native":   """            cd = a.dtype if (x.device.type == "mps" and a.dtype.is_floating_point) else x.dtype
            av = a.to(x.device, cd)
            bv = b.to(x.device, cd)
            sv = st.to(x.device, cd)
            x = x + (bv @ (av @ sv.T)).T.to(x.dtype)""",
}[os.environ["H3_CAND"]]
if OLD not in src:
    print("    ERROR: anchor not found"); sys.exit(1)
open(P, "w").write(src.replace(OLD, BODY))
print("    applied")
PYCAND
    pkill -f "main.py --listen 127.0.0.1 --port $PORT" 2>/dev/null || true
    sleep 5
    ( cd "$COMFY" && nohup python main.py --listen 127.0.0.1 --port "$PORT" > "$ROOT/comfyui.log" 2>&1 & )
    printf "    waiting for server "
    for _ in $(seq 1 150); do sleep 3; printf "."
      curl -s "http://127.0.0.1:$PORT/system_stats" >/dev/null 2>&1 && break; done; echo
    OUT=$(H3_COMFY_ROOT="$COMFY" H3_PORT="$PORT" H3_NODE_COMMIT="$NODE_COMMIT" H3_MODE="fix" H3_DIT="$DIT_FILE" H3_FFPROBE="$FFPROBE" H3_FFMPEG="$FFMPEG" H3_LOG="$ROOT/comfyui.log" \
          python "$ROOT/_run_test.py" 2>&1 | tail -25)
    echo "$OUT" | grep -E "VERDICT" || echo "    (no verdict line)"
    V=$(echo "$OUT" | grep -oE "PATCH FIXES IT|PATCH DID NOT FIX IT|BUG REPRODUCED" | head -1)
    [[ "$CAND" == "none" ]] && V="${V/PATCH DID NOT FIX IT/BUG REPRODUCED (expected - control OK)}"
    SUMMARY="$SUMMARY\n  $(printf '%-16s' "$CAND") ${V:-unknown}"
  done
  cp "$ORIG" "$NODE_DIR/__init__.py"
  pkill -f "main.py --listen 127.0.0.1 --port $PORT" 2>/dev/null || true
  printf "\n=============================================================================\n"
  printf "CANDIDATE SUMMARY  (node restored to original)\n"
  printf "  'none' is the unpatched control and SHOULD say BUG REPRODUCED\n"
  printf "$SUMMARY\n"
  printf "\nPlease paste this summary plus the ENVIRONMENT block into issue #14.\n"
  printf "=============================================================================\n"
  exit 0
fi

if [[ "$MODE" == "fix" ]]; then
  step "5b/6  Applying the proposed patch"
  if [[ -f "$NODE_DIR/__init__.py.prepatch" ]]; then
    mv "$NODE_DIR/__init__.py.prepatch" "$NODE_DIR/__init__.py"; ok "reset to unpatched first"
  fi
  python3 - "$NODE_DIR/__init__.py" <<'PYPATCH'
import sys
P = sys.argv[1]
src = open(P).read()
OLD = """            av = a.to(x.device, x.dtype)
            bv = b.to(x.device, x.dtype)
            sv = st.to(x.device, x.dtype)
            x = x + (bv @ (av @ sv.T)).T                              # [M, out]"""
NEW = """            cd = a.dtype if (x.device.type == "mps" and a.dtype.is_floating_point) else x.dtype
            av = a.to(x.device, cd)
            bv = b.to(x.device, cd)
            sv = st.to(x.device, cd)
            x = x + (bv @ (av @ sv.T)).T.to(x.dtype)                  # [M, out]"""
if NEW.splitlines()[0].strip() in src:
    print("    already patched"); sys.exit(0)
if OLD not in src:
    print("    ERROR: anchor not found - node version differs"); sys.exit(1)
open(P + ".prepatch", "w").write(src)
open(P, "w").write(src.replace(OLD, NEW))
print("    patched (original saved as __init__.py.prepatch)")
PYPATCH
  # the node is only read at startup, so a running server must be restarted
  pkill -f "main.py --listen 127.0.0.1 --port $PORT" 2>/dev/null || true
  sleep 5
fi

step "6/6  Starting ComfyUI and running the test"
STARTED=0
# Custom nodes are only loaded at startup, so never reuse a server that may
# predate a node we just cloned. Always restart.
if curl -s "http://127.0.0.1:$PORT/system_stats" >/dev/null 2>&1; then
  ok "restarting the existing server so custom nodes are (re)loaded"
  pkill -f "main.py --listen 127.0.0.1 --port $PORT" 2>/dev/null || true
  sleep 6
fi
if false; then :
else
  ( cd "$COMFY" && nohup python main.py --listen 127.0.0.1 --port "$PORT" \
      > "$ROOT/comfyui.log" 2>&1 & )
  STARTED=1
  printf "    waiting for server "
  for _ in $(seq 1 150); do
    sleep 3; printf "."
    curl -s "http://127.0.0.1:$PORT/system_stats" >/dev/null 2>&1 && break
  done; echo
  curl -s "http://127.0.0.1:$PORT/system_stats" >/dev/null 2>&1 \
    || die "ComfyUI did not start - see $ROOT/comfyui.log"
  ok "server up"
fi
if grep -q "AppleSilicon-FP8. capabilities" "$ROOT/comfyui.log" 2>/dev/null; then
  ok "AppleSilicon-FP8 active: $(grep -o 'capabilities:.*' "$ROOT/comfyui.log" | tail -1 | cut -c1-90)"
else
  warn "AppleSilicon-FP8 did NOT load - INT8 models will fail with aten::_int_mm"
fi


NOLORA=0; [[ "$MODE" == "nolora" ]] && NOLORA=1
H3_COMFY_ROOT="$COMFY" H3_PORT="$PORT" H3_NODE_COMMIT="$NODE_COMMIT" H3_MODE="$MODE" H3_NOLORA="$NOLORA" H3_DIT="$DIT_FILE" H3_FFPROBE="$FFPROBE" H3_FFMPEG="$FFMPEG" H3_LOG="$ROOT/comfyui.log" \
  python "$ROOT/_run_test.py" || true

cat <<EOF

=============================================================================
Please report the ENVIRONMENT block plus the VERDICT at:
  https://github.com/Larryvrh/ComfyUI-MiniMax-H3-Turbo/issues/14

Files live in $ROOT  ->  remove everything with:  rm -rf $ROOT
EOF
[[ "$STARTED" == "1" ]] && echo "ComfyUI still running. Stop it with: pkill -f 'main.py --listen'"
exit 0
