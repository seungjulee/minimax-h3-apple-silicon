import json, sys, time, urllib.request

API = "http://127.0.0.1:8188"
DIT = "ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors"
VIDEO_VAE = "ltx-2.5-video-vae-bf16.safetensors"
AUDIO_VAE = "ltx-2.5-audio-vae-bf16.safetensors"
TEXT_ENC = "gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors"
UPSCALER = "ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors"
NEGATIVE = "pc game, console game, video game, cartoon, childish, ugly"

STAGE1_SIGMAS = "1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0"
STAGE2_SIGMAS = "0.85, 0.7250, 0.4219, 0.0"

def build(tag, prompt, final_w=512, final_h=288, seconds=2.0, fps=24.0, seed=777):
    base_w, base_h = final_w // 2, final_h // 2
    frames = int(seconds * fps) + 1  # must be 8k+1

    g = {
      "6":  {"class_type": "UNETLoader", "inputs": {"unet_name": DIT, "weight_dtype": "default"}},
      "385": {"class_type": "VAELoader", "inputs": {"vae_name": VIDEO_VAE}},
      "386": {"class_type": "VAELoader", "inputs": {"vae_name": AUDIO_VAE}},
      "387": {"class_type": "CLIPLoader", "inputs": {"clip_name": TEXT_ENC, "type": "ltxv", "device": "default"}},
      "371": {"class_type": "LatentUpscaleModelLoader", "inputs": {"model_name": UPSCALER}},

      "364": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["387", 0], "text": prompt}},
      "373": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["387", 0], "text": NEGATIVE}},
      "365": {"class_type": "LTXVConditioning", "inputs": {
                "positive": ["364", 0], "negative": ["373", 0], "frame_rate": fps}},

      "356": {"class_type": "EmptyLTXVLatentVideo", "inputs": {
                "width": base_w, "height": base_h, "length": frames, "batch_size": 1}},
      "366": {"class_type": "LTXVEmptyLatentAudio", "inputs": {
                "frames_number": frames, "frame_rate": fps, "batch_size": 1, "audio_vae": ["386", 0]}},
      "377": {"class_type": "LTXVConcatAVLatent", "inputs": {
                "video_latent": ["356", 0], "audio_latent": ["366", 0]}},

      # stage 1: base-resolution sampling
      "339": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
      "388": {"class_type": "LTXVDualCFGGuider", "inputs": {
                "model": ["6", 0], "positive": ["365", 0], "negative": ["365", 1],
                "video_cfg": 1.0, "audio_cfg": 1.0}},
      "352": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler_ancestral"}},
      "404": {"class_type": "ManualSigmas", "inputs": {"sigmas": STAGE1_SIGMAS}},
      "344": {"class_type": "SamplerCustomAdvanced", "inputs": {
                "noise": ["339", 0], "guider": ["388", 0], "sampler": ["352", 0],
                "sigmas": ["404", 0], "latent_image": ["377", 0]}},
      "367": {"class_type": "LTXVSeparateAVLatent", "inputs": {"av_latent": ["344", 0]}},

      # latent upscale x2
      "348": {"class_type": "LTXVLatentUpsampler", "inputs": {
                "samples": ["367", 0], "upscale_model": ["371", 0], "vae": ["385", 0]}},
      "340": {"class_type": "LTXVConcatAVLatent", "inputs": {
                "video_latent": ["348", 0], "audio_latent": ["367", 1]}},

      # stage 2: upscaled-resolution sampling
      "338": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
      "391": {"class_type": "LTXVDualCFGGuider", "inputs": {
                "model": ["6", 0], "positive": ["365", 0], "negative": ["365", 1],
                "video_cfg": 1.0, "audio_cfg": 1.0}},
      "341": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler_ancestral"}},
      "395": {"class_type": "ManualSigmas", "inputs": {"sigmas": STAGE2_SIGMAS}},
      "368": {"class_type": "SamplerCustomAdvanced", "inputs": {
                "noise": ["338", 0], "guider": ["391", 0], "sampler": ["341", 0],
                "sigmas": ["395", 0], "latent_image": ["340", 0]}},
      "369": {"class_type": "LTXVSeparateAVLatent", "inputs": {"av_latent": ["368", 0]}},

      # decode
      "374": {"class_type": "VAEDecodeTiled", "inputs": {
                "samples": ["369", 0], "vae": ["385", 0],
                "tile_size": 512, "overlap": 64, "temporal_size": 64, "temporal_overlap": 16}},
      "358": {"class_type": "LTXVAudioVAEDecode", "inputs": {
                "samples": ["369", 1], "audio_vae": ["386", 0]}},
      "370": {"class_type": "CreateVideo", "inputs": {
                "images": ["374", 0], "fps": fps, "audio": ["358", 0]}},
      "75":  {"class_type": "SaveVideo", "inputs": {
                "video": ["370", 0], "filename_prefix": f"video/{tag}",
                "format": "auto", "codec": "auto"}},
    }
    return g

def run(tag, prompt, **kw):
    p = build(tag, prompt, **kw)
    t0 = time.time()
    req = urllib.request.Request(API+"/prompt", data=json.dumps({"prompt":p,"client_id":"ltx25"}).encode(),
                                  headers={"Content-Type":"application/json"})
    try:
        pid = json.load(urllib.request.urlopen(req))["prompt_id"]
    except urllib.error.HTTPError as e:
        print("FAIL-QUEUE", tag, e.read().decode()[:2000]); return None
    while True:
        time.sleep(5)
        q = json.load(urllib.request.urlopen(API+"/queue"))
        if not q.get("queue_running") and not q.get("queue_pending"): break
    el = time.time()-t0
    h = json.load(urllib.request.urlopen(f"{API}/history/{pid}")).get(pid,{})
    status = h.get("status",{}).get("status_str","?")
    files = []
    for o in h.get("outputs",{}).values():
        for a in o.get("images", o.get("audio", o.get("videos", []))):
            files.append(a["filename"])
        for a in o.get("video", []):
            files.append(a.get("filename", a))
    print(f"DONE {tag} -> {status} {el:.0f}s outputs={h.get('outputs')}")
    return files[0] if files else None

if __name__ == "__main__":
    sys.path.insert(0, "/Users/sj/code/h3_scratch")
    run("ltx25_sanity", "A red fox walking through fresh snow in a quiet pine forest at dawn, soft natural light, breath visible in the cold air.")
