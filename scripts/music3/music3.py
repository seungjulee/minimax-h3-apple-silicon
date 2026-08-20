import json, sys, time, urllib.request

API = "http://127.0.0.1:8188"
DIT = "minimax_music3_dit_fp16.safetensors"
CLIP = "minimax_music3_text_encoder_bf16.safetensors"
VAE = "minimax_music3_dav.safetensors"

def build(tag, caption, lyrics, seconds=30.0, seed=42, steps=30, cfg=1.7, top_k=50):
    return {
      "6":  {"class_type": "UNETLoader", "inputs": {"unet_name": DIT, "weight_dtype": "default"}},
      "3":  {"class_type": "CLIPLoader", "inputs": {"clip_name": CLIP, "type": "minimax", "device": "default"}},
      "7":  {"class_type": "VAELoader", "inputs": {"vae_name": VAE}},
      "13": {"class_type": "MiniMaxMusic3TextEncode", "inputs": {
                "clip": ["3", 0], "caption": caption, "lyrics": lyrics,
                "seed": seed, "max_duration": seconds, "cfg_scale": cfg, "top_k": top_k}},
      "10": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["13", 0]}},
      "15": {"class_type": "EmptyMiniMaxMusic3LatentAudio", "inputs": {"seconds": seconds, "batch_size": 1}},
      "9":  {"class_type": "KSampler", "inputs": {
                "model": ["6", 0], "positive": ["13", 0], "negative": ["10", 0],
                "latent_image": ["15", 0], "seed": seed, "steps": steps, "cfg": cfg,
                "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0}},
      "12": {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["9", 0], "vae": ["7", 0]}},
      "35": {"class_type": "SaveAudioAdvanced", "inputs": {
                "audio": ["12", 0], "filename_prefix": f"audio/{tag}",
                "format": "mp3", "format.quality": "V0"}},
    }

def run(tag, caption, lyrics, seconds=30.0, seed=42):
    p = build(tag, caption, lyrics, seconds, seed)
    t0 = time.time()
    req = urllib.request.Request(API+"/prompt", data=json.dumps({"prompt":p,"client_id":"music3"}).encode(),
                                  headers={"Content-Type":"application/json"})
    try:
        pid = json.load(urllib.request.urlopen(req))["prompt_id"]
    except urllib.error.HTTPError as e:
        print("FAIL-QUEUE", tag, e.read().decode()[:1200]); return None
    while True:
        time.sleep(5)
        q = json.load(urllib.request.urlopen(API+"/queue"))
        if not q.get("queue_running") and not q.get("queue_pending"): break
    el = time.time()-t0
    h = json.load(urllib.request.urlopen(f"{API}/history/{pid}")).get(pid,{})
    status = h.get("status",{}).get("status_str","?")
    files = []
    for o in h.get("outputs",{}).values():
        for a in o.get("audio", o.get("images", [])):
            files.append(a["filename"])
    print(f"DONE {tag} -> {status} {el:.0f}s {files}")
    return files[0] if files else None
