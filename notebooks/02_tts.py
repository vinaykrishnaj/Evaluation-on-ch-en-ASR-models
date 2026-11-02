# Notebook 2: synthesize Chinese speech from zh_ref with Edge-TTS (needs Internet on).
# Attach the zh-en-manifest dataset as input. Paste each "# %%" block into its own cell.

# %% Cell 1: install
%pip install -q edge-tts

# %% Cell 2: load manifest
import os, json, glob, asyncio, subprocess, hashlib, time, shutil
import edge_tts

MANIFEST = glob.glob("/kaggle/input/**/manifest.jsonl", recursive=True)[0]
OUT = "/kaggle/working/tts"
os.makedirs(OUT, exist_ok=True)
rows = [json.loads(l) for l in open(MANIFEST, encoding="utf-8")]
print(MANIFEST, len(rows))

# several voices so the ASR sees more than one speaker
VOICES = ["zh-CN-XiaoxiaoNeural", "zh-CN-YunxiNeural", "zh-CN-YunjianNeural", "zh-CN-XiaoyiNeural"]

def pick_voice(uid):
    return VOICES[int(hashlib.md5(uid.encode()).hexdigest(), 16) % len(VOICES)]

async def synth(text, voice, wav_path):
    mp3 = wav_path[:-4] + ".mp3"
    await edge_tts.Communicate(text, voice).save(mp3)
    # 16 kHz mono wav, the usual ASR input format
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", mp3, "-ar", "16000", "-ac", "1", wav_path], check=True)
    os.remove(mp3)

# %% Cell 3: smoke test on 5 rows (stop here if this fails)
for r in rows[:5]:
    v = pick_voice(r["id"])
    await synth(r["zh_ref"], v, f"{OUT}/{r['id']}.wav")
    print(r["id"], v, os.path.getsize(f"{OUT}/{r['id']}.wav"), "bytes")

# %% Cell 4: full run (resumable: skips files that already exist)
sem = asyncio.Semaphore(8)
failed = {}

async def worker(r):
    path = f"{OUT}/{r['id']}.wav"
    if os.path.exists(path):
        return
    async with sem:
        for attempt in range(3):
            try:
                await synth(r["zh_ref"], pick_voice(r["id"]), path)
                return
            except Exception as e:
                err = repr(e)[:150]
                await asyncio.sleep(2 * (attempt + 1))
        failed[r["id"]] = err

t0 = time.time()
await asyncio.gather(*(worker(r) for r in rows))
print("done in", round(time.time() - t0), "s; failed:", len(failed))
print(list(failed.items())[:5])

# %% Cell 5: write manifest_tts.jsonl and package (zip keeps the output file count small)
ok = []
for r in rows:
    if os.path.exists(f"{OUT}/{r['id']}.wav"):
        ok.append(dict(r, tts_audio=f"tts/{r['id']}.wav", tts_voice=pick_voice(r["id"])))
open("/kaggle/working/manifest_tts.jsonl", "w", encoding="utf-8").write("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in ok))
shutil.make_archive("/kaggle/working/tts_audio", "zip", "/kaggle/working", "tts")
shutil.rmtree(OUT)
print(len(ok), "of", len(rows), os.listdir("/kaggle/working"))
